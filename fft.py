import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from Heff import a1, a2
from chains import dt  # Asegúrate de tener definido dt o impórtalo de tu archivo de parámetros
# Asegúrate de tener definido dt o impórtalo de tu archivo de parámetros
# from chains import dt 
# ===================================================================
# 0. CONFIGURACIÓN MÍNIMA
# ===================================================================
# Parámetros geométricos reales de los sitios individuales

# Parámetros temporales
OMEGA_MAX = 2.5e14  # rad/s

LAB_COMPONENTS = (0, 1, 2)
APPLY_DEMEAN = True
APPLY_HANN = True
USE_RFFT_TIME = True

# --- NUEVA OPCIÓN ---
PLOT_INDIVIDUALS = True  # Cambia a False si solo quieres ver el compuesto RGB

# Archivos
file_fluc_1 = 'spin_history_cadena1 fluc MC = 4 Jperp=0.4.npy'
file_fluc_2 = 'spin_history_cadena2 fluc MC = 4 Jperp=0.4.npy'
file_rel_1  = 'spin_history_cadena1 relax MC = 4 Jperp=0.4.npy'
file_rel_2  = 'spin_history_cadena2 relax MC = 4 Jperp=0.4.npy'

# ===================================================================
# 1. FUNCIÓN DE CÁLCULO DE POTENCIA (BASE COMPLETA)
# ===================================================================
def compute_power_full_chain(
    spin_lab, gs_lab, components=(0, 1, 2),
    demean=True, hann=True, use_rfft_time=True,
):
    num_pasos = spin_lab.shape[0]
    n_sites = spin_lab.shape[1]
    
    window_t = None
    if hann:
        window_t = np.hanning(num_pasos).astype(np.float32)[:, np.newaxis]

    n_omega = (num_pasos // 2 + 1) if use_rfft_time else num_pasos
    power = np.zeros((n_omega, n_sites), dtype=np.float64)

    for comp in components:
        x = np.array(spin_lab[:, :, comp], dtype=np.float32, copy=True)
        x -= gs_lab[np.newaxis, :, comp].astype(np.float32, copy=False)

        if demean:
            x -= np.mean(x, axis=0, keepdims=True)
        if window_t is not None:
            x *= window_t

        if use_rfft_time:
            fft_x = np.fft.rfft(x, axis=0)           
            fft_x = np.fft.fft(fft_x, axis=1)        
            fft_x = np.fft.fftshift(fft_x, axes=(1,))
        else:
            fft_x = np.fft.fftshift(np.fft.fftn(x, axes=(0, 1)), axes=(0, 1))

        power += (fft_x.real * fft_x.real + fft_x.imag * fft_x.imag)
        del x, fft_x
        gc.collect()

    return power

# ===================================================================
# 2. CARGA Y CÁLCULO PARA AMBAS CADENAS
# ===================================================================
def load_and_compute(file_fluc, file_rel):
    print(f"-> Procesando {file_fluc}...")
    hist = np.load(file_fluc, mmap_mode='r')
    
    rel_data = np.load(file_rel, mmap_mode='r')
    gs = rel_data[-1] if rel_data.ndim == 3 else rel_data

    power = compute_power_full_chain(
        hist, gs, components=LAB_COMPONENTS,
        demean=APPLY_DEMEAN, hann=APPLY_HANN, use_rfft_time=USE_RFFT_TIME
    )
    
    num_pasos = hist.shape[0]
    n_spins = hist.shape[1]
    
    del hist, gs, rel_data
    gc.collect()
    return power, num_pasos, n_spins

power_1, num_pasos, n_spins_1 = load_and_compute(file_fluc_1, file_rel_1)
power_2, _, n_spins_2         = load_and_compute(file_fluc_2, file_rel_2)

# Convertir a escala logarítmica
log_mag_1 = np.log10(power_1 + 1e-12)
log_mag_2 = np.log10(power_2 + 1e-12)

# ===================================================================
# 3. CONSTRUCCIÓN DE EJES FÍSICOS E INTERPOLACIÓN (RGB)
# ===================================================================
if USE_RFFT_TIME:
    omega_values = np.fft.rfftfreq(num_pasos, d=dt) * 2 * np.pi
else:
    omega_values = np.fft.fftshift(np.fft.fftfreq(num_pasos, d=dt) * 2 * np.pi)

mask_w = (omega_values <= OMEGA_MAX) if USE_RFFT_TIME else (np.abs(omega_values) <= OMEGA_MAX)
omega_plot = omega_values[mask_w]

log_mag_1 = log_mag_1[mask_w, :]
log_mag_2 = log_mag_2[mask_w, :]

k_vals_1 = np.fft.fftshift(np.fft.fftfreq(n_spins_1, d=a1)) * 2 * np.pi
k_vals_2 = np.fft.fftshift(np.fft.fftfreq(n_spins_2, d=a2)) * 2 * np.pi

k_max_abs = max(np.pi/a1, np.pi/a2) * 1.05
k_common = np.linspace(-k_max_abs, k_max_abs, 800)

S1_interp = np.zeros((len(omega_plot), len(k_common)))
S2_interp = np.zeros((len(omega_plot), len(k_common)))

print("Interpolando espectros a malla K común para composición RGB...")
for i in range(len(omega_plot)):
    f1 = interp1d(k_vals_1, log_mag_1[i, :], kind='linear', bounds_error=False, fill_value=0.0)
    S1_interp[i, :] = f1(k_common)
    
    f2 = interp1d(k_vals_2, log_mag_2[i, :], kind='linear', bounds_error=False, fill_value=0.0)
    S2_interp[i, :] = f2(k_common)

# ===================================================================
# 4. NORMALIZACIÓN Y COMPOSICIÓN DEL COLORMAP
# ===================================================================
def normalize_array(arr, p_low=4, p_high=99.4):
    valid = arr[arr > np.min(arr)]
    if len(valid) == 0: return np.zeros_like(arr)
    vmin = np.percentile(valid, p_low)
    vmax = np.percentile(valid, p_high)
    norm = (arr - vmin) / (vmax - vmin)
    return np.clip(norm, 0, 1)

norm_1 = normalize_array(S1_interp)
norm_2 = normalize_array(S2_interp)

rgb_image = np.zeros((len(omega_plot), len(k_common), 3), dtype=np.float32)
rgb_image[:, :, 0] = norm_1  # ROJO: Cadena 1
rgb_image[:, :, 2] = norm_2  # AZUL: Cadena 2
rgb_image = np.maximum(rgb_image, 0.05)  

# ===================================================================
# 5. RENDERIZADO VISUAL
# ===================================================================
print("Generando gráficos...")

# ---------------------------------------------------------
# OPCIONAL: GRÁFICOS INDIVIDUALES
# ---------------------------------------------------------
if PLOT_INDIVIDUALS:
    # --- Gráfico Cadena 1 ---
    plt.figure(figsize=(10, 6))
    K1, W1 = np.meshgrid(k_vals_1, omega_plot)
    vmin1 = float(np.percentile(log_mag_1, 4))
    vmax1 = float(np.percentile(log_mag_1, 99.4))
    mesh1 = plt.pcolormesh(K1, W1, log_mag_1, cmap='plasma', vmin=vmin1, vmax=vmax1, shading='nearest')
    plt.colorbar(mesh1, label=r'$\log_{10} S_1(k, \omega)$')
    plt.axvline(-np.pi/a1, color='white', linestyle='--', alpha=0.6, label=r'BZ ($-\pi/a_1$)')
    plt.axvline( np.pi/a1, color='white', linestyle='--', alpha=0.6)
    plt.xlabel(r'$k$ $[rad/m]$')
    plt.ylabel(r'$\omega$ $[rad/s]$')
    plt.title(f'Espectro Aislado: Cadena 1 ($a_1 = {a1}$)')
    plt.xlim(-k_max_abs, k_max_abs)
    plt.ylim(0, OMEGA_MAX)
    plt.legend(loc='upper right')
    plt.tight_layout()

    # --- Gráfico Cadena 2 ---
    plt.figure(figsize=(10, 6))
    K2, W2 = np.meshgrid(k_vals_2, omega_plot)
    vmin2 = float(np.percentile(log_mag_2, 4))
    vmax2 = float(np.percentile(log_mag_2, 99.4))
    mesh2 = plt.pcolormesh(K2, W2, log_mag_2, cmap='plasma', vmin=vmin2, vmax=vmax2, shading='nearest')
    plt.colorbar(mesh2, label=r'$\log_{10} S_2(k, \omega)$')
    plt.axvline(-np.pi/a2, color='white', linestyle='-.', alpha=0.6, label=r'BZ ($-\pi/a_2$)')
    plt.axvline( np.pi/a2, color='white', linestyle='-.', alpha=0.6)
    plt.xlabel(r'$k$ $[rad/m]$')
    plt.ylabel(r'$\omega$ $[rad/s]$')
    plt.title(f'Espectro Aislado: Cadena 2 ($a_2 = {a2}$)')
    plt.xlim(-k_max_abs, k_max_abs)
    plt.ylim(0, OMEGA_MAX)
    plt.legend(loc='upper right')
    plt.tight_layout()

# ---------------------------------------------------------
# GRÁFICO COMPUESTO RGB
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))
extent = (k_common[0], k_common[-1], omega_plot[0], omega_plot[-1])
plt.imshow(rgb_image, origin='lower', aspect='auto', extent=extent, interpolation='bilinear')

plt.axvline(-np.pi/a1, color='red', linestyle='--', alpha=0.6, label=r'BZ 1 ($-\pi/a_1$)')
plt.axvline( np.pi/a1, color='red', linestyle='--', alpha=0.6)
plt.axvline(-np.pi/a2, color='dodgerblue', linestyle='-.', alpha=0.6, label=r'BZ 2 ($-\pi/a_2$)')
plt.axvline( np.pi/a2, color='dodgerblue', linestyle='-.', alpha=0.6)

red_patch = mpatches.Patch(color='red', label=r'Cadena 1 ($a_1$)')
blue_patch = mpatches.Patch(color='blue', label=r'Cadena 2 ($a_2$)')
purple_patch = mpatches.Patch(color='purple', label='Hibridación Intercadena')
handles, labels = plt.gca().get_legend_handles_labels()
handles.extend([red_patch, blue_patch, purple_patch])

plt.legend(handles=handles, loc='upper right', framealpha=0.9, fontsize=10)
plt.xlabel(r'$k$ (Momento físico absoluto) $[rad/m]$', fontsize=12)
plt.ylabel(r'$\omega$ Frecuencia $[rad/s]$', fontsize=12)
plt.title('Espectro Compuesto Moiré - Zona Extendida', fontsize=14)
plt.ylim(0, OMEGA_MAX)
plt.xlim(-k_max_abs, k_max_abs)

plt.tight_layout()
plt.show()