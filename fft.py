import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches

# ===================================================================
# 1. FUNCIÓN DE CÁLCULO DE POTENCIA (BASE COMPLETA)
# ===================================================================
def compute_power_full_chain(spin_lab, gs_lab, components=(0, 1),
                             demean=False, hann_temporal=True, hann_spatial=False,
                             use_rfft_time=True, is_pbc=False, crop_margin=0,
                             pad_factor=1, spatial_resolved=True):
    """
    Calcula el espectro de potencia 2D S(k, w) o la densidad de estados integrada.
    """
    num_pasos = spin_lab.shape[0]
    n_sites_total = spin_lab.shape[1]
    n_fft = num_pasos * pad_factor

    if not is_pbc:
        margin = int(n_sites_total * crop_margin)
        spatial_slice = slice(None) if margin == 0 else slice(margin, -margin)
        n_sites_active = n_sites_total - 2 * margin
    else:
        spatial_slice = slice(None)
        n_sites_active = n_sites_total

    window_t = np.hanning(num_pasos).astype(np.float64)[:, np.newaxis] if hann_temporal else None
    window_x = np.hanning(n_sites_active).astype(np.float64)[np.newaxis, :] if hann_spatial else None

    n_omega = (n_fft // 2 + 1) if use_rfft_time else n_fft
    power = np.zeros((n_omega, n_sites_active), dtype=np.float64) if spatial_resolved \
            else np.zeros(n_omega, dtype=np.float64)

    for comp in components:
        x = np.array(spin_lab[:, spatial_slice, comp], dtype=np.float64, copy=True)
        x -= gs_lab[np.newaxis, spatial_slice, comp].astype(np.float64, copy=False)
        if demean:
            x -= np.mean(x, axis=0, keepdims=True)
        if window_t is not None:
            x *= window_t
        if window_x is not None:
            x *= window_x

        if use_rfft_time:
            fft_x = np.fft.rfft(x, n=n_fft, axis=0)
        else:
            fft_x = np.fft.fftn(x, s=(n_fft, x.shape[1]), axes=(0, 1))

        if spatial_resolved:
            fft_x = np.fft.fftshift(np.fft.fft(fft_x, axis=1), axes=(1,)) if use_rfft_time else np.fft.fftshift(fft_x, axes=(0, 1))
            power += (fft_x.real**2 + fft_x.imag**2)
        else:
            power += np.sum(fft_x.real**2 + fft_x.imag**2, axis=1)
        del x, fft_x

    return power, n_fft


# ===================================================================
# 2. FUNCIÓN DE GRAFICACIÓN EN LA ZONA DE BRILLOUIN MOIRÉ
# ===================================================================
def plot_moire_spectrum(power_1, power_2, n_sites_1, n_sites_2, num_pasos, dt, a1, a2, 
                        gamma=1.7e11, J0_field=1.0, E_MAX=1.8, 
                        USE_RFFT_TIME=True, PLOT_INDIVIDUALS=False):
    print("-> Generando visualización espectral sincronizada con diag.py...")

    n_sites_1 = power_1.shape[1]
    n_sites_2 = power_2.shape[1]
    omega_J = gamma * J0_field

    log_mag_1 = np.log10(power_1 + 1e-12)
    log_mag_2 = np.log10(power_2 + 1e-12)

    # 1. Eje de frecuencias y conversión a energía adimensional hbar*w / J0
    if USE_RFFT_TIME:
        omega_values = np.fft.rfftfreq(num_pasos, d=dt) * 2.0 * np.pi
    else:
        omega_values = np.fft.fftshift(np.fft.fftfreq(num_pasos, d=dt) * 2.0 * np.pi)

    energy_values = omega_values / omega_J
    mask_e = (energy_values <= E_MAX) if USE_RFFT_TIME else (np.abs(energy_values) <= E_MAX)
    energy_plot = energy_values[mask_e]

    log_mag_1 = log_mag_1[mask_e, :]
    log_mag_2 = log_mag_2[mask_e, :]

    # 2. Ejes K físicos (paso atómico real d = a / 2)
    k_vals_1 = np.fft.fftshift(np.fft.fftfreq(n_sites_1, d=a1 / 2.0)) * 2.0 * np.pi
    k_vals_2 = np.fft.fftshift(np.fft.fftfreq(n_sites_2, d=a2 / 2.0)) * 2.0 * np.pi

    # Cuasimomento en la primera zona de Brillouin moiré.
    moire_reciprocal = 2.0 * np.pi * abs(1.0 / a1 - 1.0 / a2)
    k_max_abs = moire_reciprocal / 2.0
    k_common = np.linspace(-k_max_abs, k_max_abs, 1000)

    # 3. Pliegue periódico de los momentos físicos a la mBZ e interpolación.
    min_val_1 = np.min(log_mag_1)
    min_val_2 = np.min(log_mag_2)

    f1 = interp1d(k_vals_1, log_mag_1, axis=1, kind='linear', bounds_error=False, fill_value=min_val_1)
    f2 = interp1d(k_vals_2, log_mag_2, axis=1, kind='linear', bounds_error=False, fill_value=min_val_2)

    def fold_to_moire_bz(interpolator, k_values, fill_value):
        n_shifts = int(np.ceil(np.max(np.abs(k_values)) / moire_reciprocal)) + 2
        shifts = np.arange(-n_shifts, n_shifts + 1) * moire_reciprocal
        folded_intensities = [
            interpolator(k_common + shift) for shift in shifts
        ]
        return np.maximum.reduce(folded_intensities)

    S1_interp = fold_to_moire_bz(f1, k_vals_1, min_val_1)
    S2_interp = fold_to_moire_bz(f2, k_vals_2, min_val_2)

    # 4. Normalización común: log-intensidad, percentiles y clipping
    def normalize_array(arr, p_low=5.0, p_high=99.8):
        vmin = np.percentile(arr, p_low)
        vmax = np.percentile(arr, p_high)
        norm = (arr - vmin) / (vmax - vmin + 1e-12)
        return np.clip(norm, 0.0, 1.0)

    # 5. Combina las dos cadenas y aplica la normalización una sola vez
    S_total = normalize_array(np.maximum(S1_interp, S2_interp))

    # =================================================================
    # 6. Renderizado en la mBZ con escala [0, 1].
    # =================================================================
    plt.figure(figsize=(6, 12), dpi=120)
    extent = (k_common[0], k_common[-1], energy_plot[0], energy_plot[-1])

    im = plt.imshow(S_total, origin='lower', aspect='auto', extent=extent,
                    cmap='Greens', vmin=0.0, vmax=1.0, interpolation='bilinear')

    # Barra de color graduada de 0 a 1 exactamente igual a diag.py
    cbar = plt.colorbar(im, label='Normalized Spectral Intensity')
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))

    plt.axvline(-k_max_abs, color='crimson', linestyle='--', alpha=0.8, lw=1.2,
                label=r'mBZ boundary ($\pm G_{\mathrm{M}}/2$)')
    plt.axvline(k_max_abs, color='crimson', linestyle='--', alpha=0.8, lw=1.2)

    plt.ylim(0.0, E_MAX)
    plt.xlim(-k_max_abs, k_max_abs)
    plt.ylabel(r'$\hbar\omega / J_0$ (Adimensional Energy)', fontsize=12)
    plt.xlabel(r'$k$ (Physical Wavenumber) $[rad/m]$', fontsize=12)
    plt.title('LLG spectrum', fontsize=14)
    plt.legend(loc='lower center', framealpha=0.8)
    plt.tight_layout()
    plt.savefig('LLG spectrum.png', dpi=300, bbox_inches='tight')
    print("-> Figura guardada automáticamente como 'LLG_spectrum.png'.")
    plt.show()


# ===================================================================
# 3. MODO SCRIPT (EJECUCIÓN)
# ===================================================================
if __name__ == "__main__":
    from Heff import a1, a2

    dt_base = 1e-12
    SAVE_STEP = 10
    dt_efectivo = dt_base * SAVE_STEP

    GAMMA = 1.7e11           # rad/(s*T)
    J0_FIELD = 1.0           # Campo de canje base (T)
    E_MAX = 1.8              # Techo de energía idéntico a diag.py (1.8)

    LAB_COMPONENTS = (0, 1)  # Fluctuaciones transversales (Sx, Sy)
    APPLY_DEMEAN = True
    APPLY_HANN = False
    USE_RFFT_TIME = True
    PLOT_INDIVIDUALS = False 
    IS_PBC = True

    LOAD_SPECTRUM_FROM_DISK = False

    if LOAD_SPECTRUM_FROM_DISK:
        print("-> [Modo Carga Rápida] Leyendo espectros precalculados desde el disco...")
        power_1 = np.load('espectro_potencia_cadena1.npy')
        power_2 = np.load('espectro_potencia_cadena2.npy')
        n_sites_1 = power_1.shape[1]
        n_sites_2 = power_2.shape[1]
        num_pasos_inferido = (power_1.shape[0] - 1) * 2 if USE_RFFT_TIME else power_1.shape[0]

        plot_moire_spectrum(
            power_1, power_2, n_sites_1, n_sites_2, 
            num_pasos_inferido, dt_efectivo, a1, a2,
            gamma=GAMMA, J0_field=J0_FIELD, E_MAX=E_MAX,
            USE_RFFT_TIME=USE_RFFT_TIME, PLOT_INDIVIDUALS=PLOT_INDIVIDUALS
        )

    else:
        print("-> [Modo Cálculo Completo] Calculando FFT desde las historias de espín...")
        file_fluc_1 = 'spin_history_cadena1 fluc Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy'
        file_fluc_2 = 'spin_history_cadena2 fluc Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy'
        file_rel_1  = 'spin_history_cadena1 relax Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy'
        file_rel_2  = 'spin_history_cadena2 relax Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy'

        def load_and_compute(file_fluc, file_rel, is_pbc):
            print(f"-> Procesando {file_fluc}...")
            hist = np.load(file_fluc, mmap_mode='r')
            rel_data = np.load(file_rel, mmap_mode='r')
            gs = rel_data[-1] if rel_data.ndim == 3 else rel_data

            power, n_fft_time = compute_power_full_chain(
                hist, gs, components=LAB_COMPONENTS,
                demean=APPLY_DEMEAN, hann_temporal=APPLY_HANN, hann_spatial=APPLY_HANN, 
                use_rfft_time=USE_RFFT_TIME, is_pbc=is_pbc
            )

            n_sites = power.shape[1]
            num_pasos = n_fft_time
            del hist, gs, rel_data
            gc.collect()
            return power, num_pasos, n_sites

        power_1, num_pasos, n_sites_1 = load_and_compute(file_fluc_1, file_rel_1, IS_PBC)
        power_2, _, n_sites_2         = load_and_compute(file_fluc_2, file_rel_2, IS_PBC)

        plot_moire_spectrum(
            power_1, power_2, n_sites_1, n_sites_2, 
            num_pasos, dt_efectivo, a1, a2,
            gamma=GAMMA, J0_field=J0_FIELD, E_MAX=E_MAX,
            USE_RFFT_TIME=USE_RFFT_TIME, PLOT_INDIVIDUALS=PLOT_INDIVIDUALS
        )