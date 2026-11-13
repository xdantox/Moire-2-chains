import numpy as np
import matplotlib.pyplot as plt
import gc
import time

# ==========================================
# IMPORTACIONES
# ==========================================
from Heff import Kz_field as Kz_field_default
from Heff import a1 as a1_default
from Heff import n1 as n1_default
from Heff import n2 as n2_default
from Heff import u2 as u2_default
from Heff import J_perp0, tol_rel
from Heff import precompute_two_chain_couplings

from rot import execute_simulation 
from fft import compute_power_full_chain

HBAR_MEV_S = 0.6582119569e-12 

# ==========================================
# CONFIGURACIÓN DE FÍSICA Y HARDWARE
# ==========================================
dt = 1e-15          
n_nucleos = 8
IS_PBC = False      

PASOS_RELAX = 5_000
ALPHA_RELAX = 1.3    

PASOS_FLUC = 2**14
# CORRECCIÓN 1: Límite de Nyquist subido a 50 THz para evitar Aliasing
SAVE_STEP = 10
ALPHA_FLUC = 0.0      
dt_efectivo = dt * SAVE_STEP

SAVE_SPECTRUM_TO_DISK = True  

# ==========================================
# CONFIGURACIÓN DE LA MALLA ADAPTATIVA
# ==========================================
h_inicio = 1.6        
h_final  = 0.0         
dh_grueso = 0.2        
dh_fino   = 0.02       

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def create_perfect_afm(n):
    S = np.zeros((n, 3), dtype=np.float64)
    S[:, 2] = 1.0
    S[1::2, 2] = -1.0 
    return S

import numpy as np

def generate_exact_hessian_noise(n, a_eff, J0_intra, lam_intra, Kz):
    """
    Genera ruido cuántico inyectando energía uniformemente a través 
    de los autovectores exactos del Hessiano en el espacio real.
    """
    # 1. Construir la matriz de distancias reales y el Intercambio J_ij
    x = np.arange(n) * a_eff
    dx = np.abs(x[:, None] - x[None, :])
    J_ij = J0_intra * np.exp(-dx / lam_intra)
    np.fill_diagonal(J_ij, 0.0)
    J_ij[J_ij < 1e-6] = 0.0  # Tolerar para eficiencia

    # 2. Construir el Hessiano Aislado (M11) con OBC exactos
    sigma = (-1)**np.arange(n)
    diag = Kz - np.sum(J_ij * np.outer(sigma, sigma), axis=1)
    M = np.diag(diag) + J_ij

    # 3. Extraer Modos Normales (Autovectores) de la Matriz Dinámica
    Sigma_matrix = np.diag(sigma)
    eigvals, eigvecs = np.linalg.eig(Sigma_matrix @ M)

    # Ordenar por frecuencia para estabilidad
    idx = np.argsort(np.abs(np.real(eigvals)))
    V_m = np.real(eigvecs[:, idx])

    # 4. Inyectar Ruido Blanco Puro en el Espacio de Modos Normales
    # Esto garantiza que cada modo (desde el Gap hasta el más rígido)
    # reciba exactamente la misma amplitud inicial. El espectro P(w) será plano.
    ruido_modos_x = np.random.randn(n)
    ruido_modos_y = np.random.randn(n)

    # 5. Transformar de vuelta al Espacio Real (los espines)
    # Multiplicamos la matriz de autovectores por el vector de amplitudes
    ruido_real_x = V_m @ ruido_modos_x
    ruido_real_y = V_m @ ruido_modos_y

    # Devolvemos el ruido centrado para no inducir rotaciones espurias de toda la cadena
    return ruido_real_x - np.mean(ruido_real_x), ruido_real_y - np.mean(ruido_real_y)

def add_thermal_noise_perfected(S_base, ruido_x, ruido_y, noise_amp=1e-5):
    """Inyección de ruido pre-calculado a la red."""
    S_noisy = S_base.copy()
    
    rx_centered = ruido_x - ruido_x.mean()
    ry_centered = ruido_y - ruido_y.mean()

    S_noisy[:, 0] += rx_centered * noise_amp
    S_noisy[:, 1] += ry_centered * noise_amp

    norms = np.linalg.norm(S_noisy, axis=1)
    S_noisy /= norms[:, np.newaxis]
    return S_noisy

def calculate_zpe_from_spectrum(power_1, power_2, num_pasos, dt_eff):
    omega_values = np.fft.rfftfreq(num_pasos, d=dt_eff) * 2 * np.pi
    dos_total = np.sum(power_1, axis=1) + np.sum(power_2, axis=1)
    
    mask = omega_values > 1e11 
    omega_valid = omega_values[mask]
    
    # CORRECCIÓN 3: Al usar Ruido Cuántico, la FFT ya es la Densidad de Estados Pura.
    # No multiplicamos por omega^2.
    dos_valid = dos_total[mask]
    
    # Centro de masa cuántico real
    integral_E = np.sum(omega_valid * dos_valid)
    integral_norm = np.sum(dos_valid)
    
    if integral_norm == 0: return 0.0
    omega_promedio = integral_E / integral_norm
    return 0.5 * HBAR_MEV_S * omega_promedio


# ==========================================
# BUCLE MAESTRO ADAPTATIVO
# ==========================================
def main_zpe_iterator():
    print("=== INICIANDO BARRIDO ADAPTATIVO ZPE vs h ===")
    print(f"Modo Detección: Activado. Resoluciones: {dh_grueso} nm -> {dh_fino} nm\n")
    
    h_results = []
    zpe_results = []
    slope_history = []
    
    S1_seed = create_perfect_afm(n1_default)
    S2_seed = create_perfect_afm(n2_default)

    h_actual = h_inicio
    dh_actual = dh_grueso
    paso_iter = 1
    
    a2_default = a1_default * (u2_default + 1.0) # Extraer a2 de la importación

    print("-> Generando Ruido Cuántico Ab Initio (Eigendecomposition)...")
    
    # a_eff es la distancia real al primer vecino en tu cadena. 
    # Si a1_default es el largo de celda, la distancia entre espines A y B es la mitad.
    a_eff_1 = a1_default / 2.0 
    a_eff_2 = a2_default / 2.0
    
    ruido_x_1, ruido_y_1 = generate_exact_hessian_noise(
        n=n1_default, a_eff=a_eff_1, 
        J0_intra=46.75, lam_intra=0.3, Kz=0.76
    )
    
    ruido_x_2, ruido_y_2 = generate_exact_hessian_noise(
        n=n2_default, a_eff=a_eff_2, 
        J0_intra=46.75, lam_intra=0.3, Kz=0.76
    )
    RUIDO_TOLERANCIA_MEV = max(2.0 * J_perp0 * tol_rel, 1e-6)

    while h_actual >= h_final - 1e-5: 
        t0 = time.time()
        print(f"[{paso_iter}] Evaluando h = {h_actual:.3f} nm (Paso actual: {dh_actual:.3f} nm)")
        print("-" * 50)
        
        cache = precompute_two_chain_couplings(
            n1=n1_default, n2=n2_default, a1=a1_default, u2=u2_default, 
            h_sep=h_actual, use_sparse=True, is_pbc=IS_PBC
        )
        
        # --- RELAJACIÓN ---
        print("   -> Relajando red de solitones...")
        Hist_Rel_1, Hist_Rel_2 = execute_simulation(
            S1_seed, S2_seed, num_pasos=PASOS_RELAX, dt=dt, coupling_cache=cache, 
            Kz=Kz_field_default, n_iter=14, num_cores=n_nucleos, 
            save_step=PASOS_RELAX, alpha=ALPHA_RELAX 
        )
        
        S1_gs_puro = Hist_Rel_1[-1]
        S2_gs_puro = Hist_Rel_2[-1]
        
        S1_seed = S1_gs_puro.copy()
        S2_seed = S2_gs_puro.copy()
        del Hist_Rel_1, Hist_Rel_2; gc.collect()

        # --- FLUCTUACIÓN CON RUIDO CUÁNTICO ---
        print("   -> Calculando espectro térmico dinámico...")
        S1_fluc = add_thermal_noise_perfected(S1_gs_puro, ruido_x_1, ruido_y_1, noise_amp=1e-5)
        S2_fluc = add_thermal_noise_perfected(S2_gs_puro, ruido_x_2, ruido_y_2, noise_amp=1e-5)
        
        Hist_Fluc_1, Hist_Fluc_2 = execute_simulation(
            S1_fluc, S2_fluc, num_pasos=PASOS_FLUC, dt=dt, coupling_cache=cache, 
            Kz=Kz_field_default, n_iter=14, num_cores=n_nucleos, 
            save_step=SAVE_STEP, alpha=ALPHA_FLUC 
        )

        # --- ZPE ---
        P1, _ = compute_power_full_chain(Hist_Fluc_1, gs_lab=S1_gs_puro, is_pbc=IS_PBC)
        P2, _ = compute_power_full_chain(Hist_Fluc_2, gs_lab=S2_gs_puro, is_pbc=IS_PBC)
        
        if SAVE_SPECTRUM_TO_DISK:
            h_str = f"{h_actual:.3f}".replace('.', '_')
            print(f"   -> Guardando espectros (power_spectrum_c*_h{h_str}.npy)...")
            np.save(f"power_spectrum_c1_h{h_str}.npy", P1)
            np.save(f"power_spectrum_c2_h{h_str}.npy", P2)

        zpe_val = calculate_zpe_from_spectrum(P1, P2, num_pasos=Hist_Fluc_1.shape[0], dt_eff=dt_efectivo)
        
        h_results.append(h_actual)
        zpe_results.append(zpe_val)
        
        print(f"   => ZPE Absoluta = {zpe_val:.6e} meV")
        
        del Hist_Fluc_1, Hist_Fluc_2, P1, P2, S1_gs_puro, S2_gs_puro, cache; gc.collect()
        
        if len(zpe_results) > 1:
            delta_E = zpe_results[-1] - zpe_results[-2]
            delta_h = h_results[-1] - h_results[-2]
            
            pendiente_actual = delta_E / delta_h
            
            if len(slope_history) > 0:
                pendiente_previa = slope_history[-1]
                
                alerta_geometrica = (pendiente_actual < 0.6 * pendiente_previa) or (pendiente_actual < 0)
                alerta_fisica_real = abs(delta_E) > RUIDO_TOLERANCIA_MEV
                
                if alerta_geometrica and alerta_fisica_real:
                    if dh_actual == dh_grueso: 
                        print("\n   [!!!] ALERTA TOPOLÓGICA [!!!]")
                        print(f"   Cambio real de energía detectado (|dE| = {abs(delta_E):.6f} > {RUIDO_TOLERANCIA_MEV:.6f}).")
                        print(f"   Cambiando resolución de malla: {dh_grueso} -> {dh_fino} nm\n")
                        dh_actual = dh_fino
                        
            slope_history.append(pendiente_actual)
        
        t1 = time.time()
        print(f"   (Tiempo del ciclo: {t1-t0:.1f} s)")

        h_actual -= dh_actual
        h_actual = round(h_actual, 5) 
        paso_iter += 1

    print("\n=== BARRIDO COMPLETADO ===")
    
    h_array = np.array(h_results)
    zpe_array = np.array(zpe_results)
    delta_zpe = zpe_array - zpe_array[0]
    
    datos_out = np.column_stack((h_array, delta_zpe))
    np.savetxt("Resultados_Delta_ZPE_vs_h.txt", datos_out, header="h(nm) Delta_ZPE(meV)")
    print("Datos guardados en 'Resultados_Delta_ZPE_vs_h.txt'")
    
    plt.figure(figsize=(9, 6))
    plt.plot(h_array, delta_zpe, 'o-', color='indigo', linewidth=2, markersize=6, label=r'$\Delta$ZPE Calculada')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5) 
    plt.gca().invert_xaxis()
    
    plt.title(r'Fuerza de Casimir Cuántica (Malla Adaptativa)', fontsize=15)
    plt.xlabel(r'Separación intercadena $h$ (nm)', fontsize=13)
    plt.ylabel(r'$\Delta$ZPE por espín (meV)', fontsize=13)
    
    idx_fino = np.where(np.diff(h_array) > -dh_grueso + 1e-4)[0]
    if len(idx_fino) > 0:
        h_inicio_pozo = h_array[idx_fino[0]]
        plt.axvspan(h_inicio_pozo, h_final, color='gold', alpha=0.15, label=r'Régimen Físico Interactivo')
        
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_zpe_iterator()