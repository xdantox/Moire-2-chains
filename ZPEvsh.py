import numpy as np
import matplotlib.pyplot as plt
import gc
import time
from concurrent.futures import ProcessPoolExecutor

from Heff import Kz_field as Kz_field_default
from Heff import a1 as a1_default
from Heff import n1 as n1_default
from Heff import n2 as n2_default
from Heff import u2 as u2_default
from Heff import J_perp0, tol_rel
from Heff import precompute_two_chain_couplings
from Heff import J0_intra_field, lambda_intra, lambda_perp, Kz_meV, muB_meV_T

from rot import execute_simulation
from fft import compute_power_full_chain

HBAR_MEV_S = 0.6582119569e-12
LLG_ENERGY_SCALE = HBAR_MEV_S * 1.7e11 / muB_meV_T

# ==========================================
# CONFIGURACIÓN DE FÍSICA Y HARDWARE
# ==========================================
dt = 1e-15
IS_PBC = False

PASOS_RELAX = 11_000
ALPHA_RELAX = 1.3

PASOS_FLUC = 2**16        # Restaurado al óptimo para alta resolución espectral
SAVE_STEP = 10          
ALPHA_FLUC = 0.0
dt_efectivo = dt * SAVE_STEP

# Activa o desactiva el cálculo y la visualización del Delta ZPE hessiano.
CALCULAR_DELTA_HESSIANO = False

# ==========================================
# PARÁMETROS DE PARALELIZACIÓN (Optimizado)
# ==========================================
N_SEEDS = 2               
CORES_PER_SEED = 12       # 2 workers * 12 cores = 24 hilos (Saturación perfecta)
PAD_FACTOR = 4            

SAVE_SPECTRUM_TO_DISK = False

# ==========================================
# MALLA TOPOLÓGICA PRE-CALCULADA
# ==========================================
h_gruesos = np.arange(1.7, 1.0, -0.35)
h_finos   = np.arange(1.0, -0.01, -0.03)
h_grid    = np.unique(np.round(np.concatenate((h_gruesos, h_finos)), 6))[::-1]

# ==========================================
# FUNCIONES AUXILIARES DE FÍSICA
# ==========================================
def create_perfect_afm(n):
    S = np.zeros((n, 3), dtype=np.float64)
    S[:, 2] = 1.0
    S[1::2, 2] = -1.0
    return S

def compute_hessian_eigenvectors(cache, n1, n2, Kz_mev, sigma1, sigma2):
    J1 = cache.J_intra_1.toarray() if cache.use_sparse else cache.J_intra_1
    J2 = cache.J_intra_2.toarray() if cache.use_sparse else cache.J_intra_2
    W12 = cache.W_12.toarray()     if cache.use_sparse else cache.W_12
    W21 = W12.T
    conv = muB_meV_T  
    J1_meV  = J1  * conv
    J2_meV  = J2  * conv
    W12_meV = W12 * conv
    W21_meV = W21 * conv
    Kz_val  = Kz_mev  

    diag1 = (2.0 * Kz_val - np.sum(J1_meV  * np.outer(sigma1, sigma1), axis=1) - np.sum(W12_meV * np.outer(sigma1, sigma2), axis=1))
    diag2 = (2.0 * Kz_val - np.sum(J2_meV  * np.outer(sigma2, sigma2), axis=1) - np.sum(W21_meV * np.outer(sigma2, sigma1), axis=1))

    M11 = np.diag(diag1) + J1_meV
    M22 = np.diag(diag2) + J2_meV
    M_full = np.block([[M11, W12_meV], [W21_meV, M22]])

    Sigma = np.diag(np.concatenate([sigma1, sigma2]))
    eigvals, V = np.linalg.eig(Sigma @ M_full)
    
    return np.real(V)

def project_noise_with_eigenvectors(V, n1, n2, seed=42):
    rng = np.random.default_rng(seed)
    N_total = n1 + n2
    phases = rng.uniform(0.0, 2.0 * np.pi, N_total)

    xi_x = np.cos(phases)
    xi_y = np.sin(phases)

    s_x = V @ xi_x
    s_y = V @ xi_y

    return s_x[:n1], s_y[:n1], s_x[n1:], s_y[n1:]

def add_thermal_noise(S_base, ruido_x, ruido_y, noise_amp=1e-5):
    S_noisy = S_base.copy()
    S_noisy[:, 0] += ruido_x * noise_amp
    S_noisy[:, 1] += ruido_y * noise_amp
    norms = np.linalg.norm(S_noisy, axis=1, keepdims=True)
    return S_noisy / norms

def calculate_zpe_from_dos(dos_1, dos_2, n_fft, dt_eff, N1, N2, dt_integrador=1e-15):
    omega_num = np.fft.rfftfreq(n_fft, d=dt_eff) * 2 * np.pi
    dw = omega_num[1] - omega_num[0] 
    dos = dos_1 + dos_2

    N_total = N1 + N2
    norm = np.sum(dos) * dw
    if norm == 0:
        return 0.0, 0.0
    dos_norm = dos * (N_total / norm)

    # --- CORRECCIÓN ANALÍTICA EXACTA (Transformada de Cayley) ---
    # En lugar de usar la serie de Taylor (dt^2), aplicamos el mapeo exacto
    # para aislar el problema de la FFT y despreocuparnos de la integración temporal.
    omega_fis = (2.0 / dt_integrador) * np.tan(omega_num * dt_integrador / 2.0)

    zpe_corregida = (0.5 * HBAR_MEV_S / N_total) * np.sum(omega_fis * dos_norm) * dw

    # Ya no hay error_shift residual de integración temporal, devolvemos 0 por compatibilidad
    return zpe_corregida, 0.0

def calculate_zpe_hessian_exact(cache, n1, n2, Kz_mev, sigma1, sigma2):
    conv = muB_meV_T
    J1  = (cache.J_intra_1.toarray() if cache.use_sparse else cache.J_intra_1) * conv
    J2  = (cache.J_intra_2.toarray() if cache.use_sparse else cache.J_intra_2) * conv
    W12 = (cache.W_12.toarray()      if cache.use_sparse else cache.W_12)      * conv
    W21 = W12.T

    diag1 = (2 * Kz_mev - np.sum(J1  * np.outer(sigma1, sigma1), axis=1) - np.sum(W12 * np.outer(sigma1, sigma2), axis=1))
    diag2 = (2 * Kz_mev - np.sum(J2  * np.outer(sigma2, sigma2), axis=1) - np.sum(W21 * np.outer(sigma2, sigma1), axis=1))

    M = np.block([[np.diag(diag1) + J1,  W12], [W21,  np.diag(diag2) + J2]])
    Sigma = np.diag(np.concatenate([sigma1, sigma2]))
    eigvals = np.linalg.eigvals(Sigma @ M)
    omega_m = np.abs(np.real(eigvals))
    N_total = n1 + n2
    return 0.5 * LLG_ENERGY_SCALE * np.sum(omega_m) / N_total


# ==========================================
# TRABAJADOR PARALELO (Subproceso ultra-ligero)
# ==========================================
def run_single_seed_worker(rx1, ry1, rx2, ry2, cache, S1_gs, S2_gs):
    S1_fluc = add_thermal_noise(S1_gs, rx1, ry1, noise_amp=1e-5)
    S2_fluc = add_thermal_noise(S2_gs, rx2, ry2, noise_amp=1e-5)

    Hist_Fluc_1, Hist_Fluc_2 = execute_simulation(
        S1_fluc, S2_fluc, num_pasos=PASOS_FLUC, dt=dt,
        coupling_cache=cache, Kz=Kz_field_default, n_iter=14,
        num_cores=CORES_PER_SEED, save_step=SAVE_STEP, alpha=ALPHA_FLUC
    )

    # El fix de Parseval (spatial_resolved=False) en fft.py entrega directamente la DOS.
    dos_1, n_fft_out = compute_power_full_chain(
        Hist_Fluc_1, gs_lab=S1_gs, is_pbc=IS_PBC, components=(0, 1),
        pad_factor=PAD_FACTOR, demean=False, spatial_resolved=False
    )
    dos_2, _ = compute_power_full_chain(
        Hist_Fluc_2, gs_lab=S2_gs, is_pbc=IS_PBC, components=(0, 1),
        pad_factor=PAD_FACTOR, demean=False, spatial_resolved=False
    )

    # FIX: Se eliminan P1 y P2 de la orden del recolector de basura, ya que ahora son omitidas.
    del Hist_Fluc_1, Hist_Fluc_2; gc.collect()
    return dos_1, dos_2, n_fft_out

# ==========================================
# BUCLE MAESTRO (Arquitectura Secuencial en h, Paralela en Semillas)
# ==========================================
def main_zpe_iterator():
    print(f"=== INICIANDO BARRIDO ZPE SECUENCIAL ({N_SEEDS} SEEDS EN PARALELO) ===\n")
    print(f"    Total workers en RAM: {N_SEEDS} | Núcleos por worker: {CORES_PER_SEED} | Total hilos: {N_SEEDS * CORES_PER_SEED}\n")

    h_results       = []
    zpe_llg         = []   
    zpe_hess_exact  = []   

    S1_seed = create_perfect_afm(n1_default)
    S2_seed = create_perfect_afm(n2_default)
    sigma1 = (-1)**np.arange(n1_default)
    sigma2 = (-1)**np.arange(n2_default)

    with ProcessPoolExecutor(max_workers=N_SEEDS) as executor:
        
        for paso_iter, h_val in enumerate(h_grid, 1):
            t0 = time.time()
            
            print(f"[{paso_iter}] Procesando distancia h = {h_val:.4f} nm")
            print("-" * 65)

            # --- PASO 1: ACOPLES ---
            print("   -> 1. Calculando acoples del sistema...")
            cache = precompute_two_chain_couplings(
                n1=n1_default, n2=n2_default, a1=a1_default, u2=u2_default,
                h_sep=h_val, use_sparse=True, is_pbc=IS_PBC
            )

            if CALCULAR_DELTA_HESSIANO:
                print("   -> 1b. Calculando ZPE del Hessiano exacto...")
                zpe_ref = calculate_zpe_hessian_exact(cache, n1_default, n2_default, Kz_meV, sigma1, sigma2)
                zpe_hess_exact.append(zpe_ref)
            else:
                zpe_ref = None

            # --- PASO 2: DIAGONALIZACIÓN ---
            print("   -> 2. Diagonalizando Hessiano completo (O(N^3) centralizado)...")
            V_h = compute_hessian_eigenvectors(cache, n1_default, n2_default, Kz_meV, sigma1, sigma2)
            
            # --- PASO 3: RUIDO ---
            print(f"   -> 3. Proyección instantánea de ruidos para {N_SEEDS} semillas (O(N^2))...")
            ruidos_por_seed = [project_noise_with_eigenvectors(V_h, n1_default, n2_default, seed=(42 + idx)) for idx in range(N_SEEDS)]
            del V_h; gc.collect()

            # --- PASO 4a: LLG RELAJACIÓN ---
            print("   -> 4. LLG: Relajando estado base (Determinista)...")
            Hist_Rel_1, Hist_Rel_2 = execute_simulation(
                S1_seed, S2_seed, num_pasos=PASOS_RELAX, dt=dt,
                coupling_cache=cache, Kz=Kz_field_default, n_iter=14,
                num_cores=(N_SEEDS * CORES_PER_SEED), save_step=PASOS_RELAX, alpha=ALPHA_RELAX
            )
            S1_gs = Hist_Rel_1[-1].copy()
            S2_gs = Hist_Rel_2[-1].copy()
            del Hist_Rel_1, Hist_Rel_2; gc.collect()

            # --- PASO 4b: LLG FLUCTUACIÓN EN PARALELO ---
            print("   -> 5. LLG: Evolucionando ensambles térmicos en paralelo (Estocástico + FFT)...")
            futures = []
            for idx in range(N_SEEDS):
                fut = executor.submit(
                    run_single_seed_worker,
                    ruidos_por_seed[idx][0], ruidos_por_seed[idx][1],
                    ruidos_por_seed[idx][2], ruidos_por_seed[idx][3],
                    cache, S1_gs, S2_gs
                )
                futures.append(fut)

            # --- RECOLECCIÓN Y PROCESAMIENTO ---
            dos_1_acumulada = 0
            dos_2_acumulada = 0
            n_fft_out = None

            for future in futures:
                dos_1, dos_2, n_fft = future.result()
                dos_1_acumulada += dos_1
                dos_2_acumulada += dos_2
                n_fft_out = n_fft

            dos_1_acumulada /= N_SEEDS
            dos_2_acumulada /= N_SEEDS

            if SAVE_SPECTRUM_TO_DISK:
                h_str = f"{h_val:.4f}".replace('.', '_')
                np.save(f"dos_promediada_c1_h{h_str}.npy", dos_1_acumulada)
                np.save(f"dos_promediada_c2_h{h_str}.npy", dos_2_acumulada)

            zpe_val_corregida, eps_h = calculate_zpe_from_dos(
                dos_1_acumulada, dos_2_acumulada,
                n_fft=n_fft_out, dt_eff=dt_efectivo,
                N1=n1_default, N2=n2_default,
                dt_integrador=dt 
            )
            
            zpe_llg.append(zpe_val_corregida)
            h_results.append(h_val)

            if CALCULAR_DELTA_HESSIANO:
                residual = abs(zpe_val_corregida - zpe_ref) / (abs(zpe_ref) + 1e-30)
                print(f"   [RESULTADO] ZPE LLG = {zpe_val_corregida:.6e} | Hessiano = {zpe_ref:.6e} | Discrepancia = {residual:.6f}")
            else:
                print(f"   [RESULTADO] ZPE LLG = {zpe_val_corregida:.6e}")

            t1 = time.time()
            print(f"   (Tiempo del ciclo: {t1 - t0:.1f} s)\n")

    print("=== BARRIDO COMPLETADO ===")

    # Procesamiento final e impresión de gráficos
    h_arr        = np.array(h_results)
    zpe_llg_arr  = np.array(zpe_llg)
    zpe_hess_arr = np.array(zpe_hess_exact) if CALCULAR_DELTA_HESSIANO else None

    delta_llg = zpe_llg_arr - zpe_llg_arr[0]

    if CALCULAR_DELTA_HESSIANO:
        delta_hess = zpe_hess_arr - zpe_hess_arr[0]
        datos = np.column_stack((h_arr, delta_llg, delta_hess))
        np.savetxt("Resultados_Delta_ZPE_vs_h.txt", datos,
                   header="h(nm)  Delta_ZPE_LLG(meV)  Delta_ZPE_Hessiano(meV)")
    else:
        delta_hess = None
        datos = np.column_stack((h_arr, delta_llg))
        np.savetxt("Resultados_Delta_ZPE_vs_h.txt", datos,
                   header="h(nm)  Delta_ZPE_LLG(meV)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(h_arr, delta_llg,  'o-', color='indigo',  lw=2, ms=6,
                 label=f'LLG ({N_SEEDS} Seeds, Pad={PAD_FACTOR})')
    if CALCULAR_DELTA_HESSIANO:
        axes[0].plot(h_arr, delta_hess, 's--', color='goldenrod', lw=2, ms=6,
                     label='Hessiano exacto')
    axes[0].axhline(0, color='black', ls='--', alpha=0.4)
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r'$h$ (nm)', fontsize=13)
    axes[0].set_ylabel(r'$\Delta$ZPE (meV)', fontsize=13)
    axes[0].set_title('Comparación directa', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, ls=':', alpha=0.6)

    if CALCULAR_DELTA_HESSIANO:
        disc = np.abs(zpe_llg_arr - zpe_hess_arr) / (np.abs(zpe_hess_arr) + 1e-30)
        axes[1].semilogy(h_arr, disc, 'o-', color='crimson', lw=2, ms=6)
        axes[1].set_ylabel('Discrepancia relativa', fontsize=11)
        axes[1].set_title('Error relativo del método LLG', fontsize=13)
    else:
        axes[1].text(0.5, 0.5, 'Comparación hessiana desactivada',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=12)
        axes[1].set_title('Comparación hessiana', fontsize=13)
        axes[1].set_ylabel('')
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r'$h$ (nm)', fontsize=13)
    axes[1].grid(True, ls=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("ZPE_comparacion_ensamble.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main_zpe_iterator()