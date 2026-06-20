import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# RUTINA DE ANÁLISIS AFM (AUTOPSIA DEL ORDEN DE NÉEL)
# ==============================================================================

def afm_autopsy(spins, easy_axis=2, trans_axis=1, hard_axis=0, n_range=(0, 400), title_suffix=""):
    """
    Analiza la deformación de un estado AFM calculando el Vector de Néel local.
    
    Topología Físicamente Consistente con Ansatz XZ:
      - Eje de la cadena (Longitudinal / Penalizado): X (hard_axis=0)
      - Eje Transversal en el plano fácil: Y (trans_axis=1)
      - Eje de alineación principal AFM (Eje Fácil): Z (easy_axis=2)
    """
    n_sites = spins.shape[0]
    idx = np.arange(n_sites)
    
    # 1. Transformación al Vector de Néel (Enderezar el zigzag AFM)
    # N_n = S_n * (-1)^n
    staggered_phase = (-1.0) ** idx
    N_vector = spins * staggered_phase[:, np.newaxis]
    
    # Extraer componentes del Vector de Néel
    N_easy = N_vector[:, easy_axis]       # Orden primario (Debería ser ~1 o ~-1)
    N_trans = N_vector[:, trans_axis]     # Fluctuaciones dentro del plano transversal
    S_hard = spins[:, hard_axis]          # Fuga hacia el eje longitudinal (Canting)

    # 2. Ángulo Topológico del AFM
    # Mide la orientación del vector de Néel en el plano transversal al eje fácil
    # Usamos Y (trans) y Z (easy) para ver cómo rota la fase si se sale del eje fácil
    theta_neel = np.arctan2(N_trans, N_easy) 
    theta_neel_unwrapped = np.unwrap(theta_neel)
    
    # 3. Desviación del estado homogéneo (Magnitud del defecto)
    # Si N_easy es 1 en un AFM perfecto, 1 - |N_easy| muestra dónde se rompe el orden colineal
    afm_deviation = 1.0 - np.abs(N_easy)

    # ==================== GRAFICACIÓN ====================
    
    # --- GRÁFICO 1: PERFIL DEL ORDEN DE NÉEL (Detección de Solitones) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    ax1.plot(idx, N_easy, 'o-', color="tab:blue", markersize=3, lw=1.2, label=r"Orden Primario ($N_{easy}$ en Z)")
    ax1.axhline(1.0, color='k', ls='--', alpha=0.3)
    ax1.axhline(-1.0, color='k', ls='--', alpha=0.3)
    ax1.axhline(0.0, color='k', ls='-', alpha=0.5)
    ax1.fill_between(idx, N_easy, 0, where=(N_easy > 0), color='tab:blue', alpha=0.1)
    ax1.fill_between(idx, N_easy, 0, where=(N_easy < 0), color='tab:red', alpha=0.1)
    ax1.set_ylabel("Magnitud Alternante")
    ax1.set_title(rf"Perfil del Vector de Néel (Eje Fácil Z) {title_suffix}")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(idx, afm_deviation, color="tab:purple", lw=1.5, label=r"Desviación del AFM perfecto ($1 - |N_{easy}|$)")
    ax2.set_ylabel("Defecto Local")
    ax2.set_xlabel("Sitio $n$")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    ax2.set_xlim(n_range)
    
    plt.tight_layout()
    plt.show()

    # --- GRÁFICO 2: ÁNGULO TOPOLÓGICO Y FUGA AL EJE DIFÍCIL ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    # El salto de fase topológica mostrará la pared de dominio si existe
    ax3.plot(idx, theta_neel_unwrapped, color="tab:green", lw=1.5, label=r"Fase del vector de Néel ($\theta_{N\acute{e}el}$)")
    ax3.set_ylabel(r"$\theta_{N\acute{e}el}$ [rad]")
    ax3.set_title(r"Torción del AFM y Fuga Fuera del Eje Fácil")
    ax3.grid(True, alpha=0.25)
    ax3.legend()

    # Fuga al eje penalizado X
    ax4.plot(idx, S_hard, color="tab:orange", lw=1.0, label=r"Canting hacia Eje Difícil X ($S_{hard}$)")
    ax4.axhline(0.0, color='k', ls='--', alpha=0.5)
    ax4.set_ylabel(r"$S_x$ (Magnitud)")
    ax4.set_xlabel("Sitio $n$")
    ax4.grid(True, alpha=0.25)
    ax4.legend()
    ax4.set_xlim(n_range)

    plt.tight_layout()
    plt.show()

    # --- GRÁFICO 3: FFT DEL DEFECTO (Efecto Moiré de J_perp) ---
    delta_centered = afm_deviation - np.mean(afm_deviation)
    fft_delta = np.fft.rfft(delta_centered)
    k_delta = 2.0 * np.pi * np.fft.rfftfreq(n_sites)
    mag_delta = np.abs(fft_delta) / n_sites

    fig3, ax_spec = plt.subplots(1, 1, figsize=(11, 4))
    mask = (k_delta > 0) & (k_delta <= np.pi)
    ax_spec.plot(k_delta[mask], mag_delta[mask], color="tab:red", lw=1.5)
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel(r"$k$ [rad/site]")
    ax_spec.set_ylabel(r"|FFT(Defecto)|")
    ax_spec.set_title(rf"Espectro de las Deformaciones del AFM (Moiré / $J_{{\perp}}$) {title_suffix}")
    ax_spec.grid(True, which="both", alpha=0.25)
    
    plt.tight_layout()
    plt.show()

    return N_vector, afm_deviation, theta_neel_unwrapped

# ==============================================================================
# EJEMPLO DE USO CON TUS DATOS
# ==============================================================================
if __name__ == "__main__":
    # Asegúrate de haber generado las historias con el nuevo ansatz XZ
    
    # 1. Cargar el estado relajado aislado (J_perp = 0)
    try:
        hist_aislado = np.load('spin_history_cadena1 fluc MC = 4 Jperp=0.npy', mmap_mode='r')
        estado_aislado = hist_aislado[-1].copy()
    except FileNotFoundError:
        estado_aislado = None
        print("Aviso: No se encontró el archivo de J_perp=0")

    # 2. Cargar el estado relajado acoplado (J_perp = 0.4)
    try:
        hist_acoplado = np.load('spin_history_cadena1 relax MC = 4 Jperp=0.4.npy', mmap_mode='r')
        estado_acoplado = hist_acoplado[-1].copy()
    except FileNotFoundError:
        estado_acoplado = None
        print("Aviso: No se encontró el archivo de J_perp=0.4")

    # 3. Analizar
    if estado_aislado is not None:
        print("\n--- ANALIZANDO ESTADO AISLADO (J_perp = 0) ---")
        afm_autopsy(estado_aislado, title_suffix="(Aislado, J_perp=0)")

    if estado_acoplado is not None:
        print("\n--- ANALIZANDO ESTADO ACOPLADO (J_perp = 0.4) ---")
        afm_autopsy(estado_acoplado, title_suffix="(Acoplado, Moiré, J_perp=0.4)")