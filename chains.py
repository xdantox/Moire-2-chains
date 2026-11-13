import numpy as np
import gc

# ==========================================
# IMPORTACIONES DEL SISTEMA FÍSICO
# ==========================================
from Heff import Kz_field as Kz_field_default
from Heff import a1 as a1_default
from Heff import n1 as n1_default
from Heff import n2 as n2_default
from Heff import u2 as u2_default
from Heff import precompute_two_chain_couplings

from rot import execute_simulation, alpha 
from E import evaluate_energy_history
from animation import animation_two_chains
from cadena0 import build_afm_state_from_npy, build_chain_from_direct2chain_params


# ==========================================
# 1. PARÁMETROS GLOBALES Y DE TIEMPO
# ==========================================
dt = 1e-12                 # Paso de tiempo temporal
total_time = (2**17) * dt  # Tiempo total de simulación
num_pasos = int(total_time / dt)

n_nucleos = 16             # Hilos de procesamiento paralelos
save_step = 10             # Guardar cada save_step pasos para reducir memoria
dt_efectivo = dt * save_step  # Paso efectivo entre muestras guardadas

# Parámetros físicos heredados
n1 = n1_default
n2 = n2_default
Kz_field = Kz_field_default
a1 = a1_default
a2 = a1 * (u2_default + 1.0)  # Recuperamos a2 para el graficador FFT

# ==========================================
# ---> SWITCHES DE CONTROL MAESTRO <---
# ==========================================
IS_PBC = True              # False: Condiciones Abiertas (OBC). True: Anillo (PBC).
CALC_FFT_DIRECT = False     # True: Calcula el espectro FFT directamente en RAM.
SAVE_TO_DISK = True        # True: Guarda los arrays masivos History en el disco duro.
SAVE_TO_DISK_PS = False    # True: Guarda las matrices de Espectro de Potencia en el disco duro.
EVALUATE_ENERGY = True     # True: Evalúa la energía total y por sitio a lo largo de la historia.
def main():
    # ==========================================
    # 2. PRECOMPUTACIÓN GEOMÉTRICA (MOIRÉ)
    # ==========================================
    print("-> Calculando caché topológico (Agnóstico) del sistema Moiré...")
    cache = precompute_two_chain_couplings(
        n1=n1,
        n2=n2,
        a1=a1,
        u2=u2_default,
        use_sparse=True,
        is_pbc=IS_PBC
    )

    # ==========================================
    # 3. CARGA DE CONDICIONES INICIALES
    # ==========================================
    print("-> Cargando estados iniciales para la simulación...")
    

    # ==========================================
    #  CONDICIONES INICIALES
    # ==========================================
    # Estado AFM inicial extraído de relajación previa

    q1 = 0
    q2 = 0

    params = np.array(
        [-0.0, -0.0, -1.570796, -1.570796, 0, 0, 0, 0, 0, 0],
        dtype=float,
    )

  # Generar estado para la Cadena 1

    # S1_0 = build_chain_from_direct2chain_params(
    #     n_sites=n1,
    #     q=q1,
    #     params=params,
    #     chain_id=1,
    #     noise_x=1e-5,
    #     seed=42   # Semilla fija opcional para reproducir siempre el mismo ruido
    # )


    # # Generar estado para la Cadena 2
    # S2_0 = build_chain_from_direct2chain_params(
    #     n_sites=n2,
    #     q=q2,
    #     params=params,
    #     chain_id=2,
    #     noise_x=1e-5,
    #     seed=137  # Semilla distinta para que el ruido no esté correlacionado con la cadena 1
    # )

    S1_0 = build_afm_state_from_npy(n1, "spin_history_cadena1 relax Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy", noise_x=1e-5)
    S2_0 = build_afm_state_from_npy(n2, "spin_history_cadena2 relax Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy", noise_x=1e-5)

    # Cargamos el estado base puro (SIN RUIDO) para la referencia de la FFT
    if CALC_FFT_DIRECT:
        print("-> Cargando estados base puros (referencia GS para análisis espectral)...")
        S1_gs_puro = build_afm_state_from_npy(n1, "spin_history_cadena1 relax Jperp=0 Jintra=1 PBC.npy", noise_x=0.0)
        S2_gs_puro = build_afm_state_from_npy(n2, "spin_history_cadena2 relax Jperp=0 Jintra=1 PBC.npy", noise_x=0.0)

    # ==========================================
    # 4. INTEGRACIÓN LLG (DINÁMICA NO LINEAL)
    # ==========================================
    print(f"-> Iniciando integración LLG ({num_pasos} pasos) a máxima velocidad...")
    History_S1, History_S2 = execute_simulation(
        S1_0,
        S2_0,
        num_pasos,
        dt,
        cache,
        Kz_field,
        n_iter=14,
        num_cores=n_nucleos,
        save_step=save_step,  # Guardar cada 10 pasos para reducir memoria
        alpha=alpha
    )
    print("-> Integración terminada.")
    # ==========================================
    # 5. EVALUACIÓN DE ENERGÍA TERMODINÁMICA
    # ==========================================
    if EVALUATE_ENERGY:
        # IMPORTANTE: Asumimos que tienes una función evaluate_energy_single_step en E.py 
        # que calcula la energía de un solo instante de tiempo. Si no la tienes, 
        # puedes mandar un slice de tamaño 1 a tu función actual:
        
        # Extraemos solo el instante inicial y final (Añadiendo un eje extra temporal [np.newaxis, ...])
        hist_ini_1 = History_S1[0][np.newaxis, :, :]
        hist_ini_2 = History_S2[0][np.newaxis, :, :]
        
        hist_fin_1 = History_S1[-1][np.newaxis, :, :]
        hist_fin_2 = History_S2[-1][np.newaxis, :, :]

        # Evaluamos solo 2 pasos en lugar de 131,072
        E_inicial = evaluate_energy_history(hist_ini_1, hist_ini_2, cache, Kz_field)[0]
        E_final   = evaluate_energy_history(hist_fin_1, hist_fin_2, cache, Kz_field)[0]

        total_spins = n1 + n2
        print(f"   * Energía inicial (por espín): {E_inicial / total_spins:.6f}")
        print(f"   * Energía final   (por espín): {E_final / total_spins:.6f}")
        print("\n--- Ejecutando Rutinas de Enrutamiento de Datos ---")
    else:
        print("\n--- Ejecutando Rutinas de Enrutamiento de Datos ---")
    # ==========================================
    # 6. ENRUTAMIENTO DE DATOS (POST-PROCESO)
    # ==========================================
    

    # 6.1: Procesamiento Directo en Memoria RAM
    if CALC_FFT_DIRECT:
        print("-> [RAM] Calculando Transformada de Fourier 2D (S(k, w))...")
        
        # Ignoramos el segundo valor devuelto (_) porque es n_fft (temporal)
        power_spectrum_1, _ = compute_power_full_chain(History_S1, gs_lab=S1_gs_puro, is_pbc=IS_PBC)
        power_spectrum_2, _ = compute_power_full_chain(History_S2, gs_lab=S2_gs_puro, is_pbc=IS_PBC)
        
        # Obtenemos los sitios activos REALES desde las columnas de la matriz
        n_sites_active_1 = power_spectrum_1.shape[1]
        n_sites_active_2 = power_spectrum_2.shape[1]
        
        print(f"   * Análisis FFT completado. Sitios reales: C1={n_sites_active_1}, C2={n_sites_active_2}")

        if SAVE_TO_DISK_PS: 
            print("-> [Disco] Escribiendo matrices de Espectro de Potencia (.npy)...")
            np.save("espectro_potencia_cadena1.npy", power_spectrum_1)
            np.save("espectro_potencia_cadena2.npy", power_spectrum_2)
            np.save("parametros_fft.npy", np.array([
                History_S1.shape[0], dt_efectivo, a1, a2, n_sites_active_1, n_sites_active_2
            ]))
            print("   * Espectros guardados exitosamente.")

        # Liberamos los arrays de referencia de inmediato para vaciar RAM
        del S1_gs_puro, S2_gs_puro
        gc.collect()
        
        # Graficación Interactiva
        plot_moire_spectrum(
            power_spectrum_1, power_spectrum_2, 
            n_sites_active_1, n_sites_active_2, 
            History_S1.shape[0], dt_efectivo, a1, a2
        )
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Liberación de las matrices de potencia
        del power_spectrum_1, power_spectrum_2
        gc.collect()
        
        # Graficación Interactiva
        plot_moire_spectrum(
            power_spectrum_1, power_spectrum_2, 
            n_sites_active_1, n_sites_active_2, 
            History_S1.shape[0], dt_efectivo, a1, a2
        )
        import matplotlib.pyplot as plt
        plt.close('all')
    
        # Liberación de las matrices de potencia
        del power_spectrum_1, power_spectrum_2
        gc.collect()
    # 6.2: Almacenamiento en Disco Físico
    if SAVE_TO_DISK:
        print("-> [Disco] Escribiendo trayectorias completas en disco...")
        np.save("spin_history_cadena1 fluc Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy", History_S1)
        np.save("spin_history_cadena2 fluc Jperp=Jintra=1 PBC h = 1.7 epsilon = 0.1.npy", History_S2)
        print("   * Guardado exitoso.")
    # ==========================================
    # 7. RENDERIZADO 3D FINAL
    # ==========================================
    print("\n-> Lanzando motor de animación geométrica...")
    animation_two_chains(History_S1, History_S2, dt, cache, Kz_field, visual_y_sep=1.0)
    
if __name__ == "__main__":
    main()