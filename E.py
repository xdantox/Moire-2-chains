import numpy as np

def energy_two_chains_state(S1, S2, cache, Kz_field):
    """
        Calcula la energía en unidades de campo (misma convención de Heff).
        Retorna la energía total del sistema y la energía local por cada sitio.

        Convencion coherente con Heff:
            H = -dE/dS
        por lo que los terminos intra/inter entran con signo positivo, y la anisotropia sigue el signo de Kz_field.
    """
    # ========================================================
    # A. ENERGÍA INTRACADENA (Heisenberg)
    # Factor de 0.5 para no contar el enlace i->j y j->i dos veces
    # ========================================================
    # (J_intra @ S) devuelve el campo intra que siente cada espín. 
    # Al multiplicarlo por S y sumar por filas (axis=1), obtenemos S_i . H_intra_i
    E_intra_1 = 0.5 * np.sum(S1 * (cache.J_intra_1 @ S1), axis=1)
    E_intra_2 = 0.5 * np.sum(S2 * (cache.J_intra_2 @ S2), axis=1)
    
    # ========================================================
    # B. ENERGÍA INTERCADENA (Moiré)
    # No lleva factor 0.5 aquí porque la matriz W mapea de 2 -> 1 unidireccionalmente
    # Usamos las nuevas matrices consolidadas enteras W_12 y W_21
    # ========================================================
    
    # Energía de interacción vista desde la Cadena 1
    E_inter_1 = np.sum(S1 * (cache.W_12 @ S2), axis=1)
    
    # Energía de interacción vista desde la Cadena 2 
    E_inter_2 = np.sum(S2 * (cache.W_21 @ S1), axis=1)
    
    # ========================================================
    # C. ENERGIA DE ANISOTROPIA (Eje Z, eje facil)
    # Consistente con H_ani,z = +Kz_field * S^z => E_ani = -0.5 * Kz_field * (S^z)^2
    # ========================================================
    E_ani_1 = -0.5 * Kz_field * (S1[:, 2] ** 2)
    E_ani_2 = -0.5 * Kz_field * (S2[:, 2] ** 2)
    
    # ========================================================
    # D. ENSAMBLAJE FINAL
    # ========================================================
    # Para la energía local por sitio, repartimos la energía intercadena 
    # mitad y mitad entre los dos átomos que forman el enlace.
    E_site_1 = E_intra_1 + 0.5 * E_inter_1 + E_ani_1
    E_site_2 = E_intra_2 + 0.5 * E_inter_2 + E_ani_2
    
    # La energía total del sistema es la suma de todas las energías locales
    E_total = np.sum(E_site_1) + np.sum(E_site_2)
    
    return E_total, E_site_1, E_site_2


def evaluate_energy_history(History_S1, History_S2, cache, Kz_field):
    """
    Equivalente a tu ET_PBC.
    Calcula la evolución de la energía en el tiempo a partir de la historia.
    
    History_S1: Array de dimensión (T, N1, 3)
    History_S2: Array de dimensión (T, N2, 3)
    """
    num_pasos = History_S1.shape[0]
    Energy_time_series = np.zeros(num_pasos)
    
    for t in range(num_pasos):
        E_tot, _, _ = energy_two_chains_state(
            History_S1[t], 
            History_S2[t], 
            cache, 
            Kz_field
        )
        # Si prefieres la energía promedio por sitio (como en tu E0):
        # N_total = len(History_S1[t]) + len(History_S2[t])
        # Energy_time_series[t] = E_tot / N_total
        
        Energy_time_series[t] = E_tot
        
    return Energy_time_series