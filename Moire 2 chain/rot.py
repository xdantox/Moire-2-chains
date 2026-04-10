import numpy as np
from functools import lru_cache

from Heff import H_eff_two_chains_cached, Kz_field, n1, n2, precompute_two_chain_couplings

# Parámetros globales de la dinámica
gamma = 1.7e11
alpha = 1.2


@lru_cache(maxsize=1)
def get_default_coupling_cache():
    """Construye el cache de acoples una sola vez usando parametros por defecto de Heff."""
    return precompute_two_chain_couplings(n1=n1, n2=n2)

def omega(Heff, m):
    """ Frecuencia angular de precesión/amortiguamiento (forma LLG explícita) """
    return gamma * (Heff - alpha * np.cross(m, Heff))

def apply_rodrigues_rotation(c_old, w, dt):
    """ 
    Aplica la matriz de rotación de Rodrigues de forma vectorizada a un arreglo de espines.
    c_old: (N, 3) espines
    w: (N, 3) vectores de frecuencia angular
    """
    w_norm = np.linalg.norm(w, axis=1)
    
    # Evitar divisiones por cero si w_norm es 0
    safe_norm = np.where(w_norm == 0, 1.0, w_norm)
    axis = w / safe_norm[:, None]
    angle = w_norm * dt
    
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    
    # Construcción vectorizada de matrices de rotación R (N, 3, 3)
    R_matrices = np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ]).transpose(2, 0, 1)

    # c_new = R * c_old
    c_new = np.einsum('nij,nj->ni', R_matrices, c_old)
    
    # Forzar normalización estricta por seguridad numérica
    c_new /= np.linalg.norm(c_new, axis=1)[:, None]
    return c_new


def implicit_midpoint_step_two_chains(
    S1_old,
    S2_old,
    coupling_cache,
    dt,
    Kz=Kz_field,
    n_iter=14,
):
    """
    Iterador LLG vectorial acoplado para un sistema bicapa Moiré.
    """
    if coupling_cache is None:
        coupling_cache = get_default_coupling_cache()

    # ---------------------------------------------------------
    # 1. Estimación Inicial (Predictor - Paso explícito de Euler)
    # ---------------------------------------------------------
    H1_0, H2_0 = H_eff_two_chains_cached(S1_old, S2_old, coupling_cache, Kz)
    
    w1_0 = omega(H1_0, S1_old)
    w2_0 = omega(H2_0, S2_old)
    
    S1_new = S1_old + dt * np.cross(w1_0, S1_old, axisa=1, axisb=1)
    S2_new = S2_old + dt * np.cross(w2_0, S2_old, axisa=1, axisb=1)
    
    S1_new /= np.linalg.norm(S1_new, axis=1)[:, None]
    S2_new /= np.linalg.norm(S2_new, axis=1)[:, None]

    # ---------------------------------------------------------
    # 2. Refinamiento del Punto Medio (Corrector)
    # ---------------------------------------------------------
    for _ in range(n_iter):
        # Calcular estados en el punto medio implícito t + dt/2
        S1_mid = 0.5 * (S1_old + S1_new)
        S2_mid = 0.5 * (S2_old + S2_new)
        
        S1_mid /= np.linalg.norm(S1_mid, axis=1)[:, None]
        S2_mid /= np.linalg.norm(S2_mid, axis=1)[:, None]
        
        # El campo efectivo cruza información entre ambas cadenas
        H1_mid, H2_mid = H_eff_two_chains_cached(S1_mid, S2_mid, coupling_cache, Kz)
        
        # Frecuencias en el punto medio
        w1_mid = omega(H1_mid, S1_mid)
        w2_mid = omega(H2_mid, S2_mid)
        
        # Aplicar Rotación de Rodrigues exacta desde S_old usando w_mid
        S1_new = apply_rodrigues_rotation(S1_old, w1_mid, dt)
        S2_new = apply_rodrigues_rotation(S2_old, w2_mid, dt)
        
    return S1_new, S2_new