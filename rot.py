import numpy as np
import math
import numba
from numba import njit, prange
from Heff import calculate_Heff_jit

# Parámetros globales
gamma = 1.7e11
alpha = 0

# =========================================================
# RUTINAS NUMBA AUXILIARES (Sin asignación de memoria)
# =========================================================
@njit(parallel=True, fastmath=True)
def calculate_omega_jit(H, S, gamma_val, alpha_val, w_out):
    """Calcula la frecuencia angular omega puramente escalar"""
    for i in prange(S.shape[0]):
        # Producto cruz manual para evitar np.cross
        cx = S[i, 1]*H[i, 2] - S[i, 2]*H[i, 1]
        cy = S[i, 2]*H[i, 0] - S[i, 0]*H[i, 2]
        cz = S[i, 0]*H[i, 1] - S[i, 1]*H[i, 0]
        w_out[i, 0] = gamma_val * (H[i, 0] + alpha_val * cx)
        w_out[i, 1] = gamma_val * (H[i, 1] + alpha_val * cy)
        w_out[i, 2] = gamma_val * (H[i, 2] + alpha_val * cz)

@njit(parallel=True, fastmath=True)
def midpoint_normalize_jit(S_old, S_new, S_mid):
    """Calcula el punto medio y lo proyecta a la esfera"""
    for i in prange(S_old.shape[0]):
        mx = 0.5 * (S_old[i, 0] + S_new[i, 0])
        my = 0.5 * (S_old[i, 1] + S_new[i, 1])
        mz = 0.5 * (S_old[i, 2] + S_new[i, 2])
        norm = math.sqrt(mx*mx + my*my + mz*mz)
        S_mid[i, 0] = mx / norm
        S_mid[i, 1] = my / norm
        S_mid[i, 2] = mz / norm

@njit(parallel=True, fastmath=True)
def apply_rodrigues_rotation_jit(c_old, w, dt, c_new):
    """Matriz de Rodrigues compilada"""
    for i in prange(c_old.shape[0]):
        wx, wy, wz = w[i, 0], w[i, 1], w[i, 2]
        wnorm = math.sqrt(wx*wx + wy*wy + wz*wz)
        
        if wnorm == 0.0:
            c_new[i, 0], c_new[i, 1], c_new[i, 2] = c_old[i, 0], c_old[i, 1], c_old[i, 2]
            continue
            
        ax, ay, az = wx / wnorm, wy / wnorm, wz / wnorm
        angle = wnorm * dt
        c = math.cos(angle); s = math.sin(angle); t = 1.0 - c
        
        R00 = t*ax*ax + c;       R01 = t*ax*ay - az*s; R02 = t*ax*az + ay*s
        R10 = t*ax*ay + az*s; R11 = t*ay*ay + c;       R12 = t*ay*az - ax*s
        R20 = t*ax*az - ay*s; R21 = t*ay*az + ax*s; R22 = t*az*az + c
        
        cx, cy, cz = c_old[i, 0], c_old[i, 1], c_old[i, 2]
        nx = R00*cx + R01*cy + R02*cz
        ny = R10*cx + R11*cy + R12*cz
        nz = R20*cx + R21*cy + R22*cz
        
        nnorm = math.sqrt(nx*nx + ny*ny + nz*nz)
        c_new[i, 0] = nx / nnorm
        c_new[i, 1] = ny / nnorm
        c_new[i, 2] = nz / nnorm

# =========================================================
# EL BUCLE MAESTRO: JIT TOTAL (CERO PYTHON EN LA DINÁMICA)
# =========================================================
@njit(fastmath=True)
def run_llg_jit_total(S1_0, S2_0, num_pasos, dt, Kz, n_iter, gamma_val, alpha_val,
                      J1_d, J1_i, J1_p, J2_d, J2_i, J2_p,
                      W12_d, W12_i, W12_p, W21_d, W21_i, W21_p):
    """
    Este bucle gigante se compila completamente en C.
    No asigna memoria temporal ni sufre de "Garbage Collection".
    """
    N1 = S1_0.shape[0]
    N2 = S2_0.shape[0]

    # Asignamos la memoria maestra UNA SOLA VEZ para toda la simulación
    History_S1 = np.zeros((num_pasos, N1, 3))
    History_S2 = np.zeros((num_pasos, N2, 3))
    History_S1[0] = S1_0
    History_S2[0] = S2_0

    S1_old = S1_0.copy()
    S2_old = S2_0.copy()

    # Arreglos de estado temporales que serán reciclados un millón de veces
    S1_new = np.empty_like(S1_old); S2_new = np.empty_like(S2_old)
    S1_mid = np.empty_like(S1_old); S2_mid = np.empty_like(S2_old)
    H1 = np.empty_like(S1_old);     H2 = np.empty_like(S2_old)
    w1 = np.empty_like(S1_old);     w2 = np.empty_like(S2_old)

    for j in range(num_pasos - 1):
        
        # 1. Predictor (Euler o Rodrigues inicial)
        calculate_Heff_jit(S1_old, S2_old, J1_d, J1_i, J1_p, J2_d, J2_i, J2_p,
                           W12_d, W12_i, W12_p, W21_d, W21_i, W21_p, Kz, H1, H2)
        calculate_omega_jit(H1, S1_old, gamma_val, alpha_val, w1)
        calculate_omega_jit(H2, S2_old, gamma_val, alpha_val, w2)
        
        apply_rodrigues_rotation_jit(S1_old, w1, dt, S1_new)
        apply_rodrigues_rotation_jit(S2_old, w2, dt, S2_new)

        # 2. Corrector (Punto Medio Implícito)
        for _ in range(n_iter):
            midpoint_normalize_jit(S1_old, S1_new, S1_mid)
            midpoint_normalize_jit(S2_old, S2_new, S2_mid)

            calculate_Heff_jit(S1_mid, S2_mid, J1_d, J1_i, J1_p, J2_d, J2_i, J2_p,
                               W12_d, W12_i, W12_p, W21_d, W21_i, W21_p, Kz, H1, H2)

            calculate_omega_jit(H1, S1_mid, gamma_val, alpha_val, w1)
            calculate_omega_jit(H2, S2_mid, gamma_val, alpha_val, w2)

            apply_rodrigues_rotation_jit(S1_old, w1, dt, S1_new)
            apply_rodrigues_rotation_jit(S2_old, w2, dt, S2_new)

        # Actualización de estados
        S1_old[:] = S1_new
        S2_old[:] = S2_new

        History_S1[j+1] = S1_old
        History_S2[j+1] = S2_old

    return History_S1, History_S2

# =========================================================
# FUNCIÓN ENVOLTORIO PARA LLAMAR DESDE TU MAIN.PY
# =========================================================
def execute_simulation(S1_0, S2_0, num_pasos, dt, coupling_cache, Kz, n_iter=14, num_cores=8):
    """
    Desempaqueta las matrices de SciPy y lanza el integrador JIT.
    Permite configurar dinámicamente el número de núcleos de CPU a utilizar.
    """
    import numba
    
    # 1. Configuramos el hardware según el input del usuario
    numba.set_num_threads(num_cores)
    print(f"[Hardware] Motor JIT configurado para usar {num_cores} núcleos físicos.")

    # 2. Desempaquetamos los arreglos crudos
    J1 = coupling_cache.J_intra_1
    J2 = coupling_cache.J_intra_2
    W12 = coupling_cache.W_12
    W21 = coupling_cache.W_21

    # 3. Lanzamos el motor C
    Hist_S1, Hist_S2 = run_llg_jit_total(
        S1_0, S2_0, num_pasos, dt, Kz, n_iter, gamma, alpha,
        J1.data, J1.indices, J1.indptr,
        J2.data, J2.indices, J2.indptr,
        W12.data, W12.indices, W12.indptr,
        W21.data, W21.indices, W21.indptr
    )
    return Hist_S1, Hist_S2