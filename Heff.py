import numpy as np
from dataclasses import dataclass
from numba import njit, prange

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

# =========================
# Parametros del modelo general
# =========================
J0_intra = 46.75      
J_perp0 = 0.4         
Kz_meV = 0.76         
muB_meV_T = 0.05788

J0_intra_field = J0_intra / muB_meV_T
Jp0_field = J_perp0 / muB_meV_T
Kz_field = 2.0 * Kz_meV / muB_meV_T

n1 = 1198
n2 = 1198
a1 = 1 + 0.05
a2 = 1.0
x_offset = 0.0
u2 = a2 / a1 - 1.0
h_sep = 0.0

lambda_intra = 0.3
lambda_perp = 0.3
tol_rel = 0.01

@dataclass(frozen=True)
class TwoChainCouplingCache:
    """Contenedor de acoples geometricos consolidados para JIT Total."""
    J_intra_1: object
    J_intra_2: object
    W_12: object   # Matriz consolidada Cadena 1 -> Cadena 2
    W_21: object   # Matriz consolidada Cadena 2 -> Cadena 1
    x1: np.ndarray
    x2: np.ndarray
    use_sparse: bool

# =========================
# Geometria y Distancias
# =========================
def J_kernel_exp(r, J0_field, lamb):
    return J0_field * np.exp(-r / lamb)

def pairwise_distance_1d(x_left, x_right, L=None):
    dx = np.abs(x_left[:, None] - x_right[None, :])
    if L is not None:
        dx = np.minimum(dx, L - dx)
    return dx

def _to_csr_if_available(M, use_sparse=True):
    if use_sparse and sp is not None:
        return sp.csr_matrix(M)
    return M

# =========================
# Acoples generales
# =========================
def build_J_intra_general(x, J0_field=J0_intra_field, lamb=lambda_intra, tol=tol_rel, pbc=True):
    L = None
    if pbc and x.size > 1:
        spacing = np.min(np.diff(np.sort(x)))
        L = x.max() - x.min() + spacing

    r = pairwise_distance_1d(x, x, L=L)
    J = J_kernel_exp(r, J0_field, lamb)
    np.fill_diagonal(J, 0.0)
    J[J < tol * J0_field] = 0.0
    J = np.maximum(J, J.T)
    return J

def precompute_two_chain_couplings(n1, n2, a1=a1, u2=u2, x_offset=x_offset, 
                                   J0_intra_field=J0_intra_field, Jp0_field=Jp0_field, 
                                   lambda_intra=lambda_intra, lambda_perp=lambda_perp,
                                   tol=tol_rel, h_sep=h_sep, use_sparse=True):
    # 1. Cadenas 1D continuas. a_eff es la verdadera distancia interatómica (a1/2)
    a_eff = a1 / 2.0
    x1 = np.arange(n1) * a_eff
    x2 = (np.arange(n2) * a_eff * (1.0 + u2)) + x_offset

    spacing_1 = a_eff
    L_supercell = x1.max() - x1.min() + spacing_1

    # 2. Intracadena
    J_intra_1_dense = build_J_intra_general(x1, J0_field=J0_intra_field, lamb=lambda_intra, tol=tol, pbc=True)
    J_intra_2_dense = build_J_intra_general(x2, J0_field=J0_intra_field, lamb=lambda_intra, tol=tol, pbc=True)

    # 3. Moiré Intercadena Agnóstico (W_12 unificada en un solo paso)
    dx_12 = pairwise_distance_1d(x1, x2, L=L_supercell)
    r_12 = np.sqrt(dx_12**2 + h_sep**2) if h_sep != 0.0 else dx_12
    
    W_12_dense = J_kernel_exp(r_12, Jp0_field, lambda_perp)
    W_12_dense[W_12_dense < tol * Jp0_field] = 0.0
    W_21_dense = W_12_dense.T

    return TwoChainCouplingCache(
        J_intra_1=_to_csr_if_available(J_intra_1_dense, use_sparse=use_sparse),
        J_intra_2=_to_csr_if_available(J_intra_2_dense, use_sparse=use_sparse),
        W_12=_to_csr_if_available(W_12_dense, use_sparse=use_sparse),
        W_21=_to_csr_if_available(W_21_dense, use_sparse=use_sparse),
        x1=x1, x2=x2, use_sparse=(use_sparse and sp is not None)
    )

# =========================
# RUTINAS NUMBA JIT PARA EL CAMPO
# =========================
@njit(parallel=True, fastmath=True)
def calculate_Heff_jit(S1, S2, J1_d, J1_i, J1_p, J2_d, J2_i, J2_p,
                       W12_d, W12_i, W12_p, W21_d, W21_i, W21_p, Kz, H1_out, H2_out):
    """Calcula el campo efectivo total (Intra + Inter + Anisotropía) directamente en H_out."""
    
    # Evaluar Campo para Cadena 1
    for i in prange(S1.shape[0]):
        vx = 0.0; vy = 0.0; vz = 0.0
        for k in range(J1_p[i], J1_p[i+1]):
            col = J1_i[k]
            vx += J1_d[k] * S1[col, 0]
            vy += J1_d[k] * S1[col, 1]
            vz += J1_d[k] * S1[col, 2]
        for k in range(W12_p[i], W12_p[i+1]):
            col = W12_i[k]
            vx += W12_d[k] * S2[col, 0]
            vy += W12_d[k] * S2[col, 1]
            vz += W12_d[k] * S2[col, 2]
        
        H1_out[i, 0] = -vx
        H1_out[i, 1] = -vy
        H1_out[i, 2] = -vz + Kz * S1[i, 2]

    # Evaluar Campo para Cadena 2
    for i in prange(S2.shape[0]):
        vx = 0.0; vy = 0.0; vz = 0.0
        for k in range(J2_p[i], J2_p[i+1]):
            col = J2_i[k]
            vx += J2_d[k] * S2[col, 0]
            vy += J2_d[k] * S2[col, 1]
            vz += J2_d[k] * S2[col, 2]
        for k in range(W21_p[i], W21_p[i+1]):
            col = W21_i[k]
            vx += W21_d[k] * S1[col, 0]
            vy += W21_d[k] * S1[col, 1]
            vz += W21_d[k] * S1[col, 2]

        H2_out[i, 0] = -vx
        H2_out[i, 1] = -vy
        H2_out[i, 2] = -vz + Kz * S2[i, 2]