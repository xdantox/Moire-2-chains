import numpy as np
from dataclasses import dataclass
from numba import njit, prange

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

# =========================================================
# 1. PARÁMETROS DEL MODELO GENERAL (ORIGINALES PRESERVADOS)
# =========================================================
J0_intra = 1.0      
J_perp0 = 1.0 
Kz_meV = 0.025 

J0_intra_field = J0_intra 
Jp0_field = J_perp0 
Kz_field = 2.0 * Kz_meV  # Campo anisotrópico efectivo: 0.05

# Parámetros de red
a1 = 1.0 + 0.1  # epsilon (~1.101001)
a2 = 1.0

n1 = 4000
n2 = int(round(n1 * (a1 / a2)))

x_offset = 0.0
u2 = a2 / a1 - 1.0            # Relación de desajuste original

# Constantes del manuscrito
h_sep = 1.7  
beta = 2.0
l = 1.0
tol_rel = 1e-6

# =========================================================
# 2. CONTENEDOR Y FUNCIONES DE DISTANCIA
# =========================================================
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

def J_kernel_exp(r, J0_field, r_ref, beta=beta):
    return J0_field * np.exp(beta * (1.0 - r / r_ref))

def pairwise_distance_1d(x_left, x_right, L=None):
    dx = np.abs(x_left[:, None] - x_right[None, :])
    if L is not None:
        dx = np.minimum(dx, L - dx)
    return dx

def _to_csr_if_available(M, use_sparse=True):
    if use_sparse and sp is not None:
        return sp.csr_matrix(M)
    return M

def build_J_intra_general(x, J0_field=J0_intra_field, r_ref=l, beta=beta, tol=tol_rel, pbc=True):
    L = None
    if pbc and x.size > 1:
        spacing = np.min(np.diff(np.sort(x)))
        L = x.max() - x.min() + spacing

    r = pairwise_distance_1d(x, x, L=L)
    J = J_kernel_exp(r, J0_field, r_ref=r_ref, beta=beta)
    np.fill_diagonal(J, 0.0)
    J[J < tol * J0_field] = 0.0
    J = np.maximum(J, J.T)
    return J

# =========================================================
# 3. PRE-CÓMPUTO CORREGIDO (CONSERVANDO u2 Y VARIABLES)
# =========================================================
def precompute_two_chain_couplings(n1, n2, a1=a1, u2=u2, x_offset=x_offset, 
                                   J0_intra_field=J0_intra_field, Jp0_field=Jp0_field, 
                                   l=l, beta=beta,
                                   tol=tol_rel, h_sep=h_sep, use_sparse=True, 
                                   is_pbc=True):
    
    # 1. Espaciado atómico real entre espines (a0 = a / 2)
    # Al multiplicar a_eff * (1.0 + u2) con a_eff = a1/2, se obtiene a2/2 de forma exacta
    a_eff = a1 / 2.0 
    x1 = np.arange(n1) * a_eff
    x2 = (np.arange(n2) * a_eff * (1.0 + u2)) + x_offset

    spacing_1 = a_eff
    if is_pbc:
        L_supercell = x1.max() - x1.min() + spacing_1
    else:
        L_supercell = None

    # 2. Escalas de referencia para distancia atómica a0 = l / 2 = 0.5
    r_ref_atom = l / 2.0
    # zd = 1.7 * a0 = 0.85
    h_sep_eff = h_sep * (l / 2.0) if h_sep > 1.0 else h_sep

    # 3. Matrices Intracadena
    J_intra_1_dense = build_J_intra_general(x1, J0_field=J0_intra_field, r_ref=r_ref_atom, beta=beta, tol=tol, pbc=is_pbc)
    J_intra_2_dense = build_J_intra_general(x2, J0_field=J0_intra_field, r_ref=r_ref_atom, beta=beta, tol=tol, pbc=is_pbc)

    # 4. Matrices Intercadena
    dx_12 = pairwise_distance_1d(x1, x2, L=L_supercell)
    r_12 = np.sqrt(dx_12**2 + h_sep_eff**2) if h_sep_eff != 0.0 else dx_12
    
    W_12_dense = J_kernel_exp(r_12, Jp0_field, r_ref=r_ref_atom, beta=beta)
    W_12_dense[W_12_dense < tol * Jp0_field] = 0.0
    W_21_dense = W_12_dense.T

    return TwoChainCouplingCache(
        J_intra_1=_to_csr_if_available(J_intra_1_dense, use_sparse=use_sparse),
        J_intra_2=_to_csr_if_available(J_intra_2_dense, use_sparse=use_sparse),
        W_12=_to_csr_if_available(W_12_dense, use_sparse=use_sparse),
        W_21=_to_csr_if_available(W_21_dense, use_sparse=use_sparse),
        x1=x1, x2=x2, use_sparse=(use_sparse and sp is not None)
    )

# =========================================================
# 4. RUTINAS NUMBA JIT PARA EL CAMPO EFECTIVO (Heff)
# =========================================================
@njit(parallel=True, fastmath=True)
def calculate_Heff_jit(S1, S2, J1_d, J1_i, J1_p, J2_d, J2_i, J2_p,
                       W12_d, W12_i, W12_p, W21_d, W21_i, W21_p, Kz, H1_out, H2_out):
    """Calcula el campo efectivo total (Intra + Inter + Anisotropía) directamente en H_out."""
    
    # Evaluar Campo para Cadena 1
    for i in prange(S1.shape[0]):
        vx = 0.0
        vy = 0.0
        vz = 0.0
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
        vx = 0.0
        vy = 0.0
        vz = 0.0
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