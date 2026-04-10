import numpy as np
from dataclasses import dataclass

try:
    import scipy.sparse as sp
except ImportError:  # Fallback: permite correr sin SciPy usando arreglos densos.
    sp = None

# =========================
# Parametros del modelo general
# =========================
J0_intra = 46.75      # amplitud intracadena J(r) en meV
J_perp0 = 0        # amplitud intercadena J_perp(r) en meV
Kz_meV = 0.76         # anisotropia uniaxial -K (S^z)^2 en meV

muB_meV_T = 0.05788

# Unidades de campo efectivo
J0_intra_field = J0_intra / muB_meV_T
Jp0_field = J_perp0 / muB_meV_T
Kz_field = 2.0 * Kz_meV / muB_meV_T

# Geometria
n1 = 1198
n2 = 1198
a1 = 1 + 0.001
a2 = 1.0
x_offset = 0.0
u2 = a2 / a1 - 1.0
h_sep = 0.0

# Alcance de los acoples exponenciales
lambda_intra = 2.0
lambda_perp = 8.0

# Truncado para eficiencia
# Se elimina todo acople < 1% del maximo y se limita a 4 por espin.
tol_rel = 0.01
max_couplings_per_spin = 4


@dataclass(frozen=True)
class TwoChainCouplingCache:
    """Contenedor de acoples geometricos precomputados para dos cadenas."""

    J_intra_1: object
    J_intra_2: object
    W_AA: object
    W_AB: object
    W_BA: object
    W_BB: object
    x1: np.ndarray
    x2: np.ndarray
    x1A: np.ndarray
    x1B: np.ndarray
    x2A: np.ndarray
    x2B: np.ndarray
    use_sparse: bool


# =========================
# Geometria de la base A/B
# =========================
def build_ab_positions(n_cells, a, u=0.0, d_parallel=0.0):
    """
    R_{n,A} = (1+u) * (n a) + d_parallel
    R_{n,B} = (1+u) * (n a + a/2) + d_parallel
    """
    r_n = np.arange(n_cells) * a
    tau_A = 0.0
    tau_B = 0.5 * a
    scale = 1.0 + u
    x_A = scale * (r_n + tau_A) + d_parallel
    x_B = scale * (r_n + tau_B) + d_parallel
    return x_A, x_B


def merge_scalar_AB(A, B):
    """Reconstruye [A0, B0, A1, B1, ...] para arreglos 1D."""
    out = np.empty(A.shape[0] + B.shape[0], dtype=A.dtype)
    out[0::2] = A
    out[1::2] = B
    return out


def split_AB(cadena):
    """Asume orden [A0, B0, A1, B1, ...]."""
    return cadena[0::2], cadena[1::2]


def merge_AB(A, B):
    """Reconstruye [A0, B0, A1, B1, ...] para arreglos de espines."""
    out = np.empty((A.shape[0] + B.shape[0], 3), dtype=A.dtype)
    out[0::2] = A
    out[1::2] = B
    return out


# =========================
# Kernels y truncado
# =========================
def J_kernel_exp(r, J0_field, lamb):
    return J0_field * np.exp(-r / lamb)


def pairwise_distance_1d(x_left, x_right, L=None):
    """Distancia 1D, con imagen minima opcional para PBC."""
    dx = np.abs(x_left[:, None] - x_right[None, :])
    if L is not None:
        dx = np.minimum(dx, L - dx)
    return dx


def keep_topk_per_row(W, k):
    """Conserva hasta k acoples no nulos por fila."""
    if k is None or k <= 0:
        return W

    Wk = np.zeros_like(W)
    for i in range(W.shape[0]):
        nz = np.flatnonzero(W[i] > 0.0)
        if nz.size == 0:
            continue
        if nz.size <= k:
            Wk[i, nz] = W[i, nz]
            continue
        idx = nz[np.argpartition(W[i, nz], -k)[-k:]]
        Wk[i, idx] = W[i, idx]
    return Wk


def _to_csr_if_available(M, use_sparse=True):
    """Convierte a CSR si SciPy esta disponible y se solicita."""
    if use_sparse and sp is not None:
        return sp.csr_matrix(M)
    return M


def nnz_matrix(M):
    """Numero de elementos no nulos, compatible con denso y disperso."""
    if hasattr(M, "nnz"):
        return int(M.nnz)
    return int(np.count_nonzero(M))


# =========================
# Acoples generales J(R_{m beta}^l - R_{n alpha}^l)
# =========================
def build_J_intra_general(
    x,
    J0_field=J0_intra_field,
    lamb=lambda_intra,
    tol=tol_rel,
    max_neighbors=max_couplings_per_spin,
    pbc=True,
):
    """
    Matriz intracadena todos-contra-todos, truncada por:
    1) tolerancia relativa (tol * J0_field)
    2) maximo numero de acoples por espin (fila).
    """
    L = None
    if pbc and x.size > 1:
        spacing = np.min(np.diff(np.sort(x)))
        L = x.max() - x.min() + spacing

    r = pairwise_distance_1d(x, x, L=L)
    J = J_kernel_exp(r, J0_field, lamb)
    np.fill_diagonal(J, 0.0)

    J[J < tol * J0_field] = 0.0
    J = keep_topk_per_row(J, max_neighbors)

    # Mantiene simetria J_ij = J_ji.
    J = np.maximum(J, J.T)
    return J


def build_W_interchain_blocks(
    x1A,
    x1B,
    x2A,
    x2B,
    J0_field=Jp0_field,
    lamb=lambda_perp,
    tol=tol_rel,
    max_neighbors=max_couplings_per_spin,
    h=0.0,
):
    """
    Construye los cuatro bloques intercadena:
      W_AA(i,j) = J_perp(|R^2_{j,A} - R^1_{i,A}|)
      W_AB(i,j) = J_perp(|R^2_{j,B} - R^1_{i,A}|)
      W_BA(i,j) = J_perp(|R^2_{j,A} - R^1_{i,B}|)
      W_BB(i,j) = J_perp(|R^2_{j,B} - R^1_{i,B}|)

    El truncado de vecinos se aplica por espin fuente considerando conjuntamente
    los dos canales de salida (A2 y B2).
    """

    def _build_two_channel(x_src, x_tgtA, x_tgtB):
        dx_A = np.abs(x_src[:, None] - x_tgtA[None, :])
        dx_B = np.abs(x_src[:, None] - x_tgtB[None, :])

        if h != 0.0:
            r_A = np.sqrt(dx_A * dx_A + h * h)
            r_B = np.sqrt(dx_B * dx_B + h * h)
        else:
            r_A = dx_A
            r_B = dx_B

        WA = J_kernel_exp(r_A, J0_field, lamb)
        WB = J_kernel_exp(r_B, J0_field, lamb)

        WA[WA < tol * J0_field] = 0.0
        WB[WB < tol * J0_field] = 0.0

        Wcat = np.concatenate([WA, WB], axis=1)
        Wcat = keep_topk_per_row(Wcat, max_neighbors)

        na = WA.shape[1]
        return Wcat[:, :na], Wcat[:, na:]

    W_AA, W_AB = _build_two_channel(x1A, x2A, x2B)
    W_BA, W_BB = _build_two_channel(x1B, x2A, x2B)
    return W_AA, W_AB, W_BA, W_BB


def precompute_two_chain_couplings(
    n1,
    n2,
    a1=a1,
    u2=u2,
    x_offset=x_offset,
    J0_intra_field=J0_intra_field,
    Jp0_field=Jp0_field,
    lambda_intra=lambda_intra,
    lambda_perp=lambda_perp,
    tol=tol_rel,
    max_neighbors=max_couplings_per_spin,
    h_sep=h_sep,
    use_sparse=True,
):
    """
    Precomputa TODOS los acoples geometricos una sola vez.

    Retorna un cache listo para usarse en el integrador temporal sin
    recalcular distancias ni truncados en cada dt.
    """
    if n1 % 2 != 0 or n2 % 2 != 0:
        raise ValueError("n1 y n2 deben ser pares para usar base A/B.")

    n1_cells = n1 // 2
    n2_cells = n2 // 2

    x1A, x1B = build_ab_positions(n1_cells, a1, u=0.0, d_parallel=0.0)
    x2A, x2B = build_ab_positions(n2_cells, a1, u=u2, d_parallel=x_offset)
    x1 = merge_scalar_AB(x1A, x1B)
    x2 = merge_scalar_AB(x2A, x2B)

    J_intra_1_dense = build_J_intra_general(
        x1,
        J0_field=J0_intra_field,
        lamb=lambda_intra,
        tol=tol,
        max_neighbors=max_neighbors,
        pbc=True,
    )
    J_intra_2_dense = build_J_intra_general(
        x2,
        J0_field=J0_intra_field,
        lamb=lambda_intra,
        tol=tol,
        max_neighbors=max_neighbors,
        pbc=True,
    )

    W_AA_dense, W_AB_dense, W_BA_dense, W_BB_dense = build_W_interchain_blocks(
        x1A,
        x1B,
        x2A,
        x2B,
        J0_field=Jp0_field,
        lamb=lambda_perp,
        tol=tol,
        max_neighbors=max_neighbors,
        h=h_sep,
    )

    return TwoChainCouplingCache(
        J_intra_1=_to_csr_if_available(J_intra_1_dense, use_sparse=use_sparse),
        J_intra_2=_to_csr_if_available(J_intra_2_dense, use_sparse=use_sparse),
        W_AA=_to_csr_if_available(W_AA_dense, use_sparse=use_sparse),
        W_AB=_to_csr_if_available(W_AB_dense, use_sparse=use_sparse),
        W_BA=_to_csr_if_available(W_BA_dense, use_sparse=use_sparse),
        W_BB=_to_csr_if_available(W_BB_dense, use_sparse=use_sparse),
        x1=x1,
        x2=x2,
        x1A=x1A,
        x1B=x1B,
        x2A=x2A,
        x2B=x2B,
        use_sparse=(use_sparse and sp is not None),
    )


# =========================
# Campo efectivo
# =========================
def H_intra_general(cadena, J_intra, Kz=Kz_field):
    """
    H_n = sum_m J_nm S_m + Kz S_n^z z_hat
    """
    H_total = J_intra @ cadena
    H_total[:, 2] += Kz * cadena[:, 2]
    return H_total


def H_eff_two_chains_pbc(
    S1,
    S2,
    W_AA,
    W_AB,
    W_BA,
    W_BB,
    J_intra_1,
    J_intra_2,
    Kz=Kz_field,
):
    """
    S1: (2*N1,3), S2: (2*N2,3), orden [A0,B0,A1,B1,...]
    W_AA, W_AB, W_BA, W_BB: bloques intercadena (N1,N2)
    """
    if S1.shape[0] % 2 != 0 or S2.shape[0] % 2 != 0:
        raise ValueError("S1 y S2 deben tener numero par de sitios (base A/B).")

    S1A, S1B = split_AB(S1)
    S2A, S2B = split_AB(S2)

    # Intracadena general
    H1 = H_intra_general(S1, J_intra_1, Kz=Kz)
    H2 = H_intra_general(S2, J_intra_2, Kz=Kz)

    H1A, H1B = split_AB(H1)
    H2A, H2B = split_AB(H2)

    # Intercadena por bloques AA, AB, BA, BB
    H1A += W_AA @ S2A + W_AB @ S2B
    H1B += W_BA @ S2A + W_BB @ S2B

    H2A += W_AA.T @ S1A + W_BA.T @ S1B
    H2B += W_AB.T @ S1A + W_BB.T @ S1B

    return merge_AB(H1A, H1B), merge_AB(H2A, H2B)


def H_eff_two_chains_cached(S1, S2, coupling_cache, Kz=Kz_field):
    """Evalua H_eff usando acoples precomputados (densos o CSR)."""
    return H_eff_two_chains_pbc(
        S1,
        S2,
        coupling_cache.W_AA,
        coupling_cache.W_AB,
        coupling_cache.W_BA,
        coupling_cache.W_BB,
        coupling_cache.J_intra_1,
        coupling_cache.J_intra_2,
        Kz=Kz,
    )


# =========================
# Ejemplo de uso
# =========================
if __name__ == "__main__":
    cache = precompute_two_chain_couplings(
        n1=n1,
        n2=n2,
        a1=a1,
        u2=u2,
        x_offset=x_offset,
        J0_intra_field=J0_intra_field,
        Jp0_field=Jp0_field,
        lambda_intra=lambda_intra,
        lambda_perp=lambda_perp,
        tol=tol_rel,
        max_neighbors=max_couplings_per_spin,
        h_sep=h_sep,
        use_sparse=True,
    )

    # Espines iniciales normalizados
    rng = np.random.default_rng(1234)
    S1 = rng.normal(size=(n1, 3))
    S1 /= np.linalg.norm(S1, axis=1, keepdims=True)
    S2 = rng.normal(size=(n2, 3))
    S2 /= np.linalg.norm(S2, axis=1, keepdims=True)

    # Campo efectivo total
    H1, H2 = H_eff_two_chains_cached(
        S1,
        S2,
        coupling_cache=cache,
        Kz=Kz_field,
    )

    print("H1 shape:", H1.shape, "H2 shape:", H2.shape)
    print("Acoples no nulos J1/J2:", nnz_matrix(cache.J_intra_1), nnz_matrix(cache.J_intra_2))
    print(
        "Acoples no nulos W_AA/W_AB/W_BA/W_BB:",
        nnz_matrix(cache.W_AA),
        nnz_matrix(cache.W_AB),
        nnz_matrix(cache.W_BA),
        nnz_matrix(cache.W_BB),
    )
