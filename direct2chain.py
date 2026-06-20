"""
Direct initialization for a two-chain ground state using a modulated ansatz
and the Heff-derived Hamiltonian (intra + inter + anisotropy).
"""

import math
import os
import time
from typing import Sequence, Tuple

import numpy as np

try:
	from scipy.optimize import minimize
except ImportError:
	minimize = None

from Heff import Kz_field as Kz_field_default
from Heff import a1 as a1_default
from Heff import n1 as n1_default
from Heff import n2 as n2_default
from Heff import precompute_two_chain_couplings
from Heff import u2 as u2_default
from E import energy_two_chains_state
from cadena0 import build_afm_state_from_npy

# ==========================================
# 1. GLOBAL PARAMETERS
# ==========================================
# Physical defaults from Heff.py
n1 = n1_default
n2 = n2_default
Kz_field = Kz_field_default
a1 = a1_default
u2 = u2_default

# ==========================================
# 2. MINIMIZER SETTINGS (ANSATZ)
# ==========================================
USE_MINIMIZER = True
MAX_M_POINTS = 200  # coarse grid if n1 or n2 is large
M1_VALUES = np.arange(n1)    # set to np.arange(n1) for a full scan
M2_VALUES = np.arange(n2)    # set to np.arange(n2) for a full scan

PARAM_NAMES = (
	"mz1",
	"mz2",
	"gamma1",
	"gamma2",
	"alpha1",
	"alpha2",
	"phi_ind1",
	"phi_ind2",
	"delta1",
	"delta2",
)

MZ_BOUNDS = (-0.999, 0.999)
ANGLE_BOUNDS = (-math.pi, math.pi)
DEFAULT_BOUNDS = (MZ_BOUNDS, MZ_BOUNDS) + (ANGLE_BOUNDS,) * 8

INIT_GUESS = np.array([0.0, 0.0, -0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
MINIMIZER_OPTIONS = {"maxiter": 300}

CHAIN1_BOUNDS = (MZ_BOUNDS,) + (ANGLE_BOUNDS,) * 4
CHAIN2_BOUNDS = (MZ_BOUNDS,) + (ANGLE_BOUNDS,) * 4
CHAIN2_FIXED_PHI_BOUNDS = (MZ_BOUNDS,) + (ANGLE_BOUNDS,) * 3

CHAIN1_INIT_GUESS = np.array(
	[INIT_GUESS[0], INIT_GUESS[2], INIT_GUESS[4], INIT_GUESS[6], INIT_GUESS[8]],
	dtype=float,
)
CHAIN2_INIT_GUESS = np.array(
	[INIT_GUESS[1], INIT_GUESS[3], INIT_GUESS[5], INIT_GUESS[7], INIT_GUESS[9]],
	dtype=float,
)

# Optional fallback from a saved spin history
USE_NPY_FALLBACK = False
NPY_S1_PATH = "spin_history_cadena1 relax MC = 4 J_perp = 0.4.npy"
NPY_S2_PATH = "spin_history_cadena2 relax MC = 4 J_perp = 0.4.npy"


# ==========================================
# 3. ANSATZ UTILS
# ==========================================
def _wrap_pi(x: float) -> float:
	return (x + math.pi) % (2.0 * math.pi) - math.pi


def _q_from_winding(M: float | np.ndarray, chain_length: int) -> float | np.ndarray:
	if chain_length <= 0:
		raise ValueError("chain_length must be positive")
	factor = (2.0 * math.pi) / float(chain_length)
	q = factor * np.asarray(M, dtype=float)
	if q.ndim == 0:
		return float(q)
	return q


def _build_spiral_chain(
	n_sites: int,
	q: float,
	mz: float,
	gamma: float,
	alpha_ind: float,
	phi_ind: float,
	phase_shift: float = 0.0,
	idx: np.ndarray | None = None,
	parity: np.ndarray | None = None,
) -> np.ndarray:
	if idx is None:
		idx = np.arange(n_sites, dtype=np.int64)
	if parity is None:
		parity = np.where((idx & 1) == 0, 1.0, -1.0)

	theta = idx * q + gamma * parity
	if alpha_ind != 0.0:
		theta += alpha_ind * np.sin(2.0 * q * idx + phi_ind)
	theta += phase_shift

	mz = max(-0.999, min(0.999, float(mz)))
	plane = math.sqrt(max(0.0, 1.0 - mz * mz))

	spins = np.zeros((n_sites, 3), dtype=float)
	spins[:, 0] = plane * np.cos(theta)
	spins[:, 1] = plane * np.sin(theta)
	spins[:, 2] = mz
	return spins


def _build_two_chain_state_from_params(
	q1: float,
	q2: float,
	params: np.ndarray,
	idx1: np.ndarray,
	parity1: np.ndarray,
	idx2: np.ndarray,
	parity2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	mz1, mz2, gamma1, gamma2, alpha1, alpha2, phi_ind1, phi_ind2, delta1, delta2 = params
	S1 = _build_spiral_chain(
		n_sites=idx1.size,
		q=q1,
		mz=mz1,
		gamma=gamma1,
		alpha_ind=alpha1,
		phi_ind=phi_ind1,
		phase_shift=delta1,
		idx=idx1,
		parity=parity1,
	)
	S2 = _build_spiral_chain(
		n_sites=idx2.size,
		q=q2,
		mz=mz2,
		gamma=gamma2,
		alpha_ind=alpha2,
		phi_ind=phi_ind2,
		phase_shift=delta2,
		idx=idx2,
		parity=parity2,
	)
	return S1, S2


def _wrap_params(params: np.ndarray) -> np.ndarray:
	out = np.array(params, dtype=float, copy=True)
	for i, name in enumerate(PARAM_NAMES):
		if name in ("gamma1", "gamma2", "phi_ind1", "phi_ind2", "delta1", "delta2"):
			out[i] = _wrap_pi(out[i])
	return out


def _split_init_guess(
	init_guess: Sequence[float] | np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
	if init_guess is None:
		return CHAIN1_INIT_GUESS, CHAIN2_INIT_GUESS
	vec = np.asarray(init_guess, dtype=float)
	if vec.size != len(PARAM_NAMES):
		raise ValueError("init_guess must match PARAM_NAMES length")
	chain1 = np.array([vec[0], vec[2], vec[4], vec[6], vec[8]], dtype=float)
	chain2 = np.array([vec[1], vec[3], vec[5], vec[7], vec[9]], dtype=float)
	return chain1, chain2


def energy_single_chain_state(S, J_intra, Kz):
	E_intra = 0.5 * np.sum(S * (J_intra @ S), axis=1)
	E_ani = -0.5 * Kz * (S[:, 2] ** 2)
	return float(np.sum(E_intra + E_ani))


# ==========================================
# 4. MINIMIZER CORE
# ==========================================
def _default_m_values(chain_length: int, max_points: int = MAX_M_POINTS) -> np.ndarray:
	if chain_length <= max_points:
		return np.arange(chain_length, dtype=int)
	step = max(1, int(math.ceil(chain_length / float(max_points))))
	return np.arange(0, chain_length, step, dtype=int)


def minimize_single_chain_parameters(
	M: float,
	J_intra,
	Kz: float,
	n_sites: int,
	idx: np.ndarray,
	parity: np.ndarray,
	x0: Sequence[float] | np.ndarray | None = None,
	bounds=CHAIN1_BOUNDS,
	method: str = "L-BFGS-B",
	options=None,
) -> Tuple[float, np.ndarray, bool]:
	if minimize is None:
		raise RuntimeError("scipy is required for the minimizer (scipy.optimize.minimize)")

	q = float(_q_from_winding(M, n_sites))
	init = np.zeros(5) if x0 is None else np.array(x0, dtype=float)

	def objective(vec: np.ndarray) -> float:
		mz, gamma, alpha_ind, phi_ind, delta = vec
		S = _build_spiral_chain(
			n_sites=n_sites,
			q=q,
			mz=mz,
			gamma=gamma,
			alpha_ind=alpha_ind,
			phi_ind=phi_ind,
			phase_shift=delta,
			idx=idx,
			parity=parity,
		)
		E_tot = energy_single_chain_state(S, J_intra, Kz)
		return float(E_tot / n_sites)

	res = minimize(
		objective,
		x0=init,
		method=method,
		bounds=bounds,
		options=options or MINIMIZER_OPTIONS,
	)

	if not res.success:
		return float(objective(init)), init, False

	params_opt = np.asarray(res.x, dtype=float)
	params_opt[1] = _wrap_pi(params_opt[1])
	params_opt[3] = _wrap_pi(params_opt[3])
	params_opt[4] = _wrap_pi(params_opt[4])
	return float(res.fun), params_opt, True


def minimize_single_chain_parameters_fixed_phi(
	M: float,
	phi_fixed: float,
	J_intra,
	Kz: float,
	n_sites: int,
	idx: np.ndarray,
	parity: np.ndarray,
	x0: Sequence[float] | np.ndarray | None = None,
	bounds=CHAIN2_FIXED_PHI_BOUNDS,
	method: str = "L-BFGS-B",
	options=None,
) -> Tuple[float, np.ndarray, bool]:
	if minimize is None:
		raise RuntimeError("scipy is required for the minimizer (scipy.optimize.minimize)")

	q = float(_q_from_winding(M, n_sites))
	init = np.zeros(4) if x0 is None else np.array(x0, dtype=float)

	def objective(vec: np.ndarray) -> float:
		mz, gamma, alpha_ind, delta = vec
		S = _build_spiral_chain(
			n_sites=n_sites,
			q=q,
			mz=mz,
			gamma=gamma,
			alpha_ind=alpha_ind,
			phi_ind=phi_fixed,
			phase_shift=delta,
			idx=idx,
			parity=parity,
		)
		E_tot = energy_single_chain_state(S, J_intra, Kz)
		return float(E_tot / n_sites)

	res = minimize(
		objective,
		x0=init,
		method=method,
		bounds=bounds,
		options=options or MINIMIZER_OPTIONS,
	)

	if not res.success:
		return float(objective(init)), init, False

	params_opt = np.asarray(res.x, dtype=float)
	params_opt[1] = _wrap_pi(params_opt[1])
	params_opt[3] = _wrap_pi(params_opt[3])
	return float(res.fun), params_opt, True


def scan_winding_minima_single_chain(
	J_intra,
	Kz: float,
	n_sites: int,
	M_values: Sequence[int] | np.ndarray | None = None,
	init_guess: Sequence[float] | np.ndarray | None = None,
	bounds=CHAIN1_BOUNDS,
	method: str = "L-BFGS-B",
	options=None,
	verbose: bool = True,
	phi_fixed: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if n_sites <= 0:
		raise ValueError("n_sites must be positive")

	if M_values is None:
		M_arr = _default_m_values(n_sites)
		if verbose and M_arr.size < n_sites:
			print(f"Using coarse M grid: {M_arr.size} points for n={n_sites}")
	else:
		M_arr = np.asarray(M_values, dtype=float)
		if M_arr.ndim == 0:
			M_arr = np.atleast_1d(M_arr)

	q_arr = np.asarray(_q_from_winding(M_arr, n_sites), dtype=float)
	energies = np.empty(M_arr.size, dtype=float)
	param_len = 5 if phi_fixed is None else 4
	params_hist = np.empty((M_arr.size, param_len), dtype=float)
	success = np.zeros(M_arr.size, dtype=bool)

	idx = np.arange(n_sites, dtype=np.int64)
	parity = np.where((idx & 1) == 0, 1.0, -1.0)

	guess = np.zeros(param_len) if init_guess is None else np.array(init_guess, dtype=float)
	progress_step = max(1, M_arr.size // 10) if M_arr.size > 10 else 1

	for i, M_val in enumerate(M_arr):
		if phi_fixed is None:
			e_val, opt_params, ok = minimize_single_chain_parameters(
				M_val,
				J_intra,
				Kz,
				n_sites,
				idx,
				parity,
				x0=guess,
				bounds=bounds,
				method=method,
				options=options,
			)
		else:
			e_val, opt_params, ok = minimize_single_chain_parameters_fixed_phi(
				M_val,
				phi_fixed,
				J_intra,
				Kz,
				n_sites,
				idx,
				parity,
				x0=guess,
				bounds=bounds,
				method=method,
				options=options,
			)
		energies[i] = e_val
		params_hist[i] = opt_params
		success[i] = ok
		if ok:
			guess = opt_params

		if verbose and (i + 1) % progress_step == 0:
			print(f"Minimizer progress: {100.0 * (i + 1) / M_arr.size:.1f}%")

	return M_arr, q_arr, energies, params_hist, success


def build_ground_state_from_minimizer(
	cache,
	Kz: float,
	n1: int,
	n2: int,
	M1_values: Sequence[int] | np.ndarray | None = None,
	M2_values: Sequence[int] | np.ndarray | None = None,
	init_guess: Sequence[float] | np.ndarray | None = None,
	bounds=DEFAULT_BOUNDS,
	verbose: bool = True,
):
	t0 = time.time()
	chain1_guess, chain2_guess = _split_init_guess(init_guess)

	M1_arr, q1_arr, e1_arr, params1_hist, success1 = scan_winding_minima_single_chain(
		J_intra=cache.J_intra_1,
		Kz=Kz,
		n_sites=n1,
		M_values=M1_values,
		init_guess=chain1_guess,
		bounds=CHAIN1_BOUNDS,
		verbose=verbose,
	)
	if np.any(success1):
		masked1 = np.where(success1, e1_arr, np.inf)
		idx1_best = int(np.argmin(masked1))
	else:
		idx1_best = int(np.argmin(e1_arr))

	q1_best = float(q1_arr[idx1_best])
	mz1, gamma1, alpha1, phi_ind1, delta1 = params1_hist[idx1_best]

	M2_arr, q2_arr, e2_arr, params2_hist, success2 = scan_winding_minima_single_chain(
		J_intra=cache.J_intra_2,
		Kz=Kz,
		n_sites=n2,
		M_values=M2_values,
		init_guess=chain2_guess,
		bounds=CHAIN2_BOUNDS,
		verbose=verbose,
	)
	if np.any(success2):
		masked2 = np.where(success2, e2_arr, np.inf)
		idx2_best = int(np.argmin(masked2))
	else:
		idx2_best = int(np.argmin(e2_arr))

	q2_best = float(q2_arr[idx2_best])
	mz2, gamma2, alpha2, phi_ind2, delta2 = params2_hist[idx2_best]

	idx1 = np.arange(n1, dtype=np.int64)
	idx2 = np.arange(n2, dtype=np.int64)
	parity1 = np.where((idx1 & 1) == 0, 1.0, -1.0)
	parity2 = np.where((idx2 & 1) == 0, 1.0, -1.0)

	S1 = _build_spiral_chain(
		n_sites=n1,
		q=q1_best,
		mz=mz1,
		gamma=gamma1,
		alpha_ind=alpha1,
		phi_ind=phi_ind1,
		phase_shift=delta1,
		idx=idx1,
		parity=parity1,
	)
	S2 = _build_spiral_chain(
		n_sites=n2,
		q=q2_best,
		mz=mz2,
		gamma=gamma2,
		alpha_ind=alpha2,
		phi_ind=phi_ind2,
		phase_shift=delta2,
		idx=idx2,
		parity=parity2,
	)

	E1 = energy_single_chain_state(S1, cache.J_intra_1, Kz)
	E2 = energy_single_chain_state(S2, cache.J_intra_2, Kz)
	E_per_spin = float((E1 + E2) / (n1 + n2))
	params_best = np.array(
		[mz1, mz2, gamma1, gamma2, alpha1, alpha2, phi_ind1, phi_ind2, delta1, delta2],
		dtype=float,
	)
	info_success = bool(success1[idx1_best] and success2[idx2_best])
	t1 = time.time()

	if verbose:
		print("Minimizer finished (intra-only)")
		print(f"Time: {t1 - t0:.1f}s")
		print(f"Best M1: {M1_arr[idx1_best]:.3f}")
		print(f"Best M2: {M2_arr[idx2_best]:.3f}")
		print(f"Best q1: {q1_best:.8f}, q2: {q2_best:.8f}")
		print(f"Energy per spin (intra): {E_per_spin:.8f}")
		print(", ".join(f"{name}={val:.6f}" for name, val in zip(PARAM_NAMES, params_best)))

	info = {
		"M1_best": float(M1_arr[idx1_best]),
		"M2_best": float(M2_arr[idx2_best]),
		"q1_best": q1_best,
		"q2_best": q2_best,
		"energy_per_spin": E_per_spin,
		"params": params_best,
		"success": info_success,
	}
	return S1, S2, info


# ==========================================
# 5. FALLBACK INITIALIZATION
# ==========================================
def _build_simple_afm_state(n: int, noise_x: float = 1e-5, seed: int = 1234) -> np.ndarray:
	rng = np.random.default_rng(seed)
	S = np.zeros((n, 3), dtype=float)
	S[0::2, 2] = 1.0
	S[1::2, 2] = -1.0
	noise = rng.normal(scale=noise_x, size=n)
	noise -= noise.mean()
	S[:, 0] += noise
	S /= np.linalg.norm(S, axis=1, keepdims=True)
	return S


def initialize_state(cache):
	if USE_MINIMIZER:
		try:
			S1_0, S2_0, info = build_ground_state_from_minimizer(
				cache,
				Kz=Kz_field,
				n1=n1,
				n2=n2,
				M1_values=M1_VALUES,
				M2_values=M2_VALUES,
				init_guess=INIT_GUESS,
			)
			info["method"] = "minimizer"
			return S1_0, S2_0, info
		except Exception as exc:  # noqa: BLE001 - fallback path
			print(f"Minimizer failed: {exc}")
			print("Falling back to AFM initialization")

	if USE_NPY_FALLBACK and os.path.exists(NPY_S1_PATH) and os.path.exists(NPY_S2_PATH):
		S1_0 = build_afm_state_from_npy(n1, NPY_S1_PATH, noise_x=1e-5)
		S2_0 = build_afm_state_from_npy(n2, NPY_S2_PATH, noise_x=1e-5)
		return S1_0, S2_0, {"method": "npy"}

	S1_0 = _build_simple_afm_state(n1, noise_x=1e-5, seed=1234)
	S2_0 = _build_simple_afm_state(n2, noise_x=1e-5, seed=4321)
	return S1_0, S2_0, {"method": "afm"}


# ==========================================
# 6. MAIN
# ==========================================
def main():
	print("Building coupling cache for the Moire system...")
	cache = precompute_two_chain_couplings(
		n1=n1,
		n2=n2,
		a1=a1,
		u2=u2,
		use_sparse=True,
	)

	S1_0, S2_0, init_info = initialize_state(cache)
	print(f"Init method: {init_info.get('method')}")

	E_tot, _, _ = energy_two_chains_state(S1_0, S2_0, cache, Kz_field)
	total_spins = n1 + n2
	print(f"Initial energy (per spin): {E_tot / total_spins:.6f}")


if __name__ == "__main__":
	main()
