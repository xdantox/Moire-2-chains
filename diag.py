import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1e
from scipy.linalg import cholesky, eigh

class MoireBdG:
    def __init__(self, a=1.0, epsilon=0.1, zd_over_a0=1.7, 
                 beta_d=2.0, beta_inter=2.0, J0=1.0, Jinter_0=1.0, 
                 K1=0.025, K2=0.025, he=0.0, d_shift=0.0, S=1.0, Nmax=10):
        self.a = a
        self.eps = epsilon
        self.ae = a * (1.0 + epsilon)
        self.a0 = a / 2.0
        # Escala espacial física consistente con LLG
        self.zd = zd_over_a0 * self.a0
        self.beta_d = beta_d
        self.beta_inter = beta_inter
        self.J0 = J0
        self.Jinter_0 = Jinter_0
        self.K1, self.K2 = K1, K2
        self.he = he
        self.d = d_shift
        self.S = S
        self.Nmax = Nmax
        
        self.L = (self.a, self.ae)
        self.b1 = 2.0 * np.pi / self.a
        self.b2 = 2.0 * np.pi / self.ae
        self.bM = self.b1 - self.b2
        
        self.tau = (0.0, self.a / 2.0)
        self.taup = (d_shift, self.ae / 2.0 + d_shift)
        
        # Shift intracapa estático (2K + J_AB(0) - J_AA(0))
        self.D = [2.0 * K + self.JAB(np.zeros(1), L_val)[0] - self.JAA(np.zeros(1), L_val)[0]
                  for K, L_val in zip((self.K1, self.K2), self.L)]

    def J_intra(self, r):
        return self.J0 * np.exp(-self.beta_d * (np.abs(r) / self.a0 - 1.0))

    def JAA(self, p, L_val):
        n = np.arange(1, self.Nmax + 1)[:, None]
        return 2.0 * np.sum(self.J_intra(n * L_val) * np.cos(p[None, :] * n * L_val), axis=0)

    def JAB(self, p, L_val):
        n = np.arange(-self.Nmax, self.Nmax)[:, None]
        return np.sum(self.J_intra((n + 0.5) * L_val) * np.cos(p[None, :] * (n + 0.5) * L_val), axis=0)

    def Wt(self, q):
        lam = self.beta_inter / self.a0
        s = np.sqrt(lam**2 + np.asarray(q, float)**2)
        return self.Jinter_0 * np.exp(self.beta_inter) * (2.0 * self.zd * lam / s) * k1e(self.zd * s) * np.exp(-self.zd * s)

    def build_H(self, k, M=12, jmax=2):
        L1, L2 = self.L
        Nm = 2 * M + 1
        m = np.arange(-M, M + 1)
        p = k + m * self.bM
        H = np.zeros((4 * Nm, 4 * Nm), dtype=complex)
        
        off = lambda c, s: ((c - 1) + 2 * s) * Nm
        ar = np.arange(Nm)
        
        # 1. Bloque Intracapa
        for c, L_val in ((1, L1), (2, L2)):
            dg = self.JAA(p, L_val) + self.D[c - 1]
            ab = self.JAB(p, L_val)
            H[off(c, 0) + ar, off(c, 0) + ar] += dg + self.he
            H[off(c, 1) + ar, off(c, 1) + ar] += dg - self.he
            H[off(c, 0) + ar, off(c, 1) + ar] += ab
            H[off(c, 1) + ar, off(c, 0) + ar] += ab

        # 2. Bloque Intercapa dependiente de k
        norm = 1.0 / np.sqrt(L1 * L2)
        for dj in range(-jmax, jmax + 1):
            lo, hi = max(-M, -M - dj), min(M, M - dj)
            if lo > hi:
                continue
            mm = np.arange(lo, hi + 1) + M
            mp = mm + dj
            
            # Transferencia de momento dependiente de k y de la réplica m:
            amp = self.Wt(p[mm] + dj * self.b1) * norm
            
            for s in (0, 1):
                for sp in (0, 1):
                    v = amp * np.exp(1j * dj * self.b1 * self.tau[s]) * np.exp(-1j * dj * self.b2 * self.taup[sp])
                    H[off(1, s) + mm, off(2, sp) + mp] += v
                    H[off(2, sp) + mp, off(1, s) + mm] += np.conj(v)
            
            # Potencial estático longitudinal de Moiré (dj impar)
            if dj % 2 != 0:
                v1 = -(2.0 / L2) * self.Wt(dj * self.b2) * np.exp(-1j * dj * self.b2 * self.d)
                v2 = -(2.0 / L1) * self.Wt(dj * self.b1) * np.exp(-1j * dj * self.b2 * self.d)
                for s in (0, 1):
                    H[off(1, s) + mm, off(1, s) + mp] += v1
                    H[off(2, s) + mm, off(2, s) + mp] += v2

        H *= self.S
        H = 0.5 * (H + H.conj().T)
        sig = np.concatenate([np.ones(2 * Nm), -np.ones(2 * Nm)])
        return H, sig

    def spectrum(self, k, M=12, jmax=2):
        H, sig = self.build_H(k, M, jmax)
        Lc = cholesky(H, lower=True)
        C = Lc.conj().T @ (sig[:, None] * Lc)
        ev, ph = eigh(C)
        pos = ev > 0
        ev_pos = ev[pos]
        psi = np.linalg.solve(Lc.conj().T, ph[:, pos])
        psi /= np.linalg.norm(psi, axis=0)
        return ev_pos, psi

    def peso_m0(self, psi, M=12):
        Nm = 2 * M + 1
        i0 = M
        # Proyección sobre los modos fundamentales m=0 de las cadenas
        peso_total = np.zeros(psi.shape[1])
        for blk in range(4):
            peso_total += np.abs(psi[blk * Nm + i0, :])**2
        return peso_total

def calcular_dos(energias, n_k, e_max, n_bins=240):
    """Calcula g(E) = (1/N_k) sum_{k,n} delta(E - E_n(k)).

    El histograma es la representación numérica de la delta de Dirac.
    Por tanto, la integral de la DOS sobre la energía equivale al número
    de ramas positivas dentro del intervalo de energía representado.
    """
    energias = np.asarray(energias, dtype=float)
    energias = energias[np.isfinite(energias) & (energias >= 0.0)]
    if n_k <= 0 or energias.size == 0:
        raise ValueError("Se necesitan energías positivas y un número de k válido.")

    edges = np.linspace(0.0, e_max, n_bins + 1)
    counts, _ = np.histogram(energias, bins=edges)
    bin_width = edges[1] - edges[0]
    dos = counts / (float(n_k) * bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, dos


def ejecutar_diagonalizacion(PLOT_INTENSITY=True, E_MAX=1.8, N_BINS_DOS=240):
    modelo = MoireBdG()
    # Primera zona de Brillouin moiré, consistente con fft.py.
    k_max = abs(modelo.bM) / 2.0
    k_vals = np.linspace(-k_max, k_max, 1000)
    
    all_k, all_E, all_W, all_E_dos = [], [], [], []
    
    for k in k_vals:
        ev, psi = modelo.spectrum(k, M=10, jmax=2)
        pesos = modelo.peso_m0(psi, M=10)
        all_E_dos.extend(ev)
        for E, w in zip(ev, pesos):
            if 0.0 <= E <= E_MAX:
                all_k.append(k)
                all_E.append(E)
                all_W.append(w)

    all_k = np.asarray(all_k)
    all_E = np.asarray(all_E)
    all_W = np.asarray(all_W)

    energy_dos, dos = calcular_dos(all_E_dos, len(k_vals), E_MAX, N_BINS_DOS)

    fig, (ax_dos, ax_bands) = plt.subplots(
        1, 2, figsize=(12, 6), dpi=120, sharey=True,
        gridspec_kw={'width_ratios': (1, 4), 'wspace': 0.05}
    )
    ax_dos.step(dos, energy_dos, where='mid', color='black', linewidth=1.5)
    ax_dos.fill_betweenx(energy_dos, 0.0, dos, step='mid', color='grey', alpha=0.35)
    ax_dos.set_xlabel(r'$g(E)$')
    ax_dos.set_ylabel(r'$\hbar\omega / J_0$')
    ax_dos.set_title('DOS')
    ax_dos.grid(True, axis='x', linestyle=':', alpha=0.5)

    if PLOT_INTENSITY:
        # Misma representación que fft.py: intensidad logarítmica sobre
        # una malla regular, normalizada por percentiles y recortada a [0, 1].
        n_energy = 1000
        energy_grid = np.linspace(0.0, E_MAX, n_energy)
        log_floor = np.log10(1e-12)
        intensity_grid = np.full((n_energy, len(k_vals)), log_floor)
        energy_indices = np.searchsorted(energy_grid, all_E).clip(0, n_energy - 1)
        k_indices = np.searchsorted(k_vals, all_k).clip(0, len(k_vals) - 1)
        log_intensity = np.log10(all_W + 1e-12)
        np.maximum.at(intensity_grid, (energy_indices, k_indices), log_intensity)

        vmin = np.percentile(intensity_grid, 5.0)
        vmax = np.percentile(intensity_grid, 99.8)
        intensity_grid = np.clip(
            (intensity_grid - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0
        )
        im = ax_bands.imshow(
            intensity_grid,
            origin='lower',
            aspect='auto',
            extent=(-k_max, k_max, 0.0, E_MAX),
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            interpolation='bilinear',
        )
        cbar = fig.colorbar(im, ax=ax_bands, label='Normalized Spectral Intensity')
        cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    else:
        ax_bands.scatter(all_k, all_E, color='tab:blue', s=4, edgecolors='none')

    ax_bands.axvline(-k_max, color='crimson', linestyle='--', alpha=0.8,
                     label=r'Borde mBZ ($\pm G_M/2$)')
    ax_bands.axvline(k_max, color='crimson', linestyle='--', alpha=0.8)
    ax_bands.set_ylim(0.0, E_MAX)
    ax_bands.set_xlim(-k_max, k_max)
    ax_bands.set_ylabel(r'$\hbar\omega / J_0$', fontsize=12)
    ax_bands.set_xlabel(r'$k$ [rad/m]', fontsize=12)
    ax_bands.set_title('Espectro BdG en la mBZ', fontsize=14)
    ax_bands.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    PLOT_INTENSITY = True
    ejecutar_diagonalizacion(PLOT_INTENSITY=PLOT_INTENSITY)