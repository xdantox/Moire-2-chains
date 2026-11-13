import numpy as np
import matplotlib.pyplot as plt
from Heff import  n1, n2, a1, a2
# ====================================================================
# 1. TUS DATOS NUMÉRICOS (LLG + Ruido Cuántico)
# ====================================================================
datos_txt = """
# h(nm) Delta_ZPE(meV)
1.600000000000000089e+00 0.000000000000000000e+00
1.199999999999999956e+00 -5.244796046177668813e-05
8.000000000000000444e-01 -4.463063587802906795e-04
4.000000000000000222e-01 -3.366361792883409976e-03
0.000000000000000000e+00 -2.769915340345718846e-02
"""

# Procesar texto a arrays numpy
lineas = datos_txt.strip().split('\n')
h_num = []
delta_zpe_num = []
for linea in lineas:
    partes = linea.split()
    if len(partes) == 2:
        h_num.append(float(partes[0]))
        delta_zpe_num.append(float(partes[1]))

h_num = np.array(h_num)
delta_zpe_num = np.array(delta_zpe_num)

# ====================================================================
# 2. TEORÍA AB INITIO: MATRIZ DINÁMICA EXACTA (HESSIANO) EN ESPACIO REAL
# ====================================================================
HBAR_MEV_S = 0.6582119569e-12
muB_meV_T = 0.05788
gamma = 1.7e11
J_intra = HBAR_MEV_S * gamma * 46.75 / muB_meV_T
Kz = 2 * HBAR_MEV_S * gamma * 0.76 / muB_meV_T
J_perp0 = HBAR_MEV_S * gamma * 0.4 / muB_meV_T
lam_perp = 0.3
lam_intra = 0.3
tol = 1e-6

# === SWITCH DE TOPOLOGÍA ===
IS_PBC = False  # Cambia a True para probar la matriz en condiciones de borde periódicas


N_total = n1 + n2
a_eff1 = a1 / 2.0
L1 = n1 * a_eff1  # Longitud física de la Cadena 1

if IS_PBC:
    # FORZAMIENTO DE CONMESURABILIDAD GLOBAL:
    # Ajustamos a_eff2 microscópicamente para que las longitudes calcen perfecto.
    L_common = L1
    a_eff2 = L_common / n2 
else:
    L_common = None
    a_eff2 = a2 / 2.0

x1 = np.arange(n1) * a_eff1
x2 = np.arange(n2) * a_eff2

# Función de distancia topológica (Soporta OBC y PBC)
def calc_dist(p1, p2, L_wrap=None):
    dx = np.abs(p1[:, None] - p2[None, :])
    if IS_PBC and L_wrap is not None:
        # Mapea la distancia sobre un anillo, eligiendo el camino más corto
        dx = np.minimum(dx, L_wrap - dx)
    return dx

def kernel(r, J, lamb):
    return J * np.exp(-r / lamb)

# Construir acoplamientos Intracadena (J) estáticos
dx1 = calc_dist(x1, x1, L_common)
J1 = kernel(dx1, J_intra, lam_intra)
np.fill_diagonal(J1, 0.0)
J1[J1 < tol * J_intra] = 0.0

dx2 = calc_dist(x2, x2, L_common)
J2 = kernel(dx2, J_intra, lam_intra)
np.fill_diagonal(J2, 0.0)
J2[J2 < tol * J_intra] = 0.0

sigma1 = (-1)**np.arange(n1)
sigma2 = (-1)**np.arange(n2)
Sigma = np.diag(np.concatenate([sigma1, sigma2])) 

zpe_theory_exact = []
print(f"Diagonalizando el Hessiano Magnónico ({'PBC' if IS_PBC else 'OBC'})...")

for h in h_num:
    dx12 = calc_dist(x1, x2, L_common)
    r12 = np.sqrt(dx12**2 + h**2) if h > 0.0 else dx12
    W12 = kernel(r12, J_perp0, lam_perp)
    W12[W12 < tol * J_perp0] = 0.0
    W21 = W12.T
    
    # Signos corregidos formalmente para la segunda derivada (Hessiano) de -J(Si.Sj)
    diag1 = Kz - np.sum(J1 * np.outer(sigma1, sigma1), axis=1) \
               - np.sum(W12 * np.outer(sigma1, sigma2), axis=1)
               
    diag2 = Kz - np.sum(J2 * np.outer(sigma2, sigma2), axis=1) \
               - np.sum(W21 * np.outer(sigma2, sigma1), axis=1)
               
    M11 = np.diag(diag1) + J1
    M22 = np.diag(diag2) + J2
    
    M = np.block([
        [M11, W12],
        [W21, M22]
    ])
    
    eigvals = np.linalg.eigvals(Sigma @ M)
    omega = np.abs(np.real(eigvals))
    
    # CORRECCIÓN DE LA DISPUTA DEL 0.25 vs 0.5:
    # La matriz es N_total x N_total. El arreglo 'omega' contiene exactamente 
    # los N_total grados de libertad del sistema (una vez normalizado el +- por abs).
    # Por lo tanto, usamos rigurosamente la formulación de energía de vacío bosónica (1/2 \hbar \omega).
    ZPE = 0.5 * np.sum(omega) / N_total
    
    zpe_theory_exact.append(ZPE)

zpe_theory_exact = np.array(zpe_theory_exact)
delta_zpe_theory = zpe_theory_exact - zpe_theory_exact[0]

# ====================================================================
# 3. GRÁFICO SUPERPUESTO (DOBLE VERIFICACIÓN CRUZADA)
# ====================================================================
plt.figure(figsize=(11, 7))

plt.plot(h_num, delta_zpe_num, 'o', color='indigo', markersize=8, 
         label='Simulación Estocástica (LLG + Ruido Cuántico)')

plt.plot(h_num, delta_zpe_theory, '-', color='gold', linewidth=3, alpha=0.9, 
         label=f'Solución Exacta (Hessiano en Espacio Real - {"PBC" if IS_PBC else "OBC"})')

plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.gca().invert_xaxis() 
plt.title(r'Validación Cruzada Definitiva: $\Delta$ZPE vs $h$', fontsize=16)
plt.xlabel(r'Separación Intercadena $h$ (nm)', fontsize=14)
plt.ylabel(r'$\Delta$ ZPE por espín (meV)', fontsize=14)
plt.legend(fontsize=12)

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()