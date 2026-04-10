import numpy as np

# Importamos las herramientas que hemos construido para el modelo Moiré
from Heff import Kz_field as Kz_field_default
from Heff import a1 as a1_default
from Heff import n1 as n1_default
from Heff import n2 as n2_default
from Heff import precompute_two_chain_couplings
from Heff import u2 as u2_default
from rot import implicit_midpoint_step_two_chains
from E import evaluate_energy_history
from animation import animation_two_chains

# ==========================================
# 1. PARÁMETROS GLOBALES Y DE TIEMPO
# ==========================================
dt = 1e-15          # Paso de tiempo (ajustar según estabilidad)
total_time = (2**16) * dt  # Tiempo total
num_pasos = int(total_time / dt)

# Parámetros físicos por defecto tomados de Heff.py
n1 = n1_default
n2 = n2_default
Kz_field = Kz_field_default
a1 = a1_default
u2 = u2_default


def build_afm_state(n_sites, invert=False):
    """Construye un estado Neel colineal en z: [+z, -z, +z, -z, ...]."""
    if n_sites % 2 != 0:
        raise ValueError("n_sites debe ser par para un estado AFM bipartito consistente.")

    spins = np.zeros((n_sites, 3), dtype=float)
    pattern = np.where(np.arange(n_sites) % 2 == 0, 1.0, -1.0)
    if invert:
        pattern = -pattern
    spins[:, 2] = pattern
    return spins


def main():
    # ==========================================
    # 2. PRECOMPUTACIÓN GEOMÉTRICA (MOIRÉ)
    # ==========================================
    print("Calculando caché topológico del sistema Moiré...")
    # Generamos la red estática una sola vez
    cache = precompute_two_chain_couplings(
        n1=n1,
        n2=n2,
        a1=a1,
        u2=u2,
        use_sparse=True,
    )

    # ==========================================
    # 3. CONDICIONES INICIALES
    # ==========================================
    # Estado AFM inicial: cadena 2 invertida respecto a cadena 1.
    S1_0 = build_afm_state(n1, invert=False)
    S2_0 = build_afm_state(n2, invert=True)

    # Preparamos las matrices de historia (una por cadena)
    History_S1 = np.zeros((num_pasos, n1, 3))
    History_S2 = np.zeros((num_pasos, n2, 3))

    History_S1[0] = S1_0
    History_S2[0] = S2_0

    # ==========================================
    # 4. BUCLE DE INTEGRACIÓN LLG (Punto Medio)
    # ==========================================
    print(f"Iniciando integración LLG ({num_pasos} pasos)...")
    progress_step = max(1, num_pasos // 10)

    for j in range(num_pasos - 1):
        # El integrador avanza ambas cadenas simultáneamente
        S1_new, S2_new = implicit_midpoint_step_two_chains(
            S1_0,
            S2_0,
            coupling_cache=cache,
            dt=dt,
            Kz=Kz_field,
            n_iter=14,
        )

        # Almacenamos el paso
        History_S1[j + 1] = S1_new
        History_S2[j + 1] = S2_new

        # Preparamos el siguiente iterador
        S1_0 = S1_new
        S2_0 = S2_new

        # Opcional: Barra de progreso simple
        if (j + 1) % progress_step == 0:
            print(f"Progreso: {100 * (j + 1) / num_pasos:.1f}%")

    # ==========================================
    # 5. ANÁLISIS Y GUARDADO
    # ==========================================
    print("Integración terminada. Evaluando energía...")
    # Usamos nuestra función vectorizada para evaluar la energía en toda la historia
    Energy_hist = evaluate_energy_history(History_S1, History_S2, cache, Kz_field)

    total_spins = n1 + n2
    print(f"Energía inicial (por espín): {Energy_hist[0] / total_spins:.6f} unidades de campo")
    print(f"Energía final   (por espín): {Energy_hist[-1] / total_spins:.6f} unidades de campo")

    # Guardamos los binarios para evitar recalcular
    np.save("spin_history_cadena1.npy", History_S1)
    np.save("spin_history_cadena2.npy", History_S2)

    # ==========================================
    # 6. RENDERIZADO VISUAL
    # ==========================================
    print("Lanzando animación Moiré 3D...")
    # Incluimos una separación en Y para que las flechas no colisionen visualmente
    animation_two_chains(History_S1, History_S2, dt, cache, Kz_field, visual_y_sep=1.0)

if __name__ == "__main__":
    main()