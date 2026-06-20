import numpy as np
import matplotlib.pyplot as plt

def analyze_final_state(npy_path, n_plot_sites=200):
    """
    Analiza la configuración final de una historia de espines.
    n_plot_sites: Número de sitios a graficar para hacer zoom y ver la textura.
    """
    print(f"--- Analizando configuración final de: {npy_path} ---")
    
    # 1. Cargar datos de forma segura
    data = np.load(npy_path, mmap_mode="r")
    if data.ndim == 3:
        state = data[-1].copy()  # Tomar el último paso de tiempo (estado final)
    elif data.ndim == 2:
        state = data.copy()
    else:
        raise ValueError("El archivo debe tener forma (t, n, 3) o (n, 3).")

    n_sites = state.shape[0]
    
    # 2. Métricas Globales
    # Magnetización uniforme (FM)
    m_uniform = state.mean(axis=0)
    
    # Magnetización escalonada (AFM)
    pattern = np.where(np.arange(n_sites) % 2 == 0, 1.0, -1.0)
    m_staggered = (state * pattern[:, None]).mean(axis=0)
    
    print("Órdenes Globales:")
    print(f"  |M_FM| (Uniforme)   : {np.linalg.norm(m_uniform):.6f}")
    print(f"  |M_AFM| (Escalonada): {np.linalg.norm(m_staggered):.6f}")
    print(f"  Componente Z (AFM)  : {m_staggered[2]:.6f}")

    # 3. Análisis Local: Ángulo de Pitch (Correlación a primeros vecinos)
    # Calculamos S_i · S_{i+1}
    # Usamos np.roll para comparar con el vecino derecho (con PBC)
    state_shifted = np.roll(state, shift=-1, axis=0)
    dot_products = np.sum(state * state_shifted, axis=1)
    
    # Prevenir errores numéricos de arccos fuera de [-1, 1]
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)
    
    mean_angle = angles_deg.mean()
    std_angle = angles_deg.std()
    
    print("\nTextura Magnética Local:")
    print(f"  Ángulo promedio entre vecinos: {mean_angle:.2f}°")
    print(f"  Desviación estándar del ángulo: {std_angle:.4f}°")
    
    # Diagnóstico automático
    print("\n[DIAGNÓSTICO AUTOMÁTICO]")
    if mean_angle > 179.9 and std_angle < 0.1:
        print("=> Estado Colineal Antiferromagnético (AFM) Perfecto.")
    elif mean_angle < 0.1 and std_angle < 0.1:
        print("=> Estado Colineal Ferromagnético (FM) Perfecto.")
    elif std_angle < 0.1:
        print(f"=> Espiral de Espín (Spin Spiral) Uniforme con paso de {mean_angle:.2f}°.")
    else:
        print("=> Estado Modulado (Posible Bunching / Red de Solitones Moiré).")
        print("   El ángulo entre vecinos varía a lo largo de la cadena.")

    # 4. Renderizado de Gráficos
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Rango para graficar (hacer zoom para ver bien la textura)
    x_axis = np.arange(min(n_sites, n_plot_sites))
    
    # Subplot 1: Componentes Z (Perfil AFM)
    axs[0].plot(x_axis, state[x_axis, 2], 'k.-', label='$S_z$')
    axs[0].set_ylabel("Componente Z")
    axs[0].set_title(f"Perfil de Espín (Zoom a los primeros {len(x_axis)} sitios)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Subplot 2: Componentes X e Y (Plano Transversal / Espirales)
    axs[1].plot(x_axis, state[x_axis, 0], 'r.-', label='$S_x$', alpha=0.7)
    axs[1].plot(x_axis, state[x_axis, 1], 'b.-', label='$S_y$', alpha=0.7)
    axs[1].set_ylabel("Componentes X, Y")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Subplot 3: Ángulo de Pitch (Revela Bunching/Frustración)
    axs[2].plot(x_axis, angles_deg[x_axis], 'g.-', label=r'Ángulo $\theta_{i, i+1}$')
    axs[2].set_ylabel("Ángulo (°)")
    axs[2].set_xlabel("Índice del Sitio (i)")
    axs[2].axhline(180, color='gray', linestyle='--', alpha=0.5) # Referencia AFM puro
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Pasa aquí tu archivo de relajo.
    # Ajusta n_plot_sites para hacer más o menos zoom en la cadena.
    analyze_final_state("spin_history_cadena1 relax.npy", n_plot_sites=150)