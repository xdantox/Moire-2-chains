from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import cast

from E import evaluate_energy_history

# Asume que evaluate_energy_history ya está importada/definida en el entorno

def animation_two_chains(History_S1, History_S2, dt, cache, Kz_field, visual_y_sep=1.0):
    """
    Animación 3D para un sistema Moiré de dos cadenas acopladas.
    
    Parámetros:
    - visual_y_sep: Separación artificial en el eje Y para poder ver ambas cadenas 
                    claramente, incluso si físicamente h_sep = 0.
    """
    num_pasos = History_S1.shape[0]
    n1 = History_S1.shape[1]
    n2 = History_S2.shape[1]
    total_time = num_pasos * dt

    fig = plt.figure(figsize=(12, 8))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))

    # 1. Generar posiciones 3D exactas usando el caché Moiré
    # Cadena 1 (No deformada, en Y = 0)
    pos1 = np.zeros((n1, 3))
    pos1[:, 0] = cache.x1
    pos1[:, 1] = 0.0
    pos1[:, 2] = 0.0  

    # Cadena 2 (Deformada con u2, desplazada en Y para visualización)
    # Si definiste cache.h_sep > 0, puedes usarlo en Z en lugar de 0.0
    pos2 = np.zeros((n2, 3))
    pos2[:, 0] = cache.x2
    pos2[:, 1] = visual_y_sep 
    pos2[:, 2] = 0.0

    # Límites del gráfico dinámicos
    x_min = min(pos1[:, 0].min(), pos2[:, 0].min()) - 1.0
    x_max = max(pos1[:, 0].max(), pos2[:, 0].max()) + 1.0

    def update(frame: int):
        ax.clear()
        
        # Dibujar Cadena 1 (Azul)
        ax.quiver(
            pos1[:, 0], pos1[:, 1], pos1[:, 2],
            History_S1[frame, :, 0], History_S1[frame, :, 1], History_S1[frame, :, 2],
            color='blue', length=1, normalize=True, arrow_length_ratio=0.3
        )
        
        # Dibujar Cadena 2 (Roja)
        ax.quiver(
            pos2[:, 0], pos2[:, 1], pos2[:, 2],
            History_S2[frame, :, 0], History_S2[frame, :, 1], History_S2[frame, :, 2],
            color='red', length=1, normalize=True, arrow_length_ratio=0.3
        )
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.0, visual_y_sep + 1.0)
        ax.set_zlim([-1.2, 1.2])
        
        ax.set_xlabel('Eje X (Moiré)')
        ax.set_ylabel('Eje Y (Separación)')
        ax.set_zlabel('Magnetización Z')
        ax.set_title(f'Dinámica Moiré - Tiempo: {frame * dt:.2e} s')
        
        # Ajustar ángulo de visión para apreciar ambas cadenas
        ax.view_init(elev=20., azim=-45) 
        return ()

    # Intervalo bajo para animación fluida. Si num_pasos es enorme, considera un 'step'
    ani = FuncAnimation(fig, update, frames=range(0, num_pasos, 1), interval=20, blit=False)
    plt.show()

    # 2. Evaluación de la Energía Total del Sistema Acoplado
    print("Calculando historia de energía...")
    Energy = evaluate_energy_history(History_S1, History_S2, cache, Kz_field)
    
    # Normalizar por el número TOTAL de espines
    total_spins = n1 + n2
    Energy_per_spin = Energy / total_spins

    time_axis = np.linspace(0, total_time, num_pasos)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, Energy_per_spin, color='purple', linewidth=2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Energía total por espín (unidades de campo)')
    plt.title('Conservación de Energía: Sistema Bicapa Moiré')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return