from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import cast

from E_copy import evaluate_energy_history

def animation_two_chains(History_S1, History_S2, dt, cache, Kz_field, visual_y_sep=1.0):
    """
    Animación 3D para un sistema Moiré de dos cadenas acopladas.
    Permite rotación interactiva manual.
    """
    num_pasos = History_S1.shape[0]
    n1 = History_S1.shape[1]
    n2 = History_S2.shape[1]
    total_time = num_pasos * dt

    fig = plt.figure(figsize=(12, 8))
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))

    # 1. Generar posiciones 3D exactas usando el caché Moiré
    pos1 = np.zeros((n1, 3))
    pos1[:, 0] = cache.x1
    pos1[:, 1] = 0.0
    pos1[:, 2] = 0.0  

    pos2 = np.zeros((n2, 3))
    pos2[:, 0] = cache.x2
    pos2[:, 1] = visual_y_sep 
    pos2[:, 2] = 0.0

    # === CONFIGURACIÓN DE ZOOM PARA EVITAR EL "EFECTO BOSQUE" ===
    # Puedes subir este número si quieres ver más cadena, pero 40 suele ser el límite visual óptimo
    zoom_sites = 100
    
    # Límites del gráfico dinámicos ajustados al zoom
    x_min = min(pos1[0, 0], pos2[0, 0]) - 1.0
    x_max = max(pos1[zoom_sites, 0], pos2[zoom_sites, 0]) + 1.0

    def update(frame: int):
        ax.clear()
        
        # Dibujar Cadena 1 (Azul) - Guardamos el objeto en quiv1
        quiv1 = ax.quiver(
            pos1[:zoom_sites, 0], pos1[:zoom_sites, 1], pos1[:zoom_sites, 2],
            History_S1[frame, :zoom_sites, 0], History_S1[frame, :zoom_sites, 1], History_S1[frame, :zoom_sites, 2],
            color='blue', length=1, normalize=True, arrow_length_ratio=0.3
        )
        
        # Dibujar Cadena 2 (Roja) - Guardamos el objeto en quiv2
        quiv2 = ax.quiver(
            pos2[:zoom_sites, 0], pos2[:zoom_sites, 1], pos2[:zoom_sites, 2],
            History_S2[frame, :zoom_sites, 0], History_S2[frame, :zoom_sites, 1], History_S2[frame, :zoom_sites, 2],
            color='red', length=1, normalize=True, arrow_length_ratio=0.3
        )
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.5, visual_y_sep + 1.5)
        ax.set_zlim([-1.2, 1.2])
        
        # Forzar proporciones para no aplastar la hélice
        ax.set_box_aspect((x_max - x_min, visual_y_sep + 3.0, 2.4))
        
        ax.set_xlabel('Eje X (Moiré)')
        ax.set_ylabel('Eje Y (Separación)')
        ax.set_zlabel('Magnetización Z')
        ax.set_title(f'Dinámica Moiré - Tiempo: {frame * dt:.2e} s')
        
        # EL SECRETO PARA LA ROTACIÓN MANUAL ESTÁ AQUÍ: 
        # Devolver los objetos gráficos
        return quiv1, quiv2

    # El return en update permite interactuar mientras corre
    ani = FuncAnimation(fig, update, frames=range(0, num_pasos, 1), interval=20, blit=False)
    plt.show()

    # =================================================================
    # 2. Evaluación de la Energía Total del Sistema Acoplado
    # =================================================================
    print("Calculando historia de energía...")
    Energy = evaluate_energy_history(History_S1, History_S2, cache, Kz_field)
    
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