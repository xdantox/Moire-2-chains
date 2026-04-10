import numpy as np

def cadena0spinhistory(n):
    Spin_history = np.load('D_plane = 0 relax.npy',mmap_mode='r')  # Carga del historial de spins
    num_pasos = Spin_history.shape[0]
    base = Spin_history[num_pasos-1]
    result = []
    pattern_len = len(base)
    noise_x = 0.00001 * np.random.randn(n)
    noise_x -= noise_x.mean()  # centrar ruido (evita drift global)
    for i in range(n):
        vec = base[i % pattern_len].copy()  # PBC via índice modular
        vec[0] += noise_x[i]                # agregar ruido en eje x
        vec /= np.linalg.norm(vec)          # normalizar
        result.append(vec)
    return np.array(result)


