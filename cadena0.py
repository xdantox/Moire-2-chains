import numpy as np
import math


def build_afm_state_from_npy(n, npy_path, frame=-1, noise_x=0.00001):
    """Construye un estado usando la base de una cadena .npy con PBC y ruido en x."""
    data = np.load(npy_path, mmap_mode="r")
    if data.ndim == 3:
        base = data[frame]
    elif data.ndim == 2:
        base = data
    else:
        raise ValueError("El .npy debe tener forma (n, 3) o (t, n, 3).")

    result = []
    pattern_len = len(base)
    noise = noise_x * np.random.randn(n)
    noise -= noise.mean()  # centrar ruido (evita drift global)
    for i in range(n):
        vec = base[i % pattern_len].copy()  # PBC via indice modular
        vec[0] += noise[i]                  # agregar ruido en eje x
        vec /= np.linalg.norm(vec)          # normalizar
        result.append(vec)
    return np.array(result)



def build_chain_from_direct2chain_params(n_sites, q, params, chain_id=1, noise_x=1e-5, seed=None):
    """
    Construye el estado base helicoidal (espiral inmensurada) a partir de los 
    parámetros arrojados por el minimizador para inicializar la simulación LLG.

    El orden de params debe ser:
    [mz1, mz2, gamma1, gamma2, alpha1, alpha2, phi_ind1, phi_ind2, delta1, delta2]
    """
    # 1. Mapeo de parámetros según el ID de la cadena
    if chain_id == 1:
        mz, gamma, alpha, phi, delta = params[0], params[2], params[4], params[6], params[8]
    elif chain_id == 2:
        mz, gamma, alpha, phi, delta = params[1], params[3], params[5], params[7], params[9]
    else:
        raise ValueError("chain_id debe ser 1 o 2")

    # 2. Reconstruir la red bipartita A/B (Dimerización geométrica)
    idx = np.arange(n_sites, dtype=np.int64)
    # parity: +1 para sitios pares (A), -1 para sitios impares (B)
    parity = np.where((idx % 2) == 0, 1.0, -1.0) 

    # 3. Construir la fase espacial (theta_n)
    # Término inmensurado + Término Staggered + Armónicos + Fase global
    theta = idx * q + gamma * parity
    if alpha != 0.0:
        theta += alpha * np.sin(2.0 * q * idx + phi)
    theta += delta
    # 4. Proyectar en la esfera de Bloch (Plano FÁCIL = XZ o YZ)
    # Como Z es el eje fácil, queremos que el seno (que oscila entre +-1) caiga en Z.
    mz = max(-0.999, min(0.999, float(mz)))  
    plane = np.sqrt(max(0.0, 1.0 - mz * mz)) 

    S = np.zeros((n_sites, 3), dtype=float)
    S[:, 0] = plane * np.cos(theta)  # Eje X (A lo largo de la cadena)
    S[:, 1] = mz                     # Eje Y (Eje transversal constante)
    S[:, 2] = plane * np.sin(theta)  # Eje Z (EJE FÁCIL: Aquí cae la alternancia +-1)

    # 5. Inyectar ruido térmico/estocástico
    if noise_x > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(scale=noise_x, size=n_sites)
        noise -= noise.mean() 
        
        # Como los espines están ahora fuertemente anclados en +-Z, 
        # inyectamos el ruido en los ejes transversales (X o Y) para 
        # permitir que la LLG explore fluctuaciones sin quedarse rígidamente atascada.
        S[:, 0] += noise  # Ruido en X
        S[:, 1] += rng.normal(scale=noise_x, size=n_sites) - noise.mean() # Ruido en Y

    # 6. Renormalización estricta ||S|| = 1
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    S /= norms

    return S