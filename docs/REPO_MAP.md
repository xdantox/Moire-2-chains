# Mapa del repositorio — Moire-2-chains

## Componentes principales

- `Heff.py`: parámetros globales, geometría, matrices de acoplamiento, caché y cálculo JIT del campo efectivo. Usa SciPy sparse si está disponible.
- `E.py`: energía por sitio/total y evaluación de historias.
- `cadena0.py`: creación de estados a partir de una historia guardada o de parámetros; la ruta `.npy` debe existir para el primer caso.
- `direct2chain.py`: ansatz y minimización de estados de dos cadenas; requiere `scipy.optimize` para el minimizador.
- `rot.py`: integrador LLG y ejecución con Numba.
- `chains.py`: script orquestador que carga estados de entrada, integra, guarda historias y llama a la visualización.

## Análisis y visualización

- `relax.py`: diagnóstico del orden de Néel y espectro espacial.
- `check.py`: métricas y visualización de un estado final.
- `fft.py`: análisis espectral de historias guardadas; importa `chains.py` para `dt`, así que puede inicializar sus módulos asociados.
- `animation.py`: animación de las dos cadenas y gráfica de energía.

## Flujo orientativo

1. Define y revisa los parámetros en `Heff.py`.
2. Prepara los estados iniciales requeridos por `chains.py` o usa las rutinas de `direct2chain.py`/`cadena0.py`.
3. Ejecuta una prueba pequeña antes de la simulación completa.
4. Analiza las historias con `check.py`, `relax.py` o `fft.py`.

## Datos

Las historias `.npy` mencionadas por `chains.py`, `cadena0.py` y los análisis son entradas locales; no se incluyen en el código fuente. Las historias completas generadas por el integrador pueden ser voluminosas. Considera guardar resultados en una carpeta de salida dedicada y documentar parámetros/semillas para cada corrida.
