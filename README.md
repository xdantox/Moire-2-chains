# Moire-2-chains

Código de investigación para modelar dos cadenas magnéticas acopladas con desajuste de red y estudiar texturas AFM, acoplamiento intercadena y dinámica de espines.

> **Estado del repositorio:** prototipo de investigación. La simulación principal actualmente depende de historias `.npy` locales y puede producir salidas grandes; revisa los requisitos de ejecución antes de lanzarla.

## Qué contiene

- geometría de dos cadenas y cálculo de acoplamientos intra/intercadena;
- construcción/optimización de estados iniciales;
- integración de la dinámica LLG con rutinas JIT de Numba;
- análisis de energía, estado AFM y espectros;
- visualización y animación de las cadenas.

## Instalación

Se recomienda Python 3.10 o posterior:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Scripts principales

- `Heff.py`: parámetros, geometría, matrices de acoplamiento y campo efectivo.
- `E.py`: energía del estado y evaluación de historias.
- `cadena0.py`: constructores de estados iniciales; una función carga un `.npy` externo.
- `direct2chain.py`: minimización/ansatz para estados de dos cadenas.
- `rot.py`: integrador de dinámica LLG.
- `chains.py`: orquestación de la simulación, guardado de historias y animación.
- `relax.py`, `check.py`, `fft.py`: análisis del orden AFM, estado final y espectro.
- `animation.py`: visualización de la dinámica.

Consulta el [mapa de archivos](docs/REPO_MAP.md) para más detalles.

## Ejecución y datos

`chains.py` espera historias `.npy` con nombres concretos para inicializar ambas cadenas. Esos archivos no están incluidos en una instalación limpia; actualiza las rutas en el script o prepara los datos correspondientes antes de ejecutar:

```bash
python chains.py
```

La corrida predeterminada usa una historia de 65 536 pasos y guarda las historias completas de ambas cadenas, lo que puede requerir bastante memoria y almacenamiento. Para una prueba inicial, reduce `num_pasos` y usa un tamaño pequeño del sistema en una copia de la configuración. No lances la configuración completa sin estimar antes memoria/tiempo.

Los scripts de análisis también pueden esperar resultados producidos previamente. Comprueba en cada script los archivos de entrada y sus parámetros.

## Reproducibilidad

- `cadena0.py` permite establecer una semilla al construir una cadena desde parámetros; `chains.py` carga datos y no define una semilla global para todas las operaciones aleatorias.
- Registra parámetros físicos, tamaño, semilla, versiones de bibliotecas y configuración de hardware junto con cada resultado.
- Las salidas `.npy` no se deben subir sin revisar su tamaño y procedencia.
- No se declara aquí una licencia; solicita permiso al autor antes de reutilizar el código si no hay una licencia publicada.
