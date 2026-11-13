# Moire-2-chains

Research code for modeling two coupled magnetic chains with lattice mismatch and studying antiferromagnetic textures, interchain coupling, and spin dynamics.

> **Repository status:** Research prototype. The main simulation currently depends on local `.npy` histories and can produce large output files. Review the run requirements before launching it.

## Overview

- Geometry of two chains and calculation of intra- and interchain couplings.
- Construction and optimization of initial states.
- Integration of spin dynamics using Numba JIT routines.
- Energy, antiferromagnetic-order, and spectral analyses.
- Visualization and animation of the chains.

## Installation

Python 3.10 or later is recommended:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Main scripts

- `Heff.py`: parameters, geometry, coupling matrices, and effective field.
- `E.py`: state energy and history evaluation.
- `cadena0.py`: initial-state constructors; one function loads an external `.npy` file.
- `direct2chain.py`: ansatz and minimization for two-chain states.
- `rot.py`: LLG dynamics integrator.
- `chains.py`: simulation driver that saves histories and launches visualization.
- `relax.py`, `check.py`, `fft.py`: antiferromagnetic-order, final-state, and spectral analyses.
- `animation.py`: dynamics visualization.

See the [repository map](docs/REPO_MAP.md) for additional details.

## Running and data requirements

`chains.py` expects `.npy` histories with specific names to initialize the chains. These files are not included in a clean checkout; update the paths in the script or prepare the required data first.

```bash
python chains.py
```

The default run uses 65,536 time steps and saves the complete histories for both chains, which can require substantial memory and disk space. For an initial test, reduce `num_pasos` and use a smaller system in a copy of the configuration. Estimate memory and runtime before launching the full configuration.

Analysis scripts may also expect previously generated results. Check each script's input files and parameters before running it.

## Reproducibility

- `cadena0.py` supports setting a seed when constructing a chain from parameters; `chains.py` loads input data and does not define a global seed for all random operations.
- Record physical parameters, system size, random seed, library versions, and hardware configuration with each result.
- Check the size and provenance of `.npy` outputs before committing them.
- This repository does not declare a license. If no license is published, contact the author before reusing the code.
