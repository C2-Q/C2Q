# REQUIREMENTS

## Operating System

Supported and exercised paths:
- macOS with Python 3.12 or 3.13 for local source execution
- Linux-compatible container path via Docker

Supported Python versions:
- Python 3.12: primary validated RCR environment
- Python 3.13: supported through conditional dependency pins
- Python 3.14+: not part of the supported path for this release

## Hardware

Minimum practical assumptions for the smoke path:
- 4 CPU cores
- 8 GB RAM

More comfortable for recommender and validation paths:
- 8 CPU cores
- 16 GB RAM

GPU hardware is not required.

## Disk

Observed local sizes in this workspace:
- parser model directory: about 477 MB
- current `artifacts/` directory: about 2.8 MB
- JSON smoke outputs: about 1.5 MB

Practical recommendation:
- at least 3 GB free for installation and smoke/reviewer paths
- more headroom for repeated report generation and Docker images

## Required Software

Source path:
- Python 3.12 or 3.13
- `venv`
- LaTeX toolchain with `pdflatex`

Docker path:
- Docker 29 or newer is recommended
- Docker Buildx is required for the documented image build path; check with `docker buildx version`
- on Linux/WSL, Docker commands may require `sudo` unless the user has access to the Docker daemon through the `docker` group

Python packages are pinned through:
- [pyproject.toml](pyproject.toml)
- [requirements-artifact.txt](requirements-artifact.txt)
- [requirements-dev.txt](requirements-dev.txt)

The core scientific stack is pinned to NumPy 2-compatible versions for both Python 3.12 and 3.13 because Qiskit 2.5 requires NumPy 2 or newer. NumPy is pinned to 2.4.6 rather than the newest 2.x release so that the optional cloud-provider dependency chain remains resolvable on Python 3.12. The Qiskit stack is pinned to Qiskit 2.5.0 and Qiskit Aer 0.17.2 for both supported Python versions.

## External Data / Models

Required for parser-backed paths:
- parser model archive (preferred automation source): [saved_models_2025_12.zip (GitHub Release)](https://github.com/C2-Q/C2Q/releases/download/v1.0-artifact/saved_models_2025_12.zip)
- archival copy: [saved_models_2025_12.zip (Zenodo)](https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1)

Referenced paper evaluation data record:
- [C2|Q> Dataset: Reports and Evaluation Inputs (v1.0.0)](https://doi.org/10.5281/zenodo.17071667)

## Credentials / External Services

Primary reviewer path requirements:
- no cloud credentials required
- no private quantum hardware required
- no paid external services required

Optional only:
- cloud-provider SDK extras for IBM / IonQ / Quantinuum / AWS integrations
- cloud-provider SDK extras are currently Python 3.12-only because parts of
  the provider dependency chain do not yet support Python 3.13 consistently

These are not part of the required RCR path.
