# REQUIREMENTS

## Operating System

Supported and exercised paths:
- macOS with Python 3.12 for local source execution
- Linux-compatible container path via Docker

Recommended Python version:
- Python 3.12
- Python 3.13+ is not part of the supported path for this release

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
- Python 3.12
- `venv`
- LaTeX toolchain with `pdflatex`

Docker path:
- Docker 29 or newer is recommended

Python packages are pinned through:
- [pyproject.toml](/Users/mac/Documents/GitHub/C2Q/pyproject.toml)
- [requirements-artifact.txt](/Users/mac/Documents/GitHub/C2Q/requirements-artifact.txt)
- [requirements-dev.txt](/Users/mac/Documents/GitHub/C2Q/requirements-dev.txt)

## External Data / Models

Required for parser-backed paths:
- parser model archive: [saved_models_2025_12.zip](https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1)

Referenced paper evaluation data record:
- [C2Q data record on Zenodo](https://zenodo.org/records/18780001)

## Credentials / External Services

Primary reviewer path requirements:
- no cloud credentials required
- no private quantum hardware required
- no paid external services required

Optional only:
- cloud-provider SDK extras for IBM / IonQ / Quantinuum / AWS integrations

These are not part of the required RCR path.
