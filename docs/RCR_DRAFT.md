# RCR Draft

## Abstract

This artifact accompanies the TOSEM paper on C2|Q> and provides reviewer-oriented support for reproducing the paper’s core computational results. The artifact includes the framework source code, a released parser model archive, deterministic JSON inputs, validation scripts, a smoke reproduction path, a paper-scale reproduction path, and experiment-specific output directories under `artifacts/`.

## Overview

Paper:
- [INSERT DOI]
- [INSERT FINAL TOSEM TITLE]

Repository:
- C2|Q> source tree

Primary artifact goals:
- verify that the framework can construct quantum-ready representations from supported inputs
- reproduce the recommender multi-device variation experiment
- reproduce the smoke-scale paper workflow
- reproduce implementation-level and algorithmic/structural validation outputs

## Citation and Summary of the TOSEM Paper

[INSERT CITATION]

Short summary:
- C2|Q> bridges classical problem specifications and quantum software artifacts through parser, reduction, generation, execution, and reporting components.

## Key Claims in the TOSEM Paper

- [INSERT CLAIM 1]
- [INSERT CLAIM 2]
- [INSERT CLAIM 3]
- [INSERT CLAIM 4]

See also:
- [docs/CLAIMS_MAP.md](CLAIMS_MAP.md)

## Key Results Supported by This Artifact

- parser-backed smoke pipeline output
- recommender multi-device variation outputs
- smoke-scale report reproduction outputs
- dataset validation CSV outputs
- supplementary JSON DSL smoke outputs

## Artifact Inventory

| Component | Location | Purpose |
|---|---|---|
| Framework source | `src/` | Main implementation |
| Parser model setup helper | `tools/setup_model.py` | Model installation and checking |
| Paper reproduction runner | `tools/reproduce_paper.py` | Smoke and paper-scale report reproduction |
| Recommender experiment pipeline | `scripts/recommender_maxcut_pipeline.py` | Experiment 2 regeneration |
| Smoke pipeline | `scripts/artifact_smoke.py` | Fast reviewer verification |
| Dataset validation | `src/parser/validate_dataset.py` | Experiment 4 regeneration |
| JSON DSL interface | `src/json_engine.py` | Supplementary JSON-based reproduction path |

## Prerequisites and Requirements

- Python 3.12
- LaTeX / `pdflatex`
- released parser model archive
- optional Docker for isolated reviewer execution

See:
- [INSTALL.md](../INSTALL.md)
- [REQUIREMENTS.md](../REQUIREMENTS.md)

## Setup

1. Create and activate a Python 3.12 virtual environment.
2. Install the project with:

```bash
python -m pip install -e ".[dev]"
```

3. Install the released parser model with `tools/setup_model.py`.
4. Run:

```bash
make doctor
```

## Steps to Reproduce

### Minimal Reviewer Path

```bash
make smoke
make reproduce-json-smoke
```

### Main Experiment Paths

```bash
make recommender-maxcut
make reproduce-smoke
make validate-dataset
```

### Extended Path

```bash
make reproduce-paper
```

This extended run is time-consuming and expected to take roughly 10 hours.

## Output Locations

- smoke output: `artifacts/smoke/`
- JSON smoke output: `artifacts/reproduce/json/smoke/`
- recommender output: `artifacts/recommender_maxcut/`
- validation output: `artifacts/parser_validation/`
- paper smoke / paper full output: `artifacts/reproduce/{smoke,paper}/`

## Limitations

- parser training notebook provenance is archived but not part of the minimal reviewer path
- optional cloud-provider integrations are outside the required artifact path
- the full paper run is significantly slower than the smoke path
- [INSERT ANY FINAL CAMERA-READY LIMITATIONS]
