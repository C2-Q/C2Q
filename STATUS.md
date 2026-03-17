# STATUS

This file states the intended ACM artifact-badging position for the current C2|Q> artifact.

## Targeted Badges

### Artifacts Available

Status:
- justified

Why:
- source repository is available
- parser model is archived separately on Zenodo
- paper evaluation data record is archived on Zenodo
- primary artifact commands and output locations are documented

### Artifacts Evaluated – Functional

Status:
- justified for the primary reviewer path

Why:
- local smoke path is implemented and validated
- JSON-DSL smoke path is implemented and validated
- recommender experiment path is scripted and produces deterministic output files
- dataset validation path is scripted and produces stable CSV outputs
- the main path does not require cloud credentials or live hardware

Primary functional commands:
- `make smoke`
- `make reproduce-json-smoke`
- `make recommender-maxcut`
- `make validate-dataset`
- `make reproduce-smoke`

### Artifacts Evaluated – Reusable

Status:
- possible, but not claimed as the primary badge target yet

Why not stronger:
- the full paper run is long-running
- Docker is useful and documented, but the shortest validated reviewer path is still the source checkout path
- platform coverage has been exercised primarily on macOS plus Linux-style container assumptions, not a broad OS matrix

## Honest Limitations

- `make reproduce-paper` is time-consuming and should not be treated as a first-pass reviewer check.
- parser training provenance is archived, but the notebook path is not part of the minimal verification route.
- optional cloud-provider integrations exist, but they are outside the core reproducibility path.
