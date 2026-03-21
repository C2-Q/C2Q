# STATUS

This file states the ACM artifact badges claimed for the C2|Q> artifact,
submitted alongside the TOSEM RCR report.

## Claimed Badges

### Artifacts Available
All author-created artifacts are permanently archived with persistent
identifiers and released under open licenses. The framework source is on
GitHub (Apache-2.0). Evaluation data and archived outputs are on Zenodo
(DOI: 10.5281/zenodo.17071667; CC-BY-4.0). The parser model is archived
on Zenodo (DOI: 10.5281/zenodo.19061126; Apache-2.0).

### Artifacts Evaluated -- Functional
The artifacts are documented, consistent with the paper's three experiments,
complete, and exercisable. Experiments 2 and 3 are directly runnable via
documented make targets; smoke checks complete within 30 minutes on a clean
machine and produce concrete, verifiable outputs. The Docker route eliminates
host-environment variability. Experiment 1 is supported through the released
training notebook and pretrained model, reflecting the cost of full retraining
rather than a gap in the artifact.

Primary functional commands:
- make smoke
- make reproduce-json-smoke
- make reproduce-smoke
- make reproduce-paper
- make recommender-maxcut
- make validate-dataset

### Artifacts Evaluated -- Reusable
The artifact significantly exceeds minimal functionality. The modular pipeline
(encoder, deployment, decoder) is documented independently. Multiple execution
routes (Docker, source checkout, PyPI) lower the barrier for different
environments. All dependencies are pinned (pyproject.toml uses == throughout).
The make target structure separates smoke checks, paper reproduction, and
dataset validation. Archived outputs allow comparison without re-running.
Environment assumptions and platform boundaries are documented in REQUIREMENTS.

## Honest Limitations
- make reproduce-paper takes approximately 10 hours on the reference machine
  (Apple M1 Max) and is not suitable as a first-pass check.
- Parser training provenance is archived but retraining is not part of the
  minimal verification route; the released model is the primary artifact.
- Optional cloud-provider integrations exist but are outside the core
  reproducibility path.
- Real-hardware validation (IBM Brisbane, Finland Helmi) requires device
  access and cannot be independently replicated.
