# C2|Q> Dataset: Reports, Inputs, and Usability Baseline (v1.0.0)

This directory contains repository-local copies of the benchmark inputs and
supporting materials used by the C2|Q> artifact:

> Boshuai Ye, Arif Ali Khan, Teemu Pihkakoski, Peng Liang,
> Muhammad Azeem Akbar, Matti Silveri, and Lauri Malmi. 2026.
> *C2|Q>: A Robust Framework for Bridging Classical and Quantum Software
> Development.* ACM Transactions on Software Engineering and Methodology.
> DOI: 10.1145/3803018.

The long-term archival record for the evaluation data and archived outputs is
Zenodo DOI: 10.5281/zenodo.17071667.

## Contents

- `inputs/python/` - Python code snippets used as encoder evaluation inputs.
- `inputs/json/` - structured JSON instances used for deterministic checks.
- `inputs/json_dsl/` - generated JSON-DSL examples maintained from
  `src/json_engine.py`.
- `inputs/json_dsl_smoke/` - curated smoke subset with one example each for
  `ADD`, `Factor`, `MaxCut`, and `MIS`.
- `usability_comparison/` - manual Qiskit baseline materials used for the
  proxy-based usability comparison in the paper.

Generated reports and validation outputs are written under `artifacts/` when
the reproduction commands are run. Archived reports and evaluation outputs are
preserved in the Zenodo data record.

## Reproducing Artifacts

From the repository root, run the model-free JSON smoke check:

```bash
make reproduce-json-smoke
```

The full JSON report path is:

```bash
make reproduce-json-paper
```

The Python-code smoke path requires the parser model:

```bash
MODEL_PATH=src/parser/saved_models_2025_12 make reproduce-smoke
```

The full Python-code paper path is:

```bash
MODEL_PATH=src/parser/saved_models_2025_12 make reproduce-paper
```

The full Python-code run may generate up to 434 reports and is time-consuming.
The full JSON run takes roughly two hours on the reference machine.

## Parser Model

The parser model is distributed separately because it is a large trained
checkpoint. The preferred scripted source is the GitHub Release asset:

- https://github.com/C2-Q/C2Q/releases/download/v1.0-artifact/saved_models_2025_12.zip

The archival model record is:

- https://doi.org/10.5281/zenodo.19061126

Install or verify the model from the repository root with:

```bash
make model-setup
make model-check
```

If the archive was downloaded manually:

```bash
make model-setup MODEL_ARCHIVE=/path/to/saved_models_2025_12.zip
```

## Output Locations

- JSON smoke reports: `artifacts/reproduce/json/smoke/`
- JSON paper reports: `artifacts/reproduce/json/paper/`
- Python-code smoke reports: `artifacts/reproduce/smoke/reports/`
- Python-code paper reports: `artifacts/reproduce/paper/reports/`
- Recommender outputs: `artifacts/recommender_maxcut/`
- Parser validation outputs: `artifacts/parser_validation/`

## License

- Data, reports, and usability materials: CC BY 4.0.
- Software framework: Apache-2.0.

## Citation

If you use this dataset, please cite the paper and the archived data record:

> Boshuai Ye and Arif Ali Khan. 2025.
> *C2|Q> Dataset: Reports and Evaluation Inputs (v1.0.0).*
> Zenodo. DOI: 10.5281/zenodo.17071667.
