# INSTALL

This file gives the shortest reviewer-oriented installation path for the C2|Q> artifact.

## Reviewer Entry Paths

Two supported entry paths are documented:

- Lowest setup burden: Docker
- Fastest local path: source checkout with Python 3.12

For the main TOSEM RCR reproduction path, use the source checkout route.
Use Docker as the lowest-barrier preliminary verification route.

## Option A: Docker (Lowest Setup Burden)

This path avoids installing Python 3.12 on the host machine.

```bash
git clone https://github.com/C2-Q/C2Q.git
cd C2Q
make docker-build
```

Minimal Docker verification:

```bash
make docker-reproduce-json-smoke
```

Docker writes outputs back into the repository under `artifacts/`.
`make docker-reproduce-json-smoke` is the lowest-setup check because it does not require a local Python installation or the parser model.

If you want the next Docker check after that, install the parser model archive and then run:

```bash
make docker-smoke
```

## Option B: Source Checkout (Fastest Local Path)

1. Clone the repository and enter it:

```bash
git clone https://github.com/C2-Q/C2Q.git
cd C2Q
```

2. Confirm that Python 3.12 is installed:

macOS / Linux:

```bash
python3.12 --version
```

Windows PowerShell:

```powershell
py -3.12 --version
```

If `python3.12` is not available:
- macOS:
  - `brew install python@3.12`
  - or install Python 3.12 from [python.org downloads](https://www.python.org/downloads/)
- Windows:
  - install Python 3.12 from [python.org downloads](https://www.python.org/downloads/)
  - then check with `py -3.12 --version`
- Linux:
  - install Python 3.12 from your distribution packages
  - then check with `python3.12 --version`
  - if Python 3.12 is difficult to install locally, use the Docker reviewer path instead

3. Create a clean Python 3.12 virtual environment:

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

4. If you need parser-backed commands, install the parser model archive.
Preferred one-command setup:

```bash
make model-setup
```

For review use, the recommended path is:
- run `make model-setup`
- or download the model zip in a browser and then run `make model-setup MODEL_ARCHIVE=/path/to/saved_models_2025_12.zip`

Reason:
- the default automation target is now the GitHub Release asset and works with scripted download
- the Zenodo record remains the archival source
- browser download plus local archive install is still the most robust fallback route

If you already downloaded the archive to a custom location:

```bash
make model-setup MODEL_ARCHIVE=/path/to/saved_models_2025_12.zip
```

Most robust manual route:

```bash
python tools/setup_model.py --archive /path/to/saved_models_2025_12.zip --model-path src/parser/saved_models_2025_12
```

Model archives:
- [saved_models_2025_12.zip (GitHub Release)](https://github.com/C2-Q/C2Q/releases/download/v1.0-artifact/saved_models_2025_12.zip)
- [saved_models_2025_12.zip (Zenodo archival copy)](https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1)

5. Verify the environment:

```bash
make doctor
```

If you only need the model-free JSON report path, you can skip the parser-model step and go directly to:

```bash
make reproduce-json-smoke
```

## Minimal Verification

Model-free minimal check:

```bash
make reproduce-json-smoke
```

Model-backed minimal check:

```bash
make smoke
```

Expected outputs:
- [artifacts/smoke/summary.json](artifacts/smoke/summary.json)
- [artifacts/reproduce/json/smoke](artifacts/reproduce/json/smoke)

The smoke summary should describe a `MaxCut` instance with a 4x4 QUBO and a 4-qubit QAOA circuit.

## Main Paper-Backed Commands

Commands that require the parser model:

```bash
make reproduce-smoke
make validate-dataset
make reproduce-paper
```

Commands that do not require the parser model:

```bash
make reproduce-json-smoke
make reproduce-json-paper
make recommender-maxcut
```

Time-consuming path:

```bash
make reproduce-paper
```

`make reproduce-paper` is expected to take roughly 10 hours.

## Docker Path

Additional Docker commands:

```bash
make docker-recommender-maxcut
make docker-validate-dataset
make docker-paper
make docker-reproduce-json-paper
```

Docker uses `/tmp/c2q-venv` inside the container and writes outputs back into the repository via the bind mount.

## Notes

- The primary reviewer path does not require cloud credentials.
- Optional cloud-provider integrations are not required for the core artifact path.
- Full parser training is notebook-driven and not part of the minimal verification path.
