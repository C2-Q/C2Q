PYTHON ?= $(shell if command -v python3.12 >/dev/null 2>&1; then echo python3.12; else echo python3; fi)
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_READY := $(VENV)/.deps-installed
REQ_FILE := $(if $(wildcard requirements-lock.txt),requirements-lock.txt,requirements.txt)

PRIMARY_CSV ?= src/parser/python_programs.csv
BACKUP_CSV ?= src/parser/data.csv
MODEL_PATH ?= src/parser/saved_models_2025_12
OUTPUT_ROOT ?= artifacts/reproduce
TIME_LIMIT_SECS ?= 300
MODEL_SETUP_URL ?= https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link

.PHONY: venv doctor verify verify-model lock-deps model-check model-download reproduce-paper reproduce-smoke clean-reproduce

$(VENV_READY): requirements.txt $(REQ_FILE)
	@set -e; \
	echo "[venv] preparing $(VENV) using $$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	if ! $(PYTHON) -m venv --clear $(VENV); then \
		echo "[venv] standard bootstrap failed, retrying without ensurepip"; \
		$(PYTHON) -m venv --clear --without-pip $(VENV); \
		$(PYTHON) -m pip --python $(VENV) install --upgrade pip; \
	fi; \
	$(VENV_PIP) install --upgrade pip; \
	echo "[venv] installing dependencies from $(REQ_FILE)"; \
	$(VENV_PIP) install -r $(REQ_FILE); \
	touch $(VENV_READY)

venv: $(VENV_READY)

doctor:
	$(PYTHON) tools/doctor.py --model-path $(MODEL_PATH)

verify: venv
	PYTHONPATH=. $(VENV_PY) -m pytest -m "unit and not model and not paper" -p no:warnings

verify-model: venv model-check
	C2Q_MODEL_PATH=$(MODEL_PATH) PYTHONPATH=. $(VENV_PY) -m pytest src/tests/test_c2q.py -m model -p no:warnings

lock-deps: venv
	$(VENV_PY) -m pip freeze --exclude-editable > requirements-lock.txt

model-check:
	$(PYTHON) tools/setup_model.py --model-path $(MODEL_PATH)

model-download:
	$(PYTHON) tools/setup_model.py --model-path $(MODEL_PATH) --download --url $(MODEL_SETUP_URL)

reproduce-paper: venv model-check
	$(VENV_PY) tools/reproduce_paper.py \
		--mode paper \
		--primary-csv $(PRIMARY_CSV) \
		--backup-csv $(BACKUP_CSV) \
		--model-path $(MODEL_PATH) \
		--time-limit-secs $(TIME_LIMIT_SECS) \
		--output-root $(OUTPUT_ROOT)

reproduce-smoke: venv model-check
	$(VENV_PY) tools/reproduce_paper.py \
		--mode smoke \
		--max-cases 4 \
		--primary-csv $(PRIMARY_CSV) \
		--backup-csv $(BACKUP_CSV) \
		--model-path $(MODEL_PATH) \
		--time-limit-secs $(TIME_LIMIT_SECS) \
		--output-root $(OUTPUT_ROOT)

clean-reproduce:
	rm -rf $(OUTPUT_ROOT)
