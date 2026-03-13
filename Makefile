PYTHON ?= $(shell if command -v python3.12 >/dev/null 2>&1; then echo python3.12; else echo python3; fi)
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_READY := $(VENV)/.deps-installed

PRIMARY_CSV ?= src/parser/python_programs.csv
BACKUP_CSV ?= src/parser/data.csv
MODEL_PATH ?= src/parser/saved_models_2025_12
OUTPUT_ROOT ?= artifacts/reproduce
TIME_LIMIT_SECS ?= 300
MODEL_SETUP_URL ?= https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link

.PHONY: venv model-check model-download reproduce-paper reproduce-smoke clean-reproduce

$(VENV_READY): requirements.txt
	@set -e; \
	echo "[venv] preparing $(VENV) using $$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	if ! $(PYTHON) -m venv --clear $(VENV); then \
		echo "[venv] standard bootstrap failed, retrying without ensurepip"; \
		$(PYTHON) -m venv --clear --without-pip $(VENV); \
		$(PYTHON) -m pip --python $(VENV) install --upgrade pip; \
	fi; \
	$(VENV_PIP) install --upgrade pip; \
	$(VENV_PIP) install -r requirements.txt; \
	touch $(VENV_READY)

venv: $(VENV_READY)

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
