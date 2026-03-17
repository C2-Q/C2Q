PYTHON ?= python3.12
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV_PY) -m pip
VENV_READY := $(VENV)/.deps-installed
REQ_FILE ?= $(if $(wildcard requirements-dev.txt),requirements-dev.txt,$(if $(wildcard requirements-lock.txt),requirements-lock.txt,requirements.txt))

PRIMARY_CSV ?= src/parser/python_programs.csv
BACKUP_CSV ?= src/parser/data.csv
MODEL_PATH ?= src/parser/saved_models_2025_12
MODEL_ARCHIVE ?=
OUTPUT_ROOT ?= artifacts/reproduce
TIME_LIMIT_SECS ?= 300
MODEL_SETUP_URL ?= https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1
RECOMMENDER_OUTPUT_DIR ?= artifacts/recommender_maxcut
RECOMMENDER_MIN_QUBITS ?= 4
RECOMMENDER_MAX_QUBITS ?= 58
RECOMMENDER_STEP ?= 2
RECOMMENDER_GRAPH_SEED ?= 100
RECOMMENDER_QAOA_LAYERS ?= 1
VALIDATION_OUTPUT_ROOT ?= artifacts/parser_validation
SMOKE_OUTPUT_DIR ?= artifacts/smoke
JSON_DSL_EXAMPLES_ROOT ?= src/c2q-dataset/inputs/json_dsl
JSON_DSL_SMOKE_ROOT ?= src/c2q-dataset/inputs/json_dsl_smoke
JSON_DSL_REPORTS_DIR ?= artifacts/json_dsl_reports
JSON_REPRODUCE_ROOT ?= artifacts/reproduce/json
DOCKER_IMAGE ?= c2q:latest
DOCKER_VENV ?= /tmp/c2q-venv
DOCKER_REQ_FILE ?= requirements-artifact.txt
DOCKER_RUN := docker run --rm -v "$(CURDIR)":/workspace -w /workspace $(DOCKER_IMAGE)

SETUP_MODEL_ARGS := --model-path $(MODEL_PATH)

.PHONY: venv doctor verify verify-model lock-deps model-check model-setup model-download smoke reproduce-paper reproduce-smoke recommender-maxcut validate-dataset json-dsl-examples json-dsl-check json-dsl-self-test json-dsl-smoke-cases json-dsl-reports reproduce-json-smoke reproduce-json-paper reproduce-json-full docker docker-build docker-model-download docker-smoke docker-paper docker-recommender-maxcut docker-validate-dataset docker-reproduce-json-smoke docker-reproduce-json-paper clean-reproduce

$(VENV_PY):
	@set -e; \
	if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "[venv] required interpreter '$(PYTHON)' not found."; \
		echo "[venv] install Python 3.12 for the source path, or use the lowest-setup reviewer path:"; \
		echo "       make docker-build && make docker-reproduce-json-smoke"; \
		exit 1; \
	fi; \
	if [ "$$($(PYTHON) -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" != "3.12" ]; then \
		echo "[venv] unsupported interpreter '$$($(PYTHON) -c 'import sys; print(sys.executable)')'"; \
		echo "[venv] C2Q source/reviewer path requires Python 3.12."; \
		echo "[venv] use 'PYTHON=python3.12 make ...' or use Docker instead."; \
		exit 1; \
	fi; \
	echo "[venv] preparing $(VENV) using $$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	if ! $(PYTHON) -m venv --clear $(VENV); then \
		echo "[venv] standard bootstrap failed, retrying without ensurepip"; \
		$(PYTHON) -m venv --clear --without-pip $(VENV); \
		$(PYTHON) -m pip --python $(VENV) install --upgrade pip; \
	fi

venv: $(VENV_PY) requirements.txt $(REQ_FILE)
	@set -e; \
	if [ ! -f "$(VENV_READY)" ]; then \
		$(VENV_PIP) install --upgrade pip; \
		echo "[venv] installing dependencies from $(REQ_FILE)"; \
		$(VENV_PIP) install -r $(REQ_FILE); \
		touch "$(VENV_READY)"; \
	elif [ "$(notdir $(REQ_FILE))" = "requirements-artifact.txt" ] && ! $(VENV_PY) -c "import qiskit_aer, transformers" >/dev/null 2>&1; then \
		echo "[venv] detected incomplete artifact environment, forcing dependency reinstall"; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install --force-reinstall -r $(REQ_FILE); \
		touch "$(VENV_READY)"; \
	elif [ "$(notdir $(REQ_FILE))" = "requirements-dev.txt" ] && ! $(VENV_PY) -c "import pytest, qiskit_aer, transformers" >/dev/null 2>&1; then \
		echo "[venv] detected incomplete dev environment, forcing dependency reinstall"; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install --force-reinstall -r $(REQ_FILE); \
		touch "$(VENV_READY)"; \
	elif [ "$(notdir $(REQ_FILE))" != "requirements-artifact.txt" ] && ! $(VENV_PY) -c "import pytest, qiskit_aer" >/dev/null 2>&1; then \
		echo "[venv] detected incomplete environment, forcing dependency reinstall"; \
		$(VENV_PIP) install --upgrade pip; \
		$(VENV_PIP) install --force-reinstall -r $(REQ_FILE); \
		touch "$(VENV_READY)"; \
	else \
		echo "[venv] environment already ready"; \
	fi

doctor:
	$(PYTHON) tools/doctor.py $(SETUP_MODEL_ARGS)

verify: venv
	PYTHONPATH=. $(VENV_PY) -m pytest -m "unit and not model and not paper" -p no:warnings

verify-model: venv model-check
	C2Q_MODEL_PATH=$(MODEL_PATH) PYTHONPATH=. $(VENV_PY) -m pytest src/tests/test_c2q.py -m model -p no:warnings

lock-deps: venv
	$(VENV_PY) -m pip freeze --exclude-editable > requirements-lock.txt

model-check:
	$(PYTHON) tools/setup_model.py $(SETUP_MODEL_ARGS)

model-setup:
	@set -e; \
	args="--auto --model-path $(MODEL_PATH) --url $(MODEL_SETUP_URL)"; \
	if [ -n "$(MODEL_ARCHIVE)" ]; then \
		args="$$args --archive \"$(MODEL_ARCHIVE)\""; \
	fi; \
	eval "$(PYTHON) tools/setup_model.py $$args"

model-download:
	$(PYTHON) tools/setup_model.py $(SETUP_MODEL_ARGS) --download --url $(MODEL_SETUP_URL)

smoke: venv model-check
	PYTHONPATH=. $(VENV_PY) scripts/artifact_smoke.py \
		--model-path $(MODEL_PATH) \
		--output-dir $(SMOKE_OUTPUT_DIR)

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

recommender-maxcut: venv
	PYTHONPATH=. $(VENV_PY) scripts/recommender_maxcut_pipeline.py \
		--output-dir $(RECOMMENDER_OUTPUT_DIR) \
		--min-qubits $(RECOMMENDER_MIN_QUBITS) \
		--max-qubits $(RECOMMENDER_MAX_QUBITS) \
		--step $(RECOMMENDER_STEP) \
		--graph-seed $(RECOMMENDER_GRAPH_SEED) \
		--qaoa-layers $(RECOMMENDER_QAOA_LAYERS)

validate-dataset: venv model-check
	PYTHONPATH=. $(VENV_PY) src/parser/validate_dataset.py \
		--mode all \
		--csv-path $(PRIMARY_CSV) \
		--backup-csv-path $(BACKUP_CSV) \
		--model-path $(MODEL_PATH) \
		--out-root $(VALIDATION_OUTPUT_ROOT)

json-dsl-examples: venv
	PYTHONPATH=. $(VENV_PY) -m src.json_engine \
		--generate_examples \
		--examples-root $(JSON_DSL_EXAMPLES_ROOT) \
		--n-per-family 10 \
		--clean-output

json-dsl-check:
	@test -d "$(JSON_DSL_EXAMPLES_ROOT)" || { echo "[json-dsl] missing $(JSON_DSL_EXAMPLES_ROOT). Run 'make json-dsl-examples' first."; exit 1; }
	@test -f "$(JSON_DSL_EXAMPLES_ROOT)/add/add_01.json" || { echo "[json-dsl] expected $(JSON_DSL_EXAMPLES_ROOT)/add/add_01.json. Run 'make json-dsl-examples' first."; exit 1; }
	@test -f "$(JSON_DSL_EXAMPLES_ROOT)/maxcut/maxcut_01.json" || { echo "[json-dsl] expected $(JSON_DSL_EXAMPLES_ROOT)/maxcut/maxcut_01.json. Run 'make json-dsl-examples' first."; exit 1; }

json-dsl-self-test: venv
	PYTHONPATH=. $(VENV_PY) -m src.json_engine \
		--self_test_examples \
		--examples-root $(JSON_DSL_EXAMPLES_ROOT)

json-dsl-smoke-cases: json-dsl-check
	rm -rf $(JSON_DSL_SMOKE_ROOT)
	mkdir -p $(JSON_DSL_SMOKE_ROOT)
	for rel in add/add_01.json factor/factor_01.json maxcut/maxcut_01.json mis/mis_01.json; do \
		mkdir -p "$(JSON_DSL_SMOKE_ROOT)/$$(dirname $$rel)"; \
		cp "$(JSON_DSL_EXAMPLES_ROOT)/$$rel" "$(JSON_DSL_SMOKE_ROOT)/$$rel"; \
	done

json-dsl-reports: venv
	PYTHONPATH=. $(VENV_PY) -m src.json_engine \
		--batch_report \
		--examples-root $(JSON_DSL_EXAMPLES_ROOT) \
		--reports-output-dir $(JSON_DSL_REPORTS_DIR)

reproduce-json-smoke: venv json-dsl-smoke-cases
	rm -rf $(JSON_REPRODUCE_ROOT)/smoke
	PYTHONPATH=. $(VENV_PY) -m src.json_engine \
		--batch_report \
		--examples-root $(JSON_DSL_SMOKE_ROOT) \
		--reports-output-dir $(JSON_REPRODUCE_ROOT)/smoke

reproduce-json-paper: venv json-dsl-check
	rm -rf $(JSON_REPRODUCE_ROOT)/paper
	PYTHONPATH=. $(VENV_PY) -m src.json_engine \
		--batch_report \
		--examples-root $(JSON_DSL_EXAMPLES_ROOT) \
		--reports-output-dir $(JSON_REPRODUCE_ROOT)/paper

reproduce-json-full: reproduce-json-paper

docker: docker-build

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-model-download: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) model-download

docker-smoke: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) smoke

docker-paper: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) reproduce-paper

docker-reproduce-json-smoke: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) reproduce-json-smoke

docker-reproduce-json-paper: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) reproduce-json-paper

docker-recommender-maxcut: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) recommender-maxcut

docker-validate-dataset: docker-build
	$(DOCKER_RUN) make VENV=$(DOCKER_VENV) REQ_FILE=$(DOCKER_REQ_FILE) validate-dataset

clean-reproduce:
	rm -rf $(OUTPUT_ROOT)
