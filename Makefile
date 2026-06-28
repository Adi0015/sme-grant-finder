# RAISE — common tasks. Uses a local Python 3.13 venv at ./venv-index.
PY := venv-index/bin/python
PIP := venv-index/bin/pip
STREAMLIT := venv-index/bin/streamlit

.PHONY: help setup build-index run-app demo run-eval gold faithfulness benchmark docker-build docker-run clean

help:
	@echo "setup        - create venv-index (py3.13) + install requirements"
	@echo "build-index  - build the Chroma vector index from data/calls.json + data/projects.json"
	@echo "run-app      - launch the Streamlit UI (needs index + Ollama)"
	@echo "demo         - launch the UI in demo mode (no index/LLM needed): open ?demo=1"
	@echo "gold         - (re)generate the gold labeling sheet data/gold/to_label.csv"
	@echo "run-eval     - retrieval metrics (needs data/gold/labeled.csv) + faithfulness"
	@echo "benchmark    - dense vs hybrid re-rank, logged to MLflow"
	@echo "docker-build / docker-run - container image for the app"

setup:
	python3.13 -m venv venv-index
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

build-index:
	$(PY) src/build_index.py

run-app:
	$(STREAMLIT) run app.py

demo:
	@echo "Open http://localhost:8501/?demo=1 after this starts:"
	$(STREAMLIT) run app.py

gold:
	$(PY) src/build_gold.py

run-eval:
	$(PY) src/eval_retrieval.py
	$(PY) src/eval_faithfulness.py

benchmark:
	$(PY) src/eval_benchmark.py

docker-build:
	docker build -t raise-app .

docker-run:
	docker run --rm -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 -v "$(PWD)/data:/app/data" raise-app

clean:
	rm -rf __pycache__ src/__pycache__ data/*.log data/*.err
