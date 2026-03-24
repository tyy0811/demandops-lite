.PHONY: download prepare train evaluate benchmark serve test lint clean pipeline

download:
	.venv/bin/python scripts/download_data.py

prepare:
	.venv/bin/python scripts/prepare_data.py

train:
	.venv/bin/python scripts/train.py

evaluate:
	.venv/bin/python scripts/evaluate.py

benchmark:
	.venv/bin/python scripts/benchmark.py

serve:
	.venv/bin/uvicorn demandops.serving.app:app --host 0.0.0.0 --port 8001 --reload

test:
	.venv/bin/pytest tests/ -v

lint:
	.venv/bin/ruff check demandops/ tests/
	.venv/bin/ruff format --check demandops/ tests/

format:
	.venv/bin/ruff check --fix demandops/ tests/
	.venv/bin/ruff format demandops/ tests/

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet
	rm -rf artifacts/models/* artifacts/zone_universe.json
	rm -rf artifacts/reports/*
	rm -rf mlruns/
	rm -rf __pycache__ .pytest_cache

pipeline: download prepare train evaluate benchmark
