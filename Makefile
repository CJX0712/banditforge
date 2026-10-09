# BanditForge Makefile — convenience targets.
PY ?= python

.PHONY: test lint fmt demo bench ci clean

test:
	$(PY) -m pytest tests -q -W ignore::UserWarning

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

demo:
	$(PY) examples/run_demo.py

bench:
	$(PY) cli.py benchmark --seeds 10 --rounds 3000

ci: lint test demo

clean:
	$(PY) -m pytest tests --cov= --cov-report=term >/dev/null 2>&1 || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
