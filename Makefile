PY ?= python

.PHONY: install test lint coverage demo clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q -W ignore::UserWarning

lint:
	$(PY) -m ruff check banditforge tests

coverage:
	$(PY) -m pytest tests/ -q -W ignore::UserWarning --cov=banditforge --cov-report=term

demo:
	$(PY) banditforge/examples/run_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage __pycache__ banditforge/__pycache__
