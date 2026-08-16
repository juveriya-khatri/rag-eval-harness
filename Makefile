PY ?= python3
OUT ?= out

.PHONY: help install demo test cov gate baseline clean lint

help:
	@echo "make install   - install package in editable mode with dev extras"
	@echo "make demo      - run the bundled evaluation and write $(OUT)/report.html"
	@echo "make test      - run the test suite"
	@echo "make cov       - run tests with coverage"
	@echo "make baseline  - freeze current results as baseline.json"
	@echo "make gate      - fail if metrics regress against baseline.json"
	@echo "make clean     - remove build/output artifacts"

install:
	$(PY) -m pip install -e ".[dev]"

demo:
	$(PY) -m ragval run --dataset bundled --k 1,3,5,10 \
		--html $(OUT)/report.html --json $(OUT)/results.json --junit $(OUT)/junit.xml

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=ragval --cov-report=term-missing

baseline:
	$(PY) -m ragval baseline --results $(OUT)/results.json --out baseline.json

gate:
	$(PY) -m ragval gate --results $(OUT)/results.json --config ragval.config.json --baseline baseline.json

clean:
	rm -rf $(OUT) build dist *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
