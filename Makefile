.PHONY: test demo install install-all bench

install:
	pip install -e .

install-all:
	pip install -e ../compressionX
	pip install -e ".[all]"

test:
	pip install -e ../compressionX -q
	pytest -v

bench:
	pip install cma numpy -q
	python benchmarks/compare_openfugu.py

demo:
	triverse demo
