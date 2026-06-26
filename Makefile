.PHONY: test demo install

install:
	pip install -e ".[dev]"

test:
	pytest -v

demo:
	triverse demo
