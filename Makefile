.PHONY: setup test lint format clean

## Install dependencies
setup:
	uv sync

## Run the test suite (no Ollama needed)
test:
	uv run python -m unittest discover -s tests -v

## Check formatting and lint
lint:
	uv run ruff format --check
	uv run ruff check

## Apply formatting and lint fixes
format:
	uv run ruff check --fix
	uv run ruff format

## Delete compiled Python files
clean:
	find . -type f -name "*.py[co]" -not -path "./.venv/*" -delete
	find . -type d -name "__pycache__" -not -path "./.venv/*" -delete
