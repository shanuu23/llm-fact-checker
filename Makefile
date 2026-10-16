.PHONY: setup test lint format clean app app-package app-source-package

## Start the single-user local evaluator (no tunnel; inference is opt-in in the UI)
app:
	bash "Launch ModelLens.command"

## Build a Mac app ZIP (ad-hoc signed; normal macOS approval still required)
app-package:
	.venv/bin/python scripts/package_app.py --macos-app

## Developer-only source ZIP; downloaded .command files may be blocked by Finder
app-source-package:
	.venv/bin/python scripts/package_app.py

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
