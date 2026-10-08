.PHONY: env lint typecheck test qa build

env:
	uv sync --locked --python 3.12

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

typecheck:
	uv run --locked pyright --project . --warnings
	uv run --locked pyright --verifytypes mlx_reason_ct --ignoreexternal

test:
	uv run --locked python -m pytest -q

qa: lint typecheck test

build:
	uv build
