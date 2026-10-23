.DEFAULT_GOAL := qa
TESTS ?= tests
PYTEST_ARGS ?=

.PHONY: env lint typecheck test qa build

env:
	uv sync --locked --python 3.13

lint:
	uv run --locked --no-sync --offline ruff check .
	uv run --locked --no-sync --offline ruff format --check .

typecheck:
	uv run --locked --no-sync --offline pyright --project . --warnings
	bash -o pipefail -c 'rg --files --hidden --no-ignore --glob "*.py" --glob "*.pyi" --null scripts src tests | xargs -0 uv run --locked --no-sync --offline ruff check --config pyproject.toml --select TID251,ANN401 --ignore-noqa --no-force-exclude --'

test:
	uv run --locked --no-sync --offline python -m pytest -q $(TESTS) $(PYTEST_ARGS)

qa: lint typecheck

build:
	uv build

.PHONY: verifytypes
verifytypes:
	uv run --locked --no-sync --offline pyright --verifytypes mlx_reason_ct --ignoreexternal
