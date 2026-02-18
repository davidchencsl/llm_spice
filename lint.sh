#!/bin/bash

set -e

uv run ruff format
uv run ruff check --fix
uv run pyright
uv run pytest -m "not slow"