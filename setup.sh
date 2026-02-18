#!/bin/bash

set -e

sudo apt update
sudo apt install -y build-essential rustc

if which uv >/dev/null 2>&1; then
    echo "uv is installed."
else
    echo "uv is NOT installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

if [[ "$(uname)" == "Linux" ]]; then
    uv python pin pypy@3.10
elif [[ "$(uname)" == "Darwin" ]]; then # macOS
    uv python pin 3.10
elif [[ "$(uname)" == *"MINGW"* ]]; then # Git Bash on Windows
    uv python pin 3.10
fi

uv run python scripts/download_trace.py