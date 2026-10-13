#!/usr/bin/env bash
# Launch the talking pumpkin. Extra args pass through, e.g. ./run.sh --type
cd "$(dirname "$0")"
exec .venv/bin/python pumpkin.py "$@"
