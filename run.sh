#!/usr/bin/env bash
# Launch the talking pumpkin. Extra args pass through, e.g. ./run.sh --type
cd "$(dirname "$0")"
SITE=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
for d in "$SITE"/nvidia/*/lib; do LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"; done
export LD_LIBRARY_PATH
exec .venv/bin/python pumpkin.py "$@"
