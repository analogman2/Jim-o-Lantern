#!/usr/bin/env bash
# Jim's control panel on http://127.0.0.1:8790 (this PC only). Ctrl+C stops the panel and Jim.
cd "$(dirname "$0")"
exec .venv/bin/python jim_panel.py "$@"
