#!/usr/bin/env bash
# Starts the pumpkin. Settings come from jim.toml (copy jim.example.toml); options given here win.
# With brain "claude", asks for the Anthropic API key (hidden) if it isn't set.
cd "$(dirname "$0")"
if [ ! -f jim.toml ]; then
  echo "No jim.toml yet: copy jim.example.toml to jim.toml and set your WLED address." >&2
fi
uses_claude=$(.venv/bin/python -c "
import sys, tomllib
try: print(tomllib.load(open('jim.toml', 'rb')).get('brain', ''))
except Exception: print('')" 2>/dev/null)
if { [[ " $* " == *" --brain claude "* ]] || [ "$uses_claude" = claude ]; } && [ -z "$ANTHROPIC_API_KEY" ]; then
  read -rsp "Paste your Anthropic API key (input hidden) and press Enter: " ANTHROPIC_API_KEY
  echo
  export ANTHROPIC_API_KEY
fi
exec ./run.sh "$@"
