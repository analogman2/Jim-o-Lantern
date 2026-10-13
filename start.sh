#!/usr/bin/env bash
# Starts the pumpkin. With --brain claude, asks for the Anthropic API key (hidden) if it isn't set.
cd "$(dirname "$0")"
if [[ " $* " == *" --brain claude "* ]] && [ -z "$ANTHROPIC_API_KEY" ]; then
  read -rsp "Paste your Anthropic API key (input hidden) and press Enter: " ANTHROPIC_API_KEY
  echo
  export ANTHROPIC_API_KEY
fi
exec ./run.sh "$@"
