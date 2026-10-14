#!/usr/bin/env bash
# Starts the pumpkin. Options in ./jim.args (one or more per line, this machine only) come first,
# then any given here. With --brain claude, asks for the Anthropic API key (hidden) if it isn't set.
cd "$(dirname "$0")"
saved=()
if [ -f jim.args ]; then
  read -ra saved <<< "$(grep -v '^\s*#' jim.args | tr '\n' ' ')"
fi
if [[ " ${saved[*]} $* " == *" --brain claude "* ]] && [ -z "$ANTHROPIC_API_KEY" ]; then
  read -rsp "Paste your Anthropic API key (input hidden) and press Enter: " ANTHROPIC_API_KEY
  echo
  export ANTHROPIC_API_KEY
fi
exec ./run.sh "${saved[@]}" "$@"
