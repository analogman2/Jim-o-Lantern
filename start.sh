#!/usr/bin/env bash
# Starts the first Jim (or the one given with --jim-id). Settings: ~/.config/jim/config.toml (see config.example.toml);
# options given here win. With brain "claude", asks for the Anthropic API key (hidden) if it isn't set.
cd "$(dirname "$0")"
uses_claude=$(.venv/bin/python -c "
import jimconfig as J
try: print(J.jim_settings(J.load())[1].get('brain', ''))
except Exception: print('')" 2>/dev/null)
if { [[ " $* " == *" --brain claude "* ]] || [ "$uses_claude" = claude ]; } && [ -z "$ANTHROPIC_API_KEY" ]; then
  read -rsp "Paste your Anthropic API key (input hidden) and press Enter: " ANTHROPIC_API_KEY
  echo
  export ANTHROPIC_API_KEY
fi
exec ./run.sh "$@"
