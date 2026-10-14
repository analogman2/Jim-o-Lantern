#!/usr/bin/env bash
# Save Jim's camera address (with its password) to a private file outside the repo.
# Usage: ./set_camera.sh [camera-ip] [user] [stream-path]
# Anything left out comes from the [camera] section of jim.toml.
set -euo pipefail
cd "$(dirname "$0")"
cfg() {
  .venv/bin/python -c "
import tomllib
try: print(tomllib.load(open('jim.toml', 'rb')).get('camera', {}).get('$1', ''))
except Exception: print('')"
}
ip=${1:-$(cfg ip)}
user=${2:-$(cfg user)}
path=${3:-$(cfg path)}
path=${path:-h264Preview_01_main}   # Reolink main stream; other brands differ
if [ -z "$ip" ] || [ -z "$user" ]; then
  echo "Set ip and user under [camera] in jim.toml, or run: $0 <camera-ip> <user> [stream-path]" >&2
  exit 1
fi
read -rsp "Password for $user@$ip (typing is hidden): " pass
echo
mkdir -p ~/.config/jim
umask 077
printf 'rtsp://%s:%s@%s:554/%s\n' "$user" "$pass" "$ip" "$path" > ~/.config/jim/rtsp
chmod 600 ~/.config/jim/rtsp
echo "Saved to ~/.config/jim/rtsp (readable only by you). Jim will use it automatically."
