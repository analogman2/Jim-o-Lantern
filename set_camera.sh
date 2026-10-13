#!/usr/bin/env bash
# Save Jim's camera address (with its password) to a private file outside the repo.
# Usage: ./set_camera.sh <camera-ip> <user> [stream-path]
# The default stream path suits Reolink cameras; other brands use other paths.
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <camera-ip> <user> [stream-path]   (e.g. $0 192.168.1.20 admin)" >&2
  exit 1
fi
ip=$1
user=$2
path=${3:-h264Preview_01_main}
read -rsp "Password for $user@$ip (typing is hidden): " pass
echo
mkdir -p ~/.config/jim
umask 077
printf 'rtsp://%s:%s@%s:554/%s\n' "$user" "$pass" "$ip" "$path" > ~/.config/jim/rtsp
chmod 600 ~/.config/jim/rtsp
echo "Saved to ~/.config/jim/rtsp (readable only by you). Jim will use it automatically."
