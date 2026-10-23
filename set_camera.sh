#!/usr/bin/env bash
# Save a security camera for Jim: its address goes in ~/.config/jim/config.toml, its password in secrets.toml (only you
# can read it), and the first Jim looks through it. The control panel's Camera tab does the same.
# Usage: ./set_camera.sh <camera-ip> <user> [stream-path]      (Reolink: h264Preview_01_main; an NVR: _02_, _03_, ...)
set -euo pipefail
cd "$(dirname "$0")"
if [ $# -lt 2 ]; then
  echo "Usage: $0 <camera-ip> <user> [stream-path]" >&2
  exit 1
fi
read -rsp "Password for $2@$1 (typing is hidden): " pass
echo
JIM_CAM_PASS="$pass" .venv/bin/python - "$1" "$2" "${3:-h264Preview_01_main}" <<'PY'
import os, sys
import jimconfig as J
ip, user, path = sys.argv[1:4]
cid = "security-camera"
J.update(lambda cfg: cfg["cameras"].__setitem__(cid, {"name": "Security camera", "kind": "rtsp", "ip": ip, "user": user, "path": path}))
sec = J.load_secrets(); sec["cameras"][cid] = {"password": os.environ["JIM_CAM_PASS"]}; J.save_secrets(sec)
first = J.jim_ids()[0]
J.update(lambda cfg: cfg["jims"][first].__setitem__("camera_id", cid))
print(f"Saved. {J.CONFIG} has the camera, {J.SECRETS} its password; Jim looks through it from his next start.")
PY
