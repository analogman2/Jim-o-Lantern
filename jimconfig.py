"""All of Jim's settings live in two files in ~/.config/jim/ (outside the repo, so they're never committed):

  config.toml   everything except passwords: each Jim and his settings ([jims.<id>]), the saved cameras
                ([cameras.<id>]), mics ([mics.<id>]) and speakers ([speakers.<id>]), and the control panel's
                dashboard layout ([panel]). The control panel writes it; you can edit it by hand too.
  secrets.toml  only the camera passwords ([cameras.<id>] password = "..."); only you can read it.

Older setups (jim.toml in the repo, cameras.json, mics.json, speakers.json, instances.json, dashboard.json and the
rtsp file) are moved into these two files once by migrate()."""
import json, os, re, shutil, threading, time, tomllib, urllib.parse
from pathlib import Path

DIR = Path(os.environ.get("JIM_CONFIG_DIR") or Path.home() / ".config" / "jim")
CONFIG = DIR / "config.toml"
SECRETS = DIR / "secrets.toml"
HEADER = ("# Jim-o-Lantern settings (everything except passwords, which live in secrets.toml).\n"
          "# Written by the control panel; editing by hand is fine while the panel isn't saving.\n")
lock = threading.RLock()


def use(path):
    """Read and write this config file (and the secrets.toml next to it) instead of the default ones."""
    global DIR, CONFIG, SECRETS
    CONFIG = Path(path).expanduser().resolve()
    DIR, SECRETS = CONFIG.parent, CONFIG.parent / "secrets.toml"


def read_toml(path):
    try:
        return tomllib.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------- a small TOML writer (stdlib only reads TOML)
def _key(k):
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def dumps(data, header=""):
    """dicts become [tables], lists of dicts [[arrays of tables]]; None values are left out (TOML has no null)."""
    out = [header] if header else []

    def table(d, path):
        scalars = {k: v for k, v in d.items() if v is not None and not isinstance(v, dict)
                   and not (isinstance(v, list) and v and all(isinstance(x, dict) for x in v))}
        if path and (scalars or not any(isinstance(v, dict) for v in d.values())):
            out.append(f"\n[{'.'.join(_key(p) for p in path)}]")
        out.extend(f"{_key(k)} = {_value(v)}" for k, v in scalars.items())
        for k, v in d.items():
            if isinstance(v, dict):
                table(v, path + [k])
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                for item in v:
                    out.append(f"\n[[{'.'.join(_key(p) for p in path + [k])}]]")
                    out.extend(f"{_key(a)} = {_value(b)}" for a, b in item.items() if b is not None)
    table(data, [])
    return "\n".join(out).lstrip("\n") + "\n"


def _write(path, data, header="", private=False):
    DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    text = dumps(data, header)
    tomllib.loads(text)                                  # never save a file Jim couldn't read
    old = os.umask(0o077 if private else 0o022)
    try:
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(text)
        if private:
            tmp.chmod(0o600)
        tmp.replace(path)
    finally:
        os.umask(old)


# ---------------------------------------------------------------- reading and writing
def load():
    """The whole config, with empty sections filled in; the first Jim exists even in a fresh setup."""
    cfg = read_toml(CONFIG)
    for k in ("jims", "cameras", "mics", "speakers", "panel"):
        cfg.setdefault(k, {})
    if not cfg["jims"]:
        cfg["jims"]["jim"] = {"name": "Jim", "watch_port": 8780}
    return cfg


def save(cfg):
    with lock:
        _write(CONFIG, cfg, HEADER)


def load_secrets():
    s = read_toml(SECRETS)
    s.setdefault("cameras", {})
    return s


def save_secrets(s):
    with lock:
        _write(SECRETS, s, "# Camera passwords for Jim-o-Lantern. Keep this file private (mode 600).\n", private=True)


def update(fn):
    """Read, change (fn(cfg)), and write the config as one step."""
    with lock:
        cfg = load()
        result = fn(cfg)
        save(cfg)
        return result


def jim_ids(cfg=None):
    return list((cfg or load())["jims"])


def jim_settings(cfg, jim_id=None):
    """(id, settings) of this Jim (the first one if jim_id is empty or unknown)."""
    jims = cfg["jims"]
    jid = jim_id if jim_id in jims else next(iter(jims))
    return jid, dict(jims[jid])


def camera(cam_id, cfg=None):
    """A saved camera with its password (from secrets.toml), or None."""
    c = (cfg or load())["cameras"].get(cam_id)
    if not c:
        return None
    return dict(c, id=cam_id, password=load_secrets()["cameras"].get(cam_id, {}).get("password", ""))


def camera_url(c):
    """A security camera's RTSP address, password included (never print it)."""
    q = lambda v: urllib.parse.quote(v or "", safe="")
    return f"rtsp://{q(c.get('user'))}:{q(c.get('password'))}@{c['ip']}:554/{c.get('path') or 'h264Preview_01_main'}"


def device_url(cam_id, cfg=None):
    """What a Jim opens for this saved camera: the USB device, or the RTSP address. None if unknown."""
    c = camera(cam_id, cfg)
    if not c:
        return None
    return c.get("device") if c.get("kind") == "usb" else camera_url(c)


# ---------------------------------------------------------------- moving an older setup into the two files
def migrate(repo):
    """Once: gather jim.toml (and jims/*.toml), cameras/mics/speakers/instances/dashboard JSON and the rtsp file into
    config.toml + secrets.toml. Files with passwords are deleted afterwards; the rest go to ~/.config/jim/old/."""
    if CONFIG.exists():
        return False
    repo = Path(repo)
    js = lambda name: (lambda p: json.loads(p.read_text()) if p.exists() else None)(DIR / name)
    cfg = {"jims": {}, "cameras": {}, "mics": {}, "speakers": {}, "panel": {}}
    sec = {"cameras": {}}
    for c in js("cameras.json") or []:
        c = dict(c)
        cid, pw = c.pop("id"), c.pop("password", None)
        cfg["cameras"][cid] = c
        if pw:
            sec["cameras"][cid] = {"password": pw}
    rtsp = DIR / "rtsp"
    if rtsp.exists() and not any(c.get("kind") == "rtsp" for c in cfg["cameras"].values()):
        u = urllib.parse.urlsplit(rtsp.read_text().strip())
        cfg["cameras"]["security-camera"] = {"name": "Security camera", "kind": "rtsp", "ip": u.hostname,
                                             "user": urllib.parse.unquote(u.username or ""), "path": u.path.lstrip("/")}
        sec["cameras"]["security-camera"] = {"password": urllib.parse.unquote(u.password or "")}
    rtsp_cam = next((cid for cid, c in cfg["cameras"].items() if c.get("kind") == "rtsp"), "")
    for m in js("mics.json") or []:
        m = dict(m); cfg["mics"][m.pop("id")] = {k: v for k, v in m.items() if v is not None}
    for x in js("speakers.json") or []:
        x = dict(x); cfg["speakers"][x.pop("id")] = x
    layout = js("dashboard.json")
    if layout:
        cfg["panel"]["layout"] = [{k: v for k, v in p.items() if v is not None} for p in layout]
    for i in js("instances.json") or [{"id": "jim", "name": "Jim", "config": "jim.toml", "watch_port": 8780}]:
        path = Path(i["config"]) if Path(i["config"]).is_absolute() else repo / i["config"]
        old = read_toml(path)
        s = {k: v for k, v in old.items() if not isinstance(v, dict)}
        if not s.get("camera_id"):                       # the old way of choosing a camera
            src = s.get("camera_source", "auto")
            if src == "usb":
                s["camera_id"] = next((cid for cid, c in cfg["cameras"].items()
                                       if c.get("kind") == "usb" and c.get("device") == s.get("usb_camera", "/dev/video0")), "")
            elif src != "none":
                s["camera_id"] = rtsp_cam
        for k in ("camera_source", "usb_camera", "watch_port", "name"):
            s.pop(k, None)
        cfg["jims"][i["id"]] = {"name": i["name"], "watch_port": i["watch_port"], **s}
    save(cfg)
    save_secrets(sec)
    old_dir = DIR / "old"
    old_dir.mkdir(mode=0o700, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for name in ("mics.json", "speakers.json", "instances.json", "dashboard.json"):
        if (DIR / name).exists():
            shutil.move(str(DIR / name), str(old_dir / f"{name}.{stamp}"))
    old_settings = [repo / "jim.toml", *sorted((repo / "jims").glob("*.toml*"))]
    for p in old_settings:
        if p.exists():
            shutil.move(str(p), str(old_dir / f"{p.name}.{stamp}"))
    for name in ("cameras.json", "rtsp"):                # their passwords are in secrets.toml now
        if (DIR / name).exists():
            (DIR / name).unlink()
    return True

