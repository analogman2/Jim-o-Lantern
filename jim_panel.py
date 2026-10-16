#!/usr/bin/env python3
"""Jim's control panel: http://127.0.0.1:8790 on this PC (only).

Start, stop and restart Jim; watch what he sees, hears and says; set his camera, mic, voice and replies (saved to
jim.toml, comments kept); preview his voice; and play back recorded visits. The forms come from Jim's own options
(pumpkin.build_parser), so a new option shows up here by adding it to a group below.

Jim only runs while you start him here (or with ./start.sh); stopping the panel stops him.
Usage: ./panel.sh [--port 8790]
"""
import argparse, base64, collections, itertools, json, os, re, shutil, signal, subprocess, sys, threading, time, tomllib, urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pumpkin as P  # noqa: E402
import jimconfig as J  # noqa: E402  (config.toml + secrets.toml in ~/.config/jim/)

_tl = threading.local()                           # the Jim the current request is about (?jim=<id>)


def instances():
    """The Jims in config.toml, in order: [{id, name, watch_port}]."""
    return [{"id": k, "name": v.get("name", k), "watch_port": v.get("watch_port", 8780)} for k, v in J.load()["jims"].items()]


def inst(i=None):
    return i or getattr(_tl, "inst", None) or instances()[0]


def watch_url(i=None):
    """Jim's own viewer (one port per Jim); the panel reads it."""
    return f"http://127.0.0.1:{inst(i)['watch_port']}"


def as_jim(i):
    """Run the following settings calls for this Jim (a context manager)."""
    class Ctx:
        def __enter__(self):
            self.old = getattr(_tl, "inst", None); _tl.inst = i
        def __exit__(self, *a):
            _tl.inst = self.old
    return Ctx()
PYTHON = HERE / ".venv" / "bin" / "python"

# tab -> [(section title, [option names])]; anything not listed isn't shown
GROUPS = {
    "camera": [("Picture sent to the AI", ["camera_crop", "camera_width"])],
    "mic": [],                                   # saved mics ([mics] in config.toml) set mic, min_rms, vad_ratio, silence, wled_gain
    "voice": [("Voice", ["voice_design", "gentle_voice_design", "pitch", "intonation", "echo", "grit", "lip_offset"]),
              ("Engine", ["tts", "speech_url"])],
    "responses": [("Personality", ["personality", "house", "temperature", "max_words"]),
                  ("Brain", ["brain", "ollama_model", "ollama_url", "keep_loaded"]),
                  ("Timing", ["reset_after", "eyes_timeout", "snooze", "start_delay"]),
                  ("Two Jims (with another Jim beside him)", ["banter", "banter_max"]),
                  ("Face (LEDs)", ["wled", "brightness", "flip_x", "flip_y"])],
    "recordings": [("Recording", ["record", "record_keep_days", "record_dir"])],
}
MANAGED = {k for groups in GROUPS.values() for _, keys in groups for k in keys}
# 1-10 sliders: real value = base + (slider - 1) * step, chosen so each default lands on a whole step;
# "auto" = the value Jim uses when the setting is left out (for the qwen voice)
SLIDERS = {"pitch": (0.7, 0.06, 1.0), "intonation": (0.85, 0.15, 1.0), "echo": (0.0, 0.05, 0.0), "grit": (0.0, 0.06, 0.0),
           "lip_offset": (0.0, 0.025, None), "temperature": (0.2, 0.15, None), "max_words": (5, 5, None),
           "reset_after": (10, 10, None), "eyes_timeout": (10, 10, None), "snooze": (5, 5, None), "start_delay": (0, 15, None),
           "brightness": (0.1, 0.1, None), "banter": (0.0, 0.1, None), "banter_max": (1, 1, None)}
VOLUME = (0.25, 0.125)                           # the volume slider: 1-10 -> 0.25 .. 1.375; 7 = 1.0 (as made)
LONG = {"voice_design", "gentle_voice_design", "house"}
PICK = {"mic": "mics", "speaker": "speakers", "usb_camera": "cameras", "ollama_model": "models"}


# ---------------------------------------------------------------- settings (jim.toml)
def options():
    """name -> {help, default, kind, choices} from Jim's own argument parser."""
    out = {}
    for a in P.build_parser()._actions:
        if not a.option_strings or a.dest == "help":
            continue
        kind = "bool" if isinstance(a, argparse._StoreTrueAction) else \
            {int: "int", float: "float"}.get(a.type, "str")
        choices = list(a.choices) if a.choices and kind == "str" else None
        out[a.dest] = {"help": a.help or "", "default": a.default, "kind": kind, "choices": choices}
    return out


def load_config(i=None):
    """The selected Jim's settings ([jims.<id>] in config.toml)."""
    return dict(J.load()["jims"].get(inst(i)["id"], {}))


def write_settings(values, section=None):
    """Set keys in the selected Jim's section of config.toml."""
    jid = inst()["id"]
    J.update(lambda cfg: cfg["jims"].setdefault(jid, {}).update({k: v for k, v in values.items()}))


def coerce(spec, v):
    if spec["kind"] == "bool":
        return bool(v)
    if spec["kind"] == "int":
        return int(v)
    if spec["kind"] == "float":
        return float(v)
    v = str(v)
    if spec["choices"] and v not in spec["choices"]:
        raise ValueError(f"must be one of {', '.join(spec['choices'])}")
    return v


# ---------------------------------------------------------------- the security camera (password file)



def grab_frame(source, usb, url=None):
    """One JPEG from the security camera or a USB webcam. Errors never include the camera address."""
    if source == "usb":
        holder = webcam_holder(usb)
        if holder:                                                # a Jim has the webcam: use his newest frame
            with urllib.request.urlopen(holder + "/live.jpg", timeout=3) as r:
                return r.read()
        src = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", "848x480", "-i", usb, "-vf", "select=gte(n\\,8),scale=768:-2"]
        secret = None
    else:
        if not url:
            raise ValueError("no security camera chosen")
        secret = url
        src = ["-rtsp_transport", "tcp", "-timeout", "8000000", "-i", secret, "-vf", "scale=768:-2"]
    try:
        r = subprocess.run(["ffmpeg", "-loglevel", "error", *src, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                           capture_output=True, timeout=25)
    except subprocess.TimeoutExpired:
        raise ValueError("the camera didn't send a picture within 25 s")
    if r.stdout[:2] == b"\xff\xd8":
        return r.stdout
    err = r.stderr.decode(errors="replace").strip().splitlines()
    msg = err[-1] if err else "no picture"
    if secret:
        msg = msg.replace(secret, "<camera>")
        pw = urllib.parse.urlsplit(secret).password
        if pw:
            msg = msg.replace(pw, "***").replace(urllib.parse.unquote(pw), "***")
    if "401" in msg:
        msg = "the camera refused the user name or password (401)"
    if "busy" in msg.lower():
        msg = "the webcam is busy (Jim is using it: see the Live tab)"
    raise ValueError(msg[-200:])


def stream_cmd(source, usb, url=None):
    """ffmpeg reading a camera continuously as JPEGs (5 a second), or None for the security camera if none is saved."""
    if source == "usb":
        src = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", "848x480", "-i", usb]
    else:
        if not url:
            return None
        src = ["-rtsp_transport", "tcp", "-timeout", "8000000", "-i", url]
    return ["ffmpeg", "-loglevel", "quiet", *src, "-an", "-vf", "fps=5,scale=768:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-"]


def jims_camera(i=None):
    """Which camera this Jim is using right now: ("usb", device) or ("rtsp", camera id); None if he isn't running or
    is blind."""
    if not cur_jim(i).running():
        return None
    c = J.load()["cameras"].get(load_config(i).get("camera_id") or "")
    if not c:
        return None
    return ("usb", c.get("device")) if c.get("kind") == "usb" else ("rtsp", load_config(i)["camera_id"])


def webcam_holder(dev):
    """The viewer of a running Jim who has this USB webcam open (only one program can), to borrow his frames."""
    for i in instances():
        if jims_camera(i) == ("usb", dev):
            return watch_url(i)
    return None


# ---------------------------------------------------------------- saved cameras and mics






def new_id(name, taken):
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "item"
    i, out = 1, base
    while out in taken:
        i += 1; out = f"{base}-{i}"
    return out


def cameras():
    """Saved cameras ([cameras] in config.toml) with their passwords (secrets.toml)."""
    cfg, sec = J.load(), J.load_secrets()
    return [dict(v, id=k, password=sec["cameras"].get(k, {}).get("password", "")) for k, v in cfg["cameras"].items()]


def save_cameras(cams):
    """Cameras go to config.toml, their passwords to secrets.toml."""
    J.update(lambda cfg: cfg.__setitem__("cameras", {c["id"]: {k: v for k, v in c.items() if k not in ("id", "password")} for c in cams}))
    sec = J.load_secrets()
    sec["cameras"] = {c["id"]: {"password": c["password"]} for c in cams if c.get("password")}
    J.save_secrets(sec)


def save_list(section, items):
    """Saved mics or speakers ([mics] / [speakers] in config.toml)."""
    J.update(lambda cfg: cfg.__setitem__(section, {x["id"]: {k: v for k, v in x.items() if k != "id" and v is not None} for x in items}))


def active_camera(cams=None):
    cid = load_config().get("camera_id") or ""
    return cid if any(c["id"] == cid for c in (cams if cams is not None else cameras())) else None


def public_camera(c):
    out = {k: v for k, v in c.items() if k != "password"}
    if c["kind"] == "rtsp":
        out["password_saved"] = bool(c.get("password"))
    return out


def camera_url(c):
    return J.camera_url(c)


def save_camera_profile(b):
    cams = cameras()
    old = next((c for c in cams if c["id"] == b.get("id")), None)
    kind = b.get("kind")
    name = str(b.get("name") or "").strip()[:40] or ("USB webcam" if kind == "usb" else "Camera")
    if kind == "usb":
        dev = str(b.get("device") or "")
        if not re.fullmatch(r"/dev/video\d+", dev):
            raise ValueError("pick a webcam device")
        c = {"name": name, "kind": "usb", "device": dev}
    elif kind == "rtsp":
        ip, user, path = (str(b.get(k) or "").strip() for k in ("ip", "user", "path"))
        path = path.lstrip("/") or "h264Preview_01_main"
        if not re.fullmatch(r"[\w.-]+", ip) or not re.fullmatch(r"[\w.@-]+", user) or not re.fullmatch(r"[\w./-]*", path):
            raise ValueError("check the IP, user and stream path")
        password = b.get("password") or (old or {}).get("password") or ""
        if not password:
            raise ValueError("enter the camera's password")
        c = {"name": name, "kind": "rtsp", "ip": ip, "user": user, "path": path, "password": password}
    else:
        raise ValueError("kind must be rtsp or usb")
    if old:
        old.clear(); old.update(c, id=b["id"]); c = old
    else:
        c["id"] = new_id(name, {x["id"] for x in cams}); cams.append(c)
    save_cameras(cams)
    if c["kind"] == "rtsp":
        threading.Thread(target=probe_camera_audio, args=([c],), daemon=True).start()
    if active_camera(cams) == c["id"] or b.get("use"):
        use_camera(c["id"])
    return c


def nvr_channels(c):
    """The cameras behind a security camera's address when it's an NVR: [{channel, path, name, online, saved}].
    Names come from the recorder's own API (Reolink); without it, each channel is tried over RTSP."""
    import ssl, concurrent.futures
    names = {}
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE   # the NVR's own certificate

    def call(cmd, body, token=None):
        r = urllib.request.Request(f"https://{c['ip']}/api.cgi?cmd={cmd}" + (f"&token={token}" if token else ""),
                                   json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=8, context=ctx) as resp:
            return json.loads(resp.read())
    try:
        d = call("Login", [{"cmd": "Login", "param": {"User": {"Version": "0", "userName": c["user"], "password": c["password"]}}}])
        token = d[0]["value"]["Token"]["name"]
        try:
            for ch in call("GetChannelstatus", [{"cmd": "GetChannelstatus"}], token)[0]["value"]["status"]:
                names[int(ch["channel"]) + 1] = (ch.get("name") or "", bool(ch.get("online")))
        finally:
            call("Logout", [{"cmd": "Logout"}], token)
    except Exception:
        def probe(n):
            u = dict(c, path=f"h264Preview_{n:02d}_main")
            try:
                r = subprocess.run(["ffprobe", "-v", "quiet", "-rtsp_transport", "tcp", "-timeout", "6000000", "-select_streams", "v",
                                    "-show_entries", "stream=codec_type", "-of", "csv=p=0", camera_url(u)], capture_output=True, text=True, timeout=15)
                return n, "video" in r.stdout
            except subprocess.TimeoutExpired:
                return n, False
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            names = {n: ("", ok) for n, ok in ex.map(probe, range(1, 17))}
    saved = {x["path"]: x["id"] for x in cameras() if x["kind"] == "rtsp" and x["ip"] == c["ip"]}
    out = []
    for n, (name, online) in sorted(names.items()):
        if online:
            path = f"h264Preview_{n:02d}_main"
            out.append({"channel": n, "path": path, "name": name or f"Channel {n}", "saved": saved.get(path)})
    return out


def add_nvr_channels(c, chosen):
    """Save the chosen NVR channels as cameras (same address and login, each its own stream path)."""
    cams, added = cameras(), []
    for ch in chosen:
        n = int(ch["channel"])
        if not 1 <= n <= 64:
            continue
        path = f"h264Preview_{n:02d}_main"
        if any(x["kind"] == "rtsp" and x["ip"] == c["ip"] and x["path"] == path for x in cams):
            continue
        name = str(ch.get("name") or f"Channel {n}").strip()[:40]
        new = {"id": new_id(name, {x["id"] for x in cams}), "name": name, "kind": "rtsp", "ip": c["ip"], "user": c["user"],
               "password": c["password"], "path": path}
        cams.append(new); added.append(new["id"])
    save_cameras(cams)
    threading.Thread(target=probe_camera_audio, args=([x for x in cams if x["id"] in added],), daemon=True).start()
    return added


def use_camera(cid):
    """Make this the camera the selected Jim looks through (he switches at once if he's running)."""
    if not any(c["id"] == cid for c in cameras()):
        raise ValueError("no such camera")
    write_settings({"camera_id": cid})
    return reload_jim()


def reload_jim():
    """Jim switches to the camera, mic and speaker now in jim.toml (no restart). False if he isn't running."""
    return jim_post("/reload", {})


MIC_KEYS = ("mic", "min_rms", "level_mode", "vad_ratio", "silence", "wled_gain")


def mics():
    """Saved microphones (a device plus how sensitive Jim is with it), [mics] in config.toml."""
    return [dict(v, id=k) for k, v in J.load()["mics"].items()]


def active_mic(ms=None):
    cfg, opts = load_config(), options()
    cur = {k: cfg.get(k, opts[k]["default"]) for k in MIC_KEYS}
    return next((m["id"] for m in (ms if ms is not None else mics()) if all(m.get(k, opts[k]["default"]) == cur[k] for k in MIC_KEYS)), None)


def save_mic_profile(b):
    ms, opts = mics(), options()
    old = next((m for m in ms if m["id"] == b.get("id")), None)
    m = {"name": str(b.get("name") or "").strip()[:40] or "Mic"}
    for k in MIC_KEYS:
        v = b.get(k, opts[k]["default"])
        m[k] = None if v in (None, "") and k == "min_rms" else coerce(opts[k], v)
    if old:
        was_active = active_mic(ms) == old["id"]
        old.clear(); old.update(m, id=b["id"]); m = old
    else:
        was_active = False
        m["id"] = new_id(m["name"], {x["id"] for x in ms}); ms.append(m)
    save_list("mics", ms)
    if was_active or b.get("use"):
        use_mic(m["id"])
    return m


def use_mic(mid):
    m = next((m for m in mics() if m["id"] == mid), None)
    if not m:
        raise ValueError("no such mic")
    opts = options()
    write_settings({k: m.get(k, opts[k]["default"]) for k in MIC_KEYS if m.get(k, opts[k]["default"]) is not None})
    return reload_jim()


def speakers():
    """Saved speakers (a playback device plus Jim's volume on it), [speakers] in config.toml."""
    return [dict(v, id=k) for k, v in J.load()["speakers"].items()]


def active_speaker(sp=None):
    dev = load_config().get("speaker", "default")
    return next((x["id"] for x in (sp if sp is not None else speakers()) if x["speaker"] == dev), None)


def save_speaker_profile(b):
    sp = speakers()
    old = next((x for x in sp if x["id"] == b.get("id")), None)
    dev = str(b.get("speaker") or "")
    if dev != "default" and not re.fullmatch(r"[\w:=,.-]+", dev):
        raise ValueError("pick a speaker")
    x = {"name": str(b.get("name") or "").strip()[:40] or "Speaker", "speaker": dev, "volume": max(0.05, min(1.5, float(b.get("volume") or 1.0)))}
    was_active = bool(old) and active_speaker(sp) == old["id"]
    if old:
        old.clear(); old.update(x, id=b["id"]); x = old
    else:
        x["id"] = new_id(x["name"], {y["id"] for y in sp}); sp.append(x)
    save_list("speakers", sp)
    if was_active or b.get("use"):
        use_speaker(x["id"])
    return x


def use_speaker(sid):
    x = next((x for x in speakers() if x["id"] == sid), None)
    if not x:
        raise ValueError("no such speaker")
    write_settings({"speaker": x["speaker"], "volume": x["volume"]})
    return reload_jim()


def jim_post(path, body):
    """Tell a running Jim something through his viewer; False if he isn't running."""
    if not cur_jim().running():
        return False
    try:
        urllib.request.urlopen(urllib.request.Request(watch_url() + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json", "X-Jim-Panel": "1"}), timeout=3).read()
        return True
    except Exception:
        return False


def speaker_test(x):
    """Jim says a test line on this speaker, at its volume (it really plays in the house)."""
    voice = load_config().get("voice_design") or P.JACK_VOICE
    wav = preview_voice({"text": "Testing, testing. Can you hear me out there?", "voice_design": voice,
                         "pitch": load_config().get("pitch") or 1.0})
    import io, wave
    with wave.open(io.BytesIO(wav)) as w:
        rate, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    pcm = (np.clip(pcm * x["volume"], -1, 1) * 32767).astype("<i2").tobytes()
    r = subprocess.run(["aplay", "-q", "-D", x["speaker"], "-f", "S16_LE", "-c", "1", "-r", str(rate), "-t", "raw"],
                       input=pcm, capture_output=True, timeout=30)
    if r.returncode:
        msg = r.stderr.decode(errors="replace").strip()
        raise ValueError("the speaker is busy (Jim is talking?)" if "busy" in msg.lower() else msg[-200:] or "couldn't play")


# ---------------------------------------------------------------- more than one Jim
def jims_summary():
    out = []
    for i in instances():
        j, state = cur_jim(i), None
        if j.running():
            try:
                with urllib.request.urlopen(watch_url(i) + "/events.json", timeout=2) as r:
                    state = json.loads(r.read()).get("state")
            except Exception:
                state = "starting"
        cfg = load_config(i)
        out.append({"id": i["id"], "name": i["name"], "running": j.running(), "state": state, "wled": cfg.get("wled"),
                    "watch_port": i["watch_port"]})
    return out


DEVICE_KEYS = ("wled", "mic", "speaker", "camera_id", "camera_source", "usb_camera")


def add_jim(name, wled, copy_from=None, personality="jim"):
    """A new Jim: his own section of config.toml (another Jim's reply and timing settings, none of that Jim's devices),
    recordings folder and viewer port. He starts blind, with the LED controller's mic and the default speaker."""
    name = str(name or "").strip()[:30] or "Jim 2"
    if not re.fullmatch(r"[\w.-]+", wled or ""):
        raise ValueError("enter his LED controller's IP address")
    ins = instances()
    if any(load_config(i).get("wled") == wled for i in ins):
        raise ValueError("another Jim already drives that LED controller")
    new = {"id": new_id(name, {i["id"] for i in ins}), "name": name, "watch_port": max(i["watch_port"] for i in ins) + 1}
    persona = P.PERSONALITIES.get(personality) or P.PERSONALITIES["jim"]
    src = load_config(next((i for i in ins if i["id"] == copy_from), None)) if copy_from else {}
    settings = {k: v for k, v in src.items() if k not in ("wled", "mic", "speaker", "camera_id", "camera_source", "usb_camera",
                                                           "name", "watch_port", "record_dir", "min_rms", "level_mode")}
    settings.update(name=name, watch_port=new["watch_port"], wled=wled, mic="wled", speaker="default", camera_id="",
                    record_dir=f"recordings-{new['id']}", personality=personality if personality in P.PERSONALITIES else "jim",
                    voice_design=persona["voice"], gentle_voice_design=persona["gentle_voice"])
    J.update(lambda cfg: cfg["jims"].__setitem__(new["id"], settings))
    return new


def remove_jim(jid):
    ins = instances()
    i = next((i for i in ins if i["id"] == jid), None)
    if not i:
        raise ValueError("no such Jim")
    if len(ins) == 1:
        raise ValueError("the last Jim can't be removed")
    cur_jim(i).stop()
    JIMS.pop(jid, None)
    J.update(lambda cfg: cfg["jims"].pop(jid, None))


def shared_with(key, value):
    """Other Jims set to use the same device (a USB mic, webcam, speaker or LED controller can only serve one)."""
    if not value or value in ("default", "none") or (key == "mic" and (value == "wled" or value.startswith("camera:"))):
        return []
    me = inst()["id"]
    return [i["name"] for i in instances() if i["id"] != me and load_config(i).get(key) == value]


# ---------------------------------------------------------------- devices
def alsa_devices(cmd):
    """[(device, description)] from arecord -L / aplay -L, the plughw: entries (they convert rates for us)."""
    try:
        out = subprocess.run([cmd, "-L"], capture_output=True, text=True, timeout=5).stdout.splitlines()
    except Exception:
        return []
    devs = []
    for i, line in enumerate(out):
        if line.startswith("plughw:"):
            desc = out[i + 1].strip() if i + 1 < len(out) and out[i + 1].startswith(" ") else ""
            devs.append((line.strip(), desc))
    return devs


def devices():
    cams = []
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            if (d / "index").read_text().strip() == "0":          # the other nodes of a webcam are metadata
                cams.append((f"/dev/{d.name}", (d / "name").read_text().strip()))
        except OSError:
            pass
    models = []
    try:
        url = load_config().get("ollama_url", "http://localhost:11434")
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3) as r:
            models = sorted((m["name"].removesuffix(":latest"), f"{m['size'] / 2**30:.1f} GB") for m in json.loads(r.read())["models"])
    except Exception:
        pass
    cam_mics = [(f"camera:{c['id']}", f"{c['name']}'s microphone (over the network)") for c in cameras()
                if c.get("kind") == "rtsp" and CAM_AUDIO.get(c["id"])]
    return {"mics": [("wled", "the LED controller's mic (patched firmware)")] + alsa_devices("arecord") + cam_mics,
            "speakers": [("default", "system default")] + alsa_devices("aplay"), "cameras": cams, "models": models}


CAM_AUDIO = {}                                   # saved camera id -> does its stream carry sound (probed in the background)


def probe_camera_audio(cams=None):
    """Which saved security cameras have a microphone (an audio track in their stream)."""
    for c in cams if cams is not None else cameras():
        if c.get("kind") != "rtsp":
            continue
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "a", "-show_entries", "stream=codec_type",
                                "-of", "csv=p=0", camera_url(c)], capture_output=True, text=True, timeout=20)
            CAM_AUDIO[c["id"]] = "audio" in r.stdout
        except Exception:
            CAM_AUDIO.setdefault(c["id"], False)


def mic_test(device, seconds=3.0):
    """Record a few seconds and report the loudness Jim's ears would measure (30 ms frames, like pumpkin.Mic)."""
    if device.startswith("camera:"):
        c = next((c for c in cameras() if c["id"] == device[7:] and c["kind"] == "rtsp"), None)
        if not c:
            raise ValueError("no such camera")
        r = subprocess.run(["ffmpeg", "-loglevel", "quiet", "-rtsp_transport", "tcp", "-allowed_media_types", "audio", "-i", camera_url(c),
                            "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, timeout=seconds + 20)
        if not r.stdout:
            raise ValueError("no sound from the camera (no microphone, or it didn't answer)")
        rate, audio = 16000, np.frombuffer(r.stdout, "<i2").astype(np.float32)
    elif device == "wled":
        host = load_config().get("wled")
        if not host:
            raise ValueError("set the LED controller's address first")
        rate, blocks = P.wled_source(host, int(load_config().get("wled_gain", 16)))
        buf, t0 = [], time.time()
        for b in blocks:
            buf.append(b.astype(np.float32))
            if time.time() - t0 > seconds:
                break
        audio = np.concatenate(buf) if buf else np.zeros(0, np.float32)
    else:
        r = subprocess.run(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(int(seconds)), "-t", "raw"],
                           capture_output=True, timeout=seconds + 10)
        if r.returncode or not r.stdout:
            msg = r.stderr.decode(errors="replace").strip()
            raise ValueError("the mic is busy (Jim is using it: see the live meter)" if "busy" in msg.lower() else msg[-200:] or "no sound")
        rate, audio = 16000, np.frombuffer(r.stdout, "<i2").astype(np.float32)
    n = int(rate * 0.03)
    frames = [audio[i:i + n] - audio[i:i + n].mean() for i in range(0, len(audio) - n, n)]
    if not frames:
        raise ValueError("no sound came in")
    rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])
    if device.startswith("camera:") and rms.max() == 0:
        raise ValueError("the camera sends a silent sound track: turn its microphone on in the camera's app (Reolink: Record Audio)")
    noise, loud = float(np.percentile(rms, 20)), float(rms.max())
    return {"noise": round(noise), "loudest": round(loud), "suggested_min_rms": int(max(150, noise * 3)),
            "levels": [round(x) for x in rms[:: max(1, len(rms) // 100)]]}


def preview_voice(b):
    """Jim's voice through the speech service, with the effects as set: a WAV for the browser to play."""
    import io, wave
    pitch = float(b.get("pitch") or 1.0)
    v = P.SpeechServiceVoice.__new__(P.SpeechServiceVoice)       # no warm-up or local fallback: just this clip
    v.url, v.instruct, v.gentle = load_config().get("speech_url", "http://127.0.0.1:7895").rstrip("/"), b["voice_design"], bool(b.get("gentle"))
    v.gentle_instruct, v.local, v.rate = b.get("gentle_voice_design") or P.JACK_GENTLE_VOICE, None, 24000
    v.intonation, v.echo, v.grit = float(b.get("intonation") or 1.0), float(b.get("echo") or 0), float(b.get("grit") or 0)
    v.play_rate = int(v.rate * pitch)
    audio = v.synth(b.get("text") or "Who dares wake me? Nice costume. Did your mom make it, or did she lose a bet?")
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(v.play_rate); w.writeframes(pcm)
    return out.getvalue()


# ---------------------------------------------------------------- Jim himself
class Jim:
    """Runs one Jim (pumpkin.py with his own config file and viewer port) as a child process and keeps his console."""

    def __init__(self, i):
        self.inst, self.proc, self.started = i, None, 0.0
        self.log = collections.deque(maxlen=600)
        self.lock = threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def others(self):
        """Copies of this Jim (same config file) not started by this panel, e.g. from ./start.sh: only one can run."""
        mine = {j.proc.pid for j in JIMS.values() if j.running()}
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
        found = []
        for l in out.splitlines():
            if not re.search(r"python\S* (-u )?pumpkin\.py", l) or int(l.split()[0]) in mine:
                continue
            m = re.search(r"--jim-id\s+(\S+)", l)
            if (m.group(1) if m else instances()[0]["id"]) == self.inst["id"]:
                found.append(int(l.split()[0]))
        return found

    def start(self):
        with self.lock:
            if self.running():
                raise ValueError(f"{self.inst['name']} is already running")
            if self.others():
                raise ValueError(f"another copy of Jim is running (pid {', '.join(map(str, self.others()))}): stop it first")
            self.log.clear()
            self.log.append(f"--- starting {self.inst['name']} at {time.strftime('%H:%M:%S')}")
            self.proc = subprocess.Popen([str(PYTHON), "-u", "pumpkin.py", "--config", str(J.CONFIG), "--name", self.inst["name"],
                                          "--jim-id", self.inst["id"],
                                          "--watch-port", str(self.inst["watch_port"])], cwd=HERE,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED="1"), text=True, bufsize=1,
                                         start_new_session=True)   # terminal signals go to the panel; it stops Jim
            self.started = time.time()
            threading.Thread(target=self.read, args=(self.proc,), daemon=True).start()

    def read(self, proc):
        for line in proc.stdout:
            self.log.append(line.rstrip("\n"))
        code = proc.wait()
        why = f"signal {signal.Signals(-code).name}" if code < 0 else f"exit {code}"
        self.log.append(f"--- {self.inst['name']} stopped ({why}) at {time.strftime('%H:%M:%S')}")

    def stop(self, pid=None):
        """Stop Jim (he unloads his chat model on the way out); pid = a copy started elsewhere."""
        if pid:
            if pid not in self.others():
                raise ValueError("that isn't a running copy of Jim")
            os.kill(pid, signal.SIGTERM)
            return
        with self.lock:
            if not self.running():
                return
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def status(self):
        return {"id": self.inst["id"], "name": self.inst["name"], "running": self.running(), "pid": self.proc.pid if self.running() else None,
                "uptime_s": round(time.time() - self.started) if self.running() else None,
                "others": self.others(), "log": list(self.log)[-250:]}


JIMS = {}                                         # id -> Jim (the processes this panel runs)


def cur_jim(i=None):
    i = inst(i)
    if i["id"] not in JIMS:
        JIMS[i["id"]] = Jim(i)
    JIMS[i["id"]].inst = i
    return JIMS[i["id"]]


# ---------------------------------------------------------------- network health (every 5 s)
NET = {}                                          # name -> {ok, ms, detail, history: [ms or None, ...]}
NET_EVERY_S = 5


def timed(fn):
    t = time.time()
    try:
        detail = fn()
        return True, round((time.time() - t) * 1000), detail
    except Exception as e:
        return False, None, type(e).__name__ if not isinstance(e, ValueError) else str(e)


def fail(msg):
    raise ValueError(msg)


def get_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def check_network():
    checks, many = {}, len(instances()) > 1
    for i in instances():
        with as_jim(i):
            for name, fn in jim_checks().items():
                checks[f"{name} · {i['name']}" if many else name] = fn
    cfg = load_config(instances()[0])
    speech = cfg.get("speech_url", "http://127.0.0.1:7895").rstrip("/")
    checks["Speech service"] = lambda: "voice + hearing" if get_json(speech + "/health").get("ok") else fail("not ok")
    ollama = cfg.get("ollama_url", "http://localhost:11434").rstrip("/")
    checks["Ollama (brain)"] = lambda: "version " + get_json(ollama + "/api/version").get("version", "?")
    record_checks(checks)


def jim_checks():
    """This Jim's own network devices: his camera and his LED controller."""
    import socket
    cfg = load_config()
    checks = {}
    source = cfg.get("camera_source", "auto")
    cam = next((c for c in cameras() if c["id"] == cfg.get("camera_id")), None)
    source = cam["kind"] if cam else "none"
    if cam:
        source = cam["kind"]
    if source == "usb":
        dev = cam["device"] if cam else cfg.get("usb_camera", "/dev/video0")
        checks["Camera"] = lambda: f"USB webcam {dev}" if os.path.exists(dev) else fail(f"{dev} missing")
    elif source != "none":
        ip = cam["ip"] if cam else ""

        def rtsp_check():
            if not ip:
                raise ValueError("not set up")
            socket.create_connection((ip, 554), timeout=3).close()
            return f"security camera {ip} (RTSP port)"
        checks["Camera"] = rtsp_check
    if cfg.get("wled"):
        def wled():
            d = get_json(f"http://{cfg['wled']}/json/info")
            w, l = d.get("wifi", {}), d.get("leds", {})
            return (f"Wi-Fi {w.get('signal', '?')}% ({w.get('rssi', '?')} dBm) · LEDs {l.get('fps', '?')} fps",
                    {"wifi": w.get("signal"), "rssi": w.get("rssi"), "led_fps": l.get("fps")})
        checks["LED controller"] = wled
    return checks


def record_checks(checks):
    for name, fn in checks.items():
        ok, ms, detail = timed(fn)
        detail, extra = detail if isinstance(detail, tuple) else (detail, {})
        old = NET.get(name, {})
        NET[name] = {"ok": ok, "ms": ms, "detail": detail, "extra": extra, "t": round(time.time()),
                     "history": (old.get("history") or [])[-71:] + [ms],                 # 6 minutes at 5 s
                     "extra_history": {k: (old.get("extra_history", {}).get(k) or [])[-71:] + [v] for k, v in extra.items()}}
    for gone in set(NET) - set(checks):
        NET.pop(gone, None)


def network_loop():
    while True:
        try:
            check_network()
        except Exception:
            pass
        time.sleep(NET_EVERY_S)


# ---------------------------------------------------------------- recordings
def record_dir(i=None):
    i = inst(i)
    default = "recordings" if i["id"] == instances()[0]["id"] else f"recordings-{i['id']}"
    d = Path(load_config(i).get("record_dir", default))
    return d if d.is_absolute() else HERE / d


def visits():
    out = []
    root = record_dir()
    for d in sorted(root.glob("*/visit.json"), reverse=True):
        try:
            v = json.loads(d.read_text())
        except (OSError, ValueError):
            continue
        es = v.get("entries", [])
        greet = next((e for e in es if e["kind"] == "greeting"), {})
        v = greet.get("verdict") or {}
        seen = v.get("seen") or (v.get("error") and "couldn't look: " + v["error"]) or \
            ("nobody clearly in view" if v.get("visible") is False else "no camera picture" if not greet.get("frame") else "")
        out.append({"id": d.parent.name, "turns": sum(e["kind"] == "turn" for e in es), "entries": len(es), "seen": seen,
                    "frame": next((e["frame"] for e in es if e.get("frame")), None),
                    "first": es[0]["text"] if es else "", "roasted": any(e.get("roasted") for e in es)})
    return out


SAFE = re.compile(r"^\d{8}-\d{6}$")
SAFE_FILE = re.compile(r"^\d{2}-(frame\.jpg|kid\.wav|jim\.wav)$")


# ---------------------------------------------------------------- HTTP
class Panel(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def local(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{A.port}", f"localhost:{A.port}")

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):   # the page went away mid-reply
            pass

    def stream(self, source, usb, url=None, limit_s=600):
        """A live preview as multipart JPEG (an <img> shows it as video) for up to limit_s. The USB webcam can only be
        opened by one program: while Jim is using it, this shows his frames instead."""
        borrow = webcam_holder(usb) if source == "usb" else None

        def frames():
            if borrow:
                while True:
                    try:
                        with urllib.request.urlopen(borrow + "/live.jpg", timeout=3) as r:
                            yield r.read()
                    except Exception:
                        return
                    time.sleep(0.2)
            cmd = stream_cmd(source, usb, url)
            if not cmd:
                return
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                buf = b""
                while chunk := proc.stdout.read(65536):
                    buf += chunk
                    while (start := buf.find(b"\xff\xd8")) >= 0 and (end := buf.find(b"\xff\xd9", start + 2)) >= 0:
                        yield buf[start:end + 2]
                        buf = buf[end + 2:]
            finally:
                proc.kill(); proc.wait()

        gen, t0 = frames(), time.time()
        first = next(gen, None)
        if first is None:
            return self.send(502, {"error": "the webcam is busy or unplugged" if source == "usb" else "the camera didn't answer"})
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for jpg in itertools.chain([first], gen):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
                if time.time() - t0 > limit_s:
                    break
        except (BrokenPipeError, ConnectionResetError):          # the page stopped watching
            pass
        finally:
            gen.close()

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def proxy(self, path, method="GET", body=None):
        r = urllib.request.Request(watch_url() + path, body, {"Content-Type": "application/json", "X-Jim-Panel": "1"}, method=method)
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return self.send(200, resp.read(), resp.headers.get("Content-Type", "application/octet-stream"))
        except Exception:
            return self.send(503, {"error": "Jim isn't running"})

    def pick_jim(self):
        jid = (urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("jim") or [""])[0]
        _tl.inst = next((i for i in instances() if i["id"] == jid), None)

    def do_GET(self):
        if not self.local():
            return self.send(403, {"error": "this panel only answers on this PC"})
        self.pick_jim()
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/jims":
            return self.send(200, {"jims": jims_summary()})
        if path == "/":
            return self.send(200, (HERE / "panel.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/jim/events.json", "/jim/mic.json", "/jim/live.jpg", "/jim/judged.jpg"):
            return self.proxy(path[4:])
        if path == "/api/status":
            return self.send(200, cur_jim().status())
        if path == "/api/layout":
            return self.send(200, {"layout": J.load()["panel"].get("layout")})
        if path == "/api/network":
            return self.send(200, {"every_s": NET_EVERY_S, "checks": NET})
        if path == "/api/settings":
            opts, cfg = options(), load_config()
            tabs = {tab: [{"title": t, "fields": [dict(opts[k], name=k, value=cfg.get(k, opts[k]["default"]), long=k in LONG, pick=PICK.get(k),
                                                       slider=dict(zip(("base", "step", "auto"), SLIDERS[k])) if k in SLIDERS else None)
                                                  for k in keys if k in opts]} for t, keys in groups] for tab, groups in GROUPS.items()}
            return self.send(200, {"tabs": tabs})
        if path == "/api/camera/stream":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            c = next((c for c in cameras() if c["id"] == q.get("cam", [""])[0]), None)
            if not c:
                return self.send(404, {"error": "no such camera"})
            return self.stream(c["kind"], c.get("device"), camera_url(c) if c["kind"] == "rtsp" else None)
        m = re.fullmatch(r"/api/cameras/([\w-]+)/channels", path)
        if m:
            c = next((c for c in cameras() if c["id"] == m.group(1) and c["kind"] == "rtsp"), None)
            if not c:
                return self.send(404, {"error": "no such security camera"})
            return self.send(200, {"channels": nvr_channels(c)})
        m = re.fullmatch(r"/api/cameras/([\w-]+)/channels/(\d+)/snapshot", path)
        if m:
            c = next((c for c in cameras() if c["id"] == m.group(1) and c["kind"] == "rtsp"), None)
            if not c:
                return self.send(404, {"error": "no such security camera"})
            try:
                return self.send(200, grab_frame("rtsp", None, camera_url(dict(c, path=f"h264Preview_{int(m.group(2)):02d}_main"))), "image/jpeg")
            except (ValueError, OSError) as e:
                return self.send(502, {"error": str(e)})
        m = re.fullmatch(r"/api/cameras/([\w-]+)/snapshot", path)
        if m:
            c = next((c for c in cameras() if c["id"] == m.group(1)), None)
            if not c:
                return self.send(404, {"error": "no such camera"})
            try:
                return self.send(200, grab_frame(c["kind"], c.get("device"), camera_url(c) if c["kind"] == "rtsp" else None), "image/jpeg")
            except (ValueError, OSError) as e:
                return self.send(502, {"error": str(e)})
        if path == "/api/cameras":
            cams = cameras()
            return self.send(200, {"cameras": [public_camera(c) for c in cams], "active": active_camera(cams), "devices": devices()["cameras"]})
        if path == "/api/speakers":
            sp = speakers()
            return self.send(200, {"speakers": sp, "active": active_speaker(sp), "devices": devices()["speakers"], "volume_slider": VOLUME})
        if path == "/api/mics":
            ms = mics()
            return self.send(200, {"mics": ms, "active": active_mic(ms), "devices": devices()["mics"]})
        if path == "/api/devices":
            return self.send(200, devices())
        if path == "/api/recordings":
            return self.send(200, {"visits": visits(), "dir": str(record_dir())})
        m = re.fullmatch(r"/api/recordings/([^/]+)", path)
        if m and SAFE.match(m.group(1)):
            f = record_dir() / m.group(1) / "visit.json"
            return self.send(200, f.read_bytes(), "application/json") if f.exists() else self.send(404, {"error": "no such visit"})
        m = re.fullmatch(r"/rec/([^/]+)/([^/]+)", path)
        if m and SAFE.match(m.group(1)) and SAFE_FILE.match(m.group(2)):
            f = record_dir() / m.group(1) / m.group(2)
            if f.exists():
                return self.send(200, f.read_bytes(), "image/jpeg" if f.suffix == ".jpg" else "audio/wav")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        # same-PC pages only, and the custom header keeps other websites' forms out
        if not self.local() or self.headers.get("X-Jim-Panel") != "1":
            return self.send(403, {"error": "forbidden"})
        self.pick_jim()
        path = urllib.parse.urlsplit(self.path).path
        try:
            b = self.body()
            if path == "/api/jims":
                return self.send(200, {"jim": add_jim(b.get("name"), str(b.get("wled") or "").strip(), b.get("copy_from"), b.get("personality") or "jim")})
            if path in ("/api/jim/start", "/api/jim/stop", "/api/jim/restart"):
                if path.endswith(("stop", "restart")):
                    cur_jim().stop(b.get("pid"))
                if path.endswith(("start", "restart")):
                    cur_jim().start()
                return self.send(200, cur_jim().status())
            if path == "/api/layout":
                if not isinstance(b, list) or not all(isinstance(x, dict) and re.fullmatch(r"\w+", str(x.get("id", ""))) for x in b):
                    raise ValueError("not a layout")
                keep = ("id", "x", "y", "w", "h", "hidden")
                layout = [{k: x[k] for k in keep if k in x and x[k] is not None} for x in b]
                J.update(lambda cfg: cfg["panel"].__setitem__("layout", layout))
                return self.send(200, {"ok": True})
            if path == "/api/mic/level":
                v = int(max(20, min(30000, float(b["min_rms"]))))
                ms, mid = mics(), active_mic()
                for m in ms:
                    if m["id"] == mid:
                        m["min_rms"], m["level_mode"] = v, "fixed"
                save_list("mics", ms)
                write_settings({"min_rms": v, "level_mode": "fixed"})
                live = jim_post("/min_rms", {"min_rms": v, "fixed": True})
                return self.send(200, {"ok": True, "min_rms": v, "live": live})
            if path in ("/api/hear", "/api/wake"):
                text = str(b.get("text") or "").strip()[:300]
                if path == "/api/hear" and not text:
                    raise ValueError("type something for Jim to hear")
                if not jim_post(path[4:], {"text": text}):
                    raise ValueError(f"{inst()['name']} isn't running")
                return self.send(200, {"ok": True})
            if path == "/api/snooze":
                return self.proxy("/snooze", "POST", json.dumps({"seconds": b.get("seconds", 10)}).encode())
            if path == "/api/settings":
                opts = options()
                vals = {}
                for k, v in b.items():
                    if k not in MANAGED or k not in opts:
                        raise ValueError(f"{k}: not a setting the panel manages")
                    try:
                        vals[k] = coerce(opts[k], v)
                    except (TypeError, ValueError) as e:
                        raise ValueError(f"{k}: {e}")
                if "personality" in vals:            # a new personality comes with its own voice
                    persona = P.PERSONALITIES[vals["personality"]]
                    vals.setdefault("voice_design", persona["voice"])
                    vals.setdefault("gentle_voice_design", persona["gentle_voice"])
                write_settings(vals)
                live_keys = {"camera_crop", "camera_source", "usb_camera"}
                if set(vals) <= live_keys:
                    return self.send(200, {"ok": True, "restart": False, "live": reload_jim()})
                return self.send(200, {"ok": True, "restart": cur_jim().running()})
            if path == "/api/cameras":
                return self.send(200, {"camera": public_camera(save_camera_profile(b)), "restart": False})
            m = re.fullmatch(r"/api/cameras/([\w-]+)/channels", path)
            if m:
                c = next((c for c in cameras() if c["id"] == m.group(1) and c["kind"] == "rtsp"), None)
                if not c:
                    raise ValueError("no such security camera")
                return self.send(200, {"added": add_nvr_channels(c, b.get("channels") or [])})
            m = re.fullmatch(r"/api/cameras/([\w-]+)/use", path)
            if m:
                c = next((c for c in cameras() if c["id"] == m.group(1)), {})
                warn = shared_with("camera_id", m.group(1)) if c.get("kind") == "usb" else []
                live = use_camera(m.group(1)); return self.send(200, {"ok": True, "live": live, "restart": False, "shared": warn})
            if path == "/api/mics":
                return self.send(200, {"mic": save_mic_profile(b), "restart": False})
            if path == "/api/speakers":
                x = save_speaker_profile(b)
                return self.send(200, {"speaker": x, "restart": False})
            m = re.fullmatch(r"/api/speakers/([\w-]+)/(use|test)", path)
            if m:
                if m.group(2) == "use":
                    x = next((x for x in speakers() if x["id"] == m.group(1)), {})
                    warn = shared_with("speaker", x.get("speaker"))
                    live = use_speaker(m.group(1)); return self.send(200, {"ok": True, "live": live, "restart": False, "shared": warn})
                x = next((x for x in speakers() if x["id"] == m.group(1)), None)
                if not x:
                    raise ValueError("no such speaker")
                speaker_test(x); return self.send(200, {"ok": True})
            if path == "/api/volume":
                v = max(0.05, min(1.5, float(b["volume"])))
                sp, sid = speakers(), active_speaker()
                for x in sp:
                    if x["id"] == sid:
                        x["volume"] = v
                save_list("speakers", sp)
                write_settings({"volume": v})
                return self.send(200, {"ok": True, "volume": v, "live": jim_post("/volume", {"volume": v})})
            m = re.fullmatch(r"/api/mics/([\w-]+)/use", path)
            if m:
                mm = next((x for x in mics() if x["id"] == m.group(1)), {})
                warn = shared_with("mic", mm.get("mic"))
                live = use_mic(m.group(1)); return self.send(200, {"ok": True, "live": live, "restart": False, "shared": warn})
            if path == "/api/mic/test":
                return self.send(200, mic_test(b.get("device") or load_config().get("mic", "wled")))
            if path == "/api/voice/preview":
                return self.send(200, preview_voice(b), "audio/wav")
        except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(500, {"error": f"{type(e).__name__}: {e}"})
        self.send(404, {"error": "not found"})

    def do_DELETE(self):
        if not self.local() or self.headers.get("X-Jim-Panel") != "1":
            return self.send(403, {"error": "forbidden"})
        self.pick_jim()
        path = urllib.parse.urlsplit(self.path).path
        m = re.fullmatch(r"/api/jims/([\w.-]+)", path)
        if m:
            try:
                remove_jim(m.group(1)); return self.send(200, {"ok": True})
            except ValueError as e:
                return self.send(400, {"error": str(e)})
        m = re.fullmatch(r"/api/(cameras|mics|speakers)/([\w-]+)", path)
        if m:
            kind = m.group(1)
            items = [x for x in {"cameras": cameras, "mics": mics, "speakers": speakers}[kind]() if x["id"] != m.group(2)]
            save_cameras(items) if kind == "cameras" else save_list(kind, items)
            return self.send(200, {"ok": True})
        if path == "/api/recordings":
            for d in record_dir().glob("*/visit.json"):
                shutil.rmtree(d.parent, ignore_errors=True)
            return self.send(200, {"ok": True})
        m = re.fullmatch(r"/api/recordings/([^/]+)", path)
        if m and SAFE.match(m.group(1)):
            shutil.rmtree(record_dir() / m.group(1), ignore_errors=True)
            return self.send(200, {"ok": True})
        self.send(404, {"error": "not found"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jim's control panel")
    ap.add_argument("--port", type=int, default=8790)
    A = ap.parse_args()

    def bye(*_):
        for j in list(JIMS.values()):                # nothing of any Jim outlives the panel
            j.stop()
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, bye)
    if J.migrate(HERE):
        print(f"Moved the old settings into {J.CONFIG} (camera passwords: {J.SECRETS})")
    threading.Thread(target=network_loop, daemon=True).start()
    threading.Thread(target=probe_camera_audio, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", A.port), Panel)
    server.daemon_threads = True
    print(f"Jim's control panel: http://127.0.0.1:{A.port}  (Ctrl+C stops the panel and Jim)")
    server.serve_forever()
