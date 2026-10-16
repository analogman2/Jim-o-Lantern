#!/usr/bin/env python3
"""Jim's control panel: http://127.0.0.1:8790 on this PC (only).

Start, stop and restart Jim; watch what he sees, hears and says; set his camera, mic, voice and replies (saved to
jim.toml, comments kept); preview his voice; and play back recorded visits. The forms come from Jim's own options
(pumpkin.build_parser), so a new option shows up here by adding it to a group below.

Jim only runs while you start him here (or with ./start.sh); stopping the panel stops him.
Usage: ./panel.sh [--port 8790]
"""
import argparse, base64, collections, json, os, re, shutil, signal, subprocess, sys, threading, time, tomllib, urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pumpkin as P  # noqa: E402

CONFIG = HERE / "jim.toml"
RTSP_FILE = Path(P.CAMERA_FILE)
WATCH_PORT = 8780                                 # Jim's own viewer; the panel reads it
WATCH = f"http://127.0.0.1:{WATCH_PORT}"
PYTHON = HERE / ".venv" / "bin" / "python"

# tab -> [(section title, [option names])]; anything not listed isn't shown
GROUPS = {
    "camera": [("Camera", ["camera_source", "usb_camera", "camera_crop", "camera_width"])],
    "mic": [("Microphone", ["mic", "min_rms", "vad_ratio", "silence", "wled_gain"])],
    "voice": [("Voice", ["speaker", "voice_design", "gentle_voice_design", "pitch", "intonation", "echo", "grit", "lip_offset"]),
              ("Engine", ["tts", "speech_url"])],
    "responses": [("Personality", ["house", "temperature", "max_words"]),
                  ("Brain", ["brain", "ollama_model", "ollama_url", "keep_loaded"]),
                  ("Timing", ["reset_after", "eyes_timeout", "snooze", "start_delay"]),
                  ("Face (LEDs)", ["wled", "brightness", "flip_x", "flip_y"])],
    "recordings": [("Recording", ["record", "record_keep_days", "record_dir"])],
}
MANAGED = {k for groups in GROUPS.values() for _, keys in groups for k in keys}
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


def load_config():
    try:
        return tomllib.loads(CONFIG.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)


def write_settings(values, section=None):
    """Set keys in jim.toml, keeping every other line and comment. A key that's commented out (# key = ...) is
    replaced in place; a new one goes at the end of its section. section=None is the top level."""
    lines = CONFIG.read_text().splitlines() if CONFIG.exists() else []
    headers = [i for i, l in enumerate(lines) if re.match(r"^\s*\[[^\]]+\]\s*$", l)]
    if section is None:
        lo, hi = 0, headers[0] if headers else len(lines)
    else:
        start = next((i for i in headers if lines[i].strip() == f"[{section}]"), None)
        if start is None:
            lines += ["", f"[{section}]"]
            start = len(lines) - 1
            headers.append(start)
        lo, hi = start + 1, next((i for i in headers if i > start), len(lines))
    for key, v in values.items():
        new = f"{key} = {toml_value(v)}"
        hit = next((i for i in range(lo, hi) if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i])), None)
        if hit is None:
            hit = next((i for i in range(lo, hi) if re.match(rf"^\s*#\s*{re.escape(key)}\s*=", lines[i])), None)
        if hit is not None:
            comment = re.search(r"\s+#[^\"']*$", lines[hit]) if not lines[hit].lstrip().startswith("#") else None
            lines[hit] = new + (comment.group(0) if comment else "")
        else:
            at = hi
            while at > lo and not lines[at - 1].strip():      # before the blank lines that end the section
                at -= 1
            lines.insert(at, new)
            hi += 1
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tomllib.loads(tmp.read_text())                     # never save a file Jim couldn't read
    tmp.replace(CONFIG)


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
def camera_settings():
    cam = load_config().get("camera", {})
    saved = None
    try:
        saved = urllib.parse.urlsplit(RTSP_FILE.read_text().strip())
    except OSError:
        pass
    return {"ip": cam.get("ip") or (saved.hostname if saved else ""), "user": cam.get("user") or (saved.username if saved else ""),
            "path": cam.get("path") or (saved.path.lstrip("/") if saved else "h264Preview_01_main"),
            "password_saved": bool(saved and saved.password)}


def save_camera(ip, user, path, password=None):
    """Like set_camera.sh: the address with the password goes to ~/.config/jim/rtsp (only you can read it); jim.toml
    keeps the rest. Without a new password the saved one is kept."""
    if not re.fullmatch(r"[\w.-]+", ip or "") or not re.fullmatch(r"[\w.@-]+", user or "") or not re.fullmatch(r"[\w./-]*", path or ""):
        raise ValueError("check the IP, user and stream path")
    if password is None:
        try:
            password = urllib.parse.unquote(urllib.parse.urlsplit(RTSP_FILE.read_text().strip()).password or "")
        except OSError:
            password = ""
    if not password:
        raise ValueError("enter the camera's password")
    write_settings({"ip": ip, "user": user, "path": path}, section="camera")
    RTSP_FILE.parent.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        RTSP_FILE.write_text(f"rtsp://{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(password, safe='')}@{ip}:554/{path}\n")
    finally:
        os.umask(old)
    RTSP_FILE.chmod(0o600)


def grab_frame(source, usb):
    """One JPEG from the security camera or a USB webcam. Errors never include the camera address."""
    if source == "usb":
        src = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", "1280x720", "-i", usb, "-vf", "select=gte(n\\,8),scale=768:-2"]
        secret = None
    else:
        try:
            secret = RTSP_FILE.read_text().strip()
        except OSError:
            raise ValueError("no security camera saved yet")
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
    return {"mics": [("wled", "the LED controller's mic (patched firmware)")] + alsa_devices("arecord"),
            "speakers": [("default", "system default")] + alsa_devices("aplay"), "cameras": cams, "models": models}


def mic_test(device, seconds=3.0):
    """Record a few seconds and report the loudness Jim's ears would measure (30 ms frames, like pumpkin.Mic)."""
    if device == "wled":
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
    """Runs pumpkin.py as a child process (with his viewer on WATCH_PORT) and keeps his console output."""

    def __init__(self):
        self.proc, self.started = None, 0.0
        self.log = collections.deque(maxlen=600)
        self.lock = threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def others(self):
        """Copies of Jim not started by this panel (from ./start.sh): only one can have the mic."""
        mine = self.proc.pid if self.running() else None
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
        return [int(l.split()[0]) for l in out.splitlines() if re.search(r"python\S* (-u )?pumpkin\.py", l) and int(l.split()[0]) != mine]

    def start(self):
        with self.lock:
            if self.running():
                raise ValueError("Jim is already running")
            if self.others():
                raise ValueError(f"another copy of Jim is running (pid {', '.join(map(str, self.others()))}): stop it first")
            self.log.clear()
            self.log.append(f"--- starting Jim at {time.strftime('%H:%M:%S')}")
            self.proc = subprocess.Popen([str(PYTHON), "-u", "pumpkin.py", "--watch-port", str(WATCH_PORT)], cwd=HERE,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED="1"), text=True, bufsize=1)
            self.started = time.time()
            threading.Thread(target=self.read, args=(self.proc,), daemon=True).start()

    def read(self, proc):
        for line in proc.stdout:
            self.log.append(line.rstrip("\n"))
        self.log.append(f"--- Jim stopped (exit {proc.wait()}) at {time.strftime('%H:%M:%S')}")

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
        return {"running": self.running(), "pid": self.proc.pid if self.running() else None,
                "uptime_s": round(time.time() - self.started) if self.running() else None,
                "others": self.others(), "log": list(self.log)[-250:]}


jim = Jim()


# ---------------------------------------------------------------- recordings
def record_dir():
    d = Path(load_config().get("record_dir", "recordings"))
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
        out.append({"id": d.parent.name, "turns": sum(e["kind"] == "turn" for e in es), "entries": len(es),
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
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def proxy(self, path, method="GET", body=None):
        r = urllib.request.Request(WATCH + path, body, {"Content-Type": "application/json", "X-Jim-Panel": "1"}, method=method)
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return self.send(200, resp.read(), resp.headers.get("Content-Type", "application/octet-stream"))
        except Exception:
            return self.send(503, {"error": "Jim isn't running"})

    def do_GET(self):
        if not self.local():
            return self.send(403, {"error": "this panel only answers on this PC"})
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            return self.send(200, (HERE / "panel.html").read_bytes(), "text/html; charset=utf-8")
        if path in ("/jim/events.json", "/jim/live.jpg", "/jim/judged.jpg"):
            return self.proxy(path[4:])
        if path == "/api/status":
            return self.send(200, jim.status())
        if path == "/api/settings":
            opts, cfg = options(), load_config()
            tabs = {tab: [{"title": t, "fields": [dict(opts[k], name=k, value=cfg.get(k, opts[k]["default"]), long=k in LONG, pick=PICK.get(k))
                                                  for k in keys if k in opts]} for t, keys in groups] for tab, groups in GROUPS.items()}
            return self.send(200, {"tabs": tabs, "camera": camera_settings()})
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
        path = urllib.parse.urlsplit(self.path).path
        try:
            b = self.body()
            if path in ("/api/jim/start", "/api/jim/stop", "/api/jim/restart"):
                if path.endswith(("stop", "restart")):
                    jim.stop(b.get("pid"))
                if path.endswith(("start", "restart")):
                    jim.start()
                return self.send(200, jim.status())
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
                write_settings(vals)
                return self.send(200, {"ok": True, "restart": jim.running()})
            if path == "/api/camera":
                save_camera(b.get("ip", "").strip(), b.get("user", "").strip(), b.get("path", "").strip().lstrip("/"),
                            b.get("password") or None)
                return self.send(200, {"ok": True, "camera": camera_settings(), "restart": jim.running()})
            if path == "/api/camera/test":
                cfg = load_config()
                source = b.get("source") or cfg.get("camera_source", "auto")
                return self.send(200, grab_frame("usb" if source == "usb" else "rtsp", b.get("usb_camera") or cfg.get("usb_camera", "/dev/video0")), "image/jpeg")
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
        path = urllib.parse.urlsplit(self.path).path
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
        jim.stop()                                   # nothing of Jim outlives the panel
        sys.exit(0)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, bye)
    server = ThreadingHTTPServer(("127.0.0.1", A.port), Panel)
    server.daemon_threads = True
    print(f"Jim's control panel: http://127.0.0.1:{A.port}  (Ctrl+C stops the panel and Jim)")
    server.serve_forever()
