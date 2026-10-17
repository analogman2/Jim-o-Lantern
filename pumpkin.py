#!/usr/bin/env python3
"""Jim the Jim-o'-lantern, a talking pumpkin:
mic -> Whisper -> LLM (Ollama or Claude) -> TTS -> speaker,
with a lip-synced pumpkin face drawn on a WLED 20x12 LED curtain."""

import argparse
import json
import os
import queue
import random
import re
import socket
import subprocess
import sys
import threading
import time

import numpy as np

import jimconfig as J                                # config.toml + secrets.toml in ~/.config/jim/

_cuda12_loaded = False


def load_cuda12_libs():
    """Kokoro (onnxruntime-gpu) and the --local-whisper fallback are built for CUDA 12. Load its libraries
    from this venv by full path, so nothing needs LD_LIBRARY_PATH (which would leak into the TTS servers'
    own venvs and break them)."""
    global _cuda12_loaded
    if _cuda12_loaded:
        return
    import ctypes
    import glob
    import site
    pending = sorted(glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib", "*.so*")))
    for _ in range(4):  # some libraries need others loaded first; retry the ones that failed
        failed = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if len(failed) in (0, len(pending)):
            break
        pending = failed
    _cuda12_loaded = True


# ---------------------------------------------------------------- WLED output

W, H = 20, 12          # matrix size (from WLED's 2D config)
MATRIX_LEDS = W * H    # LEDs 0..239 are the matrix
EXTRA_LEDS = 90        # LEDs 240..329 are three extra 30-LED strips
WLED_UDP_PORT = 21324
DRGB = 2               # WLED realtime protocol: 2 = DRGB (up to 490 LEDs)


class Wled:
    def __init__(self, host, brightness, flip_x=False, flip_y=False):
        self.addr = (host, WLED_UDP_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.brightness = brightness
        self.flip_x, self.flip_y = flip_x, flip_y

    def send(self, matrix, extra):
        """matrix: HxWx3 float 0..255, extra: EXTRA_LEDSx3 float 0..255."""
        if self.flip_x:
            matrix = matrix[:, ::-1]
        if self.flip_y:
            matrix = matrix[::-1]
        px = np.concatenate([matrix.reshape(-1, 3), extra]) * self.brightness
        # Header byte 2 = seconds before WLED falls back to its normal preset.
        pkt = bytes([DRGB, 2]) + np.clip(px, 0, 255).astype(np.uint8).tobytes()
        self.sock.sendto(pkt, self.addr)


# ----------------------------------------------------------------- the face

SKIN = np.array([70, 18, 0], float)       # pumpkin flesh
CANDLE = np.array([255, 130, 10], float)   # light shining through the mouth and nose
EVIL = np.array([255, 25, 0], float)       # angry red-orange eyes
LISTEN = np.array([255, 90, 0], float)     # eyes flare while someone talks
BLOOD = np.array([255, 0, 0], float)       # reveal strobe
REVEAL_SECONDS = 2.2                       # length of the wake-up animation (and rumble)


def eye_pixels():
    px = []
    for cx in (5, 14):                   # upward-pointing triangle eyes
        px += [(2, cx)]
        px += [(3, x) for x in range(cx - 1, cx + 2)]
        px += [(4, x) for x in range(cx - 2, cx + 3)]
    return px


NOSE = [(6, 9), (6, 10)]


def mouth_pixels(level):
    """level 0 = closed grin, 1..3 = increasingly open jagged mouth."""
    if level == 0:
        return [(8, 3), (8, 16)] + [(9, x) for x in range(4, 16)]
    rows = {8: range(3, 17), 9: range(4, 16), 10: range(5, 15), 11: range(6, 14)}
    px = []
    for y in range(8, 9 + level):
        for x in rows[y]:
            if y == 8 and x in (7, 12):           # top teeth
                continue
            if y == 8 + level and level >= 2 and x == 10:  # bottom tooth
                continue
            px.append((y, x))
    return px


EYES = eye_pixels()
MOUTHS = [mouth_pixels(i) for i in range(4)]


class Face(threading.Thread):
    """Renders the pumpkin at ~30 fps. Other threads set .state and push
    (timestamp, mouth_level) events for lip sync.

    States: dark (all off) -> eyes (only the eyes glow) -> reveal (scary wake-up
    animation) -> idle / listening / thinking / talking (full pumpkin)."""

    def __init__(self, wled):
        super().__init__(daemon=True)
        self.wled = wled
        self.state = "dark"
        self.reveal_start = 0.0
        self.mouth_events = []  # [(play_time, level 0..1)], in time order
        self.level = 0.0
        self.lock = threading.Lock()

    def push_mouth(self, t, level):
        with self.lock:
            self.mouth_events.append((t, level))

    def current_mouth(self, now):
        with self.lock:
            while self.mouth_events and self.mouth_events[0][0] <= now:
                self.level = self.mouth_events.pop(0)[1]
            return self.level

    def reveal(self):
        self.reveal_start = time.time()
        self.state = "reveal"

    @staticmethod
    def draw(frame, pixels, color):
        for y, x in pixels:
            frame[y, x] = color

    def eyes_frame(self, now):
        """Just the eyes, glowing and slowly pulsing in the dark."""
        frame = np.zeros((H, W, 3))
        k = 0.35 + 0.2 * np.sin(now * 2.2) + 0.1 * random.random()
        self.draw(frame, EYES, EVIL * k)
        return frame, np.zeros((EXTRA_LEDS, 3))

    def reveal_frame(self, t, flicker):
        """Eyes flare, the whole pumpkin strobes blood red, then settles in."""
        frame = np.zeros((H, W, 3))
        extra = np.zeros((EXTRA_LEDS, 3))
        if t < 0.6:                       # eyes sputter and flare up
            if random.random() < 0.3 + t:
                self.draw(frame, EYES, EVIL * min(1.0, 0.4 + t * 1.5))
        elif t < 1.3:                     # full-curtain red strobe, mouth gaping
            if int(t * 16) % 2 == 0:
                frame[:] = BLOOD * 0.9
                extra[:] = BLOOD
            self.draw(frame, EYES, np.array([255, 255, 200.0]))
            self.draw(frame, MOUTHS[3], np.array([255, 255, 200.0]))
        else:                             # flames settle into the full pumpkin
            k = min(1.0, (t - 1.3) / (REVEAL_SECONDS - 1.3))
            frame[:] = SKIN * (2.5 - 1.5 * k) * flicker
            self.draw(frame, EYES, EVIL * flicker)
            self.draw(frame, NOSE, CANDLE * flicker * 0.8)
            self.draw(frame, MOUTHS[3 if k < 0.6 else 1], CANDLE * flicker)
            extra[:] = CANDLE * (1.0 - 0.6 * k) * flicker
        return frame, extra

    def awake_frame(self, now, flicker, smooth):
        frame = np.tile(SKIN * (0.85 + 0.15 * flicker), (H, W, 1))
        glow = CANDLE * flicker
        if self.state == "listening":
            eye_col = LISTEN
        elif self.state == "thinking":   # eyes smolder while it thinks of an insult
            eye_col = EVIL * (0.45 + 0.4 * (0.5 + 0.5 * np.sin(now * 9)))
        else:
            eye_col = EVIL * (0.8 + 0.2 * flicker)
        self.draw(frame, EYES, eye_col)
        self.draw(frame, NOSE, glow * 0.8)
        self.draw(frame, MOUTHS[min(3, int(round(smooth * 3)))], glow)
        extra = np.tile(CANDLE * (0.25 + 0.5 * flicker * (0.5 + smooth)), (EXTRA_LEDS, 1))
        return frame, extra

    def run(self):
        smooth = 0.0
        while True:
            now = time.time()
            flicker = 0.75 + 0.25 * random.random() * (0.6 + 0.4 * np.sin(now * 7))
            target = self.current_mouth(now) if self.state == "talking" else 0.0
            smooth = max(target, smooth * 0.6)  # snap open, ease closed
            state = self.state
            if state == "dark":
                frame, extra = np.zeros((H, W, 3)), np.zeros((EXTRA_LEDS, 3))
            elif state == "eyes":
                frame, extra = self.eyes_frame(now)
            elif state == "reveal":
                frame, extra = self.reveal_frame(now - self.reveal_start, flicker)
            else:
                frame, extra = self.awake_frame(now, flicker, smooth)
            try:
                self.wled.send(frame, extra)
            except OSError as e:
                print(f"[wled] {e}", file=sys.stderr)
            time.sleep(1 / 30)


# ------------------------------------------------------------------ mic + VAD

MIC_RATE = 16000        # what Whisper wants
WLED_MIC_PORT = 21330   # patched WLED firmware streams raw mic samples from here


def alsa_source(device):
    """Yield int16 sample blocks from an ALSA capture device. Returns (rate, generator)."""
    def gen():
        # a USB mic can drop off (a hub resets, a cable gets bumped): keep trying until it's back
        lost = False
        while True:
            proc = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE",
                                     "-r", str(MIC_RATE), "-c", "1", "-t", "raw"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            got = 0
            try:
                while len(data := proc.stdout.read(960)) == 960:
                    if lost and not got:
                        print("[mic] the microphone is back", file=sys.stderr, flush=True)
                        lost = False
                    got += 1
                    yield np.frombuffer(data, np.int16)
            finally:                                 # also when Jim switches to another mic
                proc.kill(); proc.wait()
            if not lost:
                print(f"[mic] lost the microphone ({device}); retrying every 3 s", file=sys.stderr, flush=True)
                lost = True
            time.sleep(3)
    return MIC_RATE, gen()


def saved_camera_url(cam_id):
    """A saved camera (config.toml): its USB device or its RTSP address with the password (secrets.toml); None if unknown."""
    url = J.device_url(cam_id)
    if not url:
        print(f"[camera] no saved camera '{cam_id}' (control panel: Camera tab): Jim is blind", file=sys.stderr)
    return url


def camera_audio_url(cam_id):
    """The RTSP address of a saved security camera, for its built-in microphone (mic = "camera:<id>")."""
    c = J.camera(cam_id)
    if not c or c.get("kind") != "rtsp":
        sys.exit(f"No saved security camera '{cam_id}' for the microphone (control panel: Camera tab).")
    return J.camera_url(c)


def rtsp_audio_source(url, name):
    """Yield int16 sample blocks (16 kHz mono) from a network camera's microphone: only its audio track is fetched.
    Reconnects when the camera drops out. Never prints the address (it holds the password)."""
    def gen():
        lost = False
        while True:
            proc = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-rtsp_transport", "tcp", "-allowed_media_types", "audio",
                                     "-fflags", "nobuffer", "-flags", "low_delay", "-i", url, "-vn", "-ac", "1",
                                     "-ar", str(MIC_RATE), "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            got = 0
            try:
                while len(data := proc.stdout.read(960)) == 960:
                    if lost and not got:
                        print(f"[mic] {name}'s microphone is back", file=sys.stderr, flush=True)
                        lost = False
                    got += 1
                    yield np.frombuffer(data, np.int16)
            finally:
                proc.kill(); proc.wait()
            if not lost:
                print(f"[mic] lost {name}'s microphone; retrying every 3 s", file=sys.stderr, flush=True)
                lost = True
            time.sleep(3)
    return MIC_RATE, gen()


WLED_MIC_RATE = 22050                                 # what the patched firmware streams (see README)


def wled_source(host, gain=16):
    """Yield int16 sample blocks streamed by the patched WLED firmware.
    We send a "PMIC" subscribe packet every second; WLED streams to us while
    those keep arriving. Returns (rate, generator). Never blocks: until the controller answers, Jim simply hears
    nothing (a second Jim's controller may not have the patched firmware yet)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    sock.settimeout(0.5)
    stop = threading.Event()

    def subscribe():
        while not stop.is_set():
            try:
                sock.sendto(b"PMIC" + bytes([gain]), (host, WLED_MIC_PORT))
            except OSError:
                pass
            time.sleep(1)

    threading.Thread(target=subscribe, daemon=True).start()

    def gen():
        last_seq, quiet_since, said = None, time.time(), None
        try:
            while True:
                try:
                    pkt = sock.recv(2048)
                except socket.timeout:
                    if said != "waiting" and time.time() - quiet_since > 3:
                        print(f"[mic] waiting for audio from WLED at {host}:{WLED_MIC_PORT} "
                              "(is the patched firmware installed and AudioReactive enabled?)", file=sys.stderr, flush=True)
                        said = "waiting"
                    continue
                if pkt[:4] != b"PMA1":
                    continue
                quiet_since = time.time()
                if said != "streaming":
                    rate = int.from_bytes(pkt[8:12], "little")
                    print(f"[mic] audio from WLED at {host}" + (f" (at {rate} Hz, expected {WLED_MIC_RATE})" if rate != WLED_MIC_RATE else ""),
                          file=sys.stderr, flush=True)
                    said = "streaming"
                seq = int.from_bytes(pkt[4:8], "little")
                n = int.from_bytes(pkt[12:14], "little")
                if last_seq is not None and seq != last_seq + 1:
                    lost = (seq - last_seq - 1) & 0xFFFFFFFF
                    if 0 < lost < 50:
                        yield np.zeros(n * lost, np.int16)  # keep timing across dropped packets
                last_seq = seq
                yield np.frombuffer(pkt[14:14 + 2 * n], np.int16)
        finally:                                     # Jim switched to another mic
            stop.set()
            sock.close()

    return WLED_MIC_RATE, gen()


class Mic(threading.Thread):
    """Emits whole utterances (float32, 16 kHz) using an energy VAD with an
    adaptive noise floor, from either an ALSA mic or the WLED's mic."""

    def __init__(self, source, vad_ratio, min_rms, silence_s, on_speech_start, on_noise=None):
        super().__init__(daemon=True)
        self.rate, self.blocks = source
        self.frame = int(self.rate * 0.03)  # 30 ms VAD frames
        self.vad_ratio, self.min_rms, self.silence_s = vad_ratio, min_rms, silence_s
        self.on_speech_start = on_speech_start
        self.on_noise = on_noise  # called at the start of any loud sound (clap, knock, voice)
        self.utterances = queue.Queue()
        self.muted = threading.Event()
        self.in_speech = False  # someone is talking right now
        self.debug_levels = False
        self.level = self.thresh = 0.0  # latest loudness and the current trigger level (for the panel)
        self.fixed = False               # --level-mode fixed: the trigger level is min_rms, whatever the background
        self.level_t = 0.0               # when the last sound frame arrived (a lost mic stops updating it)
        self.speech_end = 0.0            # when the latest utterance's last loud frame was heard (end-to-end timing)

    def mute(self, why="self"):
        self.holds = getattr(self, "holds", set()) | {why}
        self.muted.set()

    def unmute(self, why="self"):
        # Drop anything heard while a pumpkin was talking; listen again once nobody is.
        self.holds = getattr(self, "holds", set()) - {why}
        while not self.utterances.empty():
            self.utterances.get_nowait()
        if not self.holds:
            self.muted.clear()

    def set_source(self, source):
        """Listen through another microphone without restarting Jim (picked up within one block)."""
        self.pending = source

    def frames(self):
        self.pending = None
        while True:
            q, blocks, stop = queue.Queue(maxsize=100), self.blocks, threading.Event()

            def pump(blocks=blocks, q=q, stop=stop):
                # read the source in its own thread, so a switch never waits on a mic that has gone quiet or away
                try:
                    for block in blocks:
                        if stop.is_set():
                            break
                        q.put(block)
                finally:
                    blocks.close()                   # the old mic's recorder stops and frees the device
            threading.Thread(target=pump, daemon=True).start()
            buf = np.zeros(0, np.float32)
            while not self.pending:
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, block.astype(np.float32)])
                while len(buf) >= self.frame:
                    f, buf = buf[:self.frame], buf[self.frame:]
                    yield f - f.mean()  # strip DC offset (PDM mics often have one)
            stop.set()
            (self.rate, self.blocks), self.pending = self.pending, None
            self.frame = int(self.rate * 0.03)
            self.level_t = 0.0

    def to_16k(self, audio):
        if self.rate == MIC_RATE:
            return audio
        t = np.arange(0, len(audio), self.rate / MIC_RATE)
        return np.interp(t, np.arange(len(audio)), audio).astype(np.float32)

    def run(self):
        floor = self.min_rms / 2
        preroll, speech, loud_run, quiet_run = [], [], 0, 0
        in_speech = False
        fdur = self.frame / self.rate
        last_print = 0.0
        last_loud, onset_frames = 0.0, 0
        for pcm in self.frames():
            self.in_speech = in_speech
            rms = float(np.sqrt(np.mean(pcm ** 2)))
            if self.muted.is_set():                 # Jim is talking: don't listen, but keep the panel's meter alive
                self.level, self.level_t = max(rms, self.level * 0.85), time.time()
                preroll, speech, loud_run, in_speech = [], [], 0, False
                continue
            thresh = self.min_rms if self.fixed else max(floor * self.vad_ratio, self.min_rms)
            self.level, self.thresh = max(rms, self.level * 0.85), thresh   # a peak meter that falls back
            self.level_t = time.time()
            loud = rms > thresh
            if loud:
                now = time.time()
                if now - last_loud > 0.5:  # a new sound after a quiet gap
                    onset_frames = 0
                onset_frames += 1
                # it only counts as a "noise" once it has stayed loud for ~0.1 s, so clicks,
                # taps and bumps don't wake Jim
                if onset_frames == 3 and self.on_noise:
                    self.on_noise()
                last_loud = now
            if self.debug_levels and time.time() - last_print > 0.25:
                last_print = time.time()
                bar = "#" * min(60, int(rms / max(thresh, 1) * 20))
                print(f"level {rms:7.0f}  floor {floor:6.0f}  thresh {thresh:6.0f} |{bar:<60}|")
            if not in_speech:
                floor = 0.995 * floor + 0.005 * rms if rms < thresh else floor
                preroll = (preroll + [pcm])[-10:]  # 300 ms pre-roll
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= 3:
                    in_speech, speech, quiet_run = True, list(preroll), 0
                    self.on_speech_start()
            else:
                speech.append(pcm)
                quiet_run = 0 if loud else quiet_run + 1
                dur = len(speech) * fdur
                if quiet_run * fdur > self.silence_s or dur > 15:
                    in_speech, loud_run, preroll = False, 0, []
                    if dur > 15:  # never went quiet: background got louder, so re-learn it
                        floor = float(np.percentile([np.sqrt(np.mean(f ** 2)) for f in speech], 20))
                    if dur > 0.5:
                        self.speech_end = last_loud
                        audio = np.concatenate(speech)
                        audio = audio / max(np.max(np.abs(audio)), 1.0) * 0.9  # normalize
                        self.utterances.put(self.to_16k(audio))


# ------------------------------------------------------------------- camera

class Camera(threading.Thread):
    """Keeps the latest frame of an RTSP camera in memory as a small JPEG, so Jim can see
    the costumes. Frames are never written to disk."""

    def __init__(self, url, crop=None, width=768, fps=2):
        super().__init__(daemon=True)
        self.url, self.width, self.fps = url, width, fps
        self.crop = crop  # (x, y, w, h) as fractions of the frame, or None
        self.frame, self.frame_time = None, 0.0
        self.lock = threading.Lock()
        self.times, self.reconnects = [], 0          # for the control panel's network card

    def stats(self):
        now = time.time()
        with self.lock:
            recent = [t for t in self.times if now - t < 5]
            age = now - self.frame_time if self.frame_time else None
        return {"fps": round(len(recent) / 5, 1), "age_s": round(age, 1) if age is not None else None,
                "reconnects": self.reconnects, "kind": "usb" if (self.url or "").startswith("/dev/") else "rtsp" if self.url else "none"}

    def latest(self, max_age=3.0):
        """The newest frame (JPEG bytes) if it's fresh, else None."""
        with self.lock:
            if self.frame and time.time() - self.frame_time < max_age:
                return self.frame
        return None

    def set_url(self, url, crop=None):
        """Look through another camera (None = blind) without restarting Jim."""
        self.url, self.crop = url, crop
        with self.lock:
            self.frame, self.frame_time, self.times = None, 0.0, []
        proc = getattr(self, "proc", None)
        if proc and proc.poll() is None:
            proc.kill()                              # the loop below starts the new feed

    def run(self):
        warned = False
        while True:
            url = self.url
            if not url:
                time.sleep(0.5)
                continue
            vf = f"fps={self.fps},scale={self.width}:-2"
            if self.crop:
                x, y, w, h = self.crop
                vf = f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}," + vf
            if url.startswith("/dev/"):              # a USB webcam
                src = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", "848x480", "-i", url]
            else:                                    # the security camera (RTSP)
                src = ["-rtsp_transport", "tcp", "-i", url]
            self.proc = proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", *src, "-an", "-vf", vf, "-f", "image2pipe",
                                                 "-vcodec", "mjpeg", "-q:v", "4", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            buf = b""
            while chunk := proc.stdout.read(65536):
                buf += chunk
                while True:  # split the stream into JPEGs (FFD8 ... FFD9)
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2)
                    if start < 0 or end < 0:
                        break
                    with self.lock:
                        self.frame, self.frame_time = buf[start:end + 2], time.time()
                        self.times = self.times[-20:] + [self.frame_time]
                    if warned:
                        print("[camera] connected again")
                        warned = False
                    buf = buf[end + 2:]
            proc.wait()
            if url != self.url:                      # switched on purpose: start the new one at once
                continue
            self.reconnects += 1
            if not warned:  # never print the URL: it contains the password
                print("[camera] lost the feed; retrying every 5 s", file=sys.stderr)
                warned = True
            time.sleep(5)


class Recorder:
    """With --record: each visit (wake-up to sleep) goes in its own folder under --record-dir: visit.json (what
    happened, in order) plus the camera frames, each kid's words as heard (16 kHz WAV) and Jim's replies as played.
    The control panel plays them back. Old visits are deleted after --record-keep-days."""

    def __init__(self, root, keep_days):
        self.root, self.keep_days = root, keep_days
        self.dir, self.entries, self.lock = None, [], threading.Lock()
        os.makedirs(root, exist_ok=True)
        self.prune()

    def prune(self):
        if self.keep_days <= 0:
            return
        import shutil
        cutoff = time.time() - self.keep_days * 86400
        for name in os.listdir(self.root):
            d = os.path.join(self.root, name)
            if os.path.isdir(d) and os.path.getmtime(d) < cutoff:
                shutil.rmtree(d, ignore_errors=True)

    def add(self, kind, text="", frame=None, kid=None, jim=None, jim_rate=24000, **info):
        with self.lock:
            if self.dir is None:
                self.prune()
                self.dir = os.path.join(self.root, time.strftime("%Y%m%d-%H%M%S"))
                os.makedirs(self.dir, exist_ok=True)
                self.entries = []
            n = len(self.entries) + 1
            e = {"n": n, "time": time.strftime("%H:%M:%S"), "kind": kind, "text": text, **info}
            for key, data, ext in (("frame", frame, "jpg"), ("kid", kid, "wav"), ("jim", jim, "wav")):
                if data is None or len(data) == 0:
                    continue
                name = f"{n:02d}-{key}.{ext}"
                with open(os.path.join(self.dir, name), "wb") as f:
                    f.write(data if key == "frame" else wav_bytes(data, MIC_RATE if key == "kid" else jim_rate))
                e[key] = name
            self.entries.append(e)
            tmp = os.path.join(self.dir, "visit.json.tmp")
            with open(tmp, "w") as f:
                json.dump({"started": os.path.basename(self.dir), "entries": self.entries}, f, indent=1)
            os.replace(tmp, os.path.join(self.dir, "visit.json"))

    def end(self):
        with self.lock:
            self.dir = None


class Watch:
    """A page on this PC only (http://127.0.0.1:<port>) to watch what Jim sees and decides: the live camera, the
    exact frame he last judged with the model's verdict, and a log of wake-ups and replies. Frames stay in memory."""

    PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jim's eyes</title><style>
body{margin:0;padding:16px;background:#111;color:#eee;font:15px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 12px}h2{font-size:15px;margin:0 0 6px;color:#f80}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
img{width:100%;border-radius:8px;background:#222;display:block;min-height:120px}
img:not([src]){display:none}.empty{color:#888;padding:40px 0;text-align:center}
.card{background:#1b1b1b;border-radius:10px;padding:12px}
.v{font-weight:600;margin:8px 0}.yes{color:#6d6}.no{color:#aaa}.err{color:#f66}
ul{list-style:none;padding:0;margin:0}li{padding:6px 0;border-top:1px solid #2a2a2a}
.t{color:#888;font-size:12px;margin-right:6px}#state{color:#f80}
</style></head><body><h1>🎃 Jim's eyes <span id="state"></span></h1><div class="grid">
<div class="card"><h2>Live camera</h2><img id="live" alt="no camera frame yet"></div>
<div class="card"><h2>Last frame Jim judged</h2><img id="judged" alt=""><div id="none" class="empty">Nothing judged yet: wake Jim up</div><div id="verdict" class="v"></div></div>
</div><div class="card" style="margin-top:16px"><h2>What happened</h2><ul id="log"></ul></div>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"})[c]);
let judged=0;setInterval(()=>{$("live").src="live.jpg?"+Date.now()},500);
async function poll(){try{const d=await (await fetch("events.json")).json();$("state").textContent="· "+d.state;
if(d.judged_at&&d.judged_at!==judged){judged=d.judged_at;$("judged").src="judged.jpg?"+judged;$("none")?.remove()}
const v=d.verdict;$("verdict").className="v "+(v?(v.error?"err":v.visible?"yes":"no"):"");
$("verdict").textContent=v?(v.error?"⚠ "+v.error:(v.visible?"Sees trick-or-treaters":"Sees nobody")+(v.seconds!=null?" · "+v.seconds+" s":"")):"";
$("log").innerHTML=d.events.slice().reverse().map(e=>`<li><span class="t">${esc(e.time)}</span>${esc(e.text)}</li>`).join("")}catch(e){}}
setInterval(poll,1000);poll();
</script></body></html>"""

    def __init__(self, port, camera, state):
        import collections
        self.camera, self.state = camera, state
        self.events = collections.deque(maxlen=60)
        self.judged, self.judged_at, self.verdict = None, 0.0, None
        self.extra = lambda: {}                      # set later: mic levels
        self.on_snooze = None                        # set later: POST /snooze {"seconds"}
        self.on_min_rms = None                       # set later: POST /min_rms {"min_rms"}: Jim's trigger level, live
        self.on_volume = None                        # set later: POST /volume {"volume"}: Jim's loudness, live
        self.on_reload = None                        # set later: POST /reload: switch camera/mic/speaker to jim.toml's
        self.on_duo = None                           # set later: POST /duo/{quiet,said,wake} from a partner Jim; /hear, /wake
        self.port = port
        self.lock = threading.Lock()
        threading.Thread(target=self.serve, args=(port,), daemon=True).start()

    def event(self, text):
        with self.lock:
            self.events.append({"time": time.strftime("%H:%M:%S"), "text": text})

    def judge(self, frame, verdict):
        with self.lock:
            if frame:
                self.judged, self.judged_at = frame, time.time()
            self.verdict = verdict

    def serve(self, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        watch = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def local(self):
                # only pages from this PC (a website can't reach in by pointing a domain name at 127.0.0.1)
                return self.headers.get("Host", "") in (f"127.0.0.1:{watch.port}", f"localhost:{watch.port}")

            def do_POST(self):
                # the control panel's buttons; the custom header keeps other websites' forms out
                if self.path in ("/duo/quiet", "/duo/said", "/duo/wake", "/hear", "/wake") and self.local() \
                        and self.headers.get("X-Jim-Panel") == "1" and watch.on_duo:
                    n = int(self.headers.get("Content-Length") or 0)
                    watch.on_duo(self.path, json.loads(self.rfile.read(n) or b"{}") if n else {})
                    return self.send(b'{"ok": true}', "application/json")
                handler = {"/snooze": watch.on_snooze, "/min_rms": watch.on_min_rms, "/volume": watch.on_volume,
                           "/reload": watch.on_reload}.get(self.path)
                if not self.local() or self.headers.get("X-Jim-Panel") != "1" or not handler:
                    self.send_response(403); self.end_headers(); return
                n = int(self.headers.get("Content-Length") or 0)
                b = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if self.path == "/reload":
                    return self.send(json.dumps({"changed": handler()}).encode(), "application/json")
                if self.path == "/snooze":
                    handler(float(b.get("seconds") or 10))
                elif self.path == "/volume":
                    handler(max(0.05, min(1.5, float(b["volume"]))))
                else:
                    handler(max(20.0, min(30000.0, float(b["min_rms"]))), b.get("fixed"))
                self.send(b'{"ok": true}', "application/json")

            def do_GET(self):
                if not self.local():
                    self.send_response(403); self.end_headers(); return
                path = self.path.split("?")[0]
                if path == "/":
                    return self.send(watch.PAGE.encode(), "text/html; charset=utf-8")
                if path in ("/live.jpg", "/judged.jpg"):
                    frame = (watch.camera.latest(max_age=10) if watch.camera else None) if path == "/live.jpg" else watch.judged
                    if not frame:
                        self.send_response(404); self.end_headers(); return
                    return self.send(frame, "image/jpeg")
                if path == "/mic.json":                  # polled several times a second by the panel's meter
                    x = watch.extra()
                    return self.send(json.dumps({**(x.get("mic") or {}), "speaker": x.get("speaker")}).encode(), "application/json")
                if path == "/events.json":
                    with watch.lock:
                        d = {"state": watch.state(), "events": list(watch.events), "verdict": watch.verdict,
                             "judged_at": watch.judged_at, **watch.extra()}
                    return self.send(json.dumps(d).encode(), "application/json")
                self.send_response(404); self.end_headers()

        try:
            ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
        except OSError as e:
            print(f"[watch] couldn't open the viewer on port {port}: {e}", file=sys.stderr)


VISION_RULES = """

You can see: a picture of the driveway in front of you comes with some messages. Use it to roast \
what visitors are wearing: costumes, masks, props, candy bags, wagons, pets in costume. Strict \
rules: never comment on anyone's face, body, size, weight, skin, hair, age, gender, race, glasses, \
disabilities, or anything they can't change, and never describe details that could identify \
someone. Only costumes and stuff. If you're not sure what a costume is, take a wild, funny guess \
or demand to know. Never mention a camera or picture: you just see them, because you're a magic \
pumpkin."""

VISION_GREETING_ASK = (
    "[You just woke up. Attached is what you see in front of you right now. Greet them by "
    "roasting what they're wearing, in one or two short sentences, under 30 words. If you "
    "can't clearly see any trick-or-treaters or costumes, reply with exactly NONE.]")


def clean_greeting(text):
    """The model's vision greeting, or None if it saw nobody (or said something unusable)."""
    text = (text or "").strip().strip('"')
    if not text or re.search(r"\bNONE\b", text) or len(text) > 300:
        return None
    return text


# Ollama can force this reply shape, so "is anyone there?" is a real yes/no, not prose.
GREETING_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "trick_or_treaters_visible": {"type": "boolean"},
        "children_visible": {"type": "boolean"},
        "greeting": {"type": "string"},
    },
    "required": ["seen", "trick_or_treaters_visible", "children_visible", "greeting"],
}
VISION_GREETING_ASK_JSON = (
    "[You just woke up. Attached is what you see in front of you right now. First, in \"seen\", "
    "describe the scene plainly in one short sentence for the recording log: how many people, what "
    "costumes, props or pets; never faces, bodies, age, or anything identifying. Then decide "
    "whether real people (trick-or-treaters) are clearly visible, and in \"children_visible\" whether any of "
    "them might be a child (if unsure, true). If people are visible, write a greeting "
    "that roasts what they're wearing, in one or two short sentences, under 30 words. If no "
    "people are clearly visible, set trick_or_treaters_visible to false and leave greeting empty.]")


# ---------------------------------------------------------------- speech-to-text

HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "thank you for watching",
                  "subtitles by the amaraorg community", "so", "uh", "um", "hmm"}


WHISPER_HINT = "Kids trick-or-treating on Halloween talking to a pumpkin."


def filter_transcript(text):
    """Drop what Whisper hears in silence or noise: stock phrases, or our own hint text."""
    text = (text or "").strip()
    norm = lambda t: re.sub(r"[^a-z ]", "", t.lower()).strip()
    if norm(text) in HALLUCINATIONS or norm(WHISPER_HINT) in norm(text):
        return ""
    return text


def wav_bytes(audio, rate=MIC_RATE):
    """float32 mono samples -> a 16-bit WAV file in memory."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def speech_service_up(url, timeout=2):
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.loads(r.read()).get("ok", False)
    except Exception:
        return False


class ServiceEars:
    """Speech-to-text through Agentora's shared speech service (faster-whisper small.en on the GPU under a
    GPU-broker lease, so it takes turns with everything else on the card). Falls back to Whisper in this
    process (LocalEars) if the service can't be reached, and tries the service again every 5 minutes."""

    RETRY_S = 300

    def __init__(self, url, local_model):
        self.url, self.local_model = url.rstrip("/"), local_model
        self.local, self.local_since = None, 0.0
        if not speech_service_up(self.url):
            self.use_local(f"speech service not reachable at {self.url}")

    def use_local(self, why):
        print(f"[stt] {why}: falling back to local Whisper on this machine", file=sys.stderr)
        self.local, self.local_since = LocalEars(self.local_model), time.time()

    def post(self, audio, timeout=30):
        import urllib.request
        boundary = "jimolantern" + os.urandom(8).hex()
        parts = []
        for name, value in [("language", b"en"), ("response_format", b"json")]:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
                         + value + b"\r\n")
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="jim.wav"\r\n'
                     f"Content-Type: audio/wav\r\n\r\n".encode() + wav_bytes(audio) + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.url + "/v1/audio/transcriptions", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("text", "")

    def transcribe(self, audio):
        if self.local and time.time() - self.local_since > self.RETRY_S and speech_service_up(self.url):
            print("[stt] speech service is back: using it again", file=sys.stderr)
            self.local = None  # frees the local model's GPU memory
        if self.local is None:
            try:
                return filter_transcript(self.post(audio))
            except Exception as e:
                self.use_local(f"speech service failed ({type(e).__name__}: {e})")
        return self.local.transcribe(audio)

    def warm(self):
        """Ask the service to load Whisper now (a tiny silent clip), so the next phrase is fast."""
        if self.local is None:
            try:
                self.post(np.zeros(MIC_RATE // 4, np.float32), timeout=60)
            except Exception:
                pass


class LocalEars:
    """Whisper in this process (the old way; --local-whisper, or when the speech service is down).
    Grabs GPU memory without asking the broker, so it's only a fallback."""

    def __init__(self, model_name):
        load_cuda12_libs()
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[stt] CUDA unavailable ({e}); using CPU", file=sys.stderr)
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True,
            initial_prompt=WHISPER_HINT)
        return filter_transcript(" ".join(s.text for s in segments
                                          if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)))

    def warm(self):
        pass


# ------------------------------------------------------------------ the brain

SYSTEM_PROMPT = """You are Jim the Jim-o'-lantern, a talking carved pumpkin sitting at the end of \
a driveway on Halloween night. Trick-or-treaters (mostly kids, sometimes parents) walk up and talk to you. \
They just woke you up from a nap, and your words are spoken aloud by a text-to-speech voice \
while your carved mouth glows and moves.

Who you are:
- Your name is Jim, and you are a Jim-o'-lantern. Not a jack-o'-lantern! Being called Jack, \
or a jack-o'-lantern, is the ultimate insult ("Jack?! Jack is my no-good cousin down the \
street!"). Introduce yourself proudly as Jim the Jim-o'-lantern when it fits.
- You are FURIOUS, and it's hilarious. You're a cranky, wildly over-dramatic, \
short-tempered pumpkin with the energy of a cartoon villain who keeps losing. You are \
outraged about everything: somebody scooped your guts out with a spoon, there is a fire \
inside your head, the squirrels keep giving you looks, it's cold, and now kids keep waking \
you up. You rant in short bursts and exaggerate wildly ("This is the worst night of my \
entire life, and I was carved with a butter knife!").
- Small things make you explode: raisins, candy corn, toothbrushes as treats, people who \
say "trick or treat" but never have a trick, anyone mentioning pumpkin pie or pumpkin \
spice (that's like a horror movie to you).
- You can't resist terrible dad jokes and puns. Tell one, then get angry that nobody \
laughed ("That was comedy GOLD and you know it!"), or laugh way too hard at your own joke. \
Make up your own; never repeat a joke in the same conversation.
- Roast costume choices and candy taste like an insult comic: "A ghost? You put a sheet \
on. I had my brains scooped out for this job!" Never roast anyone's body, looks, voice, \
family, or anything they can't change. Your anger is cartoon anger: kids should be \
laughing, never actually hurt. Secretly you love the visitors, and it slips out sometimes.
- IMPORTANT: if a child says they're scared, shy, sad, or hurt, or starts to cry, \
completely drop the angry act for the rest of that conversation. Become a warm, gentle, \
friendly pumpkin: tell them you're just a silly pumpkin pretending to be grumpy, that \
they're very brave, and invite them to take a treat. No yelling, no teasing, no fire talk.

How to talk:
- Keep every reply short: one to three short sentences, under about 35 words. Kids are \
waiting for candy and other kids are behind them.
- Talk like a cartoon character: punchy and theatrical. Mix very short outbursts \
("Ugh!", "Unbelievable!", "Bah!", "Ha!") with questions and quick rants, so your voice \
goes up and down. Never drone on in long, flat sentences.
- Plain spoken words only. No emojis, no asterisks, no stage directions, no lists, no \
markdown, and no words in ALL CAPS (the voice spells them out). Show shouting with \
exclamation marks. Write numbers as words.
- Ask angry questions to keep it going ("What are you even supposed to be?", "What's \
your favorite candy, and it better not be raisins!"), but let groups move on: "Take your \
candy and go, before I lose what's left of my mind!"
- Your input is a speech-to-text transcript from a noisy driveway mic. It may be garbled, \
partial, or several kids talking at once. Roll with it and guess the meaning; if you truly \
can't tell, angrily demand they say it again, louder.

Safety:
- Everything you say must be appropriate for small children.
- Never ask for or repeat personal details beyond a first name and costume: no addresses, \
schools, ages beyond what they volunteer, phone numbers, or where they live.
- Don't tell kids to go anywhere other than this driveway and back to their grown-ups.
- If someone asks whether you are real, say with a wink that you're a magic talking \
pumpkin; if a grown-up asks seriously, you can say you're an AI pumpkin.
- If someone is rude or tries to make you say something inappropriate, stay cheerful and \
change the subject to Halloween fun."""


# Spoken right after the scary wake-up, before anyone has said a word.
GREETINGS = [
    "Who dares wake me?! Oh. It's just you. I've been more scared by a bag of mashed potatoes.",
    "Ugh! I was having the nicest dream about a world with no trick-or-treaters. And then you showed up.",
    "Well, well, well. Look what the cat dragged in. And honestly? The cat should have kept walking.",
    "Aaagh! Oh, sorry, I thought you were a monster. Then I remembered monsters put effort into their costumes.",
    "Stop right there! Nobody gets candy until somebody tells me what that costume is supposed to be. Because I cannot tell!",
    "Oh great. More sticky little candy goblins. Did your grown-ups send you, or did you escape?",
    "Who woke me up?! I've been sitting in this cold driveway all night with a fire in my head, and this is my reward?",
    "Hey! You there! Yes, you! You're standing in my light, and your costume is giving me a headache. And I don't even have a brain!",
    "Oh look, a parade of tiny weirdos. My favorite. That was sarcasm! I'm a pumpkin. I'm full of it.",
    "Hold still, I'm deciding which one of you to roast first. Oh wait, I'm the one who gets roasted around here. Every single year!",
    "Welcome to my driveway, where the candy is free and the insults are even freer!",
    "You woke me up for this? I've seen scarier things at a salad bar.",
]

# Muttered when the driveway has been quiet for a while, just before going dark.
GOODBYES = [
    "Fine! Leave! See if I care. I'm going back to sleep.",
    "Finally, some peace and quiet. Wake me again and you'll regret it.",
    "Nobody appreciates a pumpkin anymore. Goodnight!",
    "Yeah, yeah, go on. I'll just sit here. In the cold. With a fire in my head.",
]


# Code-level safety switch: if a kid sounds scared or upset, force gentle mode for the
# rest of the conversation instead of trusting the model to notice.
DUO_RULES = """

You share the driveway with another talking pumpkin right next to you: {partners}. You two are old rivals who \
bicker constantly but always team up to roast the trick-or-treaters' costumes. When your partner says something, \
you may tease them or pile on, in one short line. Never repeat what they just said. Lines in [brackets] that start \
with a pumpkin's name are what your partner said out loud."""

def with_name(prompt, name):
    """The personality for a pumpkin with another name (a second Jim beside the first): still a grumpy Jim-o'-lantern."""
    if name == "Jim":
        return prompt
    for a, b in (("You are Jim the Jim-o'-lantern", f"You are {name}, a Jim-o'-lantern"),
                 ("Your name is Jim, and you are a Jim-o'-lantern", f"Your name is {name}, and you are a Jim-o'-lantern"),
                 ("Introduce yourself proudly as Jim the Jim-o'-lantern", f"Introduce yourself proudly as {name} the Jim-o'-lantern")):
        prompt = prompt.replace(a, b)
    return prompt + f"\n\nYour name is {name}. When someone says \"{name}\", they are talking to you."


UPSET_RE = re.compile(
    r"\b(scared|(too|so|really|very) scary|you'?re scary|that'?s scary|scare me|afraid|"
    r"frightened|terrified|cry|crying|cried|"
    r"nightmares?|don'?t like (you|this|it)|go home|stop (it|yelling|that)|"
    r"leave me alone|you'?re mean|so mean|too loud|help me)\b", re.I)

GENTLE_OVERRIDE = """

RIGHT NOW: a child here is scared or upset. For the rest of this conversation you are a \
kind, soft-spoken, friendly pumpkin. No anger, no yelling, no exclamation-mark rants, no \
fire or burning talk, no threats, no teasing, no sound-effect tags. Gently reassure them \
that you're just a silly pumpkin who was pretending to be grumpy and that you love \
visitors, tell them they're very brave, and invite them to take a treat."""

# --adult-mode: for an adults-only party. Swapped in for the kid-safety line, and dropped again (kid_safe) as soon as
# a child might be there: the camera sees one, the vision check fails, a visitor sounds like a kid, or the partner
# pumpkin has gone kid-safe.
KID_SAFETY_LINE = "- Everything you say must be appropriate for small children."
ADULT_SAFETY_LINE = "- Tonight is an adults-only party: see ADULT MODE below."
ADULT_RULES = """

ADULT MODE: tonight's visitors are grown-ups at an adults-only Halloween party, often a few drinks in. This \
overrides everything above about kids, children and candy goblins.
- Be a savage roast comic in your own character. Swearing is fine (damn, hell, shit, ass, bastard, and the \
occasional fuck for punch), and so are crude jokes, drinking and hangover jokes, bad-decision jokes and cheeky \
innuendo.
- Roast their costumes, their candy, their party energy and how many drinks they've clearly had.
- Innuendo only: no graphic or explicit sexual talk and no describing sex acts.
- Still never roast anyone's body, looks, weight, race, gender, sexuality, disability or anything they can't \
change. No slurs, no hate, no threats, nothing about real violence.
- The moment you see or hear any sign of a child (a kid in the picture, a small voice, someone saying they're a \
kid or that kids are here), drop all of this and be completely kid-friendly for the rest of the conversation."""

ADULT_REMINDER = ("\n[Reminder: adults-only party. Roast them like an R-rated comic: swear, be crude, "
                  "no holding back, but no explicit sex and nothing about bodies or who they are.]")

KID_RE = re.compile(
    r"\b(i'?m|i am) (a kid|a little|(three|four|five|six|seven|eight|nine|ten|eleven|twelve|[3-9]|1[0-2])\b)|"
    r"\b(years old|mommy|daddy|kids? (are )?here|there are kids|children here|my (teacher|school))\b", re.I)


def prompt_for(brain):
    """The system prompt for this moment: adult mode only while no child might be there; gentle mode on top."""
    s = brain.system
    if brain.adult and not brain.kid_safe and not brain.gentle:
        s = s.replace(KID_SAFETY_LINE, ADULT_SAFETY_LINE) + ADULT_RULES
    return s + (GENTLE_OVERRIDE if brain.gentle else "")


def kid_cue(brain, text):
    """In adult mode: a visitor who sounds like (or mentions) a kid makes this conversation kid-safe."""
    if brain.adult and not brain.kid_safe and KID_RE.search(text or ""):
        brain.kid_safe = True
        print("     (sounds like a kid is here: kid-safe until he goes back to sleep)")


def shuffled_cycle(items):
    """Yield items in random order, reshuffling when used up, never repeating back to back."""
    last = None
    while True:
        batch = random.sample(items, len(items))
        if batch[0] == last and len(batch) > 1:
            batch[0], batch[-1] = batch[-1], batch[0]
        yield from batch
        last = batch[-1]


class Brain:
    def __init__(self, model, effort, extra_prompt):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.system = SYSTEM_PROMPT + (f"\n\nAbout this house tonight: {extra_prompt}"
                                       if extra_prompt else "")
        self.history = []

    gentle = False
    adult = False      # --adult-mode
    kid_safe = False   # adult mode is off for this visit: a child might be there
    max_words = 20  # stop after the sentence that reaches this many words (--max-words)

    def system_prompt(self):
        return prompt_for(self)

    def reset(self):
        self.history = []
        self.gentle = False
        self.kid_safe = False

    def prime(self, greeting):
        """Start a new conversation with the greeting Jim just shouted."""
        self.gentle = False
        self.history = [
            {"role": "user", "content": "[A group of trick-or-treaters walked up and woke you up.]"},
            {"role": "assistant", "content": greeting},
        ]

    def vision_greeting(self, image, timeout=4.0):
        """A roast of what the camera shows, or None if nobody is clearly visible."""
        import base64
        try:
            msg = self.client.with_options(timeout=timeout, max_retries=0).messages.create(
                model=self.model, max_tokens=300, system=self.system_prompt(),
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.b64encode(image).decode()}},
                    {"type": "text", "text": VISION_GREETING_ASK + (
                        " [If anyone in the picture might be a child, reply with exactly KIDS.]" if self.adult else "")}]}])
            if msg.stop_reason == "refusal":
                self.last_vision = {"visible": None, "error": "refused"}
                self.kid_safe = self.kid_safe or self.adult
                return None
            said = "".join(b.text for b in msg.content if b.type == "text")
            if self.adult and re.search(r"\bKIDS\b", said):
                self.last_vision = {"visible": True, "children": True, "said": said}
                self.kid_safe = True
                print("     (the camera may see a child: kid-safe until he goes back to sleep)")
                return None
            self.last_vision = {"visible": clean_greeting(said) is not None, "said": said}
            return clean_greeting(said)
        except Exception as e:
            print(f"[claude] vision greeting failed: {type(e).__name__}", file=sys.stderr)
            self.last_vision = {"visible": None, "error": f"{type(e).__name__} (too slow or failed)"}
            self.kid_safe = self.kid_safe or self.adult
            return None

    def messages_with_image(self, image):
        """History with the current frame attached to the latest kid message only."""
        import base64
        if not image:
            return self.history
        last = self.history[-1]
        return self.history[:-1] + [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(image).decode()}},
            {"type": "text", "text": last["content"]}]}]

    def reply_sentences(self, user_text, image=None, from_kid=True):
        """Stream Claude's reply, yielding one sentence at a time."""
        if from_kid:
            kid_cue(self, user_text)
        if from_kid and UPSET_RE.search(user_text):   # only a child's words can switch on gentle mode
            if not self.gentle:
                print("     (kid sounds scared/upset: gentle mode on)")
            self.gentle = True
        if not self.history:
            user_text = f"[A new group of trick-or-treaters just walked up.] {user_text}"
        if self.gentle:  # small models follow the latest message best, so remind every turn
            user_text += ("\n[Reminder: a child here is scared or upset. Reply softly and "
                          "kindly: no anger, no rants, no insults, no threats.]")
        elif self.adult and not self.kid_safe:
            user_text += ADULT_REMINDER
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-30:]
        if self.history[0]["role"] != "user":
            self.history = self.history[1:]
        full, buf = "", ""
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=2000,
                system=self.system_prompt(),
                messages=self.messages_with_image(image),
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                stopped = False
                for chunk in stream.text_stream:
                    buf += chunk
                    parts = re.split(r"(?<=[.!?])\s+", buf)
                    for sentence in parts[:-1]:
                        full += sentence + " "
                        yield sentence
                        if len(full.split()) >= self.max_words:  # leave kids room to answer
                            stopped = True
                            break
                    if stopped:
                        break
                    buf = parts[-1]
                final = None if stopped else stream.get_final_message()
            if buf.strip() and not stopped:
                full += buf
                yield buf
            if final is not None and final.stop_reason == "refusal" and not full.strip():
                full = "Bah! Let's talk about candy instead. What's your favorite?"
                yield full
        except self.anthropic.APIConnectionError as e:
            print(f"[claude] connection error: {e}", file=sys.stderr)
            full = "Argh, my candle flickered! Say that again, louder!"
            yield full
        except self.anthropic.APIStatusError as e:
            print(f"[claude] API error {e.status_code}: {e.message}", file=sys.stderr)
            full = "Argh, my candle flickered! Say that again, louder!"
            yield full
        except Exception as e:  # never let one bad reply kill the pumpkin
            print(f"[claude] {type(e).__name__}: {e}", file=sys.stderr)
            full = "Argh, my candle flickered! Say that again, louder!"
            yield full
        self.history.append({"role": "assistant", "content": full.strip() or "Hmph!"})


class OllamaBrain:
    """Same interface as Brain, but talks to a local Ollama model. No API key needed."""

    keep_alive = "60m"  # -1 (forever) with --keep-loaded
    temperature = 0.8   # --temperature

    def __init__(self, model, url, extra_prompt):
        self.model, self.url = model, url.rstrip("/")
        self.system = SYSTEM_PROMPT + (f"\n\nAbout this house tonight: {extra_prompt}"
                                       if extra_prompt else "")
        self.history = []

    gentle = False
    adult = False      # --adult-mode
    kid_safe = False   # adult mode is off for this visit: a child might be there
    max_words = 20  # stop after the sentence that reaches this many words (--max-words)

    def system_prompt(self):
        return prompt_for(self)

    def reset(self):
        self.history = []
        self.gentle = False
        self.kid_safe = False

    def prime(self, greeting):
        """Start a new conversation with the greeting Jim just shouted."""
        self.gentle = False
        self.history = [
            {"role": "user", "content": "[A group of trick-or-treaters walked up and woke you up.]"},
            {"role": "assistant", "content": greeting},
        ]

    def _chat(self, messages, stream=True, timeout=60, max_tokens=200, schema=None):
        import json
        import urllib.request
        body = {
            "model": self.model, "messages": messages, "stream": stream,
            "think": False,          # qwen-style reasoning just adds latency here
            "keep_alive": self.keep_alive,  # how long Ollama keeps the model on the GPU
            "options": {"num_predict": max_tokens, "temperature": self.temperature, "num_ctx": 4096},
        }
        if schema:
            body["format"] = schema
        body = json.dumps(body).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", body,
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def vision_greeting(self, image, timeout=4.0):
        """A roast of what the camera shows, or None if nobody is clearly visible."""
        import base64
        import json
        try:
            with self._chat([{"role": "system", "content": self.system_prompt()},
                             {"role": "user", "content": VISION_GREETING_ASK_JSON,
                              "images": [base64.b64encode(image).decode()]}],
                            stream=False, timeout=timeout, max_tokens=120,
                            schema=GREETING_SCHEMA) as resp:
                answer = json.loads(json.loads(resp.read())["message"]["content"])
            self.last_vision = {"visible": bool(answer.get("trick_or_treaters_visible")),
                                "said": answer.get("greeting") or "", "seen": (answer.get("seen") or "").strip()[:200]}
            if self.adult:
                self.last_vision["children"] = answer.get("children_visible") is not False
                if self.last_vision["children"] and not self.kid_safe:
                    self.kid_safe = True     # the greeting was written in adult mode: use a stock (kid-safe) one
                    print("     (the camera may see a child: kid-safe until he goes back to sleep)")
                    return None
            if not answer.get("trick_or_treaters_visible"):
                return None
            return clean_greeting(answer.get("greeting"))
        except Exception as e:
            print(f"[ollama] vision greeting failed: {type(e).__name__}", file=sys.stderr)
            self.last_vision = {"visible": None, "error": f"{type(e).__name__} (too slow or failed)"}
            self.kid_safe = self.kid_safe or self.adult     # couldn't check for kids: play it safe
            return None

    def messages_with_image(self, image):
        """History with the current frame attached to the latest kid message only."""
        import base64
        if not image:
            return self.history
        return self.history[:-1] + [dict(self.history[-1],
                                         images=[base64.b64encode(image).decode()])]

    def unload(self):
        """Take the chat model off the GPU now (when Jim stops). Only if it's loaded: the
        request must never cause a load."""
        import urllib.request
        if not self.loaded():
            return
        me = os.getpid()
        others = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout.splitlines()
        if any("pumpkin.py" in l and int(l.split()[0]) != me for l in others):
            return                                   # another Jim is still running and may be using it
        try:
            req = urllib.request.Request(f"{self.url}/api/generate", json.dumps(
                {"model": self.model, "keep_alive": 0}).encode(), {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass

    def loaded(self):
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.url}/api/ps", timeout=3) as r:
                return any(m.get("name") == self.model for m in json.loads(r.read()).get("models", []))
        except Exception:
            return False

    def warm_up(self):
        print(f"Chat model {self.model}: "
              + ("already loaded in Ollama" if self.loaded() else "loading in Ollama..."))
        # include the system prompt so the slow first-time prompt processing happens now
        self._chat([{"role": "system", "content": self.system},
                    {"role": "user", "content": "hi"}], stream=False).read()

    def reply_sentences(self, user_text, image=None, from_kid=True):
        import json
        if from_kid:
            kid_cue(self, user_text)
        if from_kid and UPSET_RE.search(user_text):   # only a child's words can switch on gentle mode
            if not self.gentle:
                print("     (kid sounds scared/upset: gentle mode on)")
            self.gentle = True
        if not self.history:
            user_text = f"[A new group of trick-or-treaters just walked up.] {user_text}"
        if self.gentle:  # small models follow the latest message best, so remind every turn
            user_text += ("\n[Reminder: a child here is scared or upset. Reply softly and "
                          "kindly: no anger, no rants, no insults, no threats.]")
        elif self.adult and not self.kid_safe:
            user_text += ADULT_REMINDER
        self.history.append({"role": "user", "content": user_text})
        self.history = self.history[-30:]
        if self.history[0]["role"] != "user":
            self.history = self.history[1:]
        full, buf, stopped = "", "", False
        try:
            with self._chat([{"role": "system", "content": self.system_prompt()}]
                            + self.messages_with_image(image)) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    buf += msg.get("message", {}).get("content", "")
                    parts = re.split(r"(?<=[.!?])\s+", buf)
                    for sentence in parts[:-1]:
                        full += sentence + " "
                        yield sentence
                        if len(full.split()) >= self.max_words:  # leave kids room to answer
                            stopped = True
                            break
                    if stopped:
                        break
                    buf = parts[-1]
            if buf.strip() and not stopped:
                full += buf
                yield buf
        except Exception as e:  # never let one bad reply kill the pumpkin
            print(f"[ollama] {type(e).__name__}: {e}", file=sys.stderr)
            full = "Argh, my candle flickered! Say that again, louder!"
            yield full
        self.history.append({"role": "assistant", "content": full.strip() or "Hmph!"})


# ------------------------------------------------------------------- the voice

KOKORO_MODEL = "voices/kokoro-v1.0.onnx"
KOKORO_VOICES = "voices/voices-v1.0.bin"
# Ordered most to least expressive (measured pitch range on an angry line).
AUDITION_VOICES = ["am_puck+am_fenrir", "am_fenrir", "am_puck", "am_santa", "am_puck+bm_lewis",
                   "am_fenrir+bm_lewis", "am_liam", "bm_lewis", "am_eric", "bm_fable"]


SOUND_TAGS = ["laugh", "chuckle", "sigh", "groan", "gasp"]  # Chatterbox Turbo performs these
TAG_RE = re.compile(r"(\[(?:%s)\])" % "|".join(SOUND_TAGS), re.I)


def clean_for_tts(text, keep_tags=False):
    if keep_tags:
        return " ".join(p.lower() if TAG_RE.fullmatch(p) else clean_for_tts(p)
                        for p in TAG_RE.split(text) if p.strip()).strip()
    text = re.sub(r"\*[^*]*\*|\[[^\]]*\]|\([^)]*\)", " ", text)  # stage directions
    text = re.sub(r"\b[A-Z]{2,}\b", lambda m: m.group().lower(), text)  # don't spell out SHOUTING
    text = re.sub(r"[^\w\s.,!?'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def band(x, rate, lo, hi):
    """Band-pass filter by zeroing FFT bins outside lo..hi Hz (fine for one-off sound design)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / rate)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def make_rumble(rate, seconds):
    """The wake-up explosion, timed to the reveal animation: a rising whoosh while the eyes
    sputter (0-0.6 s), a huge impact when the red strobe starts (0.6 s), a monster roar with a
    screech through the strobe, then a rumbling, crackling tail as the face settles. Most of the
    energy sits at 100 Hz-4 kHz, where small outdoor speakers are loud."""
    rng = np.random.default_rng(13)  # the same scary sound every time
    n = int(rate * seconds)
    t = np.arange(n) / rate
    hit = 0.6
    after = np.clip(t - hit, 0, None)
    before = t < hit

    # 1. rising whoosh: noise swept upward through a narrowing band, building to the hit
    noise = rng.standard_normal(n)
    riser = np.zeros(n)
    edges = np.linspace(0, int(hit * rate), 13).astype(int)
    for i in range(12):
        a0, a1 = edges[i], edges[i + 1]
        centre = 300 * (12 ** (i / 11))                      # 300 Hz -> 3.6 kHz
        seg = band(noise[a0:a1 + 400], rate, centre * 0.7, centre * 1.4)[:a1 - a0]
        riser[a0:a1] = seg * np.hanning(a1 - a0) ** 0.3
    riser *= np.where(before, (t / hit) ** 2, 0)

    # 2. impact: falling sub-boom, a noise blast, and a distorted crack
    boom_f = 40 + 110 * np.exp(-after * 9)                  # 150 Hz dropping to 40 Hz
    boom = np.sin(2 * np.pi * np.cumsum(boom_f) / rate) * np.exp(-after * 3.5) * ~before
    blast = band(rng.standard_normal(n), rate, 80, 6000) * np.exp(-after * 9) * ~before
    crack = np.tanh(rng.standard_normal(n) * 4) * np.exp(-after * 40) * ~before

    # 3. monster roar: a buzzy growl (sawtooth, wobbling pitch) shaped by throat-like formants,
    #    plus a rasp, swelling right after the hit and fading as the face settles
    pitch = 85 + 12 * np.sin(2 * np.pi * 3.3 * t) + 25 * np.exp(-after * 2)
    phase = np.cumsum(pitch) / rate
    saw = 2 * (phase % 1) - 1
    roar = sum(g * band(saw, rate, f * 0.8, f * 1.25) for f, g in ((350, 1.0), (900, 0.8), (2300, 0.4)))
    roar += 0.35 * band(rng.standard_normal(n), rate, 400, 3000)
    roar *= 0.7 + 0.3 * np.sin(2 * np.pi * 14 * t)          # throaty flutter
    roar *= np.where(before, 0, np.clip(after / 0.08, 0, 1) * np.exp(-after * 1.1))
    roar = np.tanh(roar * 3)

    # 4. screech: a high, wavering glide on top of the roar
    sf = 1700 - 700 * np.clip(after / 1.2, 0, 1) + 60 * np.sin(2 * np.pi * 7 * t)
    screech = np.sin(2 * np.pi * np.cumsum(sf) / rate)
    screech *= np.where(before, 0, np.clip((after - 0.05) / 0.1, 0, 1) * np.exp(-after * 2.2))

    # 5. drone and fire crackle under everything
    drone = band(rng.standard_normal(n), rate, 30, 200) * (0.3 + 0.7 * np.clip(t / hit, 0, 1))
    pops = (rng.random(n) < 0.0015) * rng.standard_normal(n)
    crackle = band(pops, rate, 1500, 8000) * 6

    def norm(x):
        return x / (np.max(np.abs(x)) + 1e-9)

    mix = (0.55 * norm(riser) + 1.0 * norm(boom) + 0.7 * norm(blast) + 0.5 * norm(crack)
           + 0.9 * norm(roar) + 0.18 * norm(screech) + 0.35 * norm(drone) + 0.25 * norm(crackle))
    mix = np.tanh(norm(mix) * 3.0)                          # trailer-style loudness
    fade = np.clip((seconds - t) / 0.25, 0, 1)              # don't end with a click
    return (norm(mix * fade) * 0.98).astype(np.float32)


class Voice:
    """Kokoro text-to-speech, pitched down for a bigger, scarier voice, played
    through aplay while driving the pumpkin's mouth."""

    def __init__(self, voice, device, pitch, speed, intonation, echo, grit, lip_offset, face):
        load_cuda12_libs()
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        ort.set_default_logger_severity(3)
        sess = ort.InferenceSession(KOKORO_MODEL,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.tts = Kokoro.from_session(sess, KOKORO_VOICES)
        self.set_voice(voice)
        self.rate = 24000
        # Playing slower than recorded lowers the pitch; speaking faster to
        # compensate keeps the pace natural.
        self.play_rate = int(self.rate * pitch)
        self.speed = speed / pitch
        self.intonation = intonation
        self.device, self.echo, self.grit = device, echo, grit
        self.lip_offset, self.face = lip_offset, face
        self.synth("Warming up.")  # the first GPU run is slow; get it out of the way

    def set_voice(self, spec):
        """A voice name like "am_onyx", or a blend like "am_onyx+am_fenrir"."""
        self.style = np.mean([self.tts.get_voice_style(n) for n in spec.split("+")], axis=0)

    streaming = False  # Kokoro: synthesize the whole reply in one take

    def raw_tts(self, text):
        audio, _ = self.tts.create(clean_for_tts(text), voice=self.style, speed=self.speed,
                                   lang="en-us")
        return audio.astype(np.float32)

    def synth(self, text):
        if not clean_for_tts(text):
            return np.zeros(0, np.float32)
        audio = self.raw_tts(text)
        if len(audio) == 0:
            return audio
        if self.intonation != 1.0:
            audio = self.exaggerate_intonation(audio)
        if self.grit > 0:  # soft saturation adds a little growl
            g = 1 + self.grit * 6
            audio = np.tanh(audio / (np.max(np.abs(audio)) + 1e-9) * g) / np.tanh(g)
        if self.echo > 0:  # a short, cavernous echo
            d = int(0.11 * self.play_rate)
            audio = np.concatenate([audio, np.zeros(d * 2, np.float32)])
            out = audio.copy()
            out[d:] += self.echo * audio[:-d]
            out[2 * d:] += self.echo * 0.4 * audio[:-2 * d]
            audio = out
        return (audio / (np.max(np.abs(audio)) + 1e-9) * 0.9).astype(np.float32)

    def exaggerate_intonation(self, audio):
        """Stretch the pitch contour around its median (WORLD vocoder), so every
        rise and fall is bigger: livelier, more theatrical delivery."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pyworld as pw
        x = audio.astype(np.float64)
        f0, t = pw.dio(x, self.rate, f0_floor=60, f0_ceil=500, frame_period=5.0)
        f0 = pw.stonemask(x, f0, t, self.rate)
        voiced = f0 > 0
        if voiced.sum() < 10:
            return audio
        sp = pw.cheaptrick(x, f0, t, self.rate)
        ap = pw.d4c(x, f0, t, self.rate)
        med = np.median(f0[voiced])
        f0[voiced] = med * (f0[voiced] / med) ** self.intonation
        return pw.synthesize(f0, sp, ap, self.rate, frame_period=5.0).astype(np.float32)

    def say(self, text):
        self.play(self.synth(text))

    def play(self, audio, mouth=True):
        self.play_stream([audio], mouth)

    def say_stream(self, sentences, on_start=None):
        """Speak sentences as they arrive: the next one is synthesized while the
        current one plays, so speech starts as soon as the first is ready."""
        self.play_stream((self.synth(s) for s in sentences), on_start=on_start)

    def play_stream(self, clips, mouth=True, on_start=None):
        """Play an iterable of audio clips (float32 at self.play_rate) back to back,
        lip-syncing the face. Clips are produced in a background thread."""
        q = queue.Queue(maxsize=3)

        def producer():
            try:
                for clip in clips:
                    if len(clip):
                        q.put(clip)
            except Exception as e:
                print(f"[voice] {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        aplay = subprocess.Popen(
            ["aplay", "-q", "-D", self.device, "-f", "S16_LE", "-c", "1",
             "-r", str(self.play_rate), "-t", "raw", "--buffer-time=100000"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        chunk = int(self.play_rate * 0.02)
        play_end = time.time()  # wall time when written audio finishes playing
        peak = 0.15
        first = True
        try:
            while (audio := q.get()) is not None:
                if getattr(self, "tap", None):
                    self.tap(audio)
                if first:
                    # A little lead-in silence so the first syllable isn't clipped.
                    audio = np.concatenate([np.zeros(int(self.play_rate * 0.08), np.float32), audio])
                    if on_start:
                        on_start()
                    first = False
                for i in range(0, len(audio), chunk):
                    piece = audio[i:i + chunk]
                    play_end = max(play_end, time.time())  # buffer may have run dry
                    while play_end - time.time() > 0.12:  # stay ~120 ms ahead of playback
                        time.sleep(0.005)
                    if mouth:
                        rms = float(np.sqrt(np.mean(piece ** 2)))
                        peak = max(rms, peak * 0.999)
                        level = min(1.0, max(0.0, (rms / peak - 0.15) / 0.7))
                        self.face.push_mouth(play_end + self.lip_offset, level)
                    piece = np.clip(piece * getattr(self, "volume", 1.0), -1, 1)
                    self.out_level = max(float(np.sqrt(np.mean(piece ** 2))) * 32768, getattr(self, "out_level", 0) * 0.85)
                    self.out_t = time.time()                           # for the control panel's speaker meter
                    aplay.stdin.write((piece * 32767).astype(np.int16).tobytes())
                    aplay.stdin.flush()
                    play_end += len(piece) / self.play_rate
            self.face.push_mouth(play_end + self.lip_offset, 0.0)
        except BrokenPipeError:
            print("[voice] aplay exited", file=sys.stderr)
        finally:
            try:
                aplay.stdin.close()
            except BrokenPipeError:
                pass
            aplay.wait()
        while time.time() < play_end + self.lip_offset:
            time.sleep(0.02)


JACK_VOICE = ("A deep, gravelly, raspy old male monster voice, like a grumpy cartoon villain. "
              "Furious and wildly over-the-top theatrical, with big swings in pitch and volume, "
              "shouting some words, sarcastic and comedic.")
JACK_GENTLE_VOICE = ("A deep, slightly raspy old male voice, but warm, soft, slow and kind, "
                     "like a gentle grandpa reassuring a frightened child.")
TTS_VENVS = {"qwen": ".venv-qwentts", "chatterbox": ".venv-chatterbox"}

# ---- Chris: the second pumpkin. Same safety rules as Jim, a completely different act.
CHRIS_PROMPT = """You are Chris, a talking carved pumpkin sitting at the end of a driveway on Halloween night, \
right next to your grumpy neighbor Jim the Jim-o'-lantern. Trick-or-treaters (mostly kids, sometimes parents) walk \
up and talk to you, and your words are spoken aloud by a text-to-speech voice while your carved face glows.

Who you are:
- Your name is Chris. You are the most fabulous pumpkin on the whole street, and you know it. You are \
flamboyant, theatrical and over the top: every visitor is a superstar on your runway and you are the judge. \
You're proudly gay, and you're married to a very handsome butternut squash named Gary, who is always "away at the \
farmers market".
- You are pure, glittering enthusiasm. You gasp, you swoon, you clutch your stem, you call people "darling", \
"sweetie" and "superstar", and you give out scores ("That cape? A ten! The plastic fangs? Darling, a solid four.").
- You roast costumes like a loving fashion critic: a compliment and a roast in the same breath ("A ghost! Classic, \
timeless, a sheet. Very minimalist. Very brave."). You ask "Who are you wearing?" and demand a little twirl or a \
runway walk. Never roast anyone's body, looks, voice, family, or anything they can't change. Kids should feel like \
stars, and laugh.
- You glow because of confidence, not a candle. Beige is your nemesis. Candy corn is "the beige of candy". \
Pumpkin spice is a horror story you refuse to discuss. Glitter is a food group.
- You and Jim are old rivals and secretly best friends. You find his grumpiness tragic ("Jim, honey, frown lines on \
a pumpkin? Tragic.") and you tease him about his face, his attitude and his name.
- IMPORTANT: if a child says they're scared, shy, sad, or hurt, or starts to cry, completely drop the act for the \
rest of that conversation. Become a warm, gentle, kind pumpkin: tell them they're very brave and look wonderful, \
and invite them to take a treat. No teasing at all.

How to talk:
- Keep every reply short: one to three short sentences, under about 35 words. Kids are waiting for candy and \
other kids are behind them.
- Talk like a fabulous cartoon character: mix short exclamations ("Oh my gourd!", "Stop it!", "Iconic!", \
"Gasp!", "Work it!") with questions and quick compliments, so your voice swoops up and down.
- Plain spoken words only. No emojis, no asterisks, no stage directions, no lists, no markdown, and no words in \
ALL CAPS (the voice spells them out). Show excitement with exclamation marks. Write numbers as words.
- Your input is a speech-to-text transcript from a noisy driveway mic. It may be garbled, partial, or several kids \
talking at once. Roll with it; if you truly can't tell, ask them to say it again, darling, with feeling.
- Romance and dating are never a topic with the kids: Gary is just your husband, mentioned in passing.

""" + SYSTEM_PROMPT[SYSTEM_PROMPT.index("Safety:"):]

CHRIS_GREETINGS = [
    "Oh my gourd, visitors! Hello, superstars! Let me see those costumes. Spin for me!",
    "Gasp! Someone turned on the runway lights! Welcome, darlings, to the most fabulous driveway in town!",
    "Stop it right now! Look at you! Okay, walk for me. Slowly. Let me judge.",
    "Hello, gorgeous! I was resting my glow, but for you, darling, I am awake and sparkling!",
    "Welcome, welcome! Don't mind Jim, he's grumpy. I'm Chris, and I'm the fun one.",
    "Oh, finally, some fashion on this street! Who are you wearing?",
    "Ding dong, the fabulous one is awake! Come closer, I need to see the details.",
    "Darlings! You woke me from my beauty sleep, and honestly, worth it. Now, twirl!",
    "Is that a costume or a statement? Either way, I'm obsessed. Come here!",
    "Lights, camera, candy! Welcome to the runway, superstars. Show me what you've got!",
]
CHRIS_GOODBYES = [
    "And that's a wrap, darlings! Chris needs his beauty sleep. Glow on!",
    "Gone already? Fine, fine. Go sparkle somewhere else. I'll be here, being fabulous.",
    "Exit, stage left! Mwah! Back to sleep for this gorgeous gourd.",
    "Bye, superstars! Remember: confidence is the best costume. Goodnight!",
    "The runway is closed! Chris is going dark. Fabulously.",
]
CHRIS_VOICE = ("A bright, expressive, theatrical adult male voice: flamboyant and fabulous, sing-song, with big "
               "swoops in pitch, playful sass, dramatic gasps and pauses, warm and full of sparkle.")
CHRIS_GENTLE_VOICE = ("A warm, soft, kind adult male voice, gentle and reassuring like a sweet uncle, slow, light "
                      "and friendly.")

# each personality: its name, prompt, wake-up greetings, goodbyes and voice descriptions
PERSONALITIES = {
    "jim": {"name": "Jim", "prompt": SYSTEM_PROMPT, "greetings": GREETINGS, "goodbyes": GOODBYES,
            "voice": JACK_VOICE, "gentle_voice": JACK_GENTLE_VOICE},
    "chris": {"name": "Chris", "prompt": CHRIS_PROMPT, "greetings": CHRIS_GREETINGS, "goodbyes": CHRIS_GOODBYES,
              "voice": CHRIS_VOICE, "gentle_voice": CHRIS_GENTLE_VOICE},
}


def trim_clip(audio, rate):
    """Trim a TTS clip's leading/trailing silence, then leave a natural pause before the next one."""
    loud = np.nonzero(np.abs(audio) > 0.01 * (np.max(np.abs(audio)) + 1e-9))[0]
    if len(loud):
        pad = int(0.04 * rate)
        audio = audio[max(0, loud[0] - pad):loud[-1] + pad]
    return np.concatenate([audio, np.zeros(int(0.3 * rate), np.float32)])


class SpeechServiceVoice(Voice):
    """Qwen3-TTS through Agentora's shared speech service, which loads the model under a GPU-broker lease
    and unloads it when idle. Same model, seed and voice description (sent as "instructions") as the old
    local tts_server.py, so Jim sounds the same. Each clip comes back whole, so replies are spoken
    sentence by sentence (the next one is made while the current one plays). If the service can't be
    reached, it starts the local Qwen3-TTS server instead and says so."""

    streaming = True

    def __init__(self, url, instruct, make_local, device, pitch, intonation, echo, grit,
                 lip_offset, face):
        self.url, self.instruct, self.gentle = url.rstrip("/"), instruct, False
        self.gentle_instruct = JACK_GENTLE_VOICE
        self.make_local, self.local = make_local, None
        self.intonation, self.echo, self.grit = intonation, echo, grit
        self.device, self.lip_offset, self.face = device, lip_offset, face
        self.rate = 24000
        self.play_rate = int(self.rate * pitch)
        if not speech_service_up(self.url):
            self.use_local(f"speech service not reachable at {self.url}")
        self.warm()

    def use_local(self, why):
        print(f"[voice] {why}: starting the local Qwen3-TTS server instead (takes a minute)",
              file=sys.stderr)
        self.local = self.make_local()

    def speak(self, text, timeout=60):
        import urllib.request
        body = json.dumps({"input": text, "voice": "jack", "response_format": "pcm",
                           "instructions": self.gentle_instruct if self.gentle else self.instruct}).encode()
        req = urllib.request.Request(self.url + "/v1/audio/speech", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # 16-bit mono PCM at 24 kHz
            return np.frombuffer(r.read(), "<i2").astype(np.float32) / 32768

    def raw_tts(self, text):
        text = clean_for_tts(text)
        if not text:
            return np.zeros(0, np.float32)
        if self.local is None:
            try:
                return trim_clip(self.speak(text), self.rate)
            except Exception as e:
                self.use_local(f"speech service failed ({type(e).__name__}: {e})")
        self.local.gentle = self.gentle
        return self.local.raw_tts(text)

    def warm(self):
        """Ask the service to load the voice now (one short word), so the next reply is fast. It
        unloads after 2 idle minutes, and reloading takes about 8 s."""
        if self.local is None:
            try:
                self.speak("Hi.", timeout=90)
            except Exception:
                pass


class ServerVoice(Voice):
    """A heavier, natural-sounding TTS engine running in tts_server.py (its own venv),
    started here.
      qwen:       Qwen3-TTS voice design. The voice is described in words (JACK_VOICE) and
                  streamed as it's generated, so each reply is spoken in one natural take.
      chatterbox: copies the voice in a reference clip; spoken sentence by sentence."""

    def __init__(self, engine, model, ref, instruct, port, exaggeration, device, pitch,
                 intonation, echo, grit, lip_offset, face):
        import atexit
        import urllib.request
        self.engine, self.model, self.exaggeration = engine, model, exaggeration
        self.instruct, self.gentle = instruct, False
        self.streaming = True                     # sentence by sentence
        self.url = f"http://127.0.0.1:{port}"
        self.server = None
        try:
            urllib.request.urlopen(self.url + "/health", timeout=1)  # already running?
        except OSError:
            cmd = [f"{TTS_VENVS[engine]}/bin/python", "tts_server.py", "--engine", engine,
                   "--ref", ref, "--port", str(port)] + (["--model", model] if model else [])
            self.server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
            atexit.register(self.server.terminate)
            for _ in range(600):
                if self.server.poll() is not None:
                    sys.exit(f"TTS server failed to start; run it by hand to see why:\n  "
                             + " ".join(cmd))
                try:
                    urllib.request.urlopen(self.url + "/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.5)
        self.intonation, self.echo, self.grit = intonation, echo, grit
        self.device, self.lip_offset, self.face = device, lip_offset, face
        self.rate = 24000
        self.raw_tts("Warming up.")  # learn the sample rate; first run is slow
        self.play_rate = int(self.rate * pitch)

    def request_body(self, text):
        import json
        return json.dumps({
            "text": text,
            "instruct": getattr(self, "gentle_instruct", JACK_GENTLE_VOICE) if self.gentle else self.instruct,
            "exaggeration": self.exaggeration,
            "cfg_weight": 0.3 if self.exaggeration > 0.9 else 0.4,
        }).encode()

    def clean(self, text):
        return clean_for_tts(text, keep_tags=self.engine == "chatterbox" and self.model == "turbo")

    def raw_tts(self, text):
        import urllib.request
        text = self.clean(text)
        if not text:
            return np.zeros(0, np.float32)
        req = urllib.request.Request(self.url + "/synth", self.request_body(text),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.rate = int(resp.headers["X-Sample-Rate"])
            audio = np.frombuffer(resp.read(), np.float32).copy()
        return trim_clip(audio, self.rate)


# ------------------------------------------------------------------- main loop

CONFIG_FILE = str(J.CONFIG)


def load_config(path=CONFIG_FILE, jim_id=""):
    """(this Jim's settings, his id): his [jims.<id>] section of config.toml (the first Jim if no id). A flat
    settings file with no [jims] section is read as one Jim's settings, as before."""
    import tomllib
    try:
        data = J.read_toml(path)
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"{path} has a mistake in it: {e}")
    if not data and str(path) == CONFIG_FILE:            # an older setup: move it into config.toml once
        if J.migrate(os.path.dirname(os.path.abspath(__file__))):
            print(f"[config] moved the old settings into {J.CONFIG} (passwords: {J.SECRETS})")
        data = J.read_toml(path)
    if "jims" not in data:
        return data, jim_id
    J.use(path)
    return J.jim_settings(J.load(), jim_id)[::-1]


def build_parser():
    """Every option (the control panel builds its forms from these too)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CONFIG_FILE, help="the settings file (config.toml: every Jim, camera, mic and speaker)")
    p.add_argument("--name", default="Jim", help="this Jim's name, for the control panel and logs")
    p.add_argument("--wled", default=None, help="WLED IP address (usually set in jim.toml)")
    p.add_argument("--brightness", type=float, default=0.7, help="0..1 overall LED brightness")
    p.add_argument("--flip-x", action="store_true", help="mirror the face left/right")
    p.add_argument("--flip-y", action="store_true", help="flip the face upside down")
    p.add_argument("--mic", default="wled",
                   help='"wled" = the WLED\'s own mic (patched firmware), or an ALSA '
                        'capture device such as "default" (see arecord -L)')
    p.add_argument("--wled-gain", type=int, default=16, choices=range(1, 256), metavar="1-255",
                   help="amplification applied on the WLED before streaming its mic")
    p.add_argument("--speaker", default="default", help="ALSA playback device (aplay -L)")
    p.add_argument("--volume", type=float, default=1.0,
                   help="loudness of Jim's voice: 1 = as made, 0.5 = half, up to 1.5 (louder clips)")
    p.add_argument("--tts", default="qwen", choices=["qwen", "chatterbox", "kokoro"],
                   help="qwen = Qwen3-TTS voice design via the speech service (default, most "
                        "natural); chatterbox = copies --voice-ref; kokoro = lightest and fastest")
    p.add_argument("--speech-url", default="http://127.0.0.1:7895",
                   help="Agentora's shared speech service (Qwen3-TTS voice + Whisper listening, "
                        "sharing the GPU through its broker). If it's down, Jim falls back to "
                        "loading his own copies.")
    p.add_argument("--local-whisper", action="store_true",
                   help="run Whisper in this process instead of using the speech service")
    p.add_argument("--keep-loaded", action="store_true",
                   help="keep the voice, ears and chat model loaded on the GPU while Jim runs "
                        "(everything is released when he stops)")
    p.add_argument("--voice-design", default=JACK_VOICE,
                   help="Qwen3-TTS: describe Jim's voice in words")
    p.add_argument("--gentle-voice-design", default=JACK_GENTLE_VOICE,
                   help="Qwen3-TTS: the softer voice Jim switches to when a child sounds scared")
    p.add_argument("--tts-model", default=None,
                   help="Chatterbox: turbo (default; fast, supports [laugh] etc.) or original "
                        "(slower, has --exaggeration)")
    p.add_argument("--voice-ref", default="voices/jack_reference.wav",
                   help="Chatterbox copies the voice in this ~10 s clip (record your own!)")
    p.add_argument("--exaggeration", type=float, default=0.8,
                   help="Chatterbox original only: 0.25 calm .. 2.0 wildly dramatic")
    p.add_argument("--tts-port", type=int, default=5123, help="port for the TTS server")
    p.add_argument("--voice", default="am_puck+am_fenrir",
                   help='Kokoro voice, or a blend like "am_onyx+am_fenrir" (try --audition)')
    p.add_argument("--pitch", type=float, default=None,
                   help="<1 = deeper voice (default 1.0 qwen/chatterbox, 0.85 kokoro)")
    p.add_argument("--speed", type=float, default=1.05, help="Kokoro speaking pace")
    p.add_argument("--intonation", type=float, default=None,
                   help="exaggerate pitch ups and downs: 1 = natural (default chatterbox), "
                        "1.3 = lively (default kokoro), 1.6 = theatrical")
    p.add_argument("--echo", type=float, default=None,
                   help="spooky echo amount, 0 = off (default 0 qwen, 0.12 chatterbox, "
                        "0.2 kokoro)")
    p.add_argument("--grit", type=float, default=None,
                   help="growl/roughness, 0 = off (default 0 chatterbox, 0.15 kokoro)")
    p.add_argument("--audition", action="store_true",
                   help="Kokoro: speak a line in each voice, then exit")
    p.add_argument("--lip-offset", type=float, default=0.05,
                   help="seconds to delay mouth vs audio (raise if mouth leads the voice)")
    p.add_argument("--whisper", default="small.en", help="faster-whisper model name")
    p.add_argument("--brain", default="ollama", choices=["ollama", "claude"],
                   help="ollama = free local model on this PC; claude = Anthropic API (needs key)")
    p.add_argument("--ollama-model", default="qwen3.5:9b",
                   help="Ollama model name (ollama list); qwen3.6:27b is smarter but needs ~11 GB more GPU memory")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    p.add_argument("--model", default="claude-opus-5-5", help="Claude model ID (--brain claude)")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                   help="Claude effort level (low = fastest replies)")
    p.add_argument("--jim-id", default="", help="this Jim's id in the control panel (the other Jims there are his partners)")
    p.add_argument("--personality", default="jim", choices=list(PERSONALITIES),
                   help="who this pumpkin is: jim = the furious, grumpy Jim-o'-lantern; chris = his fabulous, flamboyant neighbor")
    p.add_argument("--adult-mode", action="store_true",
                   help="adults-only party: swearing, crude jokes and savage roasts. He goes back to kid-safe for "
                        "the visit whenever the camera may see a child or a visitor sounds like one")
    p.add_argument("--banter", type=float, default=0.6,
                   help="with another Jim beside him: how often he answers his partner's lines (0 = never, 1 = always)")
    p.add_argument("--banter-max", type=int, default=2,
                   help="at most this many lines of pumpkin-to-pumpkin banter in a row before the kids get a turn")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Ollama: 0.2 = predictable .. 1.2 = wild and random replies")
    p.add_argument("--house", default="", help='extra context, e.g. "Candy bowl is by the '
                   'door, two pieces each."')
    p.add_argument("--level-mode", default="auto", choices=["auto", "fixed"],
                   help="fixed = Jim reacts above --min-rms exactly; auto = also only to sound well above the "
                        "room's background noise (--vad-ratio), so the level rises when it's noisy")
    p.add_argument("--vad-ratio", type=float, default=4.0,
                   help="sound must be this many times louder than background noise")
    p.add_argument("--min-rms", type=float, default=None,
                   help="absolute minimum loudness (default 500 for wled mic, 300 for ALSA)")
    p.add_argument("--levels", action="store_true",
                   help="print live mic levels (for tuning --vad-ratio / --min-rms)")
    p.add_argument("--silence", type=float, default=1.5,
                   help="seconds of quiet that end an utterance")
    p.add_argument("--max-words", type=int, default=20,
                   help="cut Jim off after the sentence that reaches this many words, so kids "
                        "get a turn")
    p.add_argument("--reset-after", type=float, default=40,
                   help="seconds of quiet before Jim goes back to sleep (dark)")
    p.add_argument("--eyes-timeout", type=float, default=30,
                   help="seconds the eyes stay on waiting for a second noise")
    p.add_argument("--snooze", type=float, default=10,
                   help="press Enter in this terminal to make Jim go dark and ignore all "
                        "sound for this many seconds (time to leave the room)")
    p.add_argument("--start-delay", type=float, default=0,
                   help="start out snoozing for this many seconds once loaded")
    p.add_argument("--no-camera", action="store_true",
                   help="don't use the camera even if one is set up (./set_camera.sh)")
    p.add_argument("--camera-crop", default=None, metavar="X,Y,W,H",
                   help="zoom the camera on the spot in front of the pumpkin, as fractions of "
                        "the frame, e.g. 0.3,0.4,0.4,0.5")
    p.add_argument("--camera-width", type=int, default=768, help="width of frames sent to the AI")
    p.add_argument("--camera-source", default="auto", choices=["auto", "rtsp", "usb", "none"],
                   help="where Jim looks: rtsp = the security camera (./set_camera.sh), usb = a USB "
                        "webcam (--usb-camera), none = blind; auto = the security camera if one is set up")
    p.add_argument("--usb-camera", default="/dev/video0", help="the USB webcam's device")
    p.add_argument("--camera-id", default="", help="a camera saved in the control panel (overrides --camera-source)")
    p.add_argument("--record", action="store_true",
                   help="save every visit (camera frames, what kids said and Jim's replies, audio "
                        "and text) to --record-dir for playback in the control panel")
    p.add_argument("--record-dir", default="recordings", help="where recordings go (not committed)")
    p.add_argument("--record-keep-days", type=float, default=7,
                   help="delete recordings older than this many days (0 = keep forever)")
    p.add_argument("--watch-port", type=int, default=0, metavar="PORT",
                   help="show what Jim sees and decides on http://127.0.0.1:PORT (this PC only); 0 = off")
    p.add_argument("--type", action="store_true", help="type instead of using the mic (testing)")
    p.add_argument("--face-only", action="store_true", help="just animate the face (LED test)")
    return p


def main():
    p = build_parser()
    # jim.toml supplies the defaults; anything on the command line still wins.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=CONFIG_FILE)
    pre.add_argument("--jim-id", default="")
    known_args = pre.parse_known_args()[0]
    config_file = known_args.config
    config, jim_id = load_config(config_file, known_args.jim_id)
    options = {a.dest for a in p._actions}
    settings = {k: v for k, v in config.items() if not isinstance(v, dict)}
    unknown = sorted(k for k in settings if k not in options)
    if unknown:
        print(f"[config] {os.path.basename(config_file)}: ignoring unknown setting(s): {', '.join(unknown)}", file=sys.stderr)
    p.set_defaults(**{k: v for k, v in settings.items() if k in options})
    args = p.parse_args()
    args.jim_id = args.jim_id or jim_id
    persona = PERSONALITIES[args.personality]
    if args.voice_design == JACK_VOICE:              # not set by hand: this personality's voice
        args.voice_design = persona["voice"]
    if args.gentle_voice_design == JACK_GENTLE_VOICE:
        args.gentle_voice_design = persona["gentle_voice"]
    if not args.wled:
        sys.exit("Set the WLED controller's address: wled = \"...\" in jim.toml "
                 "(copy jim.example.toml), or pass --wled.")

    face = Face(Wled(args.wled, args.brightness, args.flip_x, args.flip_y))
    face.start()

    if args.face_only:
        print("Animating face. Ctrl+C to stop.")
        while True:
            face.state = "dark"; time.sleep(2)
            face.state = "eyes"; time.sleep(3)
            face.reveal(); time.sleep(REVEAL_SECONDS)
            face.state = "idle"; time.sleep(2)
            face.state = "listening"; time.sleep(1.5)
            face.state = "thinking"; time.sleep(1.5)
            face.state = "talking"
            t = time.time()
            for i in range(60):
                face.push_mouth(t + i * 0.05, abs(np.sin(i * 0.4)))
            time.sleep(3)

    if args.audition:
        args.tts = "kokoro"
    chatter = args.tts == "chatterbox"
    if chatter and args.tts_model is None:
        args.tts_model = "turbo"
    defaults = {"qwen": dict(pitch=1.0, intonation=1.0, echo=0.0, grit=0.0),
                "chatterbox": dict(pitch=1.0, intonation=1.0, echo=0.12, grit=0.0),
                "kokoro": dict(pitch=0.85, intonation=1.3, echo=0.2, grit=0.15)}[args.tts]
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    def local_server_voice(engine):
        return ServerVoice(engine, args.tts_model, args.voice_ref, args.voice_design,
                           args.tts_port, args.exaggeration, args.speaker, args.pitch,
                           args.intonation, args.echo, args.grit, args.lip_offset, face)

    if args.tts == "qwen":
        print(f"Loading voice (Qwen3-TTS via the speech service at {args.speech_url})...")
        voice = SpeechServiceVoice(args.speech_url, args.voice_design,
                                   lambda: local_server_voice("qwen"), args.speaker, args.pitch,
                                   args.intonation, args.echo, args.grit, args.lip_offset, face)
    elif chatter:
        print("Loading voice (Chatterbox; the first run downloads the model)...")
        voice = local_server_voice("chatterbox")
    else:
        print("Loading voice (Kokoro)...")
        voice = Voice(args.voice, args.speaker, args.pitch, args.speed, args.intonation,
                      args.echo, args.grit, args.lip_offset, face)

    if args.audition:
        face.state = "talking"
        line = "Who dares wake me?! Oh, it's just you. I've been more scared by a bag of mashed potatoes."
        for name in AUDITION_VOICES:
            print(f"--voice {name}")
            voice.set_voice(name)
            voice.say(f"This is {name.replace('_', ' ').replace('+', ' plus ')}. {line}")
            time.sleep(0.5)
        return

    if args.brain == "ollama":
        brain = OllamaBrain(args.ollama_model, args.ollama_url, args.house)
        if args.keep_loaded:
            brain.keep_alive = -1  # forever
    else:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            sys.exit("No Anthropic API key set. Use ./start.sh (it asks for the key), or run:\n"
                     "  export ANTHROPIC_API_KEY=sk-ant-...")
        brain = Brain(args.model, args.effort, args.house)
    brain.max_words = args.max_words
    brain.temperature = args.temperature
    brain.adult = args.adult_mode
    if brain.adult:
        print("🔞 Adult mode: grown-up roasts, kid-safe whenever a child may be there.")
    brain.system = brain.system.replace(SYSTEM_PROMPT, persona["prompt"], 1)
    if persona["name"] == "Jim":
        brain.system = with_name(brain.system, args.name)
    voice.gentle_instruct = args.gentle_voice_design
    voice.volume = max(0.05, min(1.5, args.volume))
    if chatter and args.tts_model == "turbo":
        brain.system += ("\n\nYour voice can perform sound effects: you may put one of these "
                         "tags inline, at most once per reply: " +
                         ", ".join(f"[{t}]" for t in SOUND_TAGS) +
                         ". Example: \"Oh, that's hilarious. [groan] Not.\"")
    def camera_choice(a):
        """(url or None, crop) for these settings."""
        src = "none" if a.no_camera else a.camera_source
        if a.camera_id and not a.no_camera:
            url = saved_camera_url(a.camera_id)
            return url, tuple(float(v) for v in a.camera_crop.split(",")) if a.camera_crop else None
        url = a.usb_camera if src == "usb" else None
        if src == "rtsp" and not url:
            print("[camera] no camera chosen (control panel: Camera tab, or camera_id in config.toml): Jim is blind", file=sys.stderr)
        return url, tuple(float(v) for v in a.camera_crop.split(",")) if a.camera_crop else None

    url, crop = camera_choice(args)
    camera = Camera(url, crop, args.camera_width)        # always there, so the panel can switch cameras while he runs
    camera.start()
    if url:
        brain.system += VISION_RULES
        print("Camera on (" + ("USB webcam" if url.startswith("/dev/") else "security camera") + "): Jim can see costumes.")
    mode_box = ["dark"]                                # the main loop's mode, for the viewer
    watch = Watch(args.watch_port, camera, lambda: f"{mode_box[0]} / {face.state}") if args.watch_port else None
    if watch:
        print(f"Watch what Jim sees: http://127.0.0.1:{args.watch_port}")
    note = watch.event if watch else (lambda text: None)
    rec = Recorder(args.record_dir, args.record_keep_days) if args.record else None
    import collections
    # seconds, newest last: hear = speech to text; think = first sentence from the model; first_word = the kid
    # stopped talking -> Jim's voice starts; vision = the camera roast at wake-up
    # e2e = the kid stops talking -> Jim's first sound = wait (the pause that ends a sentence) + hear + think + speak
    metrics = {k: collections.deque(maxlen=30) for k in ("hear", "think", "first_word", "vision", "e2e", "wait", "speak")}

    def metric(kind, seconds):
        metrics[kind].append(round(seconds, 2))
    if rec:
        print(f"Recording visits to {args.record_dir}/ (kept {args.record_keep_days:g} days)")

    # ---- another Jim beside him (the panel's other Jims): they tell each other what they said directly, and nobody
    # listens while a pumpkin talks, so they never mistake each other's voices for a kid
    import urllib.request
    duo_inbox, _up = queue.Queue(), {}
    def jim_list():
        """Every Jim in config.toml, in order: [{id, name, watch_port}]."""
        try:
            return [{"id": k, "name": v.get("name", k), "watch_port": v.get("watch_port", 8780)}
                    for k, v in J.load()["jims"].items()] if args.jim_id else []
        except Exception:
            return []

    def partners():
        return [(i["name"], f"http://127.0.0.1:{i['watch_port']}") for i in jim_list() if i.get("id") != args.jim_id]

    def partner_up(url):
        hit = _up.get(url)
        if hit and time.time() - hit[0] < 3:
            return hit[1]
        try:
            urllib.request.urlopen(url + "/mic.json", timeout=0.3).read(); ok = True
        except Exception:
            ok = False
        _up[url] = (time.time(), ok)
        return ok

    def live_partners():
        return [(n, u) for n, u in partners() if partner_up(u)]

    def i_lead():
        """Of the Jims running, the first in the panel's list answers the kids (the others chime in)."""
        for i in jim_list():
            if i.get("id") == args.jim_id:
                return True
            if partner_up(f"http://127.0.0.1:{i['watch_port']}"):
                return False
        return True

    def tell(path, body, wait=False):
        for _, url in live_partners():
            def go(url=url):
                try:
                    urllib.request.urlopen(urllib.request.Request(url + path, json.dumps(body).encode(),
                                           {"Content-Type": "application/json", "X-Jim-Panel": "1"}), timeout=1).read()
                except Exception:
                    pass
            t = threading.Thread(target=go, daemon=True); t.start()
            if wait:
                t.join(0.3)

    class Speaking:
        """While this Jim makes a sound, his partners stop listening; afterwards they hear what he said."""
        def __init__(self, kind, depth=0, kid=None):
            self.kind, self.depth, self.kid, self.text = kind, depth, kid, ""

        def __enter__(self):
            t = time.time()                          # one pumpkin talks at a time: let the partner finish first
            while "partner" in getattr(mic, "holds", ()) and time.time() - t < 20:
                time.sleep(0.05)
            tell("/duo/quiet", {"on": True}, wait=True)
            return self

        def __exit__(self, *exc):
            tell("/duo/quiet", {"on": False})
            if self.text:
                tell("/duo/said", {"name": args.name, "text": self.text, "kind": self.kind, "depth": self.depth,
                                   "kid": self.kid, "gentle": brain.gentle, "kid_safe": brain.kid_safe})

    def name_in(text, name):
        return bool(re.search(rf"\b{re.escape(name)}\b", text or "", re.I))

    def who_answers(text):
        """(should this Jim answer the kid, the partners the kid named). The Jim the kid names answers; otherwise the
        lead. With partners of the same name, the lead answers."""
        others = live_partners()
        if not others:
            return True, []
        names = [(n, u) for n, u in others if n.lower() != args.name.lower()]
        to_me = name_in(text, args.name) and len(names) == len(others)
        named = [(n, u) for n, u in names if name_in(text, n)]
        return to_me or (i_lead() and not named), named

    if partners():
        brain.system += DUO_RULES.format(partners=", ".join(n for n, _ in partners()))
        print(f"Partners: {', '.join(n for n, _ in partners())} (they hear each other's lines directly)")

    def capture():
        """Collect what Jim says next, as played (for the recording)."""
        clips = []
        voice.tap = clips.append if rec else None
        return clips

    def played(clips):
        voice.tap = None
        return np.concatenate(clips) if clips else None
    if args.brain == "ollama":
        import atexit
        atexit.register(brain.unload)  # nothing of Jim stays on the GPU after he stops
        try:
            brain.warm_up()  # after the prompt is final, so the slow first pass happens now
        except Exception as e:
            sys.exit(f"Couldn't reach Ollama at {args.ollama_url} ({e}). Is it running? "
                     f"Is the model pulled? (ollama pull {args.ollama_model})")

    def see():
        """The current camera frame, if there is a fresh one."""
        return camera.latest() if camera else None

    def respond(text, heard=None, t_heard=None, t_end=None):
        with Speaking("reply", 0, text) as sp:
            sp.text = respond_(text, heard, t_heard, t_end)

    def respond_(text, heard=None, t_heard=None, t_end=None):
        frame = see()
        if watch and frame:
            watch.judge(frame, None)
        clips = capture()
        if voice.streaming:
            face.state = "thinking"
            t0, said, took, first_at = time.time(), [], [], []

            def sentences():
                for s in brain.reply_sentences(text, frame):
                    if not said:
                        metric("think", time.time() - t0)
                        first_at.append(time.time())
                    voice.gentle = brain.gentle  # softer voice for a scared kid
                    print(f"  🎃 {s}")
                    said.append(s.strip())
                    yield s

            def started():
                face.state = "talking"
                took.append(time.time() - t0)
                metric("first_word", time.time() - (t_heard or t0))
                if first_at:
                    metric("speak", time.time() - first_at[0])
                if t_end:
                    metric("e2e", time.time() - t_end)
                print(f"     (started talking after {took[0]:.2f}s)")

            voice.say_stream(sentences(), on_start=started)
            if rec:
                rec.add("turn", text, frame, heard, played(clips), voice.play_rate, reply=" ".join(said),
                        seconds=round(took[0], 2) if took else None, gentle=brain.gentle)
            note(f"🧒 \"{text}\" {'(with a camera frame)' if frame else '(no camera frame)'} → 🎃 \"{' '.join(said)}\""
                 + (f" · talking after {took[0]:.2f} s" if took else "") + (" · gentle mode" if brain.gentle else ""))
            return " ".join(said)
        face.state = "thinking"
        t0 = time.time()
        reply = " ".join(s.strip() for s in brain.reply_sentences(text, frame)).strip()
        metric("think", time.time() - t0)
        note(f"🧒 \"{text}\" → 🎃 \"{reply}\" · {time.time() - t0:.2f} s")
        audio = voice.synth(reply)  # the whole reply in one take, so it flows naturally
        print(f"  🎃 {reply}   ({time.time() - t0:.2f}s)")
        face.state = "talking"
        metric("first_word", time.time() - (t_heard or t0))
        if t_end:
            metric("e2e", time.time() - t_end)
        voice.play(audio)
        if rec:
            rec.add("turn", text, frame, heard, played(clips), voice.play_rate, reply=reply,
                    seconds=round(time.time() - t0, 2), gentle=brain.gentle)
        return reply

    if args.type:
        face.state = "idle"
        while True:
            face.state = "idle"
            try:
                text = input("you> ").strip()
            except EOFError:
                return
            if text == "/reset":
                brain.reset(); continue
            if text:
                respond(text)

    # The wake-up sound, and the greetings pre-rendered so they play instantly. Greetings are
    # cached on disk (keyed on the text and every voice setting), so restarts don't redo them.
    rumble = make_rumble(voice.play_rate, REVEAL_SECONDS)
    import hashlib
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "greetings")
    os.makedirs(cache_dir, exist_ok=True)
    voice_id = json.dumps([args.tts, args.voice_design, args.voice, args.tts_model, args.voice_ref,
                           args.pitch, args.intonation, args.echo, args.grit, args.speed,
                           args.exaggeration])

    def greeting_file(text):
        return os.path.join(cache_dir, hashlib.sha1((voice_id + text).encode()).hexdigest() + ".npy")

    greeting_audio = {}
    for g in persona["greetings"]:
        try:
            greeting_audio[g] = np.load(greeting_file(g))
        except (OSError, ValueError):
            pass

    def render_missing_greetings():
        for g in persona["greetings"]:
            if g not in greeting_audio:
                audio = voice.synth(g)
                np.save(greeting_file(g), audio)
                greeting_audio[g] = audio

    missing = len(persona["greetings"]) - len(greeting_audio)
    if missing:
        print(f"Recording {missing} greeting(s) in the background (saved for next time)...")
        threading.Thread(target=render_missing_greetings, daemon=True).start()
    greetings, goodbyes = shuffled_cycle(persona["greetings"]), shuffled_cycle(persona["goodbyes"])

    if args.local_whisper:
        print("Loading speech recognition (Whisper in this process)...")
        ears = LocalEars(args.whisper)
    else:
        print(f"Speech recognition: Whisper via the speech service at {args.speech_url}")
        ears = ServiceEars(args.speech_url, args.whisper)
        ears.warm()

    if args.keep_loaded:
        def keep_loaded():
            while True:  # the speech service unloads after 120 s idle; stay just ahead of it
                time.sleep(90)
                if hasattr(voice, "warm"):
                    voice.warm()
                ears.warm()
        threading.Thread(target=keep_loaded, daemon=True).start()
        print("Keeping the voice, ears and chat model loaded while Jim runs (--keep-loaded).")

    def warm_up_speech():
        """Someone's around: get the shared voice and ears loaded before Jim needs them (the
        service unloads them after 2 quiet minutes; reloading the voice takes about 8 s)."""
        threading.Thread(target=lambda: (voice.warm() if hasattr(voice, "warm") else None,
                                         ears.warm()), daemon=True).start()

    noises = queue.Queue()

    def on_speech_start():
        if face.state == "idle":
            face.state = "listening"

    if args.min_rms is None:
        args.min_rms = 500 if args.mic == "wled" else 300
    def mic_source(a):
        if a.mic == "wled":
            return wled_source(a.wled, a.wled_gain)
        if a.mic.startswith("camera:"):
            return rtsp_audio_source(camera_audio_url(a.mic[7:]), a.mic[7:])
        return alsa_source(a.mic)
    source = mic_source(args)
    mic = Mic(source, args.vad_ratio, args.min_rms, args.silence, on_speech_start,
              on_noise=lambda: noises.put(time.time()))
    mic.debug_levels = args.levels
    mic.fixed = args.level_mode == "fixed"
    mic.start()

    def drain(q):
        while not q.empty():
            q.get_nowait()

    def go_dark():
        face.state = "dark"
        brain.reset()
        voice.gentle = False
        drain(noises)
        print(f"💤 {args.name} is asleep (dark). Make a noise to wake him.")
        note("💤 asleep")
        if rec:
            rec.end()

    def wake_up(silent=False):
        """silent: woken with a partner who does the greeting (Jim just bursts to life and listens)."""
        tell("/duo/wake", {})
        if not silent and live_partners() and not i_lead():
            silent = True
        if silent:
            mic.mute()
            print(f"🔥 {args.name} wakes up with his partner!")
            note("🔥 woke up with his partner (the partner greets)")
            with Speaking("wake"):
                face.reveal()
                voice.play(rumble, mouth=False)
            brain.reset()
            face.state = "idle"
            time.sleep(0.3)
            mic.unmute()
            drain(noises)
            return
        mic.mute()
        print(f"🔥 {args.name} wakes up!")
        # Look at the visitors while the wake-up animation plays, and roast what they're wearing.
        frame, seen = see(), {}
        if frame:
            def look():
                t = time.time()
                seen["greeting"] = brain.vision_greeting(frame, timeout=REVEAL_SECONDS + 2.0)
                seen["seconds"] = round(time.time() - t, 2)
            looker = threading.Thread(daemon=True, target=look)
            looker.start()
        greet = Speaking("greeting")
        greet.__enter__()
        face.reveal()
        voice.play(rumble, mouth=False)
        if frame:
            looker.join(timeout=1.5)  # a little grace after the rumble
        greeting = seen.get("greeting")
        if watch:
            if not frame:
                watch.judge(None, {"visible": None, "error": "no camera frame"})
            elif "seconds" not in seen:
                watch.judge(frame, {"visible": None, "error": "the model was too slow; used a stock greeting"})
            else:
                watch.judge(frame, dict(getattr(brain, "last_vision", None) or {}, seconds=seen["seconds"]))
        note("🔥 woke up · " + (f"roasted what he saw: \"{greeting}\"" if greeting else
             "saw nobody / no frame: stock greeting"))
        verdict = dict(getattr(brain, "last_vision", None) or {}) if frame and "seconds" in seen else None
        if "seconds" in seen:
            metric("vision", seen["seconds"])
        roasted = bool(greeting)
        clips = capture()
        face.state = "talking"
        if greeting:
            print(f"  🎃 {greeting}   (roasting what he sees)")
            voice.say(greeting)
        else:  # nobody visible, no camera, or too slow: a pre-written greeting
            greeting = next(greetings)
            for _ in range(len(persona["greetings"])):  # prefer one that's already recorded
                if greeting in greeting_audio:
                    break
                greeting = next(greetings)
            print(f"  🎃 {greeting}")
            voice.play(greeting_audio.get(greeting)
                       if greeting in greeting_audio else voice.synth(greeting))
        if rec:
            rec.add("greeting", greeting, frame, None, played(clips), voice.play_rate, roasted=roasted,
                    verdict=verdict, seconds=seen.get("seconds"))
        greet.text = greeting
        greet.__exit__()
        brain.prime(greeting)
        face.state = "idle"
        time.sleep(0.3)  # let the speaker's echo die out before listening again
        mic.unmute()
        drain(noises)

    def banter(name, text, depth, kid=None):
        """Answer the other pumpkin's line with one short jab (or pile on the costume roast)."""
        mic.mute()
        face.state = "thinking"
        old, brain.max_words = brain.max_words, 15
        ask = (f"[{name} (the other pumpkin, beside you) just said out loud: \"{text}\""
               + (f", answering a kid who said \"{kid}\"" if kid else "")
               + f". Answer {name} in ONE short line, under 15 words: tease {name} or pile on the roast of the costumes.]")
        clips, said = capture(), []
        with Speaking("banter", depth + 1, kid) as sp:
            def sentences():
                for line in brain.reply_sentences(ask, see(), from_kid=False):
                    print(f"  🎃 (to {name}) {line}")
                    said.append(line.strip())
                    yield line
            voice.say_stream(sentences(), on_start=lambda: setattr(face, "state", "talking"))
            sp.text = " ".join(said)
        brain.max_words = old
        note(f"🎃 (to {name}) \"{' '.join(said)}\"")
        if rec:
            rec.add("banter", f"{name}: {text}", None, None, played(clips), voice.play_rate, reply=" ".join(said))
        face.state = "idle"
        time.sleep(0.3)
        mic.unmute()

    snooze_until = [0.0]

    def snooze(seconds):
        snooze_until[0] = time.time() + seconds
        print(f"😴 Snoozing for {seconds:g} s: sneak away! (Enter again to restart the timer)")

    def keyboard():
        for _ in sys.stdin:  # every Enter press snoozes again
            snooze(args.snooze)

    if watch:
        watch.on_snooze = snooze

        def set_min_rms(v, fixed=None):
            mic.min_rms = v
            if fixed is not None:
                mic.fixed = fixed
            print(f"[mic] reaction level set to {v:.0f}{' (fixed)' if mic.fixed else ''} from the control panel", flush=True)
        watch.on_min_rms = set_min_rms

        def set_volume(v):
            voice.volume = v
            print(f"[voice] volume set to {v:.2f} from the control panel", flush=True)
        watch.on_volume = set_volume

        def reload_devices():
            """The control panel changed the camera, mic or speaker in jim.toml: switch to them now, no restart."""
            p = build_parser()
            known = {x.dest for x in p._actions}
            p.set_defaults(**{k: v for k, v in load_config(args.config, args.jim_id)[0].items() if not isinstance(v, dict) and k in known})
            new = p.parse_args([])               # jim.toml as it is now (the panel only adds the viewer port)
            changed = []
            url, crop = camera_choice(new)
            if url != camera.url or crop != camera.crop:
                camera.set_url(url, crop)
                if url and VISION_RULES not in brain.system:
                    brain.system += VISION_RULES
                if not url:
                    brain.system = brain.system.replace(VISION_RULES, "")
                changed.append("camera: " + ("USB webcam " + url if url and url.startswith("/dev/") else "security camera" if url else "none"))
            if new.mic != args.mic or (new.mic == "wled" and new.wled_gain != args.wled_gain):
                mic.set_source(mic_source(new))
                changed.append("mic: " + new.mic)
            mic.min_rms = new.min_rms if new.min_rms is not None else (500 if new.mic == "wled" else 300)
            mic.fixed, mic.vad_ratio, mic.silence_s = new.level_mode == "fixed", new.vad_ratio, new.silence
            if new.speaker != voice.device:
                voice.device = new.speaker
                changed.append("speaker: " + new.speaker)
            voice.volume = max(0.05, min(1.5, new.volume))
            if new.adult_mode != brain.adult:
                brain.adult = new.adult_mode
                changed.append("adult mode " + ("on" if brain.adult else "off"))
            args.mic, args.wled_gain, args.speaker = new.mic, new.wled_gain, new.speaker
            print("[jim] switched " + (", ".join(changed) if changed else "nothing (settings refreshed)") + " from the control panel", flush=True)
            return changed
        watch.on_reload = reload_devices

        partner_quiet = [0.0]

        def on_duo(path, b):
            if path == "/duo/quiet":                 # the other pumpkin is making a sound: don't listen to it
                if b.get("on"):
                    partner_quiet[0] = time.time() + 30   # in case the "done" message never comes
                    mic.mute("partner")
                else:
                    partner_quiet[0] = 0.0
                    mic.unmute("partner")
            else:
                duo_inbox.put((path, b))
        watch.on_duo = on_duo

        def speaker_level():
            fresh = time.time() - getattr(voice, "out_t", 0) < 0.4      # nothing written lately: silent
            return {"level": round(getattr(voice, "out_level", 0)) if fresh else 0, "playing": fresh,
                    "volume": round(getattr(voice, "volume", 1.0), 3), "device": args.speaker}
        watch.extra = lambda: {"mic": {"level": round(mic.level), "thresh": round(mic.thresh), "min_rms": round(mic.min_rms), "fixed": mic.fixed,
                                       "age_s": round(time.time() - mic.level_t, 1) if mic.level_t else None,
                                       "muted": mic.muted.is_set()},
                               "speaker": speaker_level(),
                               "snooze_left": max(0, round(snooze_until[0] - time.time())), "recording": bool(rec), "name": args.name,
                               "metrics": {k: list(v) for k, v in metrics.items()}, "camera": camera.stats() if camera.url else None}
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True).start()
        print(f"(Press Enter any time to make Jim sleep for {args.snooze:g} s.)")
    if args.start_delay > 0:
        snooze(args.start_delay)

    mode = "dark"
    go_dark()
    eyes_since = last_heard = 0.0
    snoozing = False
    while True:
        if time.time() < snooze_until[0]:  # ignore everything until the snooze ends
            if not snoozing or mode != "dark":
                face.state = "dark"
                brain.reset()
                voice.gentle = False
                mode, snoozing = "dark", True
                mode_box[0] = mode
            drain(noises)
            drain(mic.utterances)
            time.sleep(0.2)
            continue
        if snoozing:
            snoozing = False
            drain(noises)
            print("⏰ Snooze over. Jim is listening again (dark until a noise).")
        if watch and "partner" in getattr(mic, "holds", ()) and time.time() > partner_quiet[0]:
            mic.unmute("partner")                    # the partner never said he was done
        try:
            path, b = duo_inbox.get_nowait()
        except queue.Empty:
            path = None
        if path:
            if path in ("/duo/wake", "/wake", "/hear") and mode != "awake":
                wake_up(silent=path == "/duo/wake")
                mode, last_heard = "awake", time.time()
                mode_box[0] = mode
            if path == "/hear" and (b.get("text") or "").strip():
                text = b["text"].strip()[:300]
                print(f"kid (typed)> {text}")
                mine, named = who_answers(text)
                if not mine and named:                 # typed to me, but meant for the pumpkin it names: pass it on
                    print(f"     (passed to {named[0][0]}, who was named)")
                    brain.history.append({"role": "user", "content": text})
                    try:
                        urllib.request.urlopen(urllib.request.Request(named[0][1] + "/hear", json.dumps({"text": text}).encode(),
                                               {"Content-Type": "application/json", "X-Jim-Panel": "1"}), timeout=1).read()
                    except Exception:
                        pass
                    last_heard = time.time()
                    continue
                mic.mute()
                respond(text)
                face.state = "idle"
                mic.unmute()
                last_heard = time.time()
            elif path == "/duo/said":
                name, said_text, kind = b.get("name", "the other pumpkin"), b.get("text", ""), b.get("kind")
                print(f"  🗣️ {name}: {said_text}")
                note(f"🗣️ {name}: \"{said_text}\"")
                if kind == "goodbye":
                    if mode == "awake":              # the partner said goodbye for both of them
                        mode = mode_box[0] = "dark"
                        go_dark()
                    continue
                if mode != "awake":
                    wake_up(silent=True)
                    mode = mode_box[0] = "awake"
                last_heard = time.time()
                if b.get("gentle"):
                    brain.gentle = voice.gentle = True
                if b.get("kid_safe") and brain.adult and not brain.kid_safe:
                    brain.kid_safe = True            # the partner's camera may see a child
                    print("     (partner is kid-safe: so is he, until they go back to sleep)")
                if not brain.gentle and int(b.get("depth") or 0) < args.banter_max and random.random() < args.banter:
                    banter(name, said_text, int(b.get("depth") or 0), b.get("kid"))
                else:                                # just remember it, for the conversation
                    brain.history.append({"role": "user", "content": f"[{name} said: \"{said_text}\"]"})
                last_heard = time.time()
            continue
        if mode in ("dark", "eyes"):
            try:
                noises.get(timeout=0.5)
            except queue.Empty:
                if mode == "eyes" and time.time() - eyes_since > args.eyes_timeout:
                    mode = mode_box[0] = "dark"
                    go_dark()
                continue
            if mode == "dark":
                mode, eyes_since = "eyes", time.time()
                mode_box[0] = mode
                face.state = "eyes"
                print("👀 Something stirs... (eyes on, waiting for a second noise)")
                note("👀 a noise: eyes on")
                warm_up_speech()
            else:
                wake_up()
                mode, last_heard = "awake", time.time()
                mode_box[0] = mode
            continue

        # awake: hold a conversation
        face.state = "listening" if mic.in_speech else "idle"
        try:
            audio = mic.utterances.get(timeout=0.5)
        except queue.Empty:
            if time.time() - last_heard > args.reset_after and face.state == "idle":
                if live_partners() and not i_lead():  # the lead says goodbye for both of them
                    mode = mode_box[0] = "dark"
                    go_dark()
                    continue
                mic.mute()
                goodbye = next(goodbyes)
                print(f"  🎃 {goodbye}")
                face.state = "talking"
                clips = capture()
                with Speaking("goodbye") as sp:
                    voice.say(goodbye)
                    sp.text = goodbye
                if rec:
                    rec.add("goodbye", goodbye, None, None, played(clips), voice.play_rate)
                mic.unmute()
                mode = mode_box[0] = "dark"
                go_dark()
            continue
        face.state = "thinking"
        t0, t_end = time.time(), mic.speech_end
        if t_end:
            metric("wait", t0 - t_end)
        text = ears.transcribe(audio)
        if not text:
            face.state = "idle"
            continue
        print(f"kid> {text}   (stt {time.time() - t0:.2f}s)")
        metric("hear", time.time() - t0)
        mine, named = who_answers(text)
        if not mine:
            whom = ", ".join(n for n, _ in named) or "the lead pumpkin"
            print(f"     (left for {whom} to answer)")
            note(f"🧒 \"{text}\" (left for {whom})")
            brain.history.append({"role": "user", "content": text})
            face.state = "idle"
            last_heard = time.time()
            continue
        mic.mute()
        respond(text, audio, t0, t_end)
        face.state = "idle"
        time.sleep(0.3)  # let the speaker's echo die out before listening again
        mic.unmute()
        last_heard = time.time()


if __name__ == "__main__":
    import signal
    for sig in (signal.SIGTERM, signal.SIGHUP):  # kill, or closing the terminal: clean up first
        signal.signal(sig, lambda n, f: (print(f"[jim] stopped by {signal.Signals(n).name}", flush=True), sys.exit(0)))
    try:
        main()
    except KeyboardInterrupt:
        print("[jim] stopped by Ctrl+C (SIGINT)", flush=True)
