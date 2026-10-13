#!/usr/bin/env python3
"""Tiny local TTS server for the talking pumpkin.

Each engine runs in its own venv because they pin conflicting PyTorch/transformers
versions; pumpkin.py starts the right one automatically:
  qwen:       .venv-qwentts     Qwen3-TTS voice design (describe the voice in words)
  chatterbox: .venv-chatterbox  Chatterbox, copying the voice in a reference clip

POST /synth  {"text", "instruct"?, "exaggeration"?, "cfg_weight"?}  -> raw float32 PCM
POST /stream (same body) -> length-prefixed float32 chunks as they are generated
             (each chunk: uint32 little-endian byte count, then the samples)
The sample rate is in the X-Sample-Rate header. GET /health returns "ok" once loaded."""

import argparse
import json
import struct
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")


class QwenEngine:
    def __init__(self, model, seed):
        import torch
        from faster_qwen3_tts import FasterQwen3TTS
        self.torch, self.seed = torch, seed
        self.model = FasterQwen3TTS.from_pretrained(model)
        self.sr = 24000

    def stream(self, req):
        # A fixed seed keeps the designed voice consistent from reply to reply.
        self.torch.manual_seed(self.seed)
        for chunk, sr, _ in self.model.generate_voice_design_streaming(
                text=req["text"], instruct=req["instruct"], language="English"):
            self.sr = sr
            yield chunk


class ChatterboxEngine:
    def __init__(self, model, ref):
        import torch
        if model == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model
        self.torch, self.turbo = torch, model == "turbo"
        self.model = Model.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.prepare_conditionals(ref, exaggeration=0.5)
        self.sr = self.model.sr

    def stream(self, req):
        kwargs = {} if self.turbo else {"exaggeration": req.get("exaggeration", 0.8),
                                        "cfg_weight": req.get("cfg_weight", 0.4)}
        with self.torch.inference_mode():
            wav = self.model.generate(req["text"], **kwargs)
        yield wav.squeeze().float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="qwen", choices=["qwen", "chatterbox"])
    p.add_argument("--model", default=None,
                   help="qwen: HF model id; chatterbox: turbo or original")
    p.add_argument("--ref", default="voices/jack_reference.wav", help="chatterbox: voice to copy")
    p.add_argument("--seed", type=int, default=1234, help="qwen: voice consistency seed")
    p.add_argument("--port", type=int, default=5123)
    args = p.parse_args()

    if args.engine == "qwen":
        engine = QwenEngine(args.model or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", args.seed)
    else:
        engine = ChatterboxEngine(args.model or "turbo", args.ref)
    lock = threading.Lock()  # one generation at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("X-Sample-Rate", str(engine.sr))
                    self.end_headers()
                    for chunk in engine.stream(req):
                        data = chunk.astype("float32").tobytes()
                        self.wfile.write(struct.pack("<I", len(data)) + data)
                        self.wfile.flush()
                    self.wfile.write(struct.pack("<I", 0))
                else:
                    import numpy as np
                    body = np.concatenate(list(engine.stream(req))).astype("float32").tobytes()
                    self.send_response(200)
                    self.send_header("X-Sample-Rate", str(engine.sr))
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

    print(f"[tts] {args.engine} ready on port {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
