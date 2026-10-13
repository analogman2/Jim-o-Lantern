#!/usr/bin/env bash
# Recreate Jim's Python environments and download his models.
# Needs: uv (https://docs.astral.sh/uv/), an NVIDIA GPU, aplay/arecord (alsa-utils),
# and Ollama with a chat model pulled (ollama pull qwen3.5:9b).
set -euo pipefail
cd "$(dirname "$0")"

echo "== Main environment (mic, speech recognition, LEDs, Kokoro fallback voice)"
uv venv --python 3.12 .venv
uv pip install --python .venv anthropic faster-whisper numpy kokoro-onnx pyworld \
  "onnxruntime-gpu==1.22.0" "setuptools<81" \
  nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" nvidia-cuda-runtime-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12

echo "== Qwen3-TTS voice (fallback: normally Jim uses Agentora's speech service on :7895)"
uv venv --python 3.12 .venv-qwentts
uv pip install --python .venv-qwentts faster-qwen3-tts "qwen-tts-hf<0.2,>=0.1.1.post1" \
  "transformers==5.15.1" "setuptools<81"

echo "== Chatterbox voice (optional: ./start.sh --tts chatterbox)"
uv venv --python 3.11 .venv-chatterbox
uv pip install --python .venv-chatterbox chatterbox-tts "setuptools<81"

echo "== Kokoro voice files (optional: ./start.sh --tts kokoro)"
mkdir -p voices
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -f voices/kokoro-v1.0.onnx ] || curl -L -o voices/kokoro-v1.0.onnx "$base/kokoro-v1.0.onnx"
[ -f voices/voices-v1.0.bin ] || curl -L -o voices/voices-v1.0.bin "$base/voices-v1.0.bin"

echo "Done. Jim normally uses Agentora's speech service (http://127.0.0.1:7895) for his voice and"
echo "hearing; the local copies above are only used when it is unreachable."
echo "Start Jim with: ./start.sh"
