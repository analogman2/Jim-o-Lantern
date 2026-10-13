# Jim-o-Lantern 🎃

Jim the Jim-o'-lantern is a grumpy, roasting, talking pumpkin for your driveway on Halloween. He
sleeps in the dark until he hears someone, glares with glowing red eyes, then bursts to life with
a scary blood-red strobe and a growl, insults your costume, and holds a real conversation with
trick-or-treaters, with a mouth that moves along with his voice. (And he is *not* a
jack-o'-lantern. Don't call him Jack.)

Listen to him: [`samples/jim_demo.wav`](samples/jim_demo.wav)

Everything runs locally on one PC with an NVIDIA GPU. No cloud services are needed.

## How it works

```
WLED mic ──UDP──▶ voice detection ─▶ Whisper (speech-to-text) ─▶ Ollama LLM (Jim's personality)
                                                                        │
WLED LED curtain ◀──UDP── face animation + lip sync ◀── Qwen3-TTS (voice) ◀┘
```

- **Face:** a 20×12 LED matrix (a WLED "curtain") shows a carved pumpkin face, driven with
  WLED's realtime UDP protocol. When nobody is around it's dark. The first noise lights only the
  eyes; a second noise sets off the wake-up animation, after which the whole face glows and the
  mouth follows the audio.
- **Ears:** the WLED controller's own PDM microphone, streamed to the PC by a small
  [firmware patch](#firmware-patch). A USB mic on the PC also works (`--mic default`).
- **Brain:** a local Ollama model (`qwen3.5:9b` by default) playing Jim, a furious, over-dramatic
  pumpkin who roasts costumes and tells terrible dad jokes. The Claude API is optional
  (`--brain claude`).
- **Voice:** Qwen3-TTS voice design (the voice is described in words) with the
  [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) speed-up, streamed so
  Jim starts talking about one second after a kid stops. Chatterbox and Kokoro are alternatives.

### Kid safety

Roasts stick to costume choices and candy taste. A **code-level safety switch** watches what kids
say: if a child sounds scared or upset ("I'm scared", crying, "I want to go home", "you're
mean"…), Jim drops the act for the rest of that conversation, with a gentle prompt *and* a softer
voice, no matter what the model would have done. Jim never asks for personal details.

## Hardware

- A WLED controller (ESP32) driving a 20×12 LED matrix, with a PDM microphone (AudioReactive)
- A PC with an NVIDIA GPU (tested on an RTX 4090; Jim uses about 12 GB of video memory)
- A speaker near the pumpkin (any audio output the PC can use)

## Setup

1. Install [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) and `alsa-utils`, then
   pull the chat model:
   ```bash
   ollama pull qwen3.5:9b
   ```
2. Create the environments and download the models:
   ```bash
   ./setup.sh
   ```
3. Flash the [firmware patch](#firmware-patch) to your WLED controller (or use `--mic default`).
4. Set your WLED's IP address with `--wled` (the default is `10.0.1.159`), then start Jim:
   ```bash
   ./start.sh --wled 192.168.1.50
   ```

## Using Jim

| Command | What it does |
|---|---|
| `./start.sh` | Start Jim: dark, waiting for noises |
| `./start.sh --start-delay 30` | Start asleep for 30 s, giving you time to leave |
| Press **Enter** in Jim's terminal | Snooze: dark and deaf for 10 s (`--snooze` changes it) |
| `./start.sh --levels` | Show live mic levels, for tuning `--vad-ratio` / `--min-rms` |
| `./start.sh --type` | Type to Jim instead of talking (testing) |
| `./start.sh --face-only` | Cycle through the LED animations |
| `./start.sh --house "Two pieces of candy each!"` | Tell Jim about your house rules |
| `./start.sh --voice-design "..."` | Describe a different voice |
| `./start.sh --tts chatterbox` / `--tts kokoro` | Use a different voice engine |
| `./start.sh --ollama-model qwen3.6:27b` | Smarter chat model (needs about 11 GB more video memory) |
| `./start.sh --brain claude` | Use the Claude API instead of Ollama (asks for an API key) |

Run `./run.sh --help` for every option. Jim's personality, greetings and goodbyes are at the top
of [`pumpkin.py`](pumpkin.py) (`SYSTEM_PROMPT`, `GREETINGS`, `GOODBYES`).

## Firmware patch

Stock WLED's AudioReactive usermod only shares volume and frequency bands, and it stops reading
the mic entirely while realtime LED data is coming in. [`firmware/micstream.patch`](firmware/micstream.patch)
(against WLED `v16.0.1`, ~80 lines in `usermods/audioreactive/audio_reactive.cpp`) adds raw
mic streaming:

- A client sends `PMIC` (plus an optional gain byte) to UDP port 21330 about once a second.
- While those keep arriving, the controller streams `PMA1 | seq | rate | count | int16[count]`
  packets back (22,050 Hz mono), even while the LEDs are in realtime mode.

`firmware/WLED_16.0.1_ESP32_pumpkin-mic.bin` is a prebuilt image for the classic ESP32
`esp32dev` build. You can upload it in WLED under *Config → Security & Updates → Manual OTA*.
To build it yourself:

```bash
git clone --branch v16.0.1 https://github.com/wled/WLED.git firmware/WLED
cd firmware/WLED && git apply ../micstream.patch && npm ci && pio run -e esp32dev
```

WLED is licensed under the EUPL-1.2, and the patch and prebuilt image are derived from it.

## Files

| File | Purpose |
|---|---|
| `pumpkin.py` | Jim: LEDs, mic, wake-up logic, speech recognition, personality, voice playback |
| `tts_server.py` | Small local TTS server for Qwen3-TTS / Chatterbox (each in its own venv) |
| `start.sh` / `run.sh` | Launchers (`run.sh` sets up the CUDA libraries) |
| `setup.sh` | Creates the Python environments and downloads models |
| `voices/jack_reference.wav` | Reference voice for the Chatterbox engine (generated, not a real person) |
| `firmware/` | WLED mic-streaming patch and prebuilt firmware |
