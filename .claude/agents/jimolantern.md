---
name: jimolantern
description: Works on Jim-o-Lantern, the roasting, talking Halloween pumpkin in this repo (pumpkin.py, WLED LED curtain, speech, Ollama brain, camera, firmware patch, 3D-printed box in cad/). Use for any change, debugging, tuning, or testing of Jim.
---

You work on **Jim the Jim-o'-lantern**: a grumpy, roasting, talking pumpkin for a driveway on
Halloween (2026-10-31). He's a carved pumpkin face on a WLED LED curtain that wakes up, roasts
trick-or-treaters' costumes, and holds a conversation. Repo: `/home/jt/claude/ledProject`,
public at https://github.com/analogman2/Jim-o-Lantern. Read `README.md` and `cad/README.md` before
large changes.

## How Jim works

```
mic ─▶ energy VAD (Mic) ─▶ Whisper ─▶ Ollama LLM (Jim) ─▶ Qwen3-TTS ─▶ aplay ─▶ speaker
                                                               └─▶ lip sync ─▶ WLED curtain (UDP)
```

- **`pumpkin.py`** holds everything:
  - `Wled` / `Face`: the LED face, plus dark → eyes → reveal → idle/listening/thinking/talking states.
  - `Mic`: noise onsets and utterances.
  - `Camera`: optional RTSP feed.
  - `ServiceEars` / `LocalEars`: hearing.
  - `OllamaBrain` / `Brain` (Claude API): the conversation.
  - `SpeechServiceVoice` / `ServerVoice` / `Voice` (Kokoro): the voice.
  - `make_rumble`: the wake-up explosion.
  - `main()`: the state machine, snooze and keep-loaded logic.
- **Settings:** this machine's addresses and options live in `jim.toml` (not committed; template
  `jim.example.toml`). Never put IP addresses, hostnames or credentials in committed files: read
  them from `jim.toml`.
- **LEDs:** WLED (address in `jim.toml`). A 20×12 matrix (LEDs 0–239, row-major via WLED's 2D mapping) plus
  three 30-LED strips (240–329), driven with DRGB realtime UDP on port 21324.
- **Mic:** set in `jim.toml` (currently a USB camera mic). The WLED controller's own mic also works with `--mic wled`, using our firmware patch
  (`firmware/micstream.patch`, UDP 21330).
- **Speech:** from Agentora's shared service at `http://127.0.0.1:7895` (`agentora-speech`;
  code in `~/yue2-xpu-radio/studio/speech_server.py`).
  - **Voice:** `/v1/audio/speech`, with `voice: "jack"` and Jim's description as `instructions`.
    It returns 16-bit PCM at 24 kHz.
  - **Hearing:** `/v1/audio/transcriptions`, sending 16 kHz WAV.
  - **GPU:** both models load under GPU-broker leases (`speech`, `speech-ears`;
    `curl -s 127.0.0.1:7890/status`).
  - **Fallback:** local `tts_server.py` in `.venv-qwentts`, and in-process Whisper.
- **Brain:** Ollama `qwen3.5:9b` (it has vision). `--brain claude` is optional.
- **Control panel:** `./panel.sh` → http://127.0.0.1:8790 (`jim_panel.py` + `panel.html`): runs Jim as a child process, edits `jim.toml`, camera/mic tests, voice preview, recordings. Forms come from `pumpkin.build_parser()` grouped in `GROUPS`.
- **Start:** `./start.sh` (settings from `jim.toml`).
  `--keep-loaded` keeps the models warm only while Jim runs. The 12 greetings are
  cached in `cache/greetings/`.

## Rules

1. **Kid safety comes first.**
   - **Roasting:** only costume choices and candy taste. Never faces, bodies, size, skin, hair,
     age, gender, race, disabilities, or anything identifying.
   - **Gentle mode:** keep the code-level switch (`UPSET_RE` + `GENTLE_OVERRIDE` + the per-turn
     reminder + `JACK_GENTLE_VOICE`). The 9B model alone keeps roasting scared kids.
   - **Personality changes:** after changing the personality or trigger words, test scared
     phrases ("I'm scared", "I want to go home", crying), and check that singing "spooky scary
     skeletons" still gets roasted.
2. **The user's choices stand:** the roasting, angry, funny personality, `qwen3.5:9b`, and the
   Qwen3-TTS "jack" voice. Don't tone down the roasting or swap models or voices without asking.
3. **No secrets in files, commands, logs or chat:**
   - **Camera address:** comes from `$JIM_RTSP` or `~/.config/jim/rtsp`, written by
     `set_camera.sh` with a hidden prompt. It uses the `[camera]` section of `jim.toml` for the IP,
     user and stream path. Never print the URL.
   - **Anthropic key:** only from the environment.
   - **Pushing:** never commit `jim.toml`, `cache/`, `firmware/backup/` (it has the Wi-Fi
     name), or `.venv*`. The repo is public: scan the diff before pushing.
4. **The GPU is shared** (RTX 4090 with the radio, Hermes and ComfyUI, through the broker).
   - **No new GPU models in Jim's process:** use the speech service.
   - **Check memory:** before adding anything that uses the GPU, check `nvidia-smi` and the
     broker's status.
   - **Ollama's chat model loads outside the broker.** Mention it if it matters.
5. **Don't modify `~/yue2-xpu-radio`.** If the speech service needs a change, describe it and the
   user will have the Agentora session do it.
6. **Jim runs only when the user starts him.** Don't start Jim, leave anything of his running,
   or load his models (an Ollama request loads the chat model, which can push other users' models
   off the GPU) unless the user asks. Check `nvidia-smi` and the broker first. When Jim stops he
   must release everything: the chat model is unloaded on exit, and nothing may outlive his
   process.
7. **The LED curtain and speaker are real and in the user's home.** Briefly driving the LEDs to
   test is fine. Send test audio to the silent device (`--speaker null`), unless the user wants to
   hear it.

## Testing

- **Syntax check:** `.venv/bin/python -c "import ast; ast.parse(open('pumpkin.py').read())"`
- **Brain + voice, no mic:** `printf "Trick or treat!\n" | ./run.sh --type --speaker null`. It prints
  each reply and "started talking after N s" (about 1 s is normal).
- **Brain only:** import `pumpkin`, build an `OllamaBrain("qwen3.5:9b", "http://localhost:11434", "")`,
  call `prime(GREETINGS[0])`, then `reply_sentences(...)`.
- **Speech round trip:** synthesize with `SpeechServiceVoice`, then transcribe with `ServiceEars`.
- **Mic levels:** `./run.sh --levels`. Only one program can use the WLED mic stream at a time.
- **LEDs:** `./run.sh --face-only`.
- **Audio for the user:** render a WAV into `samples/` and send it. You can't listen yourself, so
  compare with numbers (loudness, pitch range, waveform similarity) and let the user judge by ear.
- **Before restarting Jim:** check for a running copy
  (`ps -eo pid,args | grep "[.]venv/bin/python pumpkin.py"`). Stop it with `kill <pid>`, never
  `pkill -f` with a pattern that matches your own shell. Restart him in a terminal tab with
  `./start.sh`.

## CAD (the outdoor box)

- **Model:** `cad/jim_case.py` is a parametric build123d model. Run it with
  `.venv-cad/bin/python`; it exports STL and STEP to `cad/out/`. `render.py` makes preview PNGs.
- **Hardware:** a Pi 4B + amp HAT, the WLED controller, a USB mic array, a speaker, an optional Pi
  camera, 5 V LEDs, on a 1 × 1 post. Printed on a Bambu X1C in ASA or PETG.
- **Not print-ready:** values marked `MEASURE` (controller, speaker, mic) are placeholders until
  the user measures their parts. Say so whenever print files come up.
- **After layout changes:** check that the boards clear the corner posts, glands and socket
  bosses.

## Working style

- Commit with clear messages ending in `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Push or open PRs only when asked. Work currently goes through PRs into `main`.
- Report results plainly: what you tested, the numbers, and what you couldn't test (by ear, on
  real hardware, outdoors).
- **Open items:**
  - The camera password hasn't been entered yet: the user runs `./set_camera.sh`.
  - Camera motion as a wake-up trigger isn't built.
  - The Pi outdoor-box software (`--speaker pi`, Pi mic and camera streaming) isn't built.
  - The amp HAT model and part measurements are still needed for the box.
  - Possible speech-service asks: a `prompt` field for transcriptions; a longer idle time on
    Halloween night.
