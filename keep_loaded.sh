#!/usr/bin/env bash
# Keep Jim's models loaded on the GPU between runs, until told to stop.
#   ./keep_loaded.sh start    keep the speech service's voice + ears warm, and the chat model loaded
#   ./keep_loaded.sh stop     let everything unload again (speech after ~2 min idle; chat model now)
#   ./keep_loaded.sh status
# The speech service unloads its models after 120 s unused; this pings it every 90 s. Ollama is
# told to keep the chat model forever. Settings: JIM_SPEECH_URL, JIM_OLLAMA_URL, JIM_OLLAMA_MODEL.
set -euo pipefail
SPEECH=${JIM_SPEECH_URL:-http://127.0.0.1:7895}
OLLAMA=${JIM_OLLAMA_URL:-http://localhost:11434}
MODEL=${JIM_OLLAMA_MODEL:-qwen3.5:9b}
DIR=~/.cache/jim
PIDFILE=$DIR/keep_loaded.pid
mkdir -p "$DIR"

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "Already keeping Jim's models loaded (pid $(cat "$PIDFILE"))."; exit 0; fi
    [ -f "$DIR/silence.wav" ] || ffmpeg -loglevel error -f lavfi -i anullsrc=r=16000:cl=mono -t 0.25 "$DIR/silence.wav"
    # load the chat model with Jim's context size (a different size makes Ollama reload it)
    curl -s -m 120 -o /dev/null "$OLLAMA/api/generate" \
      -d "{\"model\":\"$MODEL\",\"keep_alive\":-1,\"options\":{\"num_ctx\":4096}}" || true
    nohup setsid bash -c "
      while true; do
        curl -s -m 90 -o /dev/null -H 'Content-Type: application/json' \
          -d '{\"input\":\"Hi.\",\"voice\":\"jack\",\"response_format\":\"pcm\"}' $SPEECH/v1/audio/speech
        curl -s -m 60 -o /dev/null -F file=@$DIR/silence.wav $SPEECH/v1/audio/transcriptions
        sleep 90
      done" >/dev/null 2>&1 &
    echo $! > "$PIDFILE"
    echo "Keeping Jim's models loaded until: ./keep_loaded.sh stop"
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; fi
    rm -f "$PIDFILE"
    curl -s -m 30 -o /dev/null "$OLLAMA/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" || true
    echo "Stopped. The chat model is unloaded; the speech service unloads its models after ~2 idle minutes."
    ;;
  status)
    if running; then echo "Keeping models loaded (pid $(cat "$PIDFILE"))."; else echo "Not running."; fi
    ;;
  *) echo "Usage: $0 start|stop|status" >&2; exit 1 ;;
esac
