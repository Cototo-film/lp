#!/usr/bin/env bash
# Downloads the offline speech models used by analyze.py (VAD + SenseVoice ASR).
# Source: sherpa-onnx GitHub releases (redirects to objects.githubusercontent.com).
# Total download ~1GB (tar.bz2), ~230MB kept on disk (int8 model only).
set -euo pipefail
DIR="${1:-$(cd "$(dirname "$0")" && pwd)/models}"
mkdir -p "$DIR"
cd "$DIR"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
[ -f silero_vad.onnx ] || curl -sS -L -o silero_vad.onnx "$BASE/silero_vad.onnx"
if [ ! -f sense-voice/model.int8.onnx ]; then
  T=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
  curl -sS -L -o "$T.tar.bz2" "$BASE/$T.tar.bz2"
  tar xjf "$T.tar.bz2" --wildcards "*/model.int8.onnx" "*/tokens.txt"
  rm -f "$T.tar.bz2"
  mv "$T" sense-voice
fi
echo "models ready in $DIR"
ls -la "$DIR" "$DIR/sense-voice"
