#!/bin/sh
# Explicit one-time setup. App startup never installs or downloads anything.
set -eu

voice_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
voice_model="$voice_root/models/ggml-base.en.bin"
voice_sha1=137c40403d78fd54d454da0f9bd998f78703390c

if [ "$(uname -s)" != Darwin ]; then
    echo 'This setup script supports macOS. Text coaching needs no voice setup.' >&2
    exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
    echo 'Install Homebrew from https://brew.sh, then run this script again.' >&2
    exit 1
fi
if ! command -v whisper-cli >/dev/null 2>&1; then
    HOMEBREW_NO_AUTO_UPDATE=1 brew install whisper.cpp
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    HOMEBREW_NO_AUTO_UPDATE=1 brew install ffmpeg
fi
mkdir -p "$voice_root/models"
if [ -f "$voice_model" ] && printf '%s  %s\n' "$voice_sha1" "$voice_model" | shasum -a 1 -c - >/dev/null 2>&1; then
    echo 'The local English voice model is already ready.'
    exit 0
fi
voice_partial=$(mktemp "$voice_root/models/voice-download.XXXXXX")
trap 'rm -f "$voice_partial"' EXIT HUP INT TERM
curl --fail --location --retry 3 --connect-timeout 20 --max-time 600 \
    'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin' \
    --output "$voice_partial"
printf '%s  %s\n' "$voice_sha1" "$voice_partial" | shasum -a 1 -c -
mv "$voice_partial" "$voice_model"
echo 'Optional local English voice is ready. Restart the coach if it is running.'
