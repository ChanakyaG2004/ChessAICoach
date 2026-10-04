"""Optional, local speech input/output. Importing this module never opens a mic.

Install voice dependencies explicitly with ``bash scripts/setup_voice.sh``.
Recordings and generated speech live in temporary directories that are removed
after each request. No network requests are made by this module.
"""
from __future__ import annotations

from array import array
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import wave


MAX_AUDIO_BYTES = 1024 * 1024
MAX_AUDIO_SECONDS = 30
MAX_SPEECH_TEXT = 3000
_RATE = 16000


class SpeechError(RuntimeError):
    """A voice request could not be completed; text chat remains available."""


def _executable(name: str, configured: str | None = None) -> str | None:
    candidates = [configured] if configured else [shutil.which(name),
        f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}", f"/usr/bin/{name}"]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
    return None


def _validate_recording(data: bytes) -> None:
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_AUDIO_BYTES:
        raise SpeechError("Record a WAV clip no larger than 1 MiB (up to 30 seconds).")
    try:
        with wave.open(io.BytesIO(data), "rb") as recording:
            if (recording.getnchannels(), recording.getsampwidth(),
                    recording.getframerate(), recording.getcomptype()) != (1, 2, _RATE, "NONE"):
                raise SpeechError("Voice input must be mono PCM16 WAV at 16 kHz.")
            frames = recording.getnframes()
            if not _RATE // 4 <= frames <= _RATE * MAX_AUDIO_SECONDS:
                raise SpeechError("Record between 0.25 and 30 seconds of speech.")
            samples = recording.readframes(frames)
            if len(samples) != frames * 2:
                raise SpeechError("The voice recording is incomplete. Please record again.")
    except (wave.Error, EOFError, ValueError) as exc:
        raise SpeechError("The voice recording is not a readable WAV file.") from exc
    levels = array("h", samples)
    if sys.byteorder != "little":
        levels.byteswap()
    rms = math.sqrt(sum(sample * sample for sample in levels) / len(levels))
    if rms < 40:
        raise SpeechError("No audible speech was captured. Check your microphone and try again.")


class LocalSpeech:
    """Bounded whisper.cpp transcription and macOS speech-file synthesis.

    ``available`` in status means speech input is ready. ``tts_available`` is
    independent, so read-aloud can work even before a Whisper model is installed.
    """

    def __init__(self, model_path: str | Path | None = None, *,
                 whisper_path: str | None = None, say_path: str | None = None,
                 ffmpeg_path: str | None = None):
        self.whisper = _executable("whisper-cli", whisper_path or os.environ.get("WHISPER_EXECUTABLE"))
        self.say = _executable("say", say_path)
        self.ffmpeg = _executable("ffmpeg", ffmpeg_path)
        configured = model_path or os.environ.get("WHISPER_MODEL_PATH")
        self.model = Path(configured).expanduser() if configured else Path(__file__).parent / "models" / "ggml-base.en.bin"
        self._busy = threading.Lock()

    def status(self) -> dict:
        if not self.whisper:
            reason = "Local speech input needs whisper.cpp. Run bash scripts/setup_voice.sh."
        elif not self.model.is_file() or self.model.stat().st_size < 1024 * 1024:
            reason = "Local speech input needs its English model. Run bash scripts/setup_voice.sh."
        else:
            reason = "Local English voice input is ready. Audio stays on this Mac."
        return {"available": bool(self.whisper and self.model.is_file() and self.model.stat().st_size >= 1024 * 1024),
                "reason": reason, "tts_available": bool(self.say and self.ffmpeg)}

    @staticmethod
    def _run(command: list[str], timeout: float, failure: str) -> None:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise SpeechError("Voice processing took too long. Please try a shorter request.") from exc
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SpeechError(failure) from exc

    def transcribe(self, wav_bytes: bytes) -> str:
        _validate_recording(wav_bytes)
        capability = self.status()
        if not capability["available"]:
            raise SpeechError(capability["reason"])
        if not self._busy.acquire(blocking=False):
            raise SpeechError("Another voice request is still running. Try again in a moment.")
        try:
            with tempfile.TemporaryDirectory(prefix="chess-voice-") as directory:
                source = Path(directory) / "recording.wav"
                output = Path(directory) / "transcript"
                source.write_bytes(wav_bytes)
                self._run([self.whisper, "-m", str(self.model), "-f", str(source),
                           "-l", "en", "-t", "4", "-nt", "-np", "-sns", "-otxt", "-of", str(output)],
                          60, "Local speech recognition failed. You can still type your question.")
                result = output.with_suffix(".txt")
                if not result.is_file() or result.stat().st_size > 16384:
                    raise SpeechError("No usable transcript was produced. Please try again.")
                transcript = " ".join(result.read_text(encoding="utf-8").split())
                if not transcript or transcript.lower() in {"[blank_audio]", "(silence)", "[silence]"}:
                    raise SpeechError("No speech was recognized. Please try again or type your question.")
                return transcript
        finally:
            self._busy.release()

    def synthesize(self, text: str) -> bytes:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPEECH_TEXT:
            raise SpeechError("Read-aloud needs between 1 and 3000 characters of text.")
        if not self.status()["tts_available"]:
            raise SpeechError("Local read-aloud needs macOS say and ffmpeg. Text chat is still available.")
        if not self._busy.acquire(blocking=False):
            raise SpeechError("Another voice request is still running. Try again in a moment.")
        try:
            with tempfile.TemporaryDirectory(prefix="chess-read-aloud-") as directory:
                source = Path(directory) / "reply.txt"
                aiff = Path(directory) / "reply.aiff"
                output = Path(directory) / "reply.wav"
                # A file argument prevents user text from being parsed as CLI flags.
                source.write_text(" ".join(text.replace("\x00", " ").split()), encoding="utf-8")
                self._run([self.say, "-r", "175", "-o", str(aiff), "-f", str(source)], 45,
                          "macOS could not create spoken audio. Read the reply in the chat instead.")
                if not aiff.is_file() or aiff.stat().st_size > 40 * 1024 * 1024:
                    raise SpeechError("The spoken reply could not be created within the audio limit.")
                self._run([self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                           "-i", str(aiff), "-t", "240", "-ac", "1", "-ar", str(_RATE),
                           "-c:a", "pcm_s16le", str(output)], 20,
                          "The spoken reply could not be converted to playable audio.")
                if not output.is_file() or not 44 < output.stat().st_size <= 8 * 1024 * 1024:
                    raise SpeechError("The spoken reply did not contain usable audio.")
                audio = output.read_bytes()
                try:
                    with wave.open(io.BytesIO(audio), "rb") as spoken:
                        if spoken.getnframes() == 0:
                            raise SpeechError("macOS returned empty speech audio. Restart the coach with access to macOS speech services.")
                except (wave.Error, EOFError) as exc:
                    raise SpeechError("The spoken reply is not a playable WAV file.") from exc
                return audio
        finally:
            self._busy.release()
