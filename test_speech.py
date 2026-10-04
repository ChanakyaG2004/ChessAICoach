import io
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from speech import LocalSpeech, SpeechError, _validate_recording


def recording(seconds=0.5, rate=16000, channels=1, amplitude=1000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        # Alternate sign to provide a nonzero synthetic signal.
        samples = [amplitude if index % 2 else -amplitude for index in range(int(rate * seconds) * channels)]
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return output.getvalue()


class LocalSpeechTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name) / "model.bin"
        with self.model.open("wb") as model:
            model.truncate(1024 * 1024)
        self.speech = LocalSpeech(self.model, whisper_path="/usr/bin/true",
                                  say_path="/usr/bin/true", ffmpeg_path="/usr/bin/true")

    def test_rejects_invalid_silent_truncated_and_excessive_recordings(self):
        for audio in (b"not wave", recording(rate=8000), recording(channels=2),
                      recording(amplitude=0), recording()[:-100], recording(seconds=.1),
                      recording(seconds=31), b"x" * (1024 * 1024 + 1)):
            with self.subTest(size=len(audio)), self.assertRaises(SpeechError):
                _validate_recording(audio)
        _validate_recording(recording())

    def test_missing_model_disables_input_without_disabling_read_aloud(self):
        self.model.unlink()
        status = self.speech.status()
        self.assertFalse(status["available"])
        self.assertTrue(status["tts_available"])
        self.assertIn("setup_voice.sh", status["reason"])
        with self.assertRaisesRegex(SpeechError, "English model"):
            self.speech.transcribe(recording())

    def test_transcript_is_normalized_and_recording_removed(self):
        seen = []
        def run(command, **kwargs):
            source = Path(command[command.index("-f") + 1])
            self.assertTrue(source.is_file())
            seen.append(source.parent)
            output = Path(command[command.index("-of") + 1] + ".txt")
            output.write_text(" What is a better move?\n Why? \n")
            self.assertNotIn("shell", kwargs)
        with patch("speech.subprocess.run", side_effect=run):
            text = self.speech.transcribe(recording())
        self.assertEqual(text, "What is a better move? Why?")
        self.assertTrue(seen)
        self.assertFalse(seen[0].exists())

    def test_timeout_cleans_files_and_releases_busy_guard(self):
        seen = []
        def timeout(command, **kwargs):
            seen.append(Path(command[command.index("-f") + 1]).parent)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with patch("speech.subprocess.run", side_effect=timeout):
            with self.assertRaisesRegex(SpeechError, "too long"):
                self.speech.transcribe(recording())
        self.assertFalse(seen[0].exists())
        self.assertTrue(self.speech._busy.acquire(blocking=False))
        self.speech._busy.release()

    def test_busy_request_does_not_launch_another_process(self):
        self.speech._busy.acquire()
        try:
            with patch("speech.subprocess.run") as run:
                with self.assertRaisesRegex(SpeechError, "still running"):
                    self.speech.transcribe(recording())
                run.assert_not_called()
        finally:
            self.speech._busy.release()

    def test_read_aloud_passes_text_as_file_and_cleans_output(self):
        seen = []
        spoken = "-f /tmp/anything; $(do_not_execute)"
        expected = recording()
        def run(command, **kwargs):
            if "-f" in command:
                source = Path(command[command.index("-f") + 1])
                self.assertEqual(source.read_text(), spoken)
                seen.append(source.parent)
                Path(command[command.index("-o") + 1]).write_bytes(b"test aiff")
            else:
                Path(command[-1]).write_bytes(expected)
            self.assertNotIn("shell", kwargs)
        with patch("speech.subprocess.run", side_effect=run):
            audio = self.speech.synthesize(spoken)
        self.assertEqual(audio, expected)
        self.assertFalse(seen[0].exists())
        for invalid in ("", "  ", "x" * 3001):
            with self.assertRaises(SpeechError):
                self.speech.synthesize(invalid)

    def test_empty_speech_output_is_an_error(self):
        def run(command, **kwargs):
            if "-f" in command:
                Path(command[command.index("-o") + 1]).write_bytes(b"aiff header")
            else:
                # ffmpeg writes an extended WAV header even when say supplied no audio.
                Path(command[-1]).write_bytes(recording(seconds=0) + b"padding")
        with patch("speech.subprocess.run", side_effect=run):
            with self.assertRaisesRegex(SpeechError, "empty speech"):
                self.speech.synthesize("What is a better move?")


if __name__ == "__main__":
    unittest.main()
