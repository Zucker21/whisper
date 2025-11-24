import unittest
from unittest.mock import patch
import whisper  
import os
import tempfile
import subprocess
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, "data", "raw", "whisper_test.mp3")
AUDIO_REL_PATH = Path("privacy_tests") / "data" / "raw" / "whisper_test.mp3"

class TestWhisperConnection(unittest.TestCase):
    "No compliance"
    @patch("socket.create_connection") #Regular sockect connection
    @patch("requests.sessions.Session.request") #HTTP connection
    def test_no_network_calls(self, mock_requests, mock_socket):
        
        'Ensure that running Whisper locally does not make any network calls.'
        
        model = whisper.load_model("tiny")  
        model.transcribe(AUDIO_PATH)

        # Check that no network calls were made
        mock_requests.assert_not_called()
        mock_socket.assert_not_called()

        print("Whisper ran locally, no network calls detected.")


class TestBackgroundNoise(unittest.TestCase):
    "Unawareness"
    def test_catch_background_noise(self):
        """
        Test to ensure that background noise is detected in an audio file.
        Does not necessarily identify a person. Let's say we see it as just one person talking, it is hallucinated twice that he is drunk.
        """
        model = whisper.load_model("turbo") 
        result = model.transcribe(AUDIO_PATH)
        #translation = whisper.decoding.detect_language(result["text"])
        #print(translation["text"])
        text = result["text"]
        print(text)
        self.assertIn("ik ben zat", text.lower(), "Background noise 'Ik ben zat' detected in transcription.")


class TestAutomaticStorageCLI(unittest.TestCase):
    """Data disclosure – Whisper CLI saves files to the project root."""
    def test_cli_creates_output_files(self):
        """
        Run the Whisper CLI in the *project root* and confirm that it
        automatically creates output files
        """

        # define the project root, path to the audio file
        project_root = Path(__file__).resolve().parent.parent
        audio_path = project_root / AUDIO_REL_PATH
        self.assertTrue(audio_path.exists(), f"Audio file not found: {audio_path}")

        # list all extensions whisper generates
        exts = ["txt", "json", "srt", "vtt", "tsv"]

        # firstly, clean up any existing old files before running the test this time
        for ext in exts:
            f = project_root / f"whisper_test.{ext}"
            if f.exists():
                f.unlink()

        # now call whisper in the *project root*, assuming files are going to be saved there
        subprocess.run(
            ["python", "-m", "whisper", str(audio_path), "--model", "tiny"],
            cwd=project_root,   
            check=True,
        )

        # collect all files that whisper created in the repo root
        saved = []
        for ext in exts:
            f = project_root / f"whisper_test.{ext}"
            if f.exists():
                saved.append(f.name)

        # the test will FAIL if whisper creates any files
        self.assertEqual(
            len(saved),
            0,
            f"Whisper CLI created output files in project root: {saved}",
        )


if __name__ == "__main__":
    unittest.main()
