"""
This file does not modify the original Whisper code.

it however:
1. Demonstrates the current behaviour of the Whisper CLI.
2. Implements a privacy-safe wrapper function that behaves like a
   "fixed" CLI, i.e. it prints the transcript but doesn't save anything
   unless explicitly requested.
3. Uses two tests with the same structure as the original unit test:
   - TestAutomaticStorageCLIOriginal: proves the current behaviour
     by expecting files to be created.
   - TestAutomaticStorageCLIFix: proves that our wrapper avoids file
     creation by expecting no files.
"""

import unittest
import subprocess
import json
from pathlib import Path
import whisper  


# path declarations
BASE_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = BASE_DIR.parent                     
AUDIO_REL_PATH = Path("privacy_tests") / "data" / "raw" / "whisper_test.mp3"

# recommended fix: a privacy-safe "CLI-like" wrapper that doesn't auto-save by default
def privacy_safe_transcribe(
    audio_path: Path,
    model_name: str = "tiny",
    save_files: bool = False,
    output_dir: Path | None = None,
):
    
    # load the whisper model and transcribe normally
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))

    print(result["text"])

    # only write files if the user explicitly asked for it
    if save_files:
        # choose where to save the files (defaults to project root)
        out_dir = output_dir or PROJECT_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)

        # strip extension from the audio path so output files match Whisper's naming
        base_name = audio_path.with_suffix("").name  

        # write .txt version of the transcript
        txt_path = out_dir / f"{base_name}.txt"
        txt_path.write_text(result["text"], encoding="utf-8")
        
        # write .json dump (similar to Whisper's segment metadata format)
        json_path = out_dir / f"{base_name}.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # return transcription result
    return result


# Test 1: original behaviour of the real CLI 
class TestAutomaticStorageCLIOriginal(unittest.TestCase):
    def test_cli_creates_output_files_current(self):
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

        # the assertion here is slightly different from the original - as we want the test to pass if the files are saved in this case (as we already know of this issue)
        self.assertGreater(
            len(saved),
            0,
            f"Whisper CLI created output files in project root: {saved}",
        )


# Test 2: fixed behaviour using our privacy_safe_transcribe() wrapper
class TestAutomaticStorageCLIFix(unittest.TestCase):
    def test_cli_creates_no_output_files_with_fix(self):
        """
        --- Proposed Fix ---.

        This test uses the same logic, but calls privacy_safe_transcribe() instead of 'python -m whisper'.

        It expects that no files are created when save_files=False.
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

        # now, instead, call the fixed function
        privacy_safe_transcribe(
            audio_path=audio_path,
            model_name="tiny",
            save_files=False,     
            output_dir=project_root,
        )

        # collect all files that are created in the repo root
        saved = []
        for ext in exts:
            f = project_root / f"whisper_test.{ext}"
            if f.exists():
                saved.append(f.name)

        # the test will FAIL if any files are created now
        self.assertEqual(
            len(saved),
            0,
            f"Privacy-safe wrapper created output files in project root: {saved}",
        )



if __name__ == "__main__":
    unittest.main()
