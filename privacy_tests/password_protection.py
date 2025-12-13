"""
Original Whisper CLI automatically saved files w/o user knowledge
This wrapper prints transcript and saves NOTHING by default
If saving is explicitly requested, it saves ONE ENCRYPTED file instead of plaintext
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import unittest
from pathlib import Path
from typing import Optional

import whisper
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BASE_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = BASE_DIR.parent                      
AUDIO_REL_PATH = Path("privacy_tests") / "data" / "raw" / "whisper_test.mp3"

EXPECTED_PASSWORD = "demo-password"

# --- encryption helpers ---

# compute hash from string
def _key_from_password(password: str) -> bytes:
    if not password:
        raise ValueError("Password must not be empty.")
    return hashlib.sha256(password.encode("utf-8")).digest()


def encrypt_text(text: str, password: str) -> bytes:
    # Returns bytes you can write to disk [nonce (12 bytes) + ciphertext]
    key = _key_from_password(password)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, text.encode("utf-8"), None)
    return nonce + ciphertext

# --- Privacy-safe wrapper ---
def privacy_safe_transcribe(
    audio_path: Path,
    model_name: str = "tiny",
    save_encrypted: bool = False,
    password: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    """
    - Always prints transcript
    - Saves nothing by default
    - If save_encrypted=True: writes ONE encrypted file: <audio_basename>.enc
    - If password is wrong: raise ValueError and do NOT write any encrypted file
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    text = result["text"]
    print(text)

    if not save_encrypted:
        return result

    if not password:
        raise ValueError("password is required when save_encrypted=True")

    if password != EXPECTED_PASSWORD:
        raise ValueError("Incorrect password: access denied.")

    out_dir = output_dir or PROJECT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    base = audio_path.with_suffix("").name
    enc_path = out_dir / f"{base}.enc"

    blob = encrypt_text(text, password=password)

    enc_path.write_bytes(blob)
    return result

# --- TESTS ---

# Clean Whisper CLI plaintext outputs + our demo encrypted output
def _cleanup_outputs(project_root: Path, base: str = "whisper_test") -> None:
    for name in [
        f"{base}.txt", f"{base}.json", f"{base}.srt", f"{base}.vtt", f"{base}.tsv",
        f"{base}.enc",
    ]:
        p = project_root / name
        if p.exists():
            p.unlink()

# Whisper CLI auto-saves files - this test confirms that behaviour
class TestAutomaticStorageCLIOriginal(unittest.TestCase):
    def test_cli_creates_output_files_current(self):
        project_root = Path(__file__).resolve().parent.parent
        audio_path = project_root / AUDIO_REL_PATH
        self.assertTrue(audio_path.exists(), f"Audio file not found: {audio_path}")

        _cleanup_outputs(project_root)

        subprocess.run(
            ["python", "-m", "whisper", str(audio_path), "--model", "tiny"],
            cwd=project_root,
            check=True,
        )

        # confirm at least one output exists
        created = []
        for ext in ["txt", "json", "srt", "vtt", "tsv"]:
            if (project_root / f"whisper_test.{ext}").exists():
                created.append(ext)

        self.assertGreater(len(created), 0, f"Expected CLI to create outputs, found none. created={created}")


class TestEncryptionPasswordCorrect(unittest.TestCase):
    def test_correct_password_creates_encrypted_file(self):
        project_root = Path(__file__).resolve().parent.parent
        audio_path = project_root / AUDIO_REL_PATH
        self.assertTrue(audio_path.exists(), f"Audio file not found: {audio_path}")

        _cleanup_outputs(project_root)

        privacy_safe_transcribe(
            audio_path=audio_path,
            model_name="tiny",
            save_encrypted=True,
            password="demo-password",  # correct password
            output_dir=project_root,
        )

        self.assertTrue(
            (project_root / "whisper_test.enc").exists(),
            "Expected encrypted output whisper_test.enc to be created with correct password.",
        )


class TestEncryptionPasswordIncorrect(unittest.TestCase):
    def test_incorrect_password_raises_and_creates_no_file(self):
        project_root = Path(__file__).resolve().parent.parent
        audio_path = project_root / AUDIO_REL_PATH
        self.assertTrue(audio_path.exists(), f"Audio file not found: {audio_path}")

        _cleanup_outputs(project_root)

        with self.assertRaises(ValueError):
            privacy_safe_transcribe(
                audio_path=audio_path,
                model_name="tiny",
                save_encrypted=True,
                password="wrong-password",  # incorrect password
                output_dir=project_root,
            )

        self.assertFalse(
            (project_root / "whisper_test.enc").exists(),
            "Encrypted file should NOT be created when password is incorrect.",
        )


if __name__ == "__main__":
    unittest.main()
