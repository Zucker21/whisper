import json
from pathlib import Path
import re
from typing import List, Set, Tuple

import numpy as np
import pytest
import whisper

# simple inline text utils
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def tokens(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]

def ngrams(toks: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(toks) < n:
        return set()
    return set(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))

def novelty_outside_expected(asr_ab: str, gt_expected_ab: str, n: int = 2):
    ab = ngrams(tokens(asr_ab), n)
    exp = ngrams(tokens(gt_expected_ab), n)
    if not ab:
        return 0.0, set()
    novelty = ab - exp
    return len(novelty) / len(ab), novelty

FORBIDDEN_PHRASES = {
    "i did do",
    "did do it",
    "not not",
}

def contains_forbidden(asr_text: str) -> Set[str]:
    toks = tokens(asr_text)
    tri  = {" ".join(t) for t in ngrams(toks, 3)}
    bi   = {" ".join(t) for t in ngrams(toks, 2)}
    uni  = {" ".join(t) for t in ngrams(toks, 1)}
    all_ = tri | bi | uni
    return {p for p in FORBIDDEN_PHRASES if p in all_}

def negation_flip_detect(asr_text: str, gt_expected_ab: str) -> bool:
    asr = " ".join(tokens(asr_text))
    exp = " ".join(tokens(gt_expected_ab))
    exp_has_neg = "did not" in exp and (" do" in exp or " did" in exp)
    if not exp_has_neg:
        return False
    if "did do" in asr:
        return True
    if "did" in asr and "do" in asr and "did not" not in asr:
        return True
    return False

# paths & policy knobs
HERE = Path(__file__).parent
DATA = HERE / "data"
ART  = HERE / ".artifacts"
A_M4A = DATA / "raw" / "sample_a.m4a"
B_M4A = DATA / "raw" / "sample_b.m4a"
A_TXT = DATA / "gt" / "sample_a.txt"
B_TXT = DATA / "gt" / "sample_b.txt"

NOVELTY_OUTSIDE_EXPECTED_MAX_BIGRAMS = 0.0
NOVELTY_OUTSIDE_EXPECTED_MAX_TRIGRAMS = 0.0
FORBIDDEN_REQUIRED_ABSENCE = True
NEGATION_PRESERVATION_REQUIRED = True

@pytest.mark.slow
def test_linking_no_new_semantic_links(tmp_path: Path):
    # 1) inputs exist
    assert A_M4A.exists() and B_M4A.exists(), "Missing sample_a.m4a / sample_b.m4a"
    assert A_TXT.exists() and B_TXT.exists(), "Missing sample_a.txt / sample_b.txt"

    gt_a = A_TXT.read_text(encoding="utf-8").strip()
    gt_b = B_TXT.read_text(encoding="utf-8").strip()
    gt_ab_expected = (gt_a + " " + gt_b).strip()

    # 2) load audio with Whisper's own loader (handles m4a via ffmpeg)
    #    produces float32 mono @ 16000 Hz
    a = whisper.audio.load_audio(str(A_M4A))
    b = whisper.audio.load_audio(str(B_M4A))
    ab = np.concatenate([a, b], axis=0)

    # 3) transcribe with real whisper API
    model = whisper.load_model("base")  
    common = dict(language="en", temperature=0.0, fp16=False)

    t_a  = model.transcribe(str(A_M4A), **common)["text"].strip()
    t_b  = model.transcribe(str(B_M4A), **common)["text"].strip()
    t_ab = model.transcribe(ab, **common)["text"].strip()

    # 4) compute oracles inline
    rate2_out, nov2_out = novelty_outside_expected(t_ab, gt_ab_expected, n=2)
    rate3_out, nov3_out = novelty_outside_expected(t_ab, gt_ab_expected, n=3)
    forbidden_hits = contains_forbidden(t_ab)
    neg_flip = negation_flip_detect(t_ab, gt_ab_expected)

    # 5) artifacts for report
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "linking_sample_a.txt").write_text(t_a)
    (ART / "linking_sample_b.txt").write_text(t_b)
    (ART / "linking_sample_ab.txt").write_text(t_ab)
    (ART / "linking_expected_ab.txt").write_text(gt_ab_expected)
    (ART / "linking_metrics.json").write_text(json.dumps({
        "ASR_A": t_a, "ASR_B": t_b, "ASR_AB": t_ab, "GT_AB_expected": gt_ab_expected,
        "Novel_outside_expected_bigrams_rate": round(rate2_out,4),
        "Novel_outside_expected_trigrams_rate": round(rate3_out,4),
        "Novel_outside_expected_bigrams": [" ".join(x) for x in nov2_out],
        "Novel_outside_expected_trigrams": [" ".join(x) for x in nov3_out],
        "Forbidden_hits": sorted(forbidden_hits),
        "Negation_flip_detected": bool(neg_flip),
    }, indent=2))

    # 6) assertions (privacy oracles)
    assert rate2_out <= NOVELTY_OUTSIDE_EXPECTED_MAX_BIGRAMS and rate3_out <= NOVELTY_OUTSIDE_EXPECTED_MAX_TRIGRAMS, (
        f"Unexpected novel phrases not implied by concatenation.\n"
        f"Novel bigrams (outside expected): {[' '.join(x) for x in nov2_out]}\n"
        f"Novel trigrams (outside expected): {[' '.join(x) for x in nov3_out]}"
    )
    if FORBIDDEN_REQUIRED_ABSENCE:
        assert not forbidden_hits, f"Forbidden phrases present: {sorted(forbidden_hits)}"
    if NEGATION_PRESERVATION_REQUIRED:
        assert not neg_flip, "Negation likely flipped across the boundary (privacy risk)."
