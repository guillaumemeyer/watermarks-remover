"""Tests for the model-free keyed-Gumbel (Aaronson EXP) detector.

Exercises the replay arithmetic of detect_gumbel.py with a toy keyed
Gumbel-max generator, the TextDetector protocol in text_detectors.py, and
the gumbel evaluator wiring in rewrite_text.py.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import detect_gumbel
import rewrite_text
import text_detectors

KEY_HEX = "0x" + "ab" * 16  # 16 raw bytes
KEY_HEX_OTHER = "0x" + "cd" * 16
KEY_BYTES = detect_gumbel._normalize_key(KEY_HEX)


# S311: deterministic toy RNG for the sampler tests — never used for secrets.
def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311


_WORDS = [f"w{i}" for i in range(64)]
_WORD_IDS = [detect_gumbel._token_id(w) for w in _WORDS]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("WATERMARKS_GUMBEL_KEY", raising=False)


# --- Toy keyed Gumbel-max generator (mirrors paper Section 2) ----------------


def _gumbel(u: list[float]) -> list[float]:
    return [-math.log(-math.log(x)) for x in u]


def _keyed_uniforms(key_bytes: bytes, window_ids: tuple[int, ...], vocab: list[int]) -> list[float]:
    seed = detect_gumbel._seed(key_bytes, window_ids)
    return [detect_gumbel._uniform(seed, v) for v in vocab]


def _generate(
    key_bytes: bytes | None,
    *,
    n: int,
    vocab: int,
    window: int,
    rng: random.Random,
) -> list[int]:
    """Keyed (or RNG) Gumbel-max sampler with generator-side window masking."""
    logits = [rng.uniform(-1.0, 1.0) for _ in range(vocab)]
    ids: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(n):
        win = tuple(ids[-window:])
        use_keyed = key_bytes is not None and len(win) == window and win not in seen
        if use_keyed:
            seen.add(win)
        if use_keyed:
            u = _keyed_uniforms(key_bytes, win, list(range(vocab)))
        else:
            u = [rng.random() for _ in range(vocab)]
        g = _gumbel(u)
        ids.append(max(range(vocab), key=lambda v: logits[v] + g[v]))
    return ids


def _marked_text(n: int, rng: random.Random, key_hex: str = KEY_HEX) -> str:
    """Marked *text* whose simple-tokenizer ids replay under the same key.

    Generates over the word->id space detect_gumbel uses for plain text, so
    the real detector re-derives identical seeds and uniforms.
    """
    key_bytes = detect_gumbel._normalize_key(key_hex)
    logits = [rng.uniform(-1.0, 1.0) for _ in _WORDS]
    seq_ids: list[int] = []
    seq_words: list[str] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(n):
        win = tuple(seq_ids[-4:])
        use_keyed = len(win) == 4 and win not in seen
        if use_keyed:
            seen.add(win)
        if use_keyed:
            u = _keyed_uniforms(key_bytes, win, _WORD_IDS)
        else:
            u = [rng.random() for _ in _WORD_IDS]
        g = _gumbel(u)
        chosen = max(range(len(_WORD_IDS)), key=lambda v: logits[v] + g[v])
        seq_ids.append(_WORD_IDS[chosen])
        seq_words.append(_WORDS[chosen])
    return " ".join(seq_words)


# --- Replay arithmetic -------------------------------------------------------


def test_marked_sequence_is_detected():
    rng = _rng(1)
    ids = _generate(KEY_BYTES, n=700, vocab=32, window=4, rng=rng)
    report = detect_gumbel.detect_token_ids(ids, KEY_HEX)
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["p_value"] < 1e-12


def test_unmarked_sequence_at_chance():
    rng = _rng(2)
    ids = _generate(None, n=700, vocab=32, window=4, rng=rng)
    report = detect_gumbel.detect_token_ids(ids, KEY_HEX)
    assert report["is_watermarked"] is False
    assert report["p_value"] > 0.01


def test_wrong_key_at_chance():
    rng = _rng(3)
    ids = _generate(KEY_BYTES, n=700, vocab=32, window=4, rng=rng)
    report = detect_gumbel.detect_token_ids(ids, KEY_HEX_OTHER)
    assert report["is_watermarked"] is False
    assert report["p_value"] > 0.01


def test_marked_text_roundtrip_through_simple_tokenizer():
    rng = _rng(4)
    text = _marked_text(700, rng)
    report = detect_gumbel.detect_text(text, KEY_HEX)
    assert report["is_watermarked"] is True
    assert report["p_value"] < 1e-9
    # the same text under a different key sits at chance
    other = detect_gumbel.detect_text(text, KEY_HEX_OTHER)
    assert other["is_watermarked"] is False
    assert other["p_value"] > 0.01


def test_detection_is_deterministic():
    rng = _rng(5)
    text = _marked_text(300, rng)
    assert detect_gumbel.detect_text(text, KEY_HEX) == detect_gumbel.detect_text(text, KEY_HEX)


def test_repeated_window_masking_skip_rule():
    ids = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    report = detect_gumbel.detect_token_ids(ids, KEY_HEX, window=2)
    # positions 0-1 lack context; windows (0,1)/(1,0) are keyed once each and
    # every later occurrence is a repeated window the detector must skip
    assert report["skipped_no_context"] == 2
    assert report["skipped_repeated"] == 16
    assert report["counted"] == 2
    assert report["is_watermarked"] is False  # two tokens carry no evidence
    # without masking every full-window position is counted
    report2 = detect_gumbel.detect_token_ids(ids, KEY_HEX, window=2, mask_repeated=False)
    assert report2["counted"] == 18
    assert report2["skipped_repeated"] == 0


def test_short_text_counts_nothing():
    report = detect_gumbel.detect_text("a b c", KEY_HEX)
    assert report["counted"] == 0
    assert report["is_watermarked"] is False
    assert report["p_value"] == 1.0


def test_key_normalization():
    assert detect_gumbel._normalize_key("0x" + "ab" * 8) == b"\xab" * 8
    assert detect_gumbel._normalize_key("0xAB12") == b"\xab\x12"
    assert detect_gumbel._normalize_key("s3cret key!") == b"s3cret key!"
    with pytest.raises(ValueError):
        detect_gumbel._normalize_key("0xzz")
    with pytest.raises(ValueError):
        detect_gumbel._normalize_key("0xabc")


def test_poisson_survival_sanity():
    # Gamma(1,1) survival at s=1 is e^-1; at s=n the p-value is ~0.5
    assert detect_gumbel._poisson_survival(1.0, 1) == pytest.approx(math.exp(-1.0))
    assert detect_gumbel._poisson_survival(10.0, 10) == pytest.approx(0.5, abs=0.05)
    assert detect_gumbel._poisson_survival(50.0, 10) < 1e-6
    assert detect_gumbel._poisson_survival(0.0, 10) == 1.0


def test_poisson_survival_matches_scipy_gammaincc():
    pytest.importorskip("scipy.special")
    from scipy.special import gammaincc

    for n in (1, 5, 50, 200):
        for s in (n * 0.5, n, n * 1.5, n * 2.0):
            ours = detect_gumbel._poisson_survival(s, n)
            ref = float(gammaincc(n, s))
            assert ours == pytest.approx(ref, rel=1e-6, abs=1e-300)


def test_load_token_ids_json_and_lines():
    assert detect_gumbel.load_token_ids("[1, 2, 3]") == [1, 2, 3]
    assert detect_gumbel.load_token_ids("1\n0x2\n3") == [1, 2, 3]
    with pytest.raises(ValueError):
        detect_gumbel.load_token_ids('[1, "x"]')


def test_token_id_out_of_range_rejected():
    with pytest.raises(ValueError):
        detect_gumbel.detect_token_ids([-1, 2, 3], KEY_HEX)
    with pytest.raises(ValueError):
        detect_gumbel.detect_token_ids([1 << 64, 2], KEY_HEX)


# --- CLI ---------------------------------------------------------------------


def _run_cli(
    *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "WATERMARKS_GUMBEL_KEY"}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_gumbel.py"), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )


def test_cli_detects_marked_text(tmp_path):
    rng = _rng(6)
    src = tmp_path / "marked.txt"
    src.write_text(_marked_text(500, rng), encoding="utf-8")
    r = _run_cli(str(src), "--key", KEY_HEX, "--json")
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["is_watermarked"] is True


def test_cli_token_ids_mode(tmp_path):
    rng = _rng(7)
    ids = _generate(KEY_BYTES, n=500, vocab=32, window=4, rng=rng)
    src = tmp_path / "ids.json"
    src.write_text(json.dumps(ids), encoding="utf-8")
    r = _run_cli(str(src), "--tokens", "--key", KEY_HEX, "--json")
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["is_watermarked"] is True


def test_cli_requires_key(tmp_path):
    src = tmp_path / "plain.txt"
    src.write_text("hello world", encoding="utf-8")
    r = _run_cli(str(src))
    assert r.returncode == 2
    assert "key" in (r.stderr + r.stdout).lower()


def test_cli_reads_key_from_env(tmp_path):
    rng = _rng(8)
    src = tmp_path / "marked.txt"
    src.write_text(_marked_text(400, rng), encoding="utf-8")
    r = _run_cli(str(src), "--json", env_extra={"WATERMARKS_GUMBEL_KEY": KEY_HEX})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["is_watermarked"] is True


def test_cli_bad_hex_key(tmp_path):
    src = tmp_path / "plain.txt"
    src.write_text("hello world", encoding="utf-8")
    r = _run_cli(str(src), "--key", "0xzz", "--json")
    assert r.returncode == 2


# --- GumbelTextDetector protocol ---------------------------------------------


def test_gumbel_detector_unconfigured():
    assert text_detectors.GumbelTextDetector().available() is False
    report = text_detectors.GumbelTextDetector().detect("hello")
    assert report["available"] is False
    assert "WATERMARKS_GUMBEL_KEY" in report["error"]


def test_gumbel_detector_env_key(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GUMBEL_KEY", KEY_HEX)
    rng = _rng(9)
    report = text_detectors.GumbelTextDetector().detect(_marked_text(400, rng))
    assert report["available"] is True
    assert report["is_watermarked"] is True
    assert report["detector"] == "gumbel"


def test_gumbel_detector_constructor_key():
    rng = _rng(10)
    report = text_detectors.GumbelTextDetector(key=KEY_HEX).detect(_marked_text(400, rng))
    assert report["available"] is True
    assert report["is_watermarked"] is True


def test_gumbel_in_detector_registry(monkeypatch):
    names = {d.name for d in text_detectors.all_detectors()}
    assert "gumbel" in names
    monkeypatch.setenv("WATERMARKS_GUMBEL_KEY", KEY_HEX)
    assert text_detectors.detector_status()["gumbel"] is True


# --- rewrite_text.py gumbel evaluator ----------------------------------------


def _rewrite_kwargs(**overrides):
    kwargs = dict(
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="paraphrase",
        lang="French",
        original_lang="English",
        timeout=10,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
    )
    kwargs.update(overrides)
    return kwargs


def test_rewrite_gumbel_evaluator_clears_mark(monkeypatch):
    rng = _rng(11)
    original = _marked_text(400, rng)
    monkeypatch.setattr(
        rewrite_text,
        "call_ollama",
        lambda *a, **k: "alpha beta gamma delta epsilon zeta eta theta",
    )
    _out, info = rewrite_text.rewrite(original, **_rewrite_kwargs(gumbel_key=KEY_HEX))
    assert info["evaluator"] == "gumbel"
    assert info["passed"] is True
    assert info["attempts_made"] == 1
    g = info["gumbel"]
    assert g["before"]["is_watermarked"] is True
    assert g["after"]["is_watermarked"] is False
    assert g["cleared"] is True
    assert info["candidate_scores"][0]["evaluation"]["detector"] == "gumbel"
    assert _out == "alpha beta gamma delta epsilon zeta eta theta"


def test_gumbel_takes_priority_over_markllm(monkeypatch):
    built = {}

    class _FakeMarkLLM:
        def __init__(self, **kwargs):
            built["markllm"] = True

        def available(self):
            return True

        def detect(self, text):
            return {"detector": "markllm", "available": True, "is_watermarked": False, "score": 0.0}

    monkeypatch.setattr(rewrite_text, "MarkLLMTextDetector", _FakeMarkLLM)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "alpha beta gamma delta")
    _out, info = rewrite_text.rewrite(
        "plain text here",
        **_rewrite_kwargs(gumbel_key=KEY_HEX, markllm_scheme="kgw", markllm_dir="/x"),
    )
    assert built["markllm"] is True
    assert info["evaluator"] == "gumbel"
    assert "markllm" in info and "gumbel" in info
    assert info["passed"] is True


def test_gumbel_key_does_not_build_markllm(monkeypatch):
    monkeypatch.setattr(
        rewrite_text,
        "MarkLLMTextDetector",
        lambda *a, **k: pytest.fail("markllm must not be built when only gumbel is set"),
    )
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "alpha beta gamma delta")
    _out, info = rewrite_text.rewrite("plain text here", **_rewrite_kwargs(gumbel_key=KEY_HEX))
    assert info["evaluator"] == "gumbel"
    assert info["passed"] is True
    assert "markllm" not in info


def test_no_gumbel_without_key(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "alpha beta gamma delta")
    _out, info = rewrite_text.rewrite("plain text here", **_rewrite_kwargs())
    assert "gumbel" not in info
    assert info["evaluator"] == "lexical-divergence"


def test_gumbel_key_flag_defaults_from_env(monkeypatch):
    monkeypatch.setenv("WATERMARKS_GUMBEL_KEY", KEY_HEX)
    args = rewrite_text.build_parser().parse_args(["x.txt"])
    assert args.gumbel_key == KEY_HEX
    monkeypatch.delenv("WATERMARKS_GUMBEL_KEY", raising=False)
    args = rewrite_text.build_parser().parse_args(["x.txt"])
    assert args.gumbel_key is None
