"""Tests for the forensic stealth evaluation benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
BENCH = ROOT / "benchmarks"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(BENCH))


def _run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCH / "bench_forensic.py"), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _write_random_png(path: Path, w: int, h: int) -> None:
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    img.save(path)


def test_forensic_eval_result_structure():
    from bench_forensic import ForensicEvalResult

    r = ForensicEvalResult(
        engine="test",
        n_images=10,
        mean_stealth=0.3,
        detection_rate=0.2,
        mean_psnr=35.0,
        composite_score=0.56,
    )
    assert r.engine == "test"
    assert 0.0 <= r.detection_rate <= 1.0


def test_evaluate_empty_dir(tmp_path):
    from bench_forensic import evaluate_engine_outputs

    result = evaluate_engine_outputs(tmp_path, "test")
    assert result.n_images == 0
    assert result.composite_score == 0.0


def test_evaluate_with_images(tmp_path):
    from bench_forensic import evaluate_engine_outputs

    for i in range(3):
        _write_random_png(tmp_path / f"img{i}.png", 64, 64)
    result = evaluate_engine_outputs(tmp_path, "test")
    assert result.n_images == 3
    assert 0.0 <= result.mean_stealth <= 1.0
    assert 0.0 <= result.detection_rate <= 1.0


def test_cli_bad_dir():
    result = _run_bench("--eval-dir", "/nonexistent", "--json")
    assert result.returncode == 2


def test_cli_json_output(tmp_path):
    import json

    sub = tmp_path / "unknown_marknull_0.5"
    sub.mkdir()
    _write_random_png(sub / "img.png", 64, 64)
    result = _run_bench("--eval-dir", str(tmp_path), "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "results" in report
    assert "n_dirs" in report
    assert report["n_dirs"] >= 1
