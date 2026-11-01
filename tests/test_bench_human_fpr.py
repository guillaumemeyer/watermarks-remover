"""Tests for the human-text false-positive benchmark and its corpus builder.

No network: Gutenberg downloads are faked, and detector backends other than
stdlib stylometry are replaced with a stub.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_human_fpr as fpr
import fetch_human_corpus as fetch

SENTENCE = "The river was low in summer {i} and the boats sat idle at the landing all day."


def _book(n_paras: int = 30) -> str:
    body = "\n\n".join(
        f"CHAPTER {i}\n\n" + "\n".join([SENTENCE.format(i=i)] * 3) for i in range(1, n_paras + 1)
    )
    return (
        "\ufeffThe Project Gutenberg eBook of Life on the Mississippi\n"
        "licence text\n*** START OF THE PROJECT GUTENBERG EBOOK LIFE ***\n\n"
        f"{body}\n\n*** END OF THE PROJECT GUTENBERG EBOOK LIFE ***\nfooter licence\n"
    )


def test_strip_and_paragraphs_drop_boilerplate_and_headings():
    body = fetch.strip_gutenberg(_book(3))
    assert "licence" not in body and "footer" not in body
    paras = fetch.paragraphs(body)
    assert len(paras) == 3
    assert all(not p.startswith("CHAPTER") and "\n" not in p for p in paras)


def test_paragraphs_strip_italic_underscores_and_short_fragments():
    body = "_Short._\n\n" + " ".join(["_word_"] * 30)
    assert fetch.paragraphs(body) == [" ".join(["word"] * 30)]


def test_excerpts_are_deterministic_and_long_enough():
    paras = fetch.paragraphs(fetch.strip_gutenberg(_book()))
    a = fetch.excerpts(paras, 60, 3, random.Random("s"))  # noqa: S311
    b = fetch.excerpts(paras, 60, 3, random.Random("s"))  # noqa: S311
    assert a == b and len(a) == 3
    assert all(60 <= len(x.split()) <= 90 for x in a)
    assert len(set(a)) == 3


def test_build_writes_files_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "BOOKS", ((245, "Life on the Mississippi", "Mark Twain", 1883),))
    monkeypatch.setattr(fetch, "download", lambda *a, **k: _book())
    n = fetch.build(tmp_path, [60, 150], 2, 0, None, 1.0)
    assert n == 4
    rows = (tmp_path / "SOURCES.tsv").read_text().splitlines()
    assert rows[0].startswith("file\t") and len(rows) == 5
    assert (tmp_path / "gb245-60w-1.txt").is_file()


def test_build_skips_failed_downloads(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(fetch, "download", boom)
    assert fetch.build(tmp_path, [60], 1, 0, None, 1.0) == 0


def test_wilson_interval():
    assert fpr.wilson(0, 0) is None
    lo, hi = fpr.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = fpr.wilson(10, 20)
    assert lo < 0.5 < hi


def test_bin_label():
    edges = [100, 300]
    assert fpr.bin_label(99, edges) == "<100"
    assert fpr.bin_label(100, edges) == "100-299"
    assert fpr.bin_label(300, edges) == ">=300"


def test_summarize_counts_unscored_and_bins():
    docs = [("a", "w " * 50), ("b", "w " * 150), ("c", "w " * 400), ("d", "w " * 10)]
    out = fpr.summarize(docs, [0.9, 0.2, 0.7, None], 0.6, [100, 300])
    assert (out["n"], out["flagged"], out["unscored"]) == (3, 2, 1)
    assert out["bins"]["<100"]["flagged"] == 1
    assert out["bins"]["100-299"]["fpr"] == 0.0
    assert [t["file"] for t in out["top_false_positives"]] == ["a", "c"]


class _StubDetector:
    def __init__(self, backend, detector_dir=None, pangram_model=None):
        self.backend_used = "stylometry" if backend == "pangram" else backend

    def score_many(self, texts):
        return [0.8 if len(t.split()) < 100 else 0.1 for t in texts]

    def reason(self):
        return "PANGRAM_API_KEY not set" if self.backend_used == "stylometry" else None


def test_run_records_fallback_backend(monkeypatch):
    monkeypatch.setattr(fpr, "HumanLikeness", _StubDetector)
    args = argparse.Namespace(
        corpus=Path("x"),
        backend=["pangram"],
        bins=[100, 300],
        threshold=None,
        human_detector_dir=None,
        human_pangram_model=None,
    )
    report = fpr.run(args, [("short", "w " * 50), ("long", "w " * 200)])
    b = report["backends"][0]
    assert b == {"requested": "pangram", "available": False, "reason": "PANGRAM_API_KEY not set"}
    assert report["docs"][0]["scores"] == {}
    md = fpr.render_markdown(report)
    assert "## pangram: unavailable" in md and "PANGRAM_API_KEY not set" in md


def test_run_scores_available_backend(monkeypatch):
    monkeypatch.setattr(fpr, "HumanLikeness", _StubDetector)
    args = argparse.Namespace(
        corpus=Path("x"),
        backend=["stylometry"],
        bins=[100, 300],
        threshold=None,
        human_detector_dir=None,
        human_pangram_model=None,
    )
    report = fpr.run(args, [("short", "w " * 50), ("long", "w " * 200)])
    b = report["backends"][0]
    assert b["available"] and b["threshold"] == fpr.STYLOMETRY_THRESHOLD
    assert b["fpr"] == 0.5
    assert report["docs"][0]["scores"] == {"stylometry": 0.8}


def test_main_end_to_end_with_stylometry(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    human = (ROOT / "tests" / "fixtures" / "stylometry_human_sample.txt").read_text()
    (corpus / "h1.txt").write_text(human)
    (corpus / "tiny.txt").write_text("Too short to score.")
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["bench_human_fpr.py", "--corpus", str(corpus), "--out-dir", str(out)]
    )
    assert fpr.main() == 0
    report = json.loads((out / "results.json").read_text())
    b = report["backends"][0]
    assert b["available"] and b["unscored"] == 1 and b["n"] == 1
    assert "Detector false positives" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--bins", "a,b"], ["--corpus", "/nonexistent-dir"]])
def test_main_usage_errors(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["bench_human_fpr.py", *argv])
    assert fpr.main() == 2
