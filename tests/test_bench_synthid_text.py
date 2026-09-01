"""Tests for the SynthID-text removal benchmark (bench_synthid_text.py).

Mock-based: the heavy steps (MarkLLM watermark/detect, Layer B rewrite,
are faked, so the suite needs no torch, no network, and no
rewrite backend. It exercises orchestration, sanity gating, aggregation,
controls, and the JSON/CSV/Markdown outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_synthid_text as bench
from bench_synthid_text import (
    _auc,
    _best_on_frontier,
    _pareto_frontier,
    _parse_stats_json,
    _strategy_verdict,
    _weighted_score,
    aggregate,
    estimate_tokens,
    load_corpus,
    parse_float_grid,
    parse_strategy,
    parse_variants,
    parse_weight_grid,
    parse_weight_vec,
)

DETECT_POS = {"available": True, "is_watermarked": True, "score": 2.0}
DETECT_NEG = {"available": True, "is_watermarked": False, "score": -1.0}


@pytest.fixture(autouse=True)
def _clean_rewrite_env(monkeypatch):
    """Parser defaults read ambient WATERMARKS_* vars; keep the suite hermetic."""
    for var in (
        "WATERMARKS_REWRITE_BASE_URL",
        "WATERMARKS_REWRITE_MODEL",
        "WATERMARKS_REWRITE_ALLOW_REMOTE",
        "WATERMARKS_REWRITE_REASONING_EFFORT",
    ):
        monkeypatch.delenv(var, raising=False)


def _args(**overrides):
    values = dict(
        markllm_dir="fake-markllm",
        corpus=SCRIPTS.parents[1] / "benchmarks" / "corpus",
        docs=3,
        seeds=1,
        seed_base=1,
        max_new_tokens=300,
        variants="paraphrase:1,paraphrase:3",
        restamp_control=False,
        out_dir=Path("out"),
        tag="t",
        markllm_model="facebook/opt-1.3b",
        markllm_timeout=600.0,
        rewrite_backend="ollama",
        rewrite_model="llama3.2",
        rewrite_base_url="http://127.0.0.1:11434",
        rewrite_api_key=None,
        rewrite_allow_remote=False,
        rewrite_temperature=0.9,
        rewrite_loops=1,
        chars_per_token=4.0,
        cost_per_mtok_in=0.0,
        cost_per_mtok_out=0.0,
        no_worker=True,
        scheme="synthid",
        config=None,
        mode="variants",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        rewrite_level_start=0.1,
        rewrite_level_step=0.1,
        rewrite_level_max=1.0,
        level_attempts=3,
        target_margin=0.0,
        noop_lex_floor=0.05,
        human_backend="stylometry",
        human_detector_dir=None,
        human_pangram_model="pangram-4",
        intensity_grid="0.2,0.4,0.6,0.8,1.0",
        weight_grid="0.8/0.1/0.1,0.5/0.3/0.2,0.2/0.6/0.2,0.2/0.2/0.6,0.34/0.33/0.33",
        beam=4,
        max_passes=3,
        phase2_levels_per_tactic=3,
        recommend_weight="0.5/0.3/0.2",
        strategies=None,
        write_strategy_outputs=True,
        layer_a_after=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _make_bench(tmp_path, monkeypatch, patch_steps=True, **args_overrides):
    """A Benchmark with all heavy steps faked and controllable."""
    args = _args(out_dir=tmp_path, **args_overrides)
    b = bench.Benchmark(args, Path(args.markllm_dir))
    if patch_steps:
        monkeypatch.setattr(b, "watermark_sample", _fake_watermark)
        monkeypatch.setattr(
            b, "detect", lambda text: DETECT_POS if text.startswith("watermarked") else DETECT_NEG
        )
        monkeypatch.setattr(
            b,
            "rewrite",
            lambda text, tactic, candidates, **kw: (text + " rewritten", _rewrite_stats()),
        )
    return b, args


def _fake_watermark(prompt_path, seed, out_dir):
    return {
        "watermarked": (
            "watermarked sample text for seed "
            + str(seed)
            + " with the numbers 42 and 7 and enough words to pass the gate"
        ),
        "unwatermarked": "plain sample text " + str(seed),
        "watermarked_chars": 44,
        "unwatermarked_chars": 20,
        "payload": {},
    }


def _rewrite_stats(cleared=True, evaluator="markllm", attempts_made=1, passed=True):
    return {
        "mode": "rewritten",
        "evaluator": evaluator,
        "attempts_made": attempts_made,
        "passed": passed,
        "markllm": {
            "before": dict(DETECT_POS),
            "after": {
                "available": True,
                "is_watermarked": not cleared,
                "score": -0.5,
                "threshold": 0.5,
            },
            "cleared": cleared,
            "note": "same-config only",
        },
        "output_chars": 100,
        "candidate_scores": [],
    }


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_default_variants_is_paraphrase_three():
    args = bench.build_parser().parse_args(["--markllm-dir", "x", "--rewrite-model", "m"])
    assert args.variants == "paraphrase:3"


def test_parse_variants():
    assert parse_variants("paraphrase:1,backtranslate:3") == [
        ("paraphrase", 1),
        ("backtranslate", 3),
    ]
    assert parse_variants(" structural:5 ") == [("structural", 5)]
    with pytest.raises(SystemExit):
        parse_variants("paraphrase")
    with pytest.raises(SystemExit):
        parse_variants("paraphrase:0")
    with pytest.raises(SystemExit):
        parse_variants("")


def test_load_corpus(tmp_path):
    (tmp_path / "a.txt").write_text("seed one", encoding="utf-8")
    (tmp_path / "b.txt").write_text("seed two", encoding="utf-8")
    (tmp_path / "skip.me").write_text("not a txt", encoding="utf-8")
    docs = load_corpus(tmp_path, limit=10)
    assert [(d, t) for d, t in docs] == [("a", "seed one"), ("b", "seed two")]
    assert load_corpus(tmp_path, limit=1) == [("a", "seed one")]
    single = tmp_path / "b.txt"
    assert load_corpus(single, limit=10) == [("b", "seed two")]


def test_parse_stats_json_skips_warning_lines():
    stderr = "note: evaluator=markllm attempts=1/3 passed=true\n" + json.dumps(
        {"mode": "rewritten", "markllm": {"cleared": True}}
    )
    stats = _parse_stats_json(stderr)
    assert stats is not None
    assert stats["markllm"]["cleared"] is True
    assert _parse_stats_json("no json here") is None


def test_estimate_tokens():
    assert estimate_tokens("x" * 100, 4.0) == 25
    assert estimate_tokens("", 4.0) == 1


def test_aggregate_clear_rate_and_efficiency():
    rows = [
        {
            "variant": "rewrite-paraphrase:1",
            "kind": "rewrite",
            "before_pos": True,
            "after_pos": False,
            "cleared": True,
            "score_before": 2.0,
            "score_after": -1.0,
            "margin": 1.0,
            "quality": {
                "lexical_divergence": 0.8,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 200,
                "tokens_out": 200,
            },
            "seconds": 1.0,
            "usd": 0.0,
            "attempts": 1,
            "notes": [],
        },
        {
            "variant": "rewrite-paraphrase:1",
            "kind": "rewrite",
            "before_pos": True,
            "after_pos": True,
            "cleared": False,
            "score_before": 2.0,
            "score_after": 1.5,
            "margin": -1.5,
            "quality": {
                "lexical_divergence": 0.6,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 300,
                "tokens_out": 300,
            },
            "seconds": 2.0,
            "usd": 0.0,
            "attempts": 3,
            "notes": [],
        },
    ]
    agg = aggregate(rows, [("paraphrase", 1)])
    a = agg["rewrite-paraphrase:1"]
    assert a["n"] == 2
    assert a["cleared"] == 1
    assert a["clear_rate"] == 0.5
    assert a["mean_score_delta"] == 1.75  # ((2-(-1)) + (2-1.5)) / 2
    assert a["mean_tokens_out"] == 250
    assert a["mean_attempts"] == 2.0  # (1 + 3) / 2
    assert a["mean_margin"] == -0.25  # (1.0 + -1.5) / 2
    assert a["clears_per_mtok_out"] == pytest.approx(2000.0)  # 0.5 / (250/1e6)


def test_aggregate_controls_included():
    rows = [
        {
            "variant": "control",
            "kind": "control",
            "before_pos": True,
            "after_pos": True,
            "cleared": False,
            "quality": {
                "lexical_divergence": 0.0,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 100,
                "tokens_out": 100,
            },
            "seconds": 0.1,
            "usd": 0.0,
            "notes": ["no removal applied (baseline)"],
        },
        {
            "variant": "layer-a",
            "kind": "layer-a",
            "before_pos": True,
            "after_pos": True,
            "cleared": False,
            "quality": {
                "lexical_divergence": 0.0,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 100,
                "tokens_out": 100,
            },
            "seconds": 0.1,
            "usd": 0.0,
            "notes": [],
        },
    ]
    agg = aggregate(rows, [("paraphrase", 1)])
    assert list(agg) == ["control", "layer-a"]
    assert agg["control"]["clear_rate"] == 0.0
    assert agg["layer-a"]["clear_rate"] == 0.0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_generate_samples_sanity_gate(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=2)
    samples = b.generate_samples(tmp_path / "work")
    assert len(samples) == 2
    for s in samples:
        assert s["excluded"] is False
        assert s["before"]["is_watermarked"] is True


def test_sanity_gate_excludes_undetected(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1)
    monkeypatch.setattr(b, "detect", lambda text: DETECT_NEG)
    samples = b.generate_samples(tmp_path / "work")
    assert len(samples) == 1
    assert samples[0]["excluded"] is True
    assert "sanity gate" in samples[0]["excluded_reason"]


def test_run_variants_rows_and_clear_rate(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, variants="paraphrase:1")
    samples = b.generate_samples(tmp_path / "work")
    rows = b.run_variants(samples, tmp_path / "work")
    kinds = [r["kind"] for r in rows]
    assert "control" in kinds and "layer-a" in kinds and "rewrite" in kinds
    rewrite_rows = [r for r in rows if r["kind"] == "rewrite"]
    assert len(rewrite_rows) == 1
    assert rewrite_rows[0]["cleared"] is True
    assert rewrite_rows[0]["attempts"] == 1
    assert rewrite_rows[0]["evaluator"] == "markllm"
    assert rewrite_rows[0]["passed"] is True
    control = next(r for r in rows if r["kind"] == "control")
    assert control["cleared"] is False
    layer_a = next(r for r in rows if r["kind"] == "layer-a")
    assert layer_a["cleared"] is False
    assert any("no removal applied" in n for n in control["notes"])


def test_rewrite_failure_is_recorded_not_fatal(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, variants="paraphrase:1")

    def _boom(text, tactic, candidates, **kw):
        raise RuntimeError("backend down")

    monkeypatch.setattr(b, "rewrite", _boom)
    samples = b.generate_samples(tmp_path / "work")
    rows = b.run_variants(samples, tmp_path / "work")
    failed = [r for r in rows if r["kind"] == "rewrite"]
    assert len(failed) == 1
    assert failed[0]["cleared"] is None
    assert any("rewrite failed" in n for n in failed[0]["notes"])


def test_restamp_control_rows(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, variants="paraphrase:1", restamp_control=True)
    samples = b.generate_samples(tmp_path / "work")
    rows = b.run_variants(samples, tmp_path / "work")
    restamps = [r for r in rows if r["kind"] == "restamp"]
    assert len(restamps) == 1
    # fake rewrite turns plain text into "… rewritten"; detect is text-based
    assert restamps[0]["after_pos"] in (True, False)


# ---------------------------------------------------------------------------
# Outputs via main()
# ---------------------------------------------------------------------------


class _FakeBench:
    def __init__(self, args, upstream):
        self.args = args
        self.python = "python3"
        self.variants = parse_variants(args.variants)
        self.corpus = load_corpus(args.corpus, args.docs)
        self.chars_per_token = args.chars_per_token
        self.semantic = bench.SemanticEmbedder(args.semantic_model)
        self.human = bench.HumanLikeness(
            args.human_backend, args.human_detector_dir, pangram_model=args.human_pangram_model
        )

    def close_worker(self):
        pass

    def generate_samples(self, workdir):
        return [
            {
                "doc": "d1",
                "seed": 1,
                "excluded": False,
                "notes": [],
                "watermarked": "wm text",
                "unwatermarked": "plain text",
                "before": dict(DETECT_POS),
                "plain_detect": dict(DETECT_NEG),
            }
        ]

    def run_variants(self, samples, workdir):
        return [
            {
                "doc": "d1",
                "seed": 1,
                "variant": "rewrite-paraphrase:1",
                "kind": "rewrite",
                "before_pos": True,
                "after_pos": False,
                "cleared": True,
                "score_before": 2.0,
                "score_after": -1.0,
                "quality": {
                    "lexical_divergence": 0.8,
                    "length_ratio": 1.0,
                    "numbers_preserved": 1.0,
                    "urls_preserved": 1.0,
                    "tokens_in": 100,
                    "tokens_out": 100,
                },
                "seconds": 1.5,
                "usd": 0.0,
                "notes": [],
            },
            {
                "doc": "d1",
                "seed": 1,
                "variant": "control",
                "kind": "control",
                "before_pos": True,
                "after_pos": True,
                "cleared": False,
                "quality": {
                    "lexical_divergence": 0.0,
                    "length_ratio": 1.0,
                    "numbers_preserved": 1.0,
                    "tokens_in": 100,
                    "tokens_out": 100,
                },
                "seconds": 0.1,
                "usd": 0.0,
                "notes": ["no removal applied (baseline)"],
            },
            {
                "doc": "d1",
                "seed": 1,
                "variant": "layer-a",
                "kind": "layer-a",
                "before_pos": True,
                "after_pos": True,
                "cleared": False,
                "quality": {
                    "lexical_divergence": 0.0,
                    "length_ratio": 1.0,
                    "numbers_preserved": 1.0,
                    "tokens_in": 100,
                    "tokens_out": 100,
                },
                "seconds": 0.1,
                "usd": 0.0,
                "notes": [],
            },
        ]


def test_main_writes_outputs(tmp_path, monkeypatch, capsys):
    (tmp_path / "markllm" / "watermark").mkdir(parents=True)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "d1.txt").write_text("prompt", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(bench, "Benchmark", _FakeBench)
    monkeypatch.setattr(bench, "_repo_commit", lambda: "abc123")
    monkeypatch.setattr(bench, "_markllm_commit", lambda upstream: "def456")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_synthid_text.py",
            "--markllm-dir",
            str(tmp_path / "markllm"),
            "--corpus",
            str(corpus),
            "--docs",
            "1",
            "--rewrite-model",
            "llama3.2",
            "--variants",
            "paraphrase:1",
            "--out-dir",
            str(out),
            "--tag",
            "ci-test",
        ],
    )
    rc = bench.main()
    assert rc == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# SynthID-text removal benchmark — ci-test" in report
    assert "rewrite-paraphrase:1" in report
    assert "caveat" in report.lower() or "not google" in report.lower()
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["meta"]["tag"] == "ci-test"
    assert data["meta"]["repo_commit"] == "abc123"
    agg = data["aggregates"]["rewrite-paraphrase:1"]
    assert agg["clear_rate"] == 1.0
    csv = (out / "results.csv").read_text(encoding="utf-8")
    assert "doc,seed,variant" in csv
    assert "rewrite-paraphrase:1" in csv


def test_main_requires_markllm_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench_synthid_text.py", "--rewrite-model", "m"])
    assert bench.main() == 2


def test_main_rejects_remote_rewrite_without_flag(tmp_path, monkeypatch, capsys):
    (
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_synthid_text.py",
                "--markllm-dir",
                str(tmp_path),
                "--rewrite-model",
                "m",
                "--rewrite-base-url",
                "http://api.example.com",
            ],
        ),
    )
    assert bench.main() == 2


def test_main_allow_remote_from_env(tmp_path, monkeypatch):
    """WATERMARKS_REWRITE_ALLOW_REMOTE=1 satisfies the remote-URL check."""
    (tmp_path / "markllm" / "watermark").mkdir(parents=True)
    monkeypatch.setenv("WATERMARKS_REWRITE_ALLOW_REMOTE", "1")
    monkeypatch.setattr(bench, "Benchmark", _FakeBench)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_synthid_text.py",
            "--markllm-dir",
            str(tmp_path / "markllm"),
            "--rewrite-model",
            "m",
            "--rewrite-base-url",
            "http://api.example.com",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert bench.main() == 0


def test_main_rejects_nonpositive_beam_or_max_passes(tmp_path, monkeypatch, capsys):
    """A --beam or --max-passes below 1 fails fast in strategy mode.

    Otherwise --beam 0 empties the beam (and --max-passes 0 skips all Phase 2
    expansion) and the run reports a verdict from single-step candidates only.
    """
    (tmp_path / "markllm" / "watermark").mkdir(parents=True)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "d1.txt").write_text("prompt", encoding="utf-8")
    built = []

    class _RecordingBench(_FakeBench):
        def __init__(self, *args, **kwargs):
            built.append(True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bench, "Benchmark", _RecordingBench)
    base = [
        "bench_synthid_text.py",
        "--markllm-dir",
        str(tmp_path / "markllm"),
        "--corpus",
        str(corpus),
        "--rewrite-model",
        "llama3.2",
        "--mode",
        "strategy",
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--beam", "0"])
    assert bench.main() == 1
    monkeypatch.setattr(sys, "argv", [*base, "--max-passes", "0"])
    assert bench.main() == 1
    # The validation must fail before Benchmark (worker/semantic backend) starts.
    assert built == []


def test_worker_publishes_port_env(tmp_path, monkeypatch):
    """A live worker publishes WATERMARKS_MARKLLM_PORT for child processes."""

    class _PortWorker(_FakeWorker):
        def __init__(self, python, script, upstream, model, timeout, **kwargs):
            super().__init__(python, script, upstream, model, timeout, **kwargs)
            self.port = 12345
            os.environ["WATERMARKS_MARKLLM_PORT"] = str(self.port)

        def close(self):
            os.environ.pop("WATERMARKS_MARKLLM_PORT", None)

    monkeypatch.setattr(bench, "MarkLLMWorker", _PortWorker)
    monkeypatch.delenv("WATERMARKS_MARKLLM_PORT", raising=False)
    b, _ = _make_bench(tmp_path, monkeypatch, no_worker=False, patch_steps=False)
    assert os.environ.get("WATERMARKS_MARKLLM_PORT") == "12345"
    b.close_worker()
    assert "WATERMARKS_MARKLLM_PORT" not in os.environ


# ---------------------------------------------------------------------------
def test_aggregate_tolerates_list_in_notes():
    """A row whose notes contain a non-string must not crash aggregation."""
    rows = [
        {
            "variant": "rewrite-paraphrase:1",
            "kind": "rewrite",
            "before_pos": True,
            "after_pos": False,
            "cleared": True,
            "score_before": 2.0,
            "score_after": -1.0,
            "quality": {
                "lexical_divergence": 0.8,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 100,
                "tokens_out": 100,
            },
            "seconds": 1.0,
            "usd": 0.0,
            "notes": ["clean note", ["nested", "list"], {"d": 1}],
        }
    ]
    agg = aggregate(rows, [("paraphrase", 1)])
    assert agg["rewrite-paraphrase:1"]["notes"] == ["clean note"]


# Persistent MarkLLM worker
# ---------------------------------------------------------------------------


class _FakeWorker:
    info: dict = {"device": "cuda"}  # noqa: RUF012 - test double

    def __init__(self, python, script, upstream, model, timeout, *, scheme="synthid", config=None):
        pass

    def watermark(self, prompt, seed, max_new_tokens):
        return {
            "watermarked": prompt + " WM",
            "unwatermarked": prompt + " PL",
            "watermarked_chars": len(prompt) + 3,
            "unwatermarked_chars": len(prompt) + 3,
            "payload": {},
        }

    def detect(self, text):
        return dict(DETECT_POS if "WM" in text else DETECT_NEG)

    def close(self):
        pass


def test_worker_used_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MarkLLMWorker", _FakeWorker)
    b, _ = _make_bench(tmp_path, monkeypatch, no_worker=False, patch_steps=False)
    assert b.worker is not None
    assert b.detect("x WM y")["is_watermarked"] is True
    assert b.detect("plain text")["is_watermarked"] is False


def test_worker_watermark_routing(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MarkLLMWorker", _FakeWorker)
    b, _ = _make_bench(tmp_path, monkeypatch, no_worker=False, patch_steps=False)
    p = tmp_path / "prompt.txt"
    p.write_text("hello", encoding="utf-8")
    out = b.watermark_sample(p, 1, tmp_path)
    assert out["watermarked"] == "hello WM"
    assert out["unwatermarked"] == "hello PL"


def test_worker_fallback_on_start_failure(tmp_path, monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("serve unavailable")

    monkeypatch.setattr(bench, "MarkLLMWorker", _Boom)
    calls = []

    def fake_detect(py, script, upstream, text, model, timeout, *, scheme="synthid", config=None):
        calls.append(text)
        return dict(DETECT_NEG)

    monkeypatch.setattr(bench, "run_detect", fake_detect)
    b, _ = _make_bench(tmp_path, monkeypatch, no_worker=False, patch_steps=False)
    assert b.worker is None
    r = b.detect("plain text")
    assert calls == ["plain text"]
    assert r["is_watermarked"] is False


def test_worker_disabled_with_flag(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, no_worker=True)
    assert b.worker is None


def test_handle_serve_request_protocol():
    import types

    import detect_text_watermark as dt

    class FakeWM:
        def __init__(self):
            self.config = types.SimpleNamespace(gen_kwargs={})

        def generate_watermarked_text(self, prompt):
            return prompt + " WM"

        def generate_unwatermarked_text(self, prompt):
            return prompt + " PL"

        def detect_watermark(self, text, return_dict=False):
            wm = text.endswith("WM")
            return {"is_watermarked": wm, "score": 2.0 if wm else -1.0}

    wm = FakeWM()
    r = dt._handle_serve_request(
        wm, {"op": "watermark", "id": 1, "prompt": "hello", "seed": None, "max_new_tokens": 10}, 0.5
    )
    assert r["ok"] and r["watermarked"] == "hello WM" and r["id"] == 1
    r = dt._handle_serve_request(wm, {"op": "detect", "id": 2, "text": "hello WM"}, 0.5)
    assert r["ok"] and r["is_watermarked"] is True and r["score"] == 2.0
    r = dt._handle_serve_request(wm, {"op": "detect", "id": 3, "text": "hello PL"}, 0.5)
    assert r["is_watermarked"] is False
    r = dt._handle_serve_request(wm, {"op": "nope", "id": 4}, 0.5)
    assert r["ok"] is False and "unknown op" in r["error"]
    r = dt._handle_serve_request(wm, {"op": "watermark", "id": 5, "prompt": ""}, 0.5)
    assert r["ok"] is False
    r = dt._handle_serve_request(wm, {"op": "exit", "id": 6}, 0.5)
    assert r["ok"] is True


def test_run_cmd_no_rlimit_preexec(monkeypatch):
    """_run_cmd must not apply the common 4 GiB RLIMIT_AS (kills torch/CUDA)."""
    import subprocess as _sp

    calls = {}

    class _FakePopen:
        def __init__(self, *a, **k):
            calls["kwargs"] = k
            self.returncode = 0
            self.stdout = "{}"
            self.stderr = ""

    monkeypatch.setattr(_sp, "run", _FakePopen)
    import bench_synthid_text as b

    b._run_cmd(["echo", "hi"], timeout=5)
    assert "preexec_fn" not in calls["kwargs"]


# ---------------------------------------------------------------------------
# Semantic divergence
# ---------------------------------------------------------------------------


def test_semantic_embedder_score_is_one_minus_cosine():
    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0], [0.0, 1.0]]  # orthogonal -> cosine 0

    class _FakeUtil:
        def cos_sim(self, a, b):
            class _V:
                def item(self):
                    return 0.0

            return _V()

    emb = bench.SemanticEmbedder("fake-model")
    emb._model = _FakeModel()
    emb._util = _FakeUtil()
    assert emb.available() is True
    assert emb.score("original", "candidate") == 1.0


def test_semantic_embedder_graceful_when_package_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "sentence_transformers":
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    emb = bench.SemanticEmbedder("a-model")
    assert emb.available() is False
    assert emb.score("a", "b") is None
    # a second call does not keep retrying the import
    assert emb.score("a", "b") is None


def test_semantic_embedder_failsoft_after_encode_failure():
    class _FakeModel:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, normalize_embeddings=True):
            self.calls += 1
            raise RuntimeError("encode exploded")

    emb = bench.SemanticEmbedder("fake-model")
    emb._model = _FakeModel()
    emb._util = type("_U", (), {"cos_sim": lambda self, a, b: None})()
    assert emb.score("orig", "cand") is None
    assert emb.available() is False
    # fail-soft: the backend is disabled after a failed encode, so a second
    # score returns None WITHOUT re-encoding, and available() stays False.
    assert emb.score("orig", "cand") is None
    assert emb._model.calls == 1


def test_quality_includes_semantic_divergence():
    class _FakeSem:
        def score(self, original, candidate):
            return 0.42

    sem = _FakeSem()
    q = bench._quality("original text", "rewritten text", 4.0, sem)
    assert q["semantic_divergence"] == 0.42
    # no semantic backend -> the metric is present but None
    q2 = bench._quality("original text", "rewritten text", 4.0)
    assert q2["semantic_divergence"] is None


def test_aggregate_mean_semantic_divergence_skips_none():
    rows = [
        {
            "variant": "rewrite-paraphrase:1",
            "kind": "rewrite",
            "before_pos": True,
            "after_pos": False,
            "cleared": True,
            "score_before": 2.0,
            "score_after": -1.0,
            "quality": {
                "lexical_divergence": 0.8,
                "semantic_divergence": 0.2,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 100,
                "tokens_out": 100,
            },
            "seconds": 1.0,
            "usd": 0.0,
            "notes": [],
        },
        {
            "variant": "rewrite-paraphrase:1",
            "kind": "rewrite",
            "before_pos": True,
            "after_pos": False,
            "cleared": True,
            "score_before": 2.0,
            "score_after": -1.0,
            "quality": {
                "lexical_divergence": 0.6,
                "semantic_divergence": None,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "tokens_in": 200,
                "tokens_out": 200,
            },
            "seconds": 1.0,
            "usd": 0.0,
            "notes": [],
        },
    ]
    agg = aggregate(rows, [("paraphrase", 1)])
    a = agg["rewrite-paraphrase:1"]
    assert a["semantic_n"] == 1
    assert a["mean_semantic_divergence"] == 0.2


# ---------------------------------------------------------------------------
# Minimal-rewrite-level mode
# ---------------------------------------------------------------------------


def test_minimal_search_escalates_until_cleared(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, mode="minimal", level_attempts=1)
    samples = b.generate_samples(tmp_path / "work")
    levels_called = []

    def fake_rewrite(text, tactic, candidates, rewrite_level=None, **kw):
        levels_called.append(rewrite_level)
        cleared = rewrite_level is not None and rewrite_level >= 0.3
        return f"{text}|{rewrite_level} rewritten", _rewrite_stats(cleared=cleared, attempts_made=1)

    monkeypatch.setattr(b, "rewrite", fake_rewrite)
    monkeypatch.setattr(b.semantic, "score", lambda o, c: 0.25)
    rows = b.minimal_search(samples, tmp_path / "work")
    assert len(rows) == 1
    row = rows[0]
    assert row["cleared"] is True
    assert row["level"] == 0.3
    assert row["semantic_divergence"] == 0.25
    # one attempt per level, escalating 0.1 -> 0.2 -> 0.3
    assert levels_called == [0.1, 0.2, 0.3]


def test_minimal_search_always_evaluates_max_level(tmp_path, monkeypatch):
    b, _ = _make_bench(
        tmp_path,
        monkeypatch,
        docs=1,
        mode="minimal",
        level_attempts=1,
        rewrite_level_start=0.1,
        rewrite_level_step=0.2,
        rewrite_level_max=1.0,
    )
    samples = b.generate_samples(tmp_path / "work")
    levels_called = []

    def fake_rewrite(text, tactic, candidates, rewrite_level=None, **kw):
        levels_called.append(rewrite_level)
        # clears only at the configured maximum
        return f"{text}|{rewrite_level} rewritten", _rewrite_stats(
            cleared=(rewrite_level == 1.0), attempts_made=1
        )

    monkeypatch.setattr(b, "rewrite", fake_rewrite)
    monkeypatch.setattr(b.semantic, "score", lambda o, c: 0.25)
    rows = b.minimal_search(samples, tmp_path / "work")
    row = rows[0]
    assert row["cleared"] is True
    assert row["level"] == 1.0
    # 0.1, 0.3, 0.5, 0.7, 0.9 then the clamped maximum 1.0
    assert levels_called == [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def test_minimal_search_rejects_bad_level_config(tmp_path, monkeypatch):
    for overrides, err in [
        ({"rewrite_level_step": 0.0}, "--rewrite-level-step must be a positive finite number"),
        (
            {"rewrite_level_step": float("inf")},
            "--rewrite-level-step must be a positive finite number",
        ),
        (
            {"rewrite_level_step": float("nan")},
            "--rewrite-level-step must be a positive finite number",
        ),
        ({"rewrite_level_start": 0.0}, "must be in (0,1]"),
        ({"rewrite_level_max": 1.5}, "must be in (0,1]"),
    ]:
        b, _ = _make_bench(tmp_path, monkeypatch, docs=1, mode="minimal", **overrides)
        samples = b.generate_samples(tmp_path / "work")
        with pytest.raises(SystemExit) as e:
            b.minimal_search(samples, tmp_path / "work")
        assert err in str(e.value)


def test_minimal_search_picks_min_semantic_among_clearing(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, mode="minimal", level_attempts=3)
    samples = b.generate_samples(tmp_path / "work")
    sems = iter([0.9, 0.3, 0.7])

    def fake_rewrite(text, tactic, candidates, rewrite_level=None, **kw):
        return f"{text}|{rewrite_level} rewritten", _rewrite_stats(cleared=True, attempts_made=1)

    monkeypatch.setattr(b, "rewrite", fake_rewrite)
    monkeypatch.setattr(b.semantic, "score", lambda o, c: next(sems))
    rows = b.minimal_search(samples, tmp_path / "work")
    row = rows[0]
    assert row["cleared"] is True
    # all attempts clear at the first level; the smallest semantic wins
    assert row["level"] == 0.1
    assert row["semantic_divergence"] == 0.3


def test_minimal_search_records_not_cleared_when_no_level_clears(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, mode="minimal", level_attempts=1)
    samples = b.generate_samples(tmp_path / "work")

    def fake_rewrite(text, tactic, candidates, rewrite_level=None, **kw):
        return f"{text} rewritten", _rewrite_stats(cleared=False, attempts_made=1)

    monkeypatch.setattr(b, "rewrite", fake_rewrite)
    rows = b.minimal_search(samples, tmp_path / "work")
    row = rows[0]
    assert row["cleared"] is False
    assert row["level"] == 1.0
    assert row["semantic_divergence"] is None
    assert "not cleared at any level" in "; ".join(row["notes"])


def test_minimal_search_target_margin_gates_tiny_clear(tmp_path, monkeypatch):
    b, _ = _make_bench(
        tmp_path, monkeypatch, docs=1, mode="minimal", level_attempts=1, target_margin=2.0
    )
    samples = b.generate_samples(tmp_path / "work")
    levels_called = []

    def fake_rewrite(text, tactic, candidates, rewrite_level=None, **kw):
        levels_called.append(rewrite_level)
        # "clears" by the detector (is_watermarked False) but the fake margin
        # is 1.0 < target_margin 2.0, so it is gated out of the clear count.
        return f"{text}|{rewrite_level} rewritten", _rewrite_stats(cleared=True, attempts_made=1)

    monkeypatch.setattr(b, "rewrite", fake_rewrite)
    monkeypatch.setattr(b.semantic, "score", lambda o, c: 0.25)
    rows = b.minimal_search(samples, tmp_path / "work")
    row = rows[0]
    # No level can clear against target_margin=2.0 (fake margin is 1.0), so the
    # search escalates to the max and records a failure, not a hair-thin pass.
    assert row["cleared"] is False
    assert row["level"] == b.args.rewrite_level_max
    assert levels_called == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]


def test_cleared_verdict_tristate(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, mode="minimal")
    assert b._cleared_verdict({"available": True, "is_watermarked": False}) is True
    assert b._cleared_verdict({"available": True, "is_watermarked": True}) is False
    # fail-soft: unavailable detection is never a definite "still watermarked"
    assert b._cleared_verdict({"available": False, "is_watermarked": True}) is None
    assert b._cleared_verdict(None) is None
    assert b._cleared_verdict({"available": True, "is_watermarked": None}) is True


def test_minimal_search_detection_unavailable_is_excluded(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, mode="minimal", level_attempts=1)
    samples = b.generate_samples(tmp_path / "work")

    # Detection is fail-soft unavailable, so the after-report has no verdict.
    monkeypatch.setattr(
        b,
        "_rewrite_report",
        lambda stats, out_text: {"available": False, "is_watermarked": None},
    )
    rows = b.minimal_search(samples, tmp_path / "work")
    row = rows[0]
    # An unknown verdict is excluded (cleared=None), not reported as "not cleared".
    assert row["cleared"] is None
    assert "detection unavailable" in "; ".join(row["notes"])
    # aggregate_minimal drops the unknown row from the clear-rate denominator.
    agg = bench.aggregate_minimal(rows)
    assert agg["n_samples"] == 0
    assert agg["clear_rate"] is None


def test_aggregate_minimal_means_and_usage():
    rows = [
        {
            "cleared": True,
            "level": 0.2,
            "semantic_divergence": 0.1,
            "lexical_divergence": 0.3,
            "margin": 0.9,
        },
        {
            "cleared": True,
            "level": 0.6,
            "semantic_divergence": 0.4,
            "lexical_divergence": 0.5,
            "margin": 1.5,
        },
        {"cleared": False, "level": 1.0, "semantic_divergence": None, "lexical_divergence": None},
    ]
    agg = bench.aggregate_minimal(rows)
    assert agg["n_samples"] == 3
    assert agg["n_cleared"] == 2
    assert agg["clear_rate"] == pytest.approx(2 / 3, rel=1e-4)
    assert agg["mean_min_level"] == pytest.approx(0.4, rel=1e-4)
    assert agg["median_min_level"] == pytest.approx(
        0.4, rel=1e-4
    )  # conventional median of {0.2, 0.6}
    assert agg["mean_min_semantic_divergence"] == pytest.approx(0.25, rel=1e-4)
    assert agg["mean_min_lexical_divergence"] == pytest.approx(0.4, rel=1e-4)
    assert agg["mean_min_margin"] == pytest.approx(1.2, rel=1e-4)  # (0.9 + 1.5) / 2
    assert agg["level_usage"] == [(0.2, 1), (0.6, 1)]


def test_aggregate_minimal_reports_exclusions_and_duplicates():
    rows = [
        {
            "cleared": True,
            "level": 0.2,
            "semantic_divergence": 0.1,
            "lexical_divergence": 0.3,
            "margin": 0.9,
            "attempts": 3,
            "notes": [
                "identical watermarked generation as seed 1 (seed may not be applied; sanity risk)"
            ],
        },
        {
            "cleared": None,
            "level": None,
            "semantic_divergence": None,
            "lexical_divergence": None,
            "attempts": 0,
            "notes": ["watermarked sample not detected (sanity gate)"],
        },
        {
            "cleared": None,
            "level": 1.0,
            "semantic_divergence": None,
            "lexical_divergence": None,
            "attempts": 12,
            "notes": ["detection unavailable; not verified"],
        },
    ]
    agg = bench.aggregate_minimal(rows)
    assert agg["n_excluded"] == 1
    assert agg["excluded_reasons"] == {"watermarked sample not detected (sanity gate)": 1}
    assert agg["n_duplicate_generations"] == 1


def test_generate_samples_flags_identical_generations(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, seeds=2)
    same_text = (
        "watermarked identical sample text for every seed, long enough to pass "
        "the fifty character gate and also fully detected as watermarked"
    )

    def _same(prompt_path, seed, out_dir):
        return {
            "watermarked": same_text,
            "unwatermarked": "plain sample text",
            "watermarked_chars": len(same_text),
            "unwatermarked_chars": 18,
            "payload": {},
        }

    monkeypatch.setattr(b, "watermark_sample", _same)
    samples = b.generate_samples(tmp_path / "work")
    assert len(samples) == 2
    dup_notes = [n for s in samples for n in s["notes"] if "identical watermarked generation" in n]
    assert len(dup_notes) == 1


class _StubSemantic:
    def __init__(self, available=False, reason="sentence-transformers unavailable: no module"):
        self._available = available
        self._reason = reason

    def available(self):
        return self._available

    def reason(self):
        return self._reason


class _StubBench:
    def __init__(self, available=False):
        self.semantic = _StubSemantic(available=available)
        self.corpus = [("d1", "prompt")]


def test_semantic_probe_fails_fast_on_unavailable(capsys):
    code = bench._semantic_startup_probe(_StubBench(), "all-MiniLM-L6-v2", require=True)
    assert code == 2
    assert "semantic backend unavailable" in capsys.readouterr().err


def test_semantic_probe_warns_but_continues(capsys):
    code = bench._semantic_startup_probe(_StubBench(), "all-MiniLM-L6-v2", require=False)
    assert code is None
    assert "semantic backend unavailable" in capsys.readouterr().err


def test_semantic_probe_ready(capsys):
    code = bench._semantic_startup_probe(_StubBench(available=True), "all-MiniLM-L6-v2", False)
    assert code is None
    assert "semantic backend: ready" in capsys.readouterr().err


def test_render_minimal_reports_semantic_status_and_exclusions():
    config = {
        "tag": "t",
        "timestamp": "now",
        "repo_commit": "abc",
        "markllm_commit": "def",
        "markllm_model": "opt-1.3b",
        "corpus": "corpus",
        "docs": 1,
        "seeds": 1,
        "rewrite_level_start": 0.1,
        "rewrite_level_step": 0.1,
        "rewrite_level_max": 1.0,
        "level_attempts": 3,
        "target_margin": 0.05,
        "semantic_model": "all-MiniLM-L6-v2",
        "semantic_available": False,
        "semantic_reason": "sentence-transformers unavailable: no module",
        "command": "cmd",
    }
    rows = [
        {
            "doc": "d1",
            "seed": 1,
            "cleared": None,
            "level": None,
            "semantic_divergence": None,
            "lexical_divergence": None,
            "attempts": 0,
            "notes": ["watermarked sample not detected (sanity gate)"],
        }
    ]
    agg = bench.aggregate_minimal(rows)
    md = bench.render_markdown_minimal(config, [], rows, agg)
    assert "UNAVAILABLE -- sentence-transformers unavailable: no module" in md
    assert "| Samples excluded (sanity gate / generation) | 1 |" in md
    assert "| Identical generations across seeds | 0 |" in md
    assert "0.0500 points below the detection threshold" in md


# ---------------------------------------------------------------------------
# Strategy search: parse/validate, AUROC, Pareto, no-op guard, compose, human
# ---------------------------------------------------------------------------


def test_parse_strategy_valid_and_invalid():
    assert parse_strategy("chunk@0.6,paraphrase@0.3,humanize@1.0") == [
        ("chunk", 0.6),
        ("paraphrase", 0.3),
        ("humanize", 1.0),
    ]
    with pytest.raises(SystemExit):
        parse_strategy("paraphrase@0")  # intensity 0 is invalid
    with pytest.raises(SystemExit):
        parse_strategy("paraphrase@1.5")
    with pytest.raises(SystemExit):
        parse_strategy("bogus@0.5")  # unknown tactic
    with pytest.raises(SystemExit):
        parse_strategy("humanize@1.0")  # only humanize: not a removal attempt
    with pytest.raises(SystemExit):
        parse_strategy("")  # empty


def test_parse_weight_grid():
    assert parse_weight_grid("0.8/0.1/0.1,0.5/0.3/0.2") == [(0.8, 0.1, 0.1), (0.5, 0.3, 0.2)]
    # The out-of-the-box default grid must be valid (each vector sums to 1.0);
    # a 0.33/0.33/0.33 vector sums to 0.99 and is rejected.
    default_grid = "0.8/0.1/0.1,0.5/0.3/0.2,0.2/0.6/0.2,0.2/0.2/0.6,0.34/0.33/0.33"
    assert len(parse_weight_grid(default_grid)) == 5
    with pytest.raises(SystemExit):
        parse_weight_grid("0.5/0.5")  # not three components
    with pytest.raises(SystemExit):
        parse_weight_grid("0.5/0.5/0.5")  # does not sum to 1.0
    with pytest.raises(SystemExit):
        parse_weight_grid("0.33/0.33/0.33")  # sums to 0.99
    # Non-numeric, non-finite, and negative components must be rejected before
    # (or independently of) the sum check; NaN/negative vectors slip through a
    # sum-only check because NaN comparisons are false and -1/1/1 sums to 1.0.
    with pytest.raises(SystemExit):
        parse_weight_grid("a/0.5/0.5")  # non-numeric
    with pytest.raises(SystemExit):
        parse_weight_grid("nan/nan/nan")  # non-finite
    with pytest.raises(SystemExit):
        parse_weight_grid("inf/0.5/0.5")  # non-finite
    with pytest.raises(SystemExit):
        parse_weight_grid("-1/1/1")  # negative cumulative


def test_parse_weight_vec():
    assert parse_weight_vec("0.5/0.3/0.2") == (0.5, 0.3, 0.2)
    with pytest.raises(SystemExit):
        parse_weight_vec("0.5/0.3/0.2,0.1/0.2/0.7")  # more than one vector
    with pytest.raises(SystemExit):
        parse_weight_vec("0.5/0.5")  # not three components
    with pytest.raises(SystemExit):
        parse_weight_vec("0.33/0.33/0.33")  # sums to 0.99
    # A zero weight is a legitimate "ignore this axis" vector.
    assert parse_weight_vec("0.0/0.5/0.5") == (0.0, 0.5, 0.5)


def test_weighted_score():
    axis = {"robust_clear_rate": 0.8, "sem_div": 0.2, "human_like": 0.7}
    assert round(_weighted_score(axis, (1.0, 0.0, 0.0)), 4) == 0.8
    assert round(_weighted_score(axis, (0.0, 1.0, 0.0)), 4) == 0.8  # 1 - sem = 0.8
    assert round(_weighted_score(axis, (0.0, 0.0, 1.0)), 4) == 0.7
    assert (
        _weighted_score({"robust_clear_rate": None, "sem_div": 0.2, "human_like": 0.7}, (1, 0, 0))
        is None
    )


def test_best_on_frontier_shifts_with_weight():
    a = {"steps": [("chunk", 0.6)], "robust_clear_rate": 1.0, "sem_div": 0.9, "human_like": 0.1}
    b = {
        "steps": [("paraphrase", 0.3)],
        "robust_clear_rate": 0.0,
        "sem_div": 0.1,
        "human_like": 1.0,
    }
    pick_removal = _best_on_frontier([a, b], [a, b], (1.0, 0.0, 0.0))
    pick_human = _best_on_frontier([a, b], [a, b], (0.0, 0.0, 1.0))
    assert pick_removal["steps"] == [("chunk", 0.6)]
    assert pick_human["steps"] == [("paraphrase", 0.3)]
    # Empty frontier falls back to the best candidate under the weight.
    fallback = _best_on_frontier([], [a, b], (1.0, 0.0, 0.0))
    assert fallback["steps"] == [("chunk", 0.6)]


def test_strategy_verdict():
    def c(rate):
        return {"robust_clear_rate": rate, "sem_div": 0.2, "human_like": 0.7}

    assert _strategy_verdict([c(1.0)]) == "removable"
    assert _strategy_verdict([c(0.5)]) == "partial"
    assert _strategy_verdict([c(0.0)]) == "resists"
    assert _strategy_verdict([]) == "undetermined"
    assert _strategy_verdict([{"sem_div": 0.2, "human_like": 0.7}]) == "undetermined"


def test_render_strategy_verdict_resists():
    res = {
        "candidates": [{"robust_clear_rate": 0.0, "sem_div": 0.8, "human_like": 0.2}],
        "verdict": "resists",
    }
    text = bench._render_strategy_verdict(res)
    assert "resists" in text.lower()
    assert "at this token length" in text.lower()


def test_parse_float_grid():
    assert parse_float_grid("0.2,0.4,1.0") == [0.2, 0.4, 1.0]
    with pytest.raises(SystemExit):
        parse_float_grid("")  # empty grid
    with pytest.raises(SystemExit):
        parse_float_grid("0.2,x")  # non-numeric
    with pytest.raises(SystemExit):
        parse_float_grid("1.5")  # outside (0,1]


def test_auc_perfect_and_random_and_empty():
    assert _auc([2.0, 3.0], [1.0, 0.5]) == 1.0
    assert _auc([1.0, 2.0], [2.0, 1.0]) == 0.5
    assert _auc([], [1.0]) is None
    assert _auc([1.0], []) is None


def test_pareto_frontier_weight_free():
    cands = [
        {
            "steps": [("paraphrase", 0.3)],
            "robust_clear_rate": 1.0,
            "sem_div": 0.3,
            "human_like": 0.7,
        },
        {"steps": [("chunk", 0.6)], "robust_clear_rate": 1.0, "sem_div": 0.2, "human_like": 0.6},
        {
            "steps": [("backtranslate", 0.4)],
            "robust_clear_rate": 0.5,
            "sem_div": 0.1,
            "human_like": 0.8,
        },
        {
            "steps": [("structural", 0.8)],
            "robust_clear_rate": 0.0,
            "sem_div": 0.9,
            "human_like": 0.5,
        },
    ]
    front = _pareto_frontier(cands)
    names = {c["steps"][0][0] for c in front}
    assert "structural" not in names  # dominated by everything
    assert len(front) == 3


def test_aggregate_robust_and_noop():
    def _rew(cleared, robust, noop):
        return {
            "variant": "rewrite-paraphrase:1",
            "kind": "rewrite",
            "before_pos": True,
            "cleared": cleared,
            "robust_cleared": robust,
            "noop": noop,
            "score_before": 2.0,
            "score_after": -1.0 if cleared else 0.5,
            "margin": 0.5 if robust else None,
            "seconds": 1.0,
            "attempts": 1,
            "quality": {
                "lexical_divergence": 0.8,
                "semantic_divergence": None,
                "length_ratio": 1.0,
                "numbers_preserved": 1.0,
                "urls_preserved": 1.0,
                "tokens_in": 100,
                "tokens_out": 100,
            },
            "notes": [],
        }

    rows = [_rew(True, True, False), _rew(None, False, True)]
    agg = aggregate(rows, [("paraphrase", 1)])
    a = agg["rewrite-paraphrase:1"]
    assert a["clear_rate"] == 0.5  # noop row is cleared=None, excluded from numerator
    assert a["robust_clear_rate"] == 0.5
    assert a["noop_n"] == 1


def test_run_variants_noop_guard(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch, docs=1, variants="paraphrase:1")

    def _noop(text, tactic, candidates, **kw):
        return text, {**_rewrite_stats(cleared=False), "noop": True}

    monkeypatch.setattr(b, "rewrite", _noop)
    samples = b.generate_samples(tmp_path / "work")
    rows = b.run_variants(samples, tmp_path / "work")
    rew = next(r for r in rows if r["kind"] == "rewrite")
    assert rew["noop"] is True
    assert rew["cleared"] is None  # a no-op is not a removal attempt
    assert rew["robust_cleared"] is False
    assert any("no-op" in n for n in rew["notes"])


def test_compose_strategy_applies_steps_in_order(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch)
    calls = []

    def _rewrite(text, tactic, candidates, **kw):
        calls.append((tactic, kw.get("rewrite_level"), text))
        return text + f"|{tactic}", _rewrite_stats()

    monkeypatch.setattr(b, "rewrite", _rewrite)
    out, _stats = b.compose_strategy("base", [("chunk", 0.6), ("paraphrase", 0.3)], 0.0)
    assert calls[0] == ("chunk", 0.6, "base")
    assert calls[1] == ("paraphrase", 0.3, "base|chunk")  # step 1 output feeds step 2
    assert out == "base|chunk|paraphrase"


def _spot_eval(strategy):
    """Deterministic stand-in for _eval_strategy used by the strategy_search smoke test."""
    n = len(strategy)
    max_lv = max(lv for _s, lv in strategy)
    rob = min(1.0, 0.2 * n + 0.1 * max_lv)
    sem = 0.1 * n + 0.05 * (1.0 - max_lv)
    human = 0.4 + 0.1 * min(n, 2)
    return {
        "robust_clear_rate": round(rob, 4),
        "sem_div": round(sem, 4),
        "human_like": round(human, 4),
        "n": 2,
        "unverified": 0,
    }


def test_strategy_search_explores_intensity_and_order(tmp_path, monkeypatch):
    args = _args(
        out_dir=tmp_path,
        intensity_grid="0.2,0.8,1.0",
        weight_grid="0.5/0.3/0.2",
        beam=4,
        max_passes=3,
        phase2_levels_per_tactic=3,
        recommend_weight="0.5/0.3/0.2",
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))
    monkeypatch.setattr(b, "_eval_strategy", lambda strategy, _samples: _spot_eval(strategy))
    samples = [{"excluded": False, "watermarked": "watermarked:1", "before": DETECT_POS}]
    res = b.strategy_search(samples, tmp_path / "work")

    # High-intensity levels are actually explored, not frozen to one best level.
    levels_seen = {lv for c in res["candidates"] for _s, lv in c["steps"]}
    assert any(lv > 0.2 for lv in levels_seen)
    # Ordered composition never re-uses a tactic.
    for c in res["candidates"]:
        tactics = [s for s, _lv in c["steps"]]
        assert len(tactics) == len(set(tactics))
    # The recommended strategy comes from the weight-independent frontier.
    assert res["recommended"] is not None
    assert res["recommended"]["steps"]


def test_strategy_search_reuses_evaluated_strategy_across_weights(tmp_path, monkeypatch):
    """A strategy shared by two weight vectors stays eligible for the later beam.

    Regression: `_eval` used to return None for a strategy already evaluated under
    an earlier weight vector, which dropped it from the later vector's beam and
    left its descendants unexplored. Here chunk is anti-correlated: the removal
    weight vector picks chunk@0.2 while the semantic weight vector picks
    chunk@1.0, so both weights share prefixes like paraphrase@1.0+backtranslate
    @1.0 but diverge on the chunk tail. The cache must return the already-evaluated
    axis so the semantic vector keeps that prefix in its beam and explores the
    chunk@1.0 descendants.
    """
    axis = {
        "paraphrase": {1.0: (1.0, 0.1, 0.9), 0.2: (0.2, 0.9, 0.2)},
        "backtranslate": {1.0: (1.0, 0.1, 0.9), 0.2: (0.2, 0.9, 0.2)},
        "structural": {1.0: (1.0, 0.1, 0.9), 0.2: (0.2, 0.9, 0.2)},
        "humanize": {1.0: (1.0, 0.1, 0.9), 0.2: (0.2, 0.9, 0.2)},
        "chunk": {1.0: (0.1, 0.1, 0.9), 0.2: (1.0, 0.9, 0.2)},
    }

    def evaluate(strategy, samples=None):
        robust = []
        semi = []
        human = []
        for tactic, level in strategy:
            r, s, h = axis[tactic][level]
            robust.append(r)
            semi.append(s)
            human.append(h)
        return {
            "robust_clear_rate": round(sum(robust) / len(robust), 4),
            "sem_div": round(sum(semi) / len(semi), 4),
            "human_like": round(sum(human) / len(human), 4),
            "n": 2,
            "unverified": 0,
        }

    args = _args(
        out_dir=tmp_path,
        intensity_grid="0.2,1.0",
        weight_grid="1.0/0.0/0.0,0.0/1.0/0.0",
        beam=8,
        max_passes=3,
        phase2_levels_per_tactic=1,
        recommend_weight="0.5/0.3/0.2",
        humanize_intensity=1.0,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))
    monkeypatch.setattr(b, "_eval_strategy", evaluate)
    samples = [{"excluded": False, "watermarked": "watermarked:1", "before": DETECT_POS}]
    res = b.strategy_search(samples, tmp_path / "work")

    seen = {tuple(c["steps"]) for c in res["candidates"]}
    assert (("paraphrase", 1.0), ("backtranslate", 1.0), ("chunk", 1.0)) in seen


def test_normalize_strategy_humanize_last():
    """humanize steps are moved to the end (collapsed to one at min intensity)."""
    assert bench._normalize_strategy([("chunk", 0.6), ("humanize", 0.5), ("paraphrase", 0.3)]) == [
        ("chunk", 0.6),
        ("paraphrase", 0.3),
        ("humanize", 0.5),
    ]
    assert bench._normalize_strategy(
        [("humanize", 0.8), ("paraphrase", 0.3), ("humanize", 0.5)]
    ) == [("paraphrase", 0.3), ("humanize", 0.5)]
    assert bench._normalize_strategy([("paraphrase", 0.3), ("structural", 1.0)]) == [
        ("paraphrase", 0.3),
        ("structural", 1.0),
    ]


def test_split_holdout():
    """_split_holdout keeps `fraction` of docs for search, the rest for validation."""
    samples = [{"doc": f"d{i}", "seed": 1} for i in range(5)]
    train, hold = bench._split_holdout(samples, 0.6)
    assert len(train) == 3 and len(hold) == 2
    assert train and hold
    assert bench._split_holdout(samples, 0.0) == (samples, [])


def test_strategy_search_no_recommend_when_nothing_clears(tmp_path, monkeypatch):
    """When no strategy clears the mark, nothing is recommended."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        intensity_grid="0.4,1.0",
        phase2_levels_per_tactic=1,
        beam=1,
        max_passes=1,
        recommend_weight="0.5/0.3/0.2",
        coverage_floor=0.5,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))

    def evaluate(strategy, samples=None):
        return {
            "robust_clear_rate": 0.0,
            "sem_div": 0.2,
            "human_like": 0.8,
            "n": 1,
            "unverified": 0,
        }

    monkeypatch.setattr(b, "_eval_strategy", evaluate)
    samples = [{"excluded": False, "watermarked": "w", "before": DETECT_POS}]
    res = b.strategy_search(samples, tmp_path / "work")
    assert res["recommended"] is None
    assert res["verdict"] == "resists"


def test_strategy_search_recommended_ends_humanize(tmp_path, monkeypatch):
    """The recommended strategy is automatically finished with a humanize step."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        intensity_grid="0.4,1.0",
        phase2_levels_per_tactic=1,
        beam=1,
        max_passes=1,
        recommend_weight="0.5/0.3/0.2",
        humanize_intensity=0.4,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))

    def evaluate(strategy, samples=None):
        return {
            "robust_clear_rate": 1.0,
            "sem_div": 0.1,
            "human_like": 0.9,
            "n": 1,
            "unverified": 0,
        }

    monkeypatch.setattr(b, "_eval_strategy", evaluate)
    samples = [{"excluded": False, "watermarked": "w", "before": DETECT_POS}]
    res = b.strategy_search(samples, tmp_path / "work")
    rec = res["recommended"]
    assert rec is not None
    assert rec["steps"][-1][0] == "humanize"


def test_adaptive_escalates_until_clear(tmp_path, monkeypatch):
    """A resistant input is re-run with raised intensity until it robustly clears."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        target_margin=0.0,
        escalation_step=0.2,
        escalation_max=1.0,
        escalation_attempts=3,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))
    calls: list[list[tuple[str, float]]] = []

    def fake_compose(text, steps, target_margin):
        calls.append(list(steps))
        max_lv = max(lv for _t, lv in steps)
        cleared = max_lv >= 0.7
        return text, {
            "markllm": {
                "after": {
                    "available": True,
                    "is_watermarked": not cleared,
                    "score": -0.5 if cleared else 0.2,
                    "threshold": 0.5,
                },
                "cleared": cleared,
            }
        }

    monkeypatch.setattr(b, "compose_strategy", fake_compose)
    samples = [{"excluded": False, "watermarked": "w", "before": DETECT_POS}]
    res = b._adaptive_apply_strategy([("paraphrase", 0.5)], samples)
    assert res["n"] == 1
    assert res["base_clear_rate"] == 0.0
    assert res["adapt_clear_rate"] == 1.0
    assert res["rows"][0]["cleared"] is True
    assert res["rows"][0]["escalation_level"] == 1
    # It escalated once: 0.5 -> 0.7 (max level reached the clear threshold).
    assert calls[0] == [("paraphrase", 0.5)]
    assert calls[1] == [("paraphrase", 0.7)]


def test_recommended_humanized_is_search_candidate(tmp_path, monkeypatch):
    """The auto-humanized recommendation stays in the candidate set (for CSV marking)."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        intensity_grid="0.4,1.0",
        phase2_levels_per_tactic=1,
        beam=1,
        max_passes=1,
        recommend_weight="0.5/0.3/0.2",
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))

    def evaluate(strategy, samples=None):
        return {
            "robust_clear_rate": 1.0,
            "sem_div": 0.1,
            "human_like": 0.9,
            "n": 1,
            "unverified": 0,
        }

    monkeypatch.setattr(b, "_eval_strategy", evaluate)
    samples = [{"excluded": False, "watermarked": "w", "before": DETECT_POS}]
    res = b.strategy_search(samples, tmp_path / "work")
    rec = res["recommended"]
    assert rec is not None
    assert rec["steps"][-1][0] == "humanize"
    assert any(rec is c for c in res["candidates"])


def test_adaptive_excludes_unverified(tmp_path, monkeypatch):
    """An unavailable detection is recorded as unverified, not as a failed clear."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        target_margin=0.0,
        escalation_step=0.2,
        escalation_max=1.0,
        escalation_attempts=1,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))

    def fake_compose(text, steps, target_margin):
        return text, {"markllm": {"after": {"available": False}}}

    monkeypatch.setattr(b, "compose_strategy", fake_compose)
    samples = [{"excluded": False, "watermarked": "w", "before": DETECT_POS}]
    res = b._adaptive_apply_strategy([("paraphrase", 0.5)], samples)
    assert res["unverified"] == 1
    assert res["n"] == 0
    assert res["base_clear_rate"] is None
    assert res["rows"][0]["note"] == "detection unavailable"


def test_persist_strategy_outputs_writes_files(tmp_path):
    """Each strategy candidate's input/output text is written to disk for inspection."""
    args = _args(out_dir=tmp_path, mode="strategy")
    b = bench.Benchmark(args, Path(args.markllm_dir))
    cands = [
        {
            "steps": [("chunk", 0.6), ("paraphrase", 0.3)],
            "robust_clear_rate": 1.0,
            "sem_div": 0.2,
            "human_like": 0.7,
            "n": 1,
            "unverified": 0,
            "outputs": [
                {
                    "doc": "doc1",
                    "seed": 1,
                    "input": "watermarked input",
                    "output": "rewritten out",
                    "robust": True,
                    "margin": 0.4,
                    "sem": 0.2,
                    "note": None,
                }
            ],
        },
        # Candidate whose eval returned no per-sample text (e.g. a monkeypatched
        # _eval_strategy) is skipped, not crashed on.
        {
            "steps": [("humanize", 1.0)],
            "robust_clear_rate": None,
            "sem_div": None,
            "human_like": None,
            "n": 0,
            "unverified": 0,
        },
    ]
    workdir = tmp_path / "work"
    written = b._persist_strategy_outputs(cands, workdir)
    assert written == 1
    cand_dir = workdir / "strategies" / "chunk@0.6+paraphrase@0.3"
    assert (cand_dir / "input_doc1_seed1.txt").read_text() == "watermarked input"
    assert (cand_dir / "output_doc1_seed1.txt").read_text() == "rewritten out"
    # Heavy text stripped from the candidate; replaced with on-disk references.
    assert cands[0]["outputs"] is None
    assert cands[0]["output_dir"] == "work/strategies/chunk@0.6+paraphrase@0.3"
    assert (
        cands[0]["output_files"][0]["input"]
        == "work/strategies/chunk@0.6+paraphrase@0.3/input_doc1_seed1.txt"
    )
    assert cands[1].get("output_dir") is None


def test_strategy_search_persists_outputs(tmp_path, monkeypatch):
    """strategy_search writes every evaluated candidate's output, then strips it."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        intensity_grid="0.2,1.0",
        weight_grid="0.5/0.3/0.2",
        beam=2,
        max_passes=2,
        phase2_levels_per_tactic=1,
        recommend_weight="0.5/0.3/0.2",
        write_strategy_outputs=True,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))

    def evaluate(strategy, samples=None):
        return {
            "robust_clear_rate": 1.0,
            "sem_div": 0.1,
            "human_like": 0.6,
            "n": 1,
            "unverified": 0,
            "outputs": [
                {
                    "doc": "doc1",
                    "seed": 1,
                    "input": "wm input",
                    "output": "rewritten:" + "+".join(f"{s}@{lv:g}" for s, lv in strategy),
                    "robust": True,
                    "margin": 0.4,
                    "sem": 0.1,
                    "note": None,
                }
            ],
        }

    monkeypatch.setattr(b, "_eval_strategy", evaluate)
    samples = [
        {
            "excluded": False,
            "doc": "doc1",
            "seed": 1,
            "watermarked": "wm input",
            "before": DETECT_POS,
        }
    ]
    workdir = tmp_path / "work"
    res = b.strategy_search(samples, workdir)

    strategies_dir = workdir / "strategies"
    assert strategies_dir.exists()
    dirs = list(strategies_dir.glob("*"))
    assert dirs
    # The auto-humanized recommendation is persisted too (this run found one).
    rec = res["recommended"]
    extra = 1 if rec is not None and all(rec is not c for c in res["candidates"]) else 0
    assert res["strategy_outputs_written"] == len(res["candidates"]) + extra
    for c in res["candidates"]:
        assert c["outputs"] is None
        assert c["output_files"]
        for f in c["output_files"]:
            if f.get("output"):
                assert (workdir.parent / f["output"]).exists()
            if f.get("input"):
                assert (workdir.parent / f["input"]).exists()
    if rec is not None:
        assert rec["outputs"] is None
        assert rec["output_files"]


def test_strategy_search_no_write_strategy_outputs(tmp_path, monkeypatch):
    """write_strategy_outputs=False still strips candidate text from results.json."""
    args = _args(
        out_dir=tmp_path,
        mode="strategy",
        intensity_grid="0.2,1.0",
        weight_grid="0.5/0.3/0.2",
        beam=2,
        max_passes=2,
        phase2_levels_per_tactic=1,
        recommend_weight="0.5/0.3/0.2",
        write_strategy_outputs=False,
    )
    b = bench.Benchmark(args, Path(args.markllm_dir))

    def evaluate(strategy, samples=None):
        return {
            "robust_clear_rate": 1.0,
            "sem_div": 0.1,
            "human_like": 0.6,
            "n": 1,
            "unverified": 0,
            "outputs": [
                {
                    "doc": "doc1",
                    "seed": 1,
                    "input": "wm input",
                    "output": "rewritten:" + "+".join(f"{s}@{lv:g}" for s, lv in strategy),
                    "robust": True,
                    "margin": 0.4,
                    "sem": 0.1,
                    "note": None,
                }
            ],
        }

    monkeypatch.setattr(b, "_eval_strategy", evaluate)
    samples = [
        {
            "excluded": False,
            "doc": "doc1",
            "seed": 1,
            "watermarked": "wm input",
            "before": DETECT_POS,
        }
    ]
    workdir = tmp_path / "work"
    res = b.strategy_search(samples, workdir)

    # No strategy dirs are written, so results.json stays slim.
    assert res["strategy_outputs_written"] == 0
    assert not (workdir / "strategies").exists()
    for c in res["candidates"]:
        assert c["outputs"] is None  # heavy per-sample text stripped
        assert "output_files" not in c
        assert "output_dir" not in c


def test_human_likeness_backend_fallback():
    h = bench.HumanLikeness("lastde", None)  # no detector dir -> degrade to stylometry
    assert h.backend_used == "stylometry"
    assert h.reason() is not None
    s = h.score("A reasonably long piece of prose about watermarks and their removal.")
    assert s is None or isinstance(s, (int, float))
    h2 = bench.HumanLikeness("stylometry")
    assert h2.backend_used == "stylometry"


def test_human_likeness_pangram_requires_key(monkeypatch):
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    h = bench.HumanLikeness("pangram", None)
    assert h.backend_used == "stylometry"
    assert h.reason() and "PANGRAM_API_KEY" in h.reason()


def test_pangram_answer_score():
    h = bench.HumanLikeness("stylometry")
    assert h._pangram_answer_score({"fraction_human": 0.2, "fraction_ai": 0.8}) == 0.8
    assert h._pangram_answer_score({"fraction_human": None, "fraction_ai": 0.3}) == 0.3
    assert h._pangram_answer_score({"fraction_ai": -0.1}) == 0.0  # clamped
    assert h._pangram_answer_score({"fraction_ai": 1.7}) == 1.0  # clamped
    assert h._pangram_answer_score({}) is None
    assert h._pangram_answer_score(None) is None


def test_human_likeness_pangram_bulk(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "k")
    calls: dict[str, object] = {}

    def fake(method: str, path: str, body: dict | None = None) -> dict:
        if path == "/models":
            return {"models": ["pangram-4"]}
        if method == "POST" and path == "/bulk":
            calls["submit"] = body
            return {"bulk_id": "blk_1", "status": "queued"}
        if method == "GET" and path == "/bulk/blk_1":
            return {"status": "succeeded", "succeeded": 2}
        if method == "GET" and path.startswith("/bulk/blk_1/results"):
            return {
                "items": [
                    {"id": "0", "result": {"prediction_short": "AI", "fraction_human": 0.1}},
                    {"id": "1", "result": {"prediction_short": "Human", "fraction_human": 1.0}},
                ]
            }
        raise AssertionError(f"unexpected pangram call {method} {path}")

    monkeypatch.setattr(bench.HumanLikeness, "_pangram_request", staticmethod(fake))
    h = bench.HumanLikeness("pangram", None)
    assert h.backend_used == "pangram"
    assert h.score_many(["aaa", "bbb"]) == [0.9, 0.0]  # 1 - fraction_human
    assert calls["submit"]["model"] == "pangram-4"
    assert len(calls["submit"]["items"]) == 2
    assert h.reason() is None


def test_human_likeness_pangram_batches_in_one_job(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "k")
    submits: list[dict] = []

    def fake(method: str, path: str, body: dict | None = None) -> dict:
        if path == "/models":
            return {"models": ["pangram-4"]}
        if method == "POST" and path == "/bulk":
            submits.append(body or {})
            return {"bulk_id": f"blk_{len(submits)}", "status": "queued"}
        if method == "GET" and "/results" in path:
            # echo back one result per submitted item id, aligned by index
            n = len(submits[-1].get("items") or [])
            return {"items": [{"id": str(i), "result": {"fraction_human": 0.0}} for i in range(n)]}
        if method == "GET" and path.startswith("/bulk/blk_"):
            return {"status": "succeeded"}
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(bench.HumanLikeness, "_pangram_request", staticmethod(fake))
    h = bench.HumanLikeness("pangram", None)
    assert h.score_many(["x", "y", "z", ""]) == [1.0, 1.0, 1.0, None]  # empty -> None
    assert len(submits) == 1  # one bulk job for all non-empty texts


def test_human_likeness_pangram_fallback_on_error(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "k")

    def fake(method: str, path: str, body: dict | None = None) -> dict:
        if path == "/models":
            return {"models": ["pangram-4"]}
        raise OSError("network down")

    monkeypatch.setattr(bench.HumanLikeness, "_pangram_request", staticmethod(fake))
    h = bench.HumanLikeness("pangram", None)
    assert h.backend_used == "pangram"
    scores = h.score_many(["text here enough words about watermarks"])
    assert h.backend_used == "stylometry"  # degraded after batch error
    assert all(s is None or isinstance(s, float) for s in scores)


def test_human_likeness_pangram_bulk_failed_falls_back(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "k")

    def fake(method: str, path: str, body: dict | None = None) -> dict:
        if path == "/models":
            return {"models": ["pangram-4"]}
        if method == "POST" and path == "/bulk":
            return {"bulk_id": "blk_1", "status": "queued"}
        if method == "GET" and path == "/bulk/blk_1":
            return {"status": "failed"}
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(bench.HumanLikeness, "_pangram_request", staticmethod(fake))
    h = bench.HumanLikeness("pangram", None)
    assert h.backend_used == "pangram"
    scores = h.score_many(["short text"])
    # A job-level "failed" raises, so score_many degrades to stylometry.
    assert h.backend_used == "stylometry"
    assert scores == [None]  # short text -> stylometry uncalibrated
    assert h.reason() and "failed" in h.reason()  # reason persisted for the report


def test_human_likeness_pangram_model_fallback_resolved(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "k")

    def fake(method: str, path: str, body: dict | None = None) -> dict:
        if path == "/models":
            return {"models": ["default"]}
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(bench.HumanLikeness, "_pangram_request", staticmethod(fake))
    h = bench.HumanLikeness("pangram", None, pangram_model="pangram-4")
    assert h.backend_used == "pangram"
    # The resolved (allowed) selector is persisted for the metadata field.
    assert h.pangram_model == "default"


def test_render_markdown_strategy_and_csv():
    rec = {
        "steps": [("chunk", 0.6)],
        "robust_clear_rate": 1.0,
        "sem_div": 0.1,
        "human_like": 1.0,
        "n": 5,
    }
    res = {
        "candidates": [rec],
        "recommended": rec,
        "frontier": [rec],
        "intensity_curves": {
            "chunk": [{"level": 0.6, "robust_clear_rate": 1.0, "sem_div": 0.1, "human_like": 1.0}]
        },
    }
    config = {
        "tag": "t",
        "timestamp": "now",
        "repo_commit": "abc",
        "markllm_commit": "def",
        "markllm_model": "m",
        "corpus": "c",
        "docs": 1,
        "seeds": 1,
        "rewrite_backend": "b",
        "rewrite_model": "r",
        "human_backend": "stylometry",
        "human_backend_used": "stylometry",
        "semantic_model": "sm",
        "strategies": None,
        "command": "cmd",
    }
    md = bench.render_markdown_strategy(config, [], res)
    assert "Recommend" in md
    assert "chunk@0.6" in md
    assert "Pareto frontier" in md
    csv = bench._strategy_csv(res)
    assert any("chunk@0.6" in line for line in csv)
    assert csv[0].startswith("strategy,")  # header
