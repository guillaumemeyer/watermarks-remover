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
    _pareto_frontier,
    _parse_stats_json,
    aggregate,
    estimate_tokens,
    load_corpus,
    parse_float_grid,
    parse_recipe,
    parse_variants,
    parse_weight_grid,
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
        intensity_grid="0.2,0.4,0.6,0.8,1.0",
        weight_grid="0.8/0.1/0.1,0.5/0.3/0.2,0.2/0.6/0.2,0.2/0.2/0.6,0.33/0.33/0.33",
        beam=4,
        max_passes=3,
        recipes=None,
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
            lambda text, strength, candidates, **kw: (text + " rewritten", _rewrite_stats()),
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

    def _boom(text, strength, candidates, **kw):
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
        self.human = bench.HumanLikeness(args.human_backend, args.human_detector_dir)

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

    def fake_rewrite(text, strength, candidates, rewrite_level=None, **kw):
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

    def fake_rewrite(text, strength, candidates, rewrite_level=None, **kw):
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

    def fake_rewrite(text, strength, candidates, rewrite_level=None, **kw):
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

    def fake_rewrite(text, strength, candidates, rewrite_level=None, **kw):
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

    def fake_rewrite(text, strength, candidates, rewrite_level=None, **kw):
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
# Recipe search: parse/validate, AUROC, Pareto, no-op guard, compose, human
# ---------------------------------------------------------------------------


def test_parse_recipe_valid_and_invalid():
    assert parse_recipe("chunk@0.6,paraphrase@0.3,humanize@1.0") == [
        ("chunk", 0.6),
        ("paraphrase", 0.3),
        ("humanize", 1.0),
    ]
    with pytest.raises(SystemExit):
        parse_recipe("paraphrase@0")  # intensity 0 is invalid
    with pytest.raises(SystemExit):
        parse_recipe("paraphrase@1.5")
    with pytest.raises(SystemExit):
        parse_recipe("bogus@0.5")  # unknown strength
    with pytest.raises(SystemExit):
        parse_recipe("humanize@1.0")  # only humanize: not a removal attempt
    with pytest.raises(SystemExit):
        parse_recipe("")  # empty


def test_parse_weight_grid():
    assert parse_weight_grid("0.8/0.1/0.1,0.5/0.3/0.2") == [(0.8, 0.1, 0.1), (0.5, 0.3, 0.2)]
    with pytest.raises(SystemExit):
        parse_weight_grid("0.5/0.5")  # not three components
    with pytest.raises(SystemExit):
        parse_weight_grid("0.5/0.5/0.5")  # does not sum to 1.0


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

    def _noop(text, strength, candidates, **kw):
        return text, {**_rewrite_stats(cleared=False), "noop": True}

    monkeypatch.setattr(b, "rewrite", _noop)
    samples = b.generate_samples(tmp_path / "work")
    rows = b.run_variants(samples, tmp_path / "work")
    rew = next(r for r in rows if r["kind"] == "rewrite")
    assert rew["noop"] is True
    assert rew["cleared"] is None  # a no-op is not a removal attempt
    assert rew["robust_cleared"] is False
    assert any("no-op" in n for n in rew["notes"])


def test_compose_recipe_applies_steps_in_order(tmp_path, monkeypatch):
    b, _ = _make_bench(tmp_path, monkeypatch)
    calls = []

    def _rewrite(text, strength, candidates, **kw):
        calls.append((strength, kw.get("rewrite_level"), text))
        return text + f"|{strength}", _rewrite_stats()

    monkeypatch.setattr(b, "rewrite", _rewrite)
    out, _stats = b.compose_recipe("base", [("chunk", 0.6), ("paraphrase", 0.3)], 0.0)
    assert calls[0] == ("chunk", 0.6, "base")
    assert calls[1] == ("paraphrase", 0.3, "base|chunk")  # step 1 output feeds step 2
    assert out == "base|chunk|paraphrase"


def test_human_likeness_backend_fallback():
    h = bench.HumanLikeness("lastde", None)  # no detector dir -> degrade to stylometry
    assert h.backend_used == "stylometry"
    assert h.reason() is not None
    s = h.score("A reasonably long piece of prose about watermarks and their removal.")
    assert s is None or isinstance(s, (int, float))
    h2 = bench.HumanLikeness("stylometry")
    assert h2.backend_used == "stylometry"


def test_render_markdown_recipe_and_csv():
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
        "recipes": None,
        "command": "cmd",
    }
    md = bench.render_markdown_recipe(config, [], res)
    assert "Recommend" in md
    assert "chunk@0.6" in md
    assert "Pareto frontier" in md
    csv = bench._recipe_csv(res)
    assert any("chunk@0.6" in line for line in csv)
    assert csv[0].startswith("recipe,")  # header
