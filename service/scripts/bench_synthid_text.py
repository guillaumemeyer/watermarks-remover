#!/usr/bin/env python3
"""Benchmark for MarkLLM text-watermark removal (Layer B rewrite).

Orchestrates the repo's existing machinery into a reproducible, shareable
benchmark:

  1. Generate a watermarked + unwatermarked corpus with a chosen MarkLLM
     scheme (same-config generation and detection; --scheme/--config).
  2. Run removal variants (Layer A only, Layer B rewrites at chosen
     tactic x max-attempt counts; the rewrite loop stops early when an
     attempt passes evaluation) and control rows (no removal, optional
     re-stamp control on unwatermarked text).
  3. Measure removal efficiency and cost:
       - clear rate (before-positive -> after-negative) per variant
       - score suppression (mean/median delta)
       - quality (lexical divergence, length drift, number/URL survival)
       - cost (estimated tokens, wall time, optional USD at given prices)
  4. Emit results.json / results.csv / report.md for sharing.

Detection: same-config-only MarkLLM detection (reproducible,
no vendor APIs). Google retired SynthID text watermarking on its API in
Aug 2026, so no vendor tier exists.

See docs/synthid-text-benchmark.md for how to run and share.

Exit codes:
  0  benchmark completed (even with partial results; counts are reported)
  2  usage/configuration error
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which
from typing import Any
from urllib.parse import urlparse

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import eprint, safe_arg, subprocess_creationflags  # noqa: E402
from detect_text_watermark import SCHEMES  # noqa: E402  (single source of scheme names)
from rewrite_text import _lexical_divergence  # noqa: E402
from text_unicode import clean_text  # noqa: E402

_RESOLVED_SCRIPT = Path(__file__).resolve()
try:
    DEFAULT_CORPUS = _RESOLVED_SCRIPT.parents[2] / "benchmarks" / "corpus"
except IndexError:
    # Container layout (/app/bench_synthid_text.py): no repo root above us;
    # callers pass --corpus explicitly.
    DEFAULT_CORPUS = _RESOLVED_SCRIPT.parent / "benchmarks" / "corpus"
DEFAULT_MARKLLM_MODEL = "facebook/opt-1.3b"
# Default scheme for the benchmark. Overridable with --scheme (any key of
# detect_text_watermark.SCHEMES); --config overrides the scheme's config JSON
# (default: <MarkLLM checkout>/config/<ALG>.json).
DEFAULT_SCHEME = "synthid"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# MarkLLM generation/detection can take minutes on CPU (model load per call).
WATERMARK_TIMEOUT = float(os.environ.get("WATERMARKS_BENCH_WATERMARK_TIMEOUT", "900"))
DETECT_TIMEOUT = float(os.environ.get("WATERMARKS_MARKLLM_TIMEOUT", "600"))
REWRITE_TIMEOUT = float(os.environ.get("WATERMARKS_REWRITE_TIMEOUT", "300"))


def parse_variants(spec: str) -> list[tuple[str, int]]:
    """Parse a variant spec like 'paraphrase:3,backtranslate:3'.

    Each item is <tactic>:<candidates>; tactics come from rewrite_text.py
    (paraphrase, backtranslate, structural, humanize, code, chunk). candidates
    is the max rewrite attempts per input — the Layer B loop stops early as
    soon as an attempt passes evaluation.
    """
    variants: list[tuple[str, int]] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise SystemExit(f"error: bad variant {item!r}; expected <tactic>:<candidates>")
        tactic, raw_c = parts
        try:
            c = int(raw_c)
        except ValueError:
            raise SystemExit(f"error: bad candidate count in variant {item!r}") from None
        if c < 1:
            raise SystemExit(f"error: candidate count must be >= 1 in variant {item!r}")
        variants.append((tactic, c))
    if not variants:
        raise SystemExit("error: --variants must name at least one variant")
    return variants


def parse_strategy(spec: str) -> list[tuple[str, float]]:
    """Parse a strategy spec like 'chunk@0.6,paraphrase@0.3,humanize@1.0'.

    Returns an ordered list of (tactic, intensity) steps. Validates tactic
    names and that intensity lies in (0,1].
    """
    steps: list[tuple[str, float]] = []
    known = {"paraphrase", "backtranslate", "structural", "humanize", "chunk"}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "@" not in item:
            raise SystemExit(f"error: bad strategy step {item!r}; expected tactic@intensity")
        tactic, raw_level = item.rsplit("@", 1)
        tactic = tactic.strip()
        try:
            level = float(raw_level)
        except ValueError:
            raise SystemExit(f"error: bad intensity in strategy step {item!r}") from None
        if tactic not in known:
            raise SystemExit(f"error: unknown strategy tactic {tactic!r}")
        if not (0 < level <= 1):
            raise SystemExit(f"error: strategy intensity must be in (0,1], got {level} in {item!r}")
        steps.append((tactic, level))
    if not steps:
        raise SystemExit("error: empty strategy")
    if all(tactic == "humanize" for tactic, _ in steps):
        raise SystemExit(
            "error: strategy has only humanize steps; style polish is not a removal attempt"
        )
    return steps


def _strategy_slug(steps: list[tuple[str, float]]) -> str:
    """Filesystem-safe slug for an ordered strategy, e.g. 'chunk@0.6+paraphrase@0.3'."""
    return "+".join(f"{tactic}@{level:g}" for tactic, level in steps)


def parse_float_grid(spec: str) -> list[float]:
    """Parse comma-separated floats in (0, 1] into a list."""
    out: list[float] = []
    for raw in spec.split(","):
        x = raw.strip()
        if not x:
            continue
        try:
            val = float(x)
        except ValueError:
            raise SystemExit(f"error: bad float {x!r} in grid") from None
        if not (0 < val <= 1):
            raise SystemExit(f"error: intensity must be in (0,1], got {val}")
        out.append(val)
    if not out:
        raise SystemExit("error: empty intensity grid")
    return out


def parse_weight_grid(spec: str) -> list[tuple[float, float, float]]:
    """Parse comma-separated weight triples (a/b/c) summing to 1.0."""
    out: list[tuple[float, float, float]] = []
    for raw in spec.split(","):
        parts = raw.strip().split("/")
        if len(parts) != 3:
            raise SystemExit(f"error: bad weight vector {raw!r}; expected a/b/c")
        try:
            w = tuple(float(x) for x in parts)
        except ValueError:
            raise SystemExit(f"error: non-numeric weight component in {raw!r}") from None
        if not all(math.isfinite(c) for c in w):
            raise SystemExit(f"error: weight vector {raw!r} must be finite")
        if any(c < 0.0 for c in w):
            raise SystemExit(f"error: weight vector {raw!r} must be non-negative")
        if abs(sum(w) - 1.0) > 1e-6:
            raise SystemExit(f"error: weight vector {raw!r} does not sum to 1.0")
        out.append((w[0], w[1], w[2]))  # type: ignore[assignment]
    return out


def _base_url_is_loopback(base_url: str) -> bool:
    """Check if base URL points to localhost/loopback address."""
    host = urlparse(base_url).hostname or ""
    return host in LOOPBACK_HOSTS


def _venv_python(upstream: Path) -> Path | None:
    """Prefer the MarkLLM checkout's venv interpreter, like text_detectors."""
    if os.name == "nt":
        candidate = upstream / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = upstream / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _markllm_commit(upstream: Path) -> str | None:
    """Resolve current git commit SHA of MarkLLM repository."""
    git = which("git")
    if git is None:
        return None
    try:
        r = subprocess.run(
            [git, "-C", str(upstream), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=subprocess_creationflags,
        )
        if r.returncode == 0:
            return r.stdout.strip()[:12]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _repo_commit() -> str | None:
    """Resolve current git commit SHA of watermarks-remover repository."""
    git = which("git")
    if git is None:
        return None
    try:
        repo_root = SCRIPTS_DIR.parents[1] if len(SCRIPTS_DIR.parents) > 1 else SCRIPTS_DIR.parent
        r = subprocess.run(
            [git, "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=subprocess_creationflags,
        )
        if r.returncode == 0:
            return r.stdout.strip()[:12]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _run_cmd(cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    # No RLIMIT_AS here: every child is the MarkLLM harness or rewrite_text.py,
    # both of which load torch and need a large address space (the common
    # 4 GiB child cap kills CUDA init and the 5 GB fp32 model). This matches
    # text_detectors.py, which applies no address-space cap to MarkLLM by
    # default.
    """Run subprocess command with timeout and error capture."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=subprocess_creationflags,
    )


def _parse_stats_json(stderr: str) -> dict[str, Any] | None:
    """Extract the rewrite --json-stats object from stderr.

    rewrite_text.py prints warnings to stderr before the JSON, so the whole
    stream is not parseable; the stats object is the last thing written and
    starts at the first '{'.
    """
    idx = stderr.find("{")
    if idx < 0:
        return None
    try:
        data = json.loads(stderr[idx:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def run_watermark(
    python: str,
    script: Path,
    upstream: Path,
    prompt_path: Path,
    seed: int,
    max_new_tokens: int,
    out_dir: Path,
    model: str,
    timeout: float,
    *,
    scheme: str,
    config: str | None,
) -> dict[str, Any]:
    """Generate one watermarked (+ unwatermarked) sample via MarkLLM."""
    wm_path = out_dir / f"wm_seed{seed}.txt"
    plain_path = out_dir / f"plain_seed{seed}.txt"
    cmd = [
        python,
        str(script),
        "watermark",
        str(prompt_path),
        "--scheme",
        scheme,
        "--seed",
        str(seed),
        "--max-new-tokens",
        str(max_new_tokens),
        "--model",
        model,
        "--upstream-dir",
        str(upstream),
        "-o",
        str(wm_path),
        "-o2",
        str(plain_path),
        "--json",
    ]
    if config:
        cmd += ["--config", config]
    try:
        proc = _run_cmd(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "watermark generation timed out"}
    if proc.returncode != 0:
        return {
            "error": (proc.stderr or proc.stdout or "").strip()[:300] or f"exit {proc.returncode}"
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "watermark emitted non-JSON stdout"}
    try:
        watermarked = wm_path.read_text(encoding="utf-8", errors="surrogateescape")
        unwatermarked = plain_path.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as e:
        return {"error": f"could not read generated samples: {e}"}
    return {
        "watermarked": watermarked,
        "unwatermarked": unwatermarked,
        "watermarked_chars": len(watermarked),
        "unwatermarked_chars": len(unwatermarked),
        "payload": payload,
    }


def _unlink(path: str) -> None:
    """Safely remove a file if it exists."""
    with contextlib.suppress(OSError):
        os.unlink(path)


def run_detect(
    python: str,
    script: Path,
    upstream: Path,
    text: str,
    model: str,
    timeout: float,
    *,
    scheme: str,
    config: str | None,
) -> dict[str, Any]:
    """Same-config MarkLLM detection of *text*; fail-soft payload."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        cmd = [
            python,
            str(script),
            "detect",
            tmp,
            "--scheme",
            scheme,
            "--model",
            model,
            "--upstream-dir",
            str(upstream),
            "--json",
        ]
        if config:
            cmd += ["--config", config]
        try:
            proc = _run_cmd(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"available": False, "error": "MarkLLM detection timed out"}
        if proc.returncode != 0:
            return {
                "available": False,
                "error": (proc.stderr or "").strip()[:300] or f"exit {proc.returncode}",
            }
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return {"available": False, "error": "MarkLLM detection emitted non-JSON"}
    finally:
        _unlink(tmp)
    if not isinstance(payload, dict):
        return {"available": False, "error": "MarkLLM detection returned non-object"}
    payload["available"] = True
    return payload


def run_rewrite(
    python: str,
    script: Path,
    upstream: Path,
    text: str,
    *,
    backend: str,
    model: str,
    base_url: str,
    tactic: str,
    candidates: int,
    max_loops: int,
    temperature: float,
    timeout: float,
    allow_remote: bool,
    api_key: str | None,
    markllm_model: str,
    markllm_timeout: float,
    markllm_scheme: str,
    rewrite_level: float | None = None,
    target_margin: float = 0.0,
    noop_lex_floor: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run the Layer B rewrite on *text* via rewrite_text.py (real product path).

    Returns (rewritten_text, stats). Stats carry evaluator/attempts_made/
    passed plus markllm.before/after/cleared (always present: the bench passes
    --markllm-scheme, so MarkLLM drives the iterative rewrite loop). Errors
    raise RuntimeError so callers record a note.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(text)
        in_path = f.name
    out_path = in_path + ".rewritten.txt"
    env = dict(os.environ)
    if api_key:
        env["WATERMARKS_REWRITE_API_KEY"] = api_key
    cmd = [
        python,
        str(script),
        str(in_path),
        "-o",
        out_path,
        "--backend",
        backend,
        "--model",
        model,
        "--base-url",
        base_url,
        "--tactic",
        tactic,
        "--candidates",
        str(candidates),
        "--max-loops",
        str(max_loops),
        "--temperature",
        str(temperature),
        "--timeout",
        str(timeout),
        "--markllm-scheme",
        markllm_scheme,
        "--markllm-dir",
        str(upstream),
        "--markllm-model",
        markllm_model,
        "--markllm-timeout",
        str(markllm_timeout),
        "--json-stats",
    ]
    if rewrite_level is not None:
        cmd += ["--rewrite-level", str(rewrite_level)]
    if target_margin:
        cmd += ["--target-margin", str(target_margin)]
    if noop_lex_floor is not None:
        cmd += ["--noop-lex-floor", safe_arg(str(noop_lex_floor))]
    if allow_remote:
        cmd.append("--allow-remote")
    try:
        try:
            proc = _run_cmd(cmd, timeout=max(timeout + 60, markllm_timeout + 60))
        except subprocess.TimeoutExpired:
            raise RuntimeError("rewrite timed out") from None
        if proc.returncode != 0:
            raise RuntimeError(
                (proc.stderr or "").strip()[:300] or f"rewrite exit {proc.returncode}"
            )
        stats = _parse_stats_json(proc.stderr)
        if stats is None:
            raise RuntimeError("rewrite emitted no --json-stats payload")
        out_text = Path(out_path).read_text(encoding="utf-8", errors="surrogateescape")
    finally:
        _unlink(in_path)
        _unlink(out_path)
    return out_text, stats


def load_corpus(path: Path, limit: int) -> list[tuple[str, str]]:
    """Load seed prompts from *path* (a dir of .txt files or a single file)."""
    files = [path] if path.is_file() else sorted(p for p in path.glob("*.txt") if p.is_file())
    if not files:
        raise SystemExit(f"error: no .txt seed files under {path}")
    out: list[tuple[str, str]] = []
    for f in files[:limit]:
        data = f.read_text(encoding="utf-8", errors="surrogateescape").strip()
        if not data:
            continue
        if len(data.encode("utf-8", errors="surrogateescape")) > (1 << 16):
            eprint(f"warning: skipping oversized seed {f.name}")
            continue
        out.append((f.stem, data))
    if not out:
        raise SystemExit(f"error: no usable seed texts under {path}")
    return out


def _numbers_preserved(original: str, candidate: str) -> float:
    """Check if numbers from original text are retained in candidate."""
    a = set(re.findall(r"\d+", original))
    if not a:
        return 1.0
    b = set(re.findall(r"\d+", candidate))
    return len(a & b) / len(a)


def _urls_preserved(original: str, candidate: str) -> float:
    """Check if URLs from original text are retained in candidate."""
    a = set(re.findall(r"https?://\S+", original))
    if not a:
        return 1.0
    b = set(re.findall(r"https?://\S+", candidate))
    return len(a & b) / len(a)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Estimate token count from character length."""
    return max(1, int(len(text) / max(chars_per_token, 1.0)))


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def _detect_positive(d: dict[str, Any] | None) -> bool:
    """Check if watermark detection report verdict is positive."""
    return bool(d and d.get("available") and d.get("is_watermarked"))


class SemanticEmbedder:
    """Optional sentence-embedding semantic divergence (1 - cosine similarity).

    Lazily loads a SentenceTransformer the first time it is used. Any failure —
    package missing (sentence-transformers not installed), model download error,
    encode error — makes ``available()`` return False and ``score()`` return None,
    so the benchmark degrades gracefully to lexical-only output instead of failing.
    """

    def __init__(self, model_name: str) -> None:
        """init."""
        self._model_name = model_name
        self._model = None
        self._util = None
        self._failed: str | None = None

    def _load(self):
        # Fail-soft: a prior load OR encode failure disables the backend so we
        # stop retrying a failing model/encode instead of looping on it.
        """load."""
        if self._failed is not None:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer, util

            self._util = util
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:  # fail-soft: optional dependency
            self._failed = f"sentence-transformers unavailable: {e}"
            return None
        return self._model

    def available(self) -> bool:
        """Available."""
        return self._load() is not None

    def reason(self) -> str | None:
        """Why the backend is unavailable (missing package, download/encode
        error), or None once a model is loaded. Triggers the lazy load, so a
        startup probe reports the real cause instead of a silent '—' report."""
        self._load()
        return self._failed

    def score(self, original: str, candidate: str) -> float | None:
        """Score."""
        model = self._load()
        if model is None:
            return None
        try:
            emb = model.encode([original, candidate], normalize_embeddings=True)
            return round(float(1.0 - self._util.cos_sim(emb[0], emb[1]).item()), 4)
        except Exception as e:
            self._failed = f"embedding failed: {e}"
            return None


class HumanLikeness:
    """Human-likeness axis: stylometry (always on) or an optional detector.

    ``score(text)`` returns AI-likeness in [0,1] (lower = more human);
    ``human_like = 1 - score``. Backends:

    - 'stylometry' (default): stdlib-only burstiness/MATTR/AI-phrase gauge.
    - 'lastde'/'binoculars': an offline ``ai_human.py`` module in
      ``--human-detector-dir`` exposing ``score(text) -> float``.
    - 'pangram': the Pangram Labs async **bulk** API (``--human-pangram-model``,
      API key in ``PANGRAM_API_KEY``). Batching is via ``score_many`` (one bulk
      job per call); ``score`` is a one-item bulk job.

    A backend that fails to load or run degrades to stylometry, reported via
    ``reason()``. Stylometry returns ``None`` (uncalibrated) below
    ``MIN_SAMPLE_WORDS`` words; the caller excludes those from averages.
    """

    PANGRAM_BASE_URL = "https://text.external-api.pangram.com"

    def __init__(
        self, backend: str, detector_dir: str | None = None, pangram_model: str | None = None
    ) -> None:
        """HumanLikeness."""
        self.backend = backend
        self.detector_dir = detector_dir
        self.pangram_model = pangram_model or "pangram-4"
        self.backend_used = "stylometry"
        self._failed: str | None = None
        self._detector = None
        self._pangram_key: str | None = None
        self._pangram_models: list[str] | None = None
        self._load()

    def _load(self) -> None:
        """Load the requested backend (fail-soft to stylometry)."""
        if self.backend == "stylometry":
            return
        if self.backend == "pangram":
            key = os.environ.get("PANGRAM_API_KEY")
            if not key:
                self._failed = "PANGRAM_API_KEY not set"
                return
            self._pangram_key = key
            try:
                models = self._pangram_request("GET", "/models") or {}
                self._pangram_models = list(models.get("models") or [])
            except Exception as e:  # fail-soft: auth / network / bad key
                self._pangram_key = None
                self._failed = f"pangram unavailable: {e}"
                return
            allowed = self._pangram_models
            if self.pangram_model not in allowed:
                self.pangram_model = (
                    "default" if "default" in allowed else (allowed[0] if allowed else None)
                )
                if self.pangram_model is None:
                    self._failed = f"no pangram model available ({allowed})"
                    return
            self._pangram_key = key
            self.backend_used = "pangram"
            return
        if not self.detector_dir:
            self._failed = f"{self.backend} requires --human-detector-dir"
            return
        try:
            import importlib.util

            path = Path(self.detector_dir) / "ai_human.py"
            spec = importlib.util.spec_from_file_location("ai_human", path)
            if spec is None or spec.loader is None:
                raise ImportError("no ai_human.py loader")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._detector = mod.score
            self.backend_used = self.backend
        except Exception as e:  # fail-soft: optional detector backend
            self._failed = f"{self.backend} detector unavailable: {e}"

    def available(self) -> bool:
        """Available."""
        return True  # a fallback always scores

    def reason(self) -> str | None:
        """Reason."""
        return self._failed

    def _pangram_request(self, method: str, path: str, body: dict | None = None) -> dict:
        """Talk to the Pangram text API over https (no shell; urllib only)."""
        import urllib.request

        url = self.PANGRAM_BASE_URL + path
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(f"refusing non-http(s) pangram endpoint: {url}")
        req = urllib.request.Request(url, method=method)  # noqa: S310 - scheme checked above
        req.add_header("x-api-key", self._pangram_key or "")
        req.add_header("Content-Type", "application/json")
        if body is not None:
            req.data = json.dumps(body).encode("utf-8")
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def _stylometry(self, text: str) -> float | None:
        """Stdlib stylometry score."""
        try:
            from score_stylometry import score_text_stylometry

            rep = score_text_stylometry(text)
            if rep.score is None:
                return None
            return round(float(rep.score), 4)
        except Exception as e:
            self._failed = f"stylometry failed: {e}"
            return None

    def _pangram_answer_score(self, result: dict | None) -> float | None:
        """AI-likeness from a Pangram result: 1 - fraction_human (fallback fraction_ai)."""
        if not isinstance(result, dict):
            return None
        fh = result.get("fraction_human")
        if isinstance(fh, (int, float)):
            return round(min(1.0, max(0.0, 1.0 - float(fh))), 4)
        fa = result.get("fraction_ai")
        if isinstance(fa, (int, float)):
            return round(min(1.0, max(0.0, float(fa))), 4)
        return None

    def score_many(self, texts: list[str]) -> list[float | None]:
        """Score many texts; the Pangram backend uses one async bulk job.

        Non-Pangram backends fall back to per-text ``score``.
        """
        if self._pangram_key and self.backend_used == "pangram":
            try:
                return self._pangram_batch(texts)
            except Exception as e:  # fail-soft: disable pangram, fall back to stylometry
                self._failed = f"pangram error: {e}"
                self.backend_used = "stylometry"
                self._pangram_key = None
        return [self.score(t) for t in texts]

    def _pangram_batch(self, texts: list[str]) -> list[float | None]:
        """Submit/poll/fetch one bulk job covering all texts; align results by id."""
        import time

        index_of: dict[str, int] = {}
        items: list[dict] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            index_of[str(i)] = i
            items.append({"id": str(i), "text": t})
        if not items:
            return [None] * len(texts)
        submit = self._pangram_request(
            "POST", "/bulk", {"items": items, "model": self.pangram_model}
        )
        bulk_id = submit.get("bulk_id")
        if not bulk_id:
            raise RuntimeError(f"pangram bulk submit returned no bulk_id: {submit}")
        deadline = time.monotonic() + 60.0  # overall wait before giving up
        while True:
            st = self._pangram_request("GET", f"/bulk/{bulk_id}")
            status = st.get("status")
            if status in ("succeeded", "failed", "partial"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("pangram bulk job timed out")
            time.sleep(2.0)
        if status == "failed":
            # Every item failed; trigger score_many's stylometry fallback.
            raise RuntimeError("pangram bulk job failed (all items failed)")
        scores: dict[str, float | None] = {}
        offset = 0
        while True:
            page = self._pangram_request(
                "GET", f"/bulk/{bulk_id}/results?offset={offset}&limit=1000"
            )
            entries = page.get("items") or page.get("results") or []
            if not entries:
                break
            for entry in entries:
                id_ = str(
                    entry.get("id") if entry.get("id") is not None else entry.get("index") or ""
                )
                scores[id_] = self._pangram_answer_score(entry.get("result"))
            if len(entries) < 1000:
                break
            offset += len(entries)
        return [scores.get(str(i)) for i in range(len(texts))]

    def score(self, text: str) -> float | None:
        """Score human-likeness of a single text."""
        if self._pangram_key and self.backend_used == "pangram":
            return self.score_many([text])[0]
        if self._detector is not None:
            try:
                v = self._detector(text)
                return round(float(v), 4) if isinstance(v, (int, float)) else None
            except Exception as e:
                self._detector = None
                self.backend_used = "stylometry"
                self._failed = f"detector error: {e}"
        return self._stylometry(text)


def _quality(
    original: str, candidate: str, chars_per_token: float, semantic: SemanticEmbedder | None = None
) -> dict[str, Any]:
    """Compute quality metrics between original and rewritten text."""
    sem = semantic.score(original, candidate) if semantic is not None else None
    return {
        "lexical_divergence": round(_lexical_divergence(original, candidate), 4),
        "semantic_divergence": sem,
        "length_ratio": round(len(candidate) / max(len(original), 1), 4),
        "numbers_preserved": round(_numbers_preserved(original, candidate), 4),
        "urls_preserved": round(_urls_preserved(original, candidate), 4),
        "tokens_in": estimate_tokens(original, chars_per_token),
        "tokens_out": estimate_tokens(candidate, chars_per_token),
    }


def _score_of(d: dict[str, Any] | None) -> float | None:
    """Extract primary score from detection result."""
    if not d or not d.get("available"):
        return None
    s = d.get("score")
    return float(s) if isinstance(s, (int, float)) else None


class MarkLLMWorker:
    """Persistent MarkLLM serve process: one model load, many operations.

    Speaks the JSON-lines protocol of ``detect_text_watermark.py serve``
    (ready handshake, then watermark/detect/exit requests). Falls back to
    one-shot subprocesses automatically if it cannot start or dies.
    """

    def __init__(
        self,
        python: str,
        script: Path,
        upstream: Path,
        model: str,
        timeout: float,
        *,
        scheme: str,
        config: str | None,
    ) -> None:
        """init."""
        self._timeout = timeout
        cmd = [
            python,
            str(script),
            "serve",
            "--scheme",
            scheme,
            "--model",
            model,
            "--upstream-dir",
            str(upstream),
            "--port",
            "0",
        ]
        if config:
            cmd += ["--config", config]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=subprocess_creationflags,
        )
        self._stderr_tail: list[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        ready = self._read_line(timeout)
        if ready is None or not ready.get("ready"):
            self.close()
            raise RuntimeError(
                "markllm serve did not become ready" + (f": {ready.get('error')}" if ready else ""),
            )
        self.info = ready
        # Loopback port for OTHER processes (e.g. the rewrite subprocess's
        # MarkLLM detector) to reuse this resident model. The benchmark's own
        # calls go over stdin; the port is exposed so children can too.
        self.port = ready.get("port")
        if self.port is not None:
            os.environ["WATERMARKS_MARKLLM_PORT"] = str(self.port)

    def _drain_stderr(self) -> None:
        """drain stderr."""
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 200:
                self._stderr_tail.pop(0)

    def _read_line(self, timeout: float) -> dict[str, Any] | None:
        """read line."""
        q: queue.Queue[str] = queue.Queue()

        def _reader() -> None:
            """reader."""
            try:
                q.put(self._proc.stdout.readline())
            except Exception as e:
                q.put(f"__error__:{e}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise RuntimeError("markllm worker response timed out")
        line = q.get()
        if line.startswith("__error__:"):
            raise RuntimeError(line[len("__error__:") :])
        if not line:
            raise RuntimeError("markllm worker closed (EOF)")
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            raise RuntimeError(f"markllm worker emitted non-JSON: {line[:120]!r}") from None
        return data if isinstance(data, dict) else None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """request."""
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            resp = self._read_line(self._timeout)
        except Exception as e:
            hint = "; ".join(self._stderr_tail[-3:])
            raise RuntimeError(f"{e} ({hint})") from None
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error") or "markllm worker request failed")
        return resp

    def watermark(self, prompt: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
        """Watermark."""
        resp = self._request(
            {
                "op": "watermark",
                "id": seed,
                "prompt": prompt,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
            }
        )
        return {
            "watermarked": resp["watermarked"],
            "unwatermarked": resp["unwatermarked"],
            "watermarked_chars": resp["watermarked_chars"],
            "unwatermarked_chars": resp["unwatermarked_chars"],
            "payload": resp,
        }

    def detect(self, text: str) -> dict[str, Any]:
        """Detect."""
        resp = self._request({"op": "detect", "id": 0, "text": text})
        return {
            "available": True,
            "is_watermarked": resp["is_watermarked"],
            "score": resp.get("score"),
            "threshold": resp.get("threshold"),
        }

    def close(self) -> None:
        """Close."""
        os.environ.pop("WATERMARKS_MARKLLM_PORT", None)
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"op": "exit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                with contextlib.suppress(Exception):
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            with contextlib.suppress(Exception):
                stream.close()


class Benchmark:
    def __init__(self, args: argparse.Namespace, upstream: Path) -> None:
        """init."""
        self.args = args
        self.upstream = upstream
        self.script = SCRIPTS_DIR / "detect_text_watermark.py"
        self.rewrite_script = SCRIPTS_DIR / "rewrite_text.py"
        self.python = str(_venv_python(upstream) or sys.executable)
        self.variants = parse_variants(args.variants)
        self.corpus = load_corpus(args.corpus, args.docs)
        self.chars_per_token = args.chars_per_token
        self.semantic = SemanticEmbedder(args.semantic_model)
        self.human = HumanLikeness(
            args.human_backend, args.human_detector_dir, pangram_model=args.human_pangram_model
        )
        self.scheme = args.scheme
        self.config = args.config
        self.worker = None
        if not args.no_worker:
            try:
                self.worker = MarkLLMWorker(
                    self.python,
                    self.script,
                    self.upstream,
                    args.markllm_model,
                    args.markllm_timeout,
                    scheme=self.scheme,
                    config=self.config,
                )
                eprint(f"markllm worker: resident on {self.worker.info.get('device', '?')}")
            except Exception as e:
                eprint(f"markllm worker unavailable, using one-shot subprocesses: {e}")

    def _drop_worker(self) -> None:
        """drop worker."""
        if self.worker is not None:
            with contextlib.suppress(Exception):
                self.worker.close()
        self.worker = None

    def close_worker(self) -> None:
        """Close worker."""
        self._drop_worker()

    # -- step wrappers (monkeypatchable in tests) --------------------------

    def watermark_sample(self, prompt_path: Path, seed: int, out_dir: Path) -> dict[str, Any]:
        """Watermark sample."""
        if self.worker is not None:
            prompt = prompt_path.read_text(encoding="utf-8", errors="surrogateescape")
            try:
                return self.worker.watermark(prompt, seed, self.args.max_new_tokens)
            except Exception as e:
                eprint(f"markllm worker failed ({e}); falling back to one-shot")
                self._drop_worker()
        return run_watermark(
            self.python,
            self.script,
            self.upstream,
            prompt_path,
            seed,
            self.args.max_new_tokens,
            out_dir,
            self.args.markllm_model,
            WATERMARK_TIMEOUT,
            scheme=self.scheme,
            config=self.config,
        )

    def detect(self, text: str) -> dict[str, Any]:
        """Detect."""
        if self.worker is not None:
            try:
                return self.worker.detect(text)
            except Exception as e:
                eprint(f"markllm worker failed ({e}); falling back to one-shot")
                self._drop_worker()
        return run_detect(
            self.python,
            self.script,
            self.upstream,
            text,
            self.args.markllm_model,
            DETECT_TIMEOUT,
            scheme=self.scheme,
            config=self.config,
        )

    def rewrite(
        self,
        text: str,
        tactic: str,
        candidates: int,
        max_loops: int = 1,
        rewrite_level: float | None = None,
        target_margin: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        """Execute text rewrite pass across candidates and select best candidate."""
        a = self.args
        return run_rewrite(
            self.python,
            self.rewrite_script,
            self.upstream,
            text,
            backend=a.rewrite_backend,
            model=a.rewrite_model,
            base_url=a.rewrite_base_url,
            tactic=tactic,
            candidates=candidates,
            max_loops=max_loops,
            temperature=a.rewrite_temperature,
            timeout=REWRITE_TIMEOUT,
            allow_remote=a.rewrite_allow_remote,
            api_key=a.rewrite_api_key,
            markllm_model=a.markllm_model,
            markllm_timeout=a.markllm_timeout,
            markllm_scheme=self.scheme,
            rewrite_level=rewrite_level,
            target_margin=target_margin,
            noop_lex_floor=a.noop_lex_floor,
        )

    # -- phases ------------------------------------------------------------

    def generate_samples(self, workdir: Path) -> list[dict[str, Any]]:
        """Generate and sanity-check watermarked/unwatermarked pairs."""
        workdir.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        total = len(self.corpus) * self.args.seeds
        done = 0
        for doc_id, prompt in self.corpus:
            prompt_path = workdir / f"prompt_{doc_id}.txt"
            prompt_path.write_text(prompt, encoding="utf-8", errors="surrogateescape")
            # Sanity guard: seeds must produce distinct watermarked texts. An
            # identical generation across seeds means the seed is not applied
            # (MarkLLM may use its own RNG), which collapses per-seed variance.
            seen_hashes: dict[str, int] = {}
            for seed in range(self.args.seed_base, self.args.seed_base + self.args.seeds):
                sample: dict[str, Any] = {
                    "doc": doc_id,
                    "seed": seed,
                    "excluded": False,
                    "notes": [],
                }
                gen = self.watermark_sample(prompt_path, seed, workdir)
                if gen.get("error"):
                    sample.update(
                        {"excluded": True, "excluded_reason": f"generation: {gen['error']}"}
                    )
                    samples.append(sample)
                    continue
                wm_text = gen["watermarked"]
                plain_text = gen["unwatermarked"]
                wm_hash = hashlib.sha256(
                    wm_text.encode("utf-8", errors="surrogateescape")
                ).hexdigest()
                prev_seed = seen_hashes.get(wm_hash)
                if prev_seed is not None:
                    sample["notes"].append(
                        f"identical watermarked generation as seed {prev_seed} "
                        "(seed may not be applied; sanity risk)"
                    )
                else:
                    seen_hashes[wm_hash] = seed
                if len(wm_text.strip()) < 50:
                    sample.update(
                        {"excluded": True, "excluded_reason": "watermarked sample too short"}
                    )
                    samples.append(sample)
                    continue
                before = self.detect(wm_text)
                plain_detect = self.detect(plain_text)
                sample.update(
                    {
                        "watermarked": wm_text,
                        "unwatermarked": plain_text,
                        "before": before,
                        "plain_detect": plain_detect,
                    }
                )
                if not _detect_positive(before):
                    sample.update(
                        {
                            "excluded": True,
                            "excluded_reason": "watermarked sample not detected (sanity gate)",
                        }
                    )
                if _detect_positive(plain_detect):
                    sample["notes"].append("unwatermarked control detected positive (weak control)")
                samples.append(sample)
                done += 1
                status = "excluded" if sample.get("excluded") else "ok"
                eprint(f"[gen {done}/{total}] {doc_id} seed {seed}: {status}")
        return samples

    def run_variants(self, samples: list[dict[str, Any]], workdir: Path) -> list[dict[str, Any]]:
        """Run removal/control rows for every non-excluded sample."""
        rows: list[dict[str, Any]] = []
        for sample in samples:
            if sample.get("excluded"):
                rows.append(
                    {
                        "doc": sample["doc"],
                        "seed": sample["seed"],
                        "variant": "excluded",
                        "kind": "excluded",
                        "cleared": None,
                        "notes": [sample.get("excluded_reason", "excluded")],
                    }
                )
                continue
            wm_text = sample["watermarked"]
            before = sample["before"]
            base = {
                "doc": sample["doc"],
                "seed": sample["seed"],
                "score_before": _score_of(before),
                "before_pos": _detect_positive(before),
            }

            # Control: no removal (baseline stability).
            rows.append(self._row(base, "control", "control", wm_text, wm_text, sample, workdir))

            # Layer A only: deterministic Unicode scrub; must NOT clear the mark.
            layer_a_text, _layer_stats = clean_text(wm_text)
            rows.append(
                self._row(base, "layer-a", "layer-a", wm_text, layer_a_text, sample, workdir)
            )

            # Layer B rewrites.
            for tactic, candidates in self.variants:
                variant = f"rewrite-{tactic}:{candidates}"
                started = time.monotonic()
                try:
                    out_text, stats = self.rewrite(
                        wm_text, tactic, candidates, target_margin=self.args.target_margin
                    )
                    rewrite_seconds = round(time.monotonic() - started, 3)
                except RuntimeError as e:
                    rows.append(
                        {
                            **base,
                            "variant": variant,
                            "kind": "rewrite",
                            "cleared": None,
                            "after_pos": None,
                            "score_after": None,
                            "margin": None,
                            "noop": False,
                            "robust_cleared": False,
                            "notes": [f"rewrite failed: {e}"],
                        }
                    )
                    continue
                markllm_after = (stats.get("markllm") or {}).get("after")
                if not (markllm_after or {}).get("available"):
                    rows.append(
                        {
                            **base,
                            "variant": variant,
                            "kind": "rewrite",
                            "cleared": None,
                            "after_pos": None,
                            "score_after": None,
                            "margin": None,
                            "noop": False,
                            "robust_cleared": False,
                            "notes": ["rewrite markllm verification unavailable"],
                        }
                    )
                    continue
                row = self._row(
                    base,
                    variant,
                    "rewrite",
                    wm_text,
                    out_text,
                    sample,
                    workdir,
                    detect_after=False,
                )
                row["cleared"] = (stats.get("markllm") or {}).get("cleared")
                if row["cleared"] is None:
                    row["cleared"] = bool(row["before_pos"] and not _detect_positive(markllm_after))
                row["after_pos"] = _detect_positive(markllm_after)
                row["score_after"] = _score_of(markllm_after)
                row["margin"] = self._score_margin(markllm_after)
                row["seconds"] = rewrite_seconds
                row["attempts"] = stats.get("attempts_made")
                row["evaluator"] = stats.get("evaluator")
                row["passed"] = stats.get("passed")
                row["rewrite_stats"] = {
                    k: stats[k]
                    for k in (
                        "candidate_scores",
                        "output_chars",
                        "layer_a_after",
                        "evaluator",
                        "attempts_made",
                        "passed",
                        "mode",
                    )
                    if k in stats
                }
                # No-op guard: a near-verbatim output is not a removal attempt.
                # Do not report it as "0% clear" (the misleading backtranslate
                # case) — set cleared=None and exclude it from the clear rate.
                if stats.get("noop"):
                    row["noop"] = True
                    row["cleared"] = None
                    row["after_pos"] = None
                    row["score_after"] = None
                    row["margin"] = None
                    row["notes"] = [
                        *(row.get("notes") or []),
                        "rewrite returned ≈ input (no-op); not a valid removal test",
                    ]
                row["robust_cleared"] = bool(
                    row["cleared"] is True
                    and row.get("margin") is not None
                    and row["margin"] >= self.args.target_margin - 1e-9
                )
                rows.append(row)

            # Optional re-stamp control: rewrite the UNwatermarked text; a
            # positive after-detection means the backend re-stamped it (or the
            # detector false-positives post-rewrite).
            if self.args.restamp_control:
                for tactic, candidates in self.variants:
                    variant = f"restamp-{tactic}:{candidates}"
                    try:
                        out_text, _stats = self.rewrite(
                            sample["unwatermarked"],
                            tactic,
                            candidates,
                            target_margin=self.args.target_margin,
                        )
                    except RuntimeError as e:
                        rows.append(
                            {
                                **base,
                                "variant": variant,
                                "kind": "restamp",
                                "cleared": None,
                                "noop": False,
                                "robust_cleared": False,
                                "notes": [f"rewrite failed: {e}"],
                            }
                        )
                        continue
                    after = self.detect(out_text)
                    rows.append(
                        {
                            **base,
                            "variant": variant,
                            "kind": "restamp",
                            "after_pos": _detect_positive(after),
                            "score_after": _score_of(after),
                            "margin": self._score_margin(after),
                            "cleared": None,
                            "noop": False,
                            "robust_cleared": False,
                            "quality": _quality(
                                sample["unwatermarked"],
                                out_text,
                                self.chars_per_token,
                                self.semantic,
                            ),
                            "notes": (
                                ["re-stamped by rewrite backend"]
                                if _detect_positive(after)
                                else [],
                            ),
                        }
                    )
            cleared_count = sum(
                1
                for r in rows
                if r["doc"] == base["doc"] and r["seed"] == base["seed"] and r.get("cleared")
            )
            eprint(
                f"[removal] {base['doc']} seed {base['seed']}: {len(rows)} rows, {cleared_count} cleared"
            )
        return rows

    def _row(
        self,
        base: dict[str, Any],
        variant: str,
        kind: str,
        original: str,
        candidate: str,
        sample: dict[str, Any],
        workdir: Path,
        *,
        detect_after: bool = True,
    ) -> dict[str, Any]:
        # Rewrite variants already get after-detection from the rewrite's
        # --json-stats; running another MarkLLM detect here would waste a
        # model load per document.
        """row."""
        started = time.monotonic()
        after = self.detect(candidate) if detect_after else None
        seconds = round(time.monotonic() - started, 3) if detect_after else 0.0
        cleared = (
            bool(base["before_pos"] and not _detect_positive(after))
            if kind in ("control", "layer-a")
            else None
        )
        row: dict[str, Any] = {
            **base,
            "variant": variant,
            "kind": kind,
            "cleared": cleared,
            "after_pos": _detect_positive(after),
            "score_after": _score_of(after),
            "margin": self._score_margin(after),
            "quality": _quality(original, candidate, self.chars_per_token, self.semantic),
            "seconds": seconds,
            "usd": 0.0,
            "notes": [],
            "noop": False,
            "ai_style_score": self.human.score(candidate),
            "human_backend": self.human.backend_used,
        }
        if kind == "control":
            row["notes"].append("no removal applied (baseline)")
        elif kind == "layer-a":
            row["notes"].append("Layer A only; statistical marks are expected to survive")
        row["robust_cleared"] = bool(
            cleared is True
            and row.get("margin") is not None
            and row["margin"] >= self.args.target_margin - 1e-9
        )
        return row

    def _rewrite_report(self, stats: dict[str, Any], out_text: str) -> dict[str, Any]:
        """The rewrite's MarkLLM after-detection report (or a fresh detect).

        Prefer the rewrite's --json-stats (already paid for, no extra model
        load); fall back to a direct detection if the stats verdict is missing.
        """
        mk = (stats.get("markllm") or {}).get("after")
        if mk and mk.get("available"):
            return mk
        return self.detect(out_text)

    def _score_margin(self, report: dict[str, Any] | None) -> float | None:
        """margin = threshold - score for an after-detection report (None if N/A)."""
        if not report or not report.get("available"):
            return None
        score = report.get("score")
        threshold = report.get("threshold")
        if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
            return round(float(threshold) - float(score), 4)
        return None

    def _cleared_verdict(self, report: dict[str, Any] | None) -> bool | None:
        """Tri-state clearance from an after-detection report.

        True = report available and not watermarked (cleared); False = report
        available and still watermarked; None = detection unavailable (fail-soft).
        An unknown verdict is never conflated with a definite "still watermarked".
        """
        if not report or not report.get("available"):
            return None
        return not bool(report.get("is_watermarked"))

    def minimal_search(self, samples: list[dict[str, Any]], workdir: Path) -> list[dict[str, Any]]:
        """For each watermarked sample, raise the rewrite level until it clears.

        Starts at ``--rewrite-level-start`` and increments by
        ``--rewrite-level-step`` per loop (capped at ``--rewrite-level-max``).
        At each level it tries up to ``--level-attempts`` rewrites and, if any
        clears, keeps the one with the smallest semantic divergence and stops
        raising the level. One row per sample records that minimal level and the
        lexical/semantic divergence of the chosen rewrite.
        """
        a = self.args
        # Validate against the (0,1] rewrite-level contract before building the
        # list: a non-positive step would loop forever, and a level outside
        # (0,1] would make every rewrite_text.py call fail at runtime.
        if not math.isfinite(a.rewrite_level_step) or a.rewrite_level_step <= 0:
            raise SystemExit("error: --rewrite-level-step must be a positive finite number")
        if not (0 < a.rewrite_level_start <= 1) or not (0 < a.rewrite_level_max <= 1):
            raise SystemExit(
                "error: --rewrite-level-start and --rewrite-level-max must be in (0,1]"
            )
        levels: list[float] = []
        lvl = a.rewrite_level_start
        while lvl <= a.rewrite_level_max + 1e-9:
            levels.append(round(lvl, 6))
            lvl += a.rewrite_level_step
        # Always evaluate the configured maximum: a step can land just short of
        # it (e.g. start .1, step .2, max 1.0 -> .1 ... .9, then 1.1 > max).
        if levels and levels[-1] < a.rewrite_level_max - 1e-9:
            levels.append(round(a.rewrite_level_max, 6))

        rows: list[dict[str, Any]] = []
        for sample in samples:
            if sample.get("excluded"):
                rows.append(
                    {
                        "doc": sample["doc"],
                        "seed": sample["seed"],
                        "variant": "minimal",
                        "kind": "minimal",
                        "cleared": None,
                        "before_pos": _detect_positive(sample.get("before")),
                        "score_before": _score_of(sample.get("before")),
                        "level": None,
                        "lexical_divergence": None,
                        "semantic_divergence": None,
                        "attempts": 0,
                        "seconds": 0.0,
                        "notes": [
                            sample.get("excluded_reason", "excluded"),
                            *list(sample.get("notes") or []),
                        ],
                    }
                )
                continue

            wm_text = sample["watermarked"]
            base = {
                "doc": sample["doc"],
                "seed": sample["seed"],
                "score_before": _score_of(sample["before"]),
                "before_pos": _detect_positive(sample["before"]),
            }
            attempts = 0
            started = time.monotonic()
            chosen: dict[str, Any] | None = None
            level_used: float | None = None
            failed: str | None = None
            unavailable = False
            for lvl in levels:
                clears: list[dict[str, Any]] = []
                for _ in range(max(1, a.level_attempts)):
                    attempts += 1
                    try:
                        out_text, stats = self.rewrite(
                            wm_text,
                            "paraphrase",
                            1,
                            rewrite_level=lvl,
                            target_margin=a.target_margin,
                        )
                    except RuntimeError as e:
                        failed = str(e)
                        break
                    if stats.get("noop"):
                        # A level that returned ≈ the input is not a real removal;
                        # it must never be the minimal clearing level.
                        continue
                    report = self._rewrite_report(stats, out_text)
                    verdict = self._cleared_verdict(report)
                    if verdict is None:
                        unavailable = True
                        continue
                    if not verdict:
                        continue
                    margin = self._score_margin(report)
                    # A clear by a hair is not a robust removal: require the
                    # configured margin before the level counts as "cleared".
                    if margin is not None and margin < a.target_margin - 1e-9:
                        continue
                    clears.append(
                        {
                            "out": out_text,
                            "lex": _lexical_divergence(wm_text, out_text),
                            "sem": self.semantic.score(wm_text, out_text)
                            if self.semantic
                            else None,
                            "margin": margin,
                            "score_after": _score_of(report),
                        }
                    )
                if failed is not None:
                    break
                if clears:
                    chosen = min(
                        clears,
                        key=lambda c: c["sem"] if c["sem"] is not None else float("inf"),
                    )
                    level_used = lvl
                    break

            row = {
                **base,
                "variant": "minimal",
                "kind": "minimal",
                "attempts": attempts,
                "seconds": round(time.monotonic() - started, 3),
            }
            if failed is not None:
                row.update(
                    {
                        "cleared": None,
                        "level": lvl,
                        "lexical_divergence": None,
                        "semantic_divergence": None,
                        "notes": [
                            f"rewrite failed: {failed}",
                            *list(sample.get("notes") or []),
                        ],
                    }
                )
            elif chosen is None:
                if unavailable:
                    row.update(
                        {
                            "cleared": None,
                            "level": levels[-1],
                            "lexical_divergence": None,
                            "semantic_divergence": None,
                            "notes": [
                                "detection unavailable; not verified",
                                *list(sample.get("notes") or []),
                            ],
                        }
                    )
                else:
                    row.update(
                        {
                            "cleared": False,
                            "level": levels[-1],
                            "lexical_divergence": None,
                            "semantic_divergence": None,
                            "notes": [
                                "not cleared at any level",
                                *list(sample.get("notes") or []),
                            ],
                        }
                    )
            else:
                row.update(
                    {
                        "cleared": True,
                        "level": level_used,
                        "lexical_divergence": chosen["lex"],
                        "semantic_divergence": chosen["sem"],
                        "margin": chosen["margin"],
                        "score_after": chosen["score_after"],
                        "notes": list(sample.get("notes") or []),
                    }
                )
            rows.append(row)
        return rows

    # -- strategy mode --------------------------------------------------------

    def compose_strategy(
        self, text: str, steps: list[tuple[str, float]], target_margin: float
    ) -> tuple[str, dict[str, Any]]:
        """Apply an ordered strategy: each (tactic, intensity) step is one rewrite,
        feeding the previous output as the next input. Returns (final_text, stats)
        where stats is the last step's rewrite stats (carrying markllm before/after).
        Ordering is normalized so ``humanize`` (the user-facing polish) always runs
        last."""
        current = text
        stats: dict[str, Any] = {}
        for tactic, level in _normalize_strategy(steps):
            current, stats = self.rewrite(
                current, tactic, 1, rewrite_level=level, target_margin=target_margin
            )
        if self.args.layer_a_after:
            current, _layer = clean_text(current)
        return current, stats

    def _eval_strategy(
        self, strategy: list[tuple[str, float]], samples: list[dict[str, Any]]
    ) -> dict:
        """Score one strategy across all non-excluded watermarked samples.

        Axes: robust_clear_rate (↑), sem_div (↓), human_like (↑). A sample
        counts only if it was detected positive before, and only if the strategy's
        final markllm after-report is available. The return also carries
        ``outputs``: one entry per evaluated sample with the per-sample input and
        output text (persisted for inspection) plus that sample's doc/seed,
        robust verdict, margin and semantic divergence.
        """
        steps = _normalize_strategy(strategy)
        rows_in: list[dict[str, Any]] = []
        outs: list[str] = []
        score_at: list[int] = []
        per_sample: list[dict[str, Any]] = []
        for s in samples:
            if s.get("excluded"):
                continue
            orig = s["watermarked"]
            if not _detect_positive(s.get("before")):
                continue
            entry: dict[str, Any] = {
                "doc": s.get("doc"),
                "seed": s.get("seed"),
                "input": orig,
                "output": None,
                "robust": None,
                "margin": None,
                "sem": None,
                "note": None,
            }
            try:
                out, stats = self.compose_strategy(orig, steps, self.args.target_margin)
            except RuntimeError as e:
                rows_in.append({"robust": None, "sem": None, "human": None, "err": str(e)})
                entry["note"] = f"rewrite failed: {e}"
                per_sample.append(entry)
                continue
            mk = (stats.get("markllm") or {}).get("after") if stats else None
            entry["output"] = out
            if not (mk or {}).get("available"):
                rows_in.append({"robust": None, "sem": None, "human": None})
                entry["note"] = "detection unavailable"
                per_sample.append(entry)
                continue
            cleared = (stats.get("markllm") or {}).get("cleared")
            if cleared is None:
                cleared = bool(_detect_positive(s.get("before")) and not _detect_positive(mk))
            margin = self._score_margin(mk)
            robust = bool(
                cleared is True and margin is not None and margin >= self.args.target_margin - 1e-9
            )
            sem = self.semantic.score(orig, out) if self.semantic is not None else None
            rows_in.append({"robust": robust, "sem": sem, "human": None})
            entry["robust"] = robust
            entry["margin"] = margin
            entry["sem"] = sem
            per_sample.append(entry)
            outs.append(out)
            score_at.append(len(rows_in) - 1)
        if outs:
            # Batch the human-likeness scoring (one Pangram bulk job per strategy).
            scored = self.human.score_many(outs)
            for idx, val in zip(score_at, scored, strict=True):
                rows_in[idx]["human"] = val
        verified = [r for r in rows_in if r["robust"] is not None]
        n = len(verified)
        unverified = len(rows_in) - n
        if n == 0:
            return {
                "robust_clear_rate": None,
                "sem_div": None,
                "human_like": None,
                "n": 0,
                "unverified": unverified,
                "steps": steps,
                "outputs": per_sample,
            }
        rob = sum(1 for r in verified if r["robust"]) / n
        sem_vals = [r["sem"] for r in verified if r.get("sem") is not None]
        hum_vals = [r["human"] for r in verified if r.get("human") is not None]
        return {
            "robust_clear_rate": round(rob, 4),
            "sem_div": round(_mean(sem_vals), 4) if sem_vals else None,
            "human_like": round(1.0 - _mean(hum_vals), 4) if hum_vals else None,
            "n": n,
            "unverified": unverified,
            "steps": steps,
            "outputs": per_sample,
        }

    def _adaptive_apply_strategy(
        self, strategy: list[tuple[str, float]], samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Escalate the strategy per-input until each sample clears (runtime adaptivity).

        Starts at the strategy's base intensities; for any input that does not robustly
        clear, every step's intensity is raised by ``--escalation-step`` (capped at
        ``--escalation-max``) and re-run, up to ``--escalation-attempts`` rounds, until it
        clears. Returns per-sample rows plus the coverage before and after escalation.
        """
        a = self.args
        esc_step = float(getattr(a, "escalation_step", 0.1) or 0.1)
        esc_max = float(getattr(a, "escalation_max", 1.0) or 1.0)
        attempts = max(1, int(getattr(a, "escalation_attempts", 3)))
        base = _normalize_strategy(strategy)
        rows: list[dict[str, Any]] = []
        for s in samples:
            if s.get("excluded"):
                continue
            orig = s["watermarked"]
            if not _detect_positive(s.get("before")):
                continue
            level0_robust: bool | None = None
            best = {"cleared": False, "level": 0, "margin": None, "err": None}
            current = list(base)
            level = 0
            while True:
                try:
                    _out, stats = self.compose_strategy(orig, current, a.target_margin)
                except RuntimeError as e:
                    best["err"] = str(e)
                    break
                mk = (stats.get("markllm") or {}).get("after") if stats else None
                if not (mk or {}).get("available"):
                    best["err"] = "detection unavailable"
                    break
                cleared = (stats.get("markllm") or {}).get("cleared")
                if cleared is None:
                    cleared = bool(_detect_positive(s.get("before")) and not _detect_positive(mk))
                margin = self._score_margin(mk)
                robust = bool(
                    cleared is True and margin is not None and margin >= a.target_margin - 1e-9
                )
                if level == 0:
                    level0_robust = robust
                if robust:
                    best = {"cleared": True, "level": level, "margin": margin, "err": None}
                    break
                if best["margin"] is None or (margin is not None and margin > best["margin"]):
                    best = {"cleared": False, "level": level, "margin": margin, "err": None}
                level += 1
                if level > attempts or all(lv >= esc_max - 1e-9 for _t, lv in current):
                    break
                current = _escalate_steps(current, esc_step, esc_max)
            rows.append(
                {
                    "doc": s.get("doc"),
                    "seed": s.get("seed"),
                    "base_cleared": bool(level0_robust),
                    "cleared": best["cleared"],
                    "escalation_level": best["level"],
                    "margin": best["margin"],
                    "note": best["err"],
                }
            )
        # An unavailable detection is NOT a failed clear: exclude it from the rate
        # denominators while keeping it visible in the rows (matches _eval_strategy,
        # which counts such cases as unverified).
        verified = [r for r in rows if r.get("note") is None]
        n = len(verified)
        unverified = len(rows) - n
        base_clear = sum(1 for r in verified if r["base_cleared"])
        adapt_clear = sum(1 for r in verified if r["cleared"])
        levels = [r["escalation_level"] for r in verified if r["cleared"]]
        return {
            "n": n,
            "unverified": unverified,
            "base_clear_rate": round(base_clear / n, 4) if n else None,
            "adapt_clear_rate": round(adapt_clear / n, 4) if n else None,
            "mean_level": round(_mean(levels), 3) if levels else None,
            "median_level": round(sorted(levels)[len(levels) // 2], 3) if levels else None,
            "max_level": max(levels) if levels else None,
            "rows": rows,
        }

    def _scalarize(self, axis: dict, w: tuple[float, float, float]) -> float | None:
        """Scalarize an axis dict under weight vector w."""
        return _weighted_score(axis, w)

    def strategy_search(self, samples: list[dict[str, Any]], workdir: Path) -> dict[str, Any]:
        """Strategy search.

        Phase 1 sweeps each tactic's intensity grid as a single-step strategy.
        Phase 2 runs a per-weight-vector beam search that combines an *order* of
        tactics with the top `--phase2-levels-per-tactic` intensities for that
        weight vector, so both step order and intensity are explored. The
        recommended strategy is the point on the weight-independent Pareto frontier
        that best matches `--recommend-weight`; the frontier itself is unaffected
        by weights.
        """
        a = self.args
        tactics: tuple[str, ...] = (
            "paraphrase",
            "backtranslate",
            "structural",
            "humanize",
            "chunk",
        )
        grid = parse_float_grid(a.intensity_grid)
        weights = parse_weight_grid(a.weight_grid)
        recommend_w = parse_weight_vec(getattr(a, "recommend_weight", "0.5/0.3/0.2"))
        k = max(1, int(getattr(a, "phase2_levels_per_tactic", 3)))
        eval_split = float(getattr(a, "eval_split", 0.0) or 0.0)
        eval_samples, holdout = _split_holdout(samples, eval_split)
        coverage_floor = float(getattr(a, "coverage_floor", 0.5))
        humanize_intensity = float(getattr(a, "humanize_intensity", 0.4))
        evaluated: dict[tuple[tuple[str, float], ...], dict] = {}
        cands: list[dict] = []

        def _eval(strategy: list[tuple[str, float]]) -> dict | None:
            """Evaluate one candidate strategy (humanize-last normalized) once."""
            norm = _normalize_strategy(strategy)
            key = tuple(norm)
            if key in evaluated:
                return evaluated[key]
            axis = self._eval_strategy(strategy, eval_samples)
            axis["steps"] = norm
            cands.append(axis)
            evaluated[key] = axis
            return axis

        # Phase 1: per-tactic intensity sweep (single-step strategies).
        step_axes: dict[tuple[str, float], dict] = {}
        for tactic in tactics:
            for level in grid:
                axis = _eval([(tactic, level)])
                if axis is not None:
                    step_axes[(tactic, level)] = axis

        # Phase 2: per-weight top-k intensity ranking + intensity x order beam search.
        for w in weights:
            topk: dict[str, list[float]] = {}
            for tactic in tactics:
                ranked: list[tuple[float, float]] = []
                for (s, level), axis in step_axes.items():
                    if s != tactic:
                        continue
                    sc = self._scalarize(axis, w)
                    if sc is not None:
                        ranked.append((sc, level))
                ranked.sort(key=lambda t: t[0], reverse=True)
                topk[tactic] = [level for _sc, level in ranked[:k]]

            beams: list[list[tuple[str, float]]] = [
                [(s, level)] for s in tactics for level in topk[s]
            ]
            for _depth in range(1, max(1, a.max_passes)):
                candidates_beam: list[tuple[float, list]] = []
                for strategy in beams:
                    used = {s for s, _lv in strategy}
                    for tactic in tactics:
                        if tactic in used:
                            continue
                        for level in topk[tactic]:
                            cand = [*strategy, (tactic, level)]
                            axis = _eval(cand)
                            if axis is None:
                                continue
                            sc = self._scalarize(axis, w)
                            if sc is None:
                                continue
                            candidates_beam.append((sc, cand))
                if not candidates_beam:
                    break
                candidates_beam.sort(key=lambda t: t[0], reverse=True)
                beams = [cand for _sc, cand in candidates_beam[: a.beam]]

        frontier = _pareto_frontier(cands)

        # Cross-input validation: re-measure the frontier (the candidates that matter
        # for the recommendation) on the held-out documents so a strategy that only
        # overfits the search subset is not recommended.
        if holdout:
            for c in frontier:
                h = self._eval_strategy(c["steps"], holdout)
                c["holdout_robust_clear_rate"] = h["robust_clear_rate"]
                c["holdout_sem_div"] = h["sem_div"]
                c["holdout_human_like"] = h["human_like"]

        recommended = _best_on_frontier(
            frontier,
            cands,
            recommend_w,
            coverage_floor=coverage_floor,
            prefer_holdout=bool(holdout),
        )

        # Humanizer always runs last: the recommended default ends with a humanize
        # polish (re-measured on all inputs so its reported axes reflect the final
        # output the user actually sees). The humanized result must still clear the
        # coverage floor, otherwise it is not a recommendable remover.
        if recommended is not None:
            steps = [*recommended["steps"]]
            if not steps or steps[-1][0] != "humanize":
                steps = [*steps, ("humanize", humanize_intensity)]
                final = self._eval_strategy(steps, samples)
                final["steps"] = steps
                final_rate = final.get("robust_clear_rate")
                if final_rate is None or final_rate < coverage_floor - 1e-9:
                    recommended = None
                else:
                    recommended = final
                    # Keep the humanized recommendation in the candidate set so it
                    # gets a row in results.csv, is marked recommended, and is
                    # persisted for inspection.
                    cands.append(final)

        # Intensity curves per tactic (from Phase 1 single-step strategies).
        curves: dict[str, list] = {}
        for tactic in tactics:
            curve = []
            for c in cands:
                if len(c.get("steps") or []) == 1 and c["steps"][0][0] == tactic:
                    curve.append(
                        {
                            "level": c["steps"][0][1],
                            "robust_clear_rate": c["robust_clear_rate"],
                            "sem_div": c["sem_div"],
                            "human_like": c["human_like"],
                        }
                    )
            curves[tactic] = sorted(curve, key=lambda r: r["level"])

        verdict = _strategy_verdict(cands)
        # No recommendable remover: if some strategy was evaluable but none met the
        # coverage floor, the actionable outcome is that the mark resisted.
        if recommended is None and any(c.get("robust_clear_rate") is not None for c in cands):
            verdict = "resists"

        # Persist every candidate (including the auto-humanized recommendation, which
        # is now in `cands`), so the reported best strategy is always written for
        # inspection and its embedded per-sample text is stripped from results.json.
        strategy_outputs_written = self._persist_strategy_outputs(cands, workdir)

        return {
            "candidates": cands,
            "recommended": recommended,
            "frontier": frontier,
            "intensity_curves": curves,
            "verdict": verdict,
            "strategy_outputs_written": strategy_outputs_written,
        }

    # --- Strategy output persistence -------------------------------------------

    def _persist_strategy_outputs(self, cands: list[dict[str, Any]], workdir: Path | None) -> int:
        """Write each candidate's per-sample input/output text to disk for inspection.

        One directory per strategy (``<workdir>/strategies/<slug>/``) holding
        ``input_<doc>_seed<seed>.txt`` and ``output_<doc>_seed<seed>.txt``, so a
        rewritten result sits alongside its watermarked input. Returns the number
        of candidate directories written. After writing, the heavy per-sample
        text is stripped from each candidate and replaced with ``output_dir`` and a
        list of ``output_files`` ({doc, seed, path, robust, margin, sem, note}) so
        results.json stays slim while the text stays inspectable on disk.
        """
        if not getattr(self.args, "write_strategy_outputs", True) or workdir is None:
            for cand in cands:
                cand["outputs"] = None
            return 0
        strategies_dir = workdir / "strategies"
        written = 0
        for cand in cands:
            steps = cand.get("steps")
            outputs = cand.get("outputs")
            if not steps or not outputs:
                continue
            slug = _strategy_slug(steps)
            cand_dir = strategies_dir / slug
            cand_dir.mkdir(parents=True, exist_ok=True)
            files: list[dict[str, Any]] = []
            for entry in outputs:
                doc = entry.get("doc")
                seed = entry.get("seed")
                stem = f"{doc}_seed{seed}"
                if entry.get("input") is not None:
                    in_path = cand_dir / f"input_{stem}.txt"
                    in_path.write_text(entry["input"], encoding="utf-8", errors="surrogateescape")
                    files.append(
                        {
                            "doc": doc,
                            "seed": seed,
                            "input": in_path.relative_to(workdir.parent).as_posix(),
                            "output": None,
                            "robust": entry.get("robust"),
                            "margin": entry.get("margin"),
                            "sem": entry.get("sem"),
                            "note": entry.get("note"),
                        }
                    )
                if entry.get("output") is not None:
                    out_path = cand_dir / f"output_{stem}.txt"
                    out_path.write_text(entry["output"], encoding="utf-8", errors="surrogateescape")
                    files.append(
                        {
                            "doc": doc,
                            "seed": seed,
                            "input": None,
                            "output": out_path.relative_to(workdir.parent).as_posix(),
                            "robust": entry.get("robust"),
                            "margin": entry.get("margin"),
                            "sem": entry.get("sem"),
                            "note": entry.get("note"),
                        }
                    )
            cand["outputs"] = None
            cand["output_dir"] = cand_dir.relative_to(workdir.parent).as_posix()
            cand["output_files"] = files
            written += 1
        return written


# ---------------------------------------------------------------------------
# Aggregation and outputs
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    """Calculate arithmetic mean of a numeric sequence."""
    if not values:
        return None
    return sum(values) / len(values)


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """Rank-based (Mann-Whitney) area under the ROC curve.

    An item is treated as positive when its detector score is *higher*; AUC is
    the probability that a random positive outscores a random negative. 1.0 =
    perfect separation, 0.5 = indistinguishable, 0.0 = perfectly inverted.
    """
    if not pos or not neg:
        return None
    n1 = len(pos)
    n2 = len(neg)
    values = sorted(pos + neg)
    n = len(values)
    rank_of: dict[float, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n and values[j] == values[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            rank_of[values[k]] = avg
        i = j
    rank_sum_pos = sum(rank_of[v] for v in pos if v in rank_of)
    u = rank_sum_pos - n1 * (n1 + 1) / 2.0
    return round(u / (n1 * n2), 4)


def compute_auroc(
    samples: list[dict[str, Any]], rows: list[dict[str, Any]], variants: list[tuple[str, int]]
) -> dict[str, Any]:
    """Population-level AUROC: baseline (orig wm vs plain) and per-variant post-removal.

    Post-removal AUROC uses the rewritten/watermarked after-scores (rewrite-*)
    vs the rewritten/unwatermarked after-scores (restamp-*); it needs
    --restamp-control and degrades to None per variant when unavailable.
    """
    ok = [s for s in samples if not s.get("excluded")]
    base_pos = [_score_of(s.get("before")) for s in ok]
    base_pos = [v for v in base_pos if v is not None]
    base_neg = [_score_of(s.get("plain_detect")) for s in ok]
    base_neg = [v for v in base_neg if v is not None]
    out: dict[str, Any] = {"baseline_auroc": _auc(base_pos, base_neg), "post": {}}
    for tactic, _c in variants:
        key = f"rewrite-{tactic}:{_c}"
        restamp_key = f"restamp-{tactic}:{_c}"
        pos = [
            r["score_after"]
            for r in rows
            if r.get("variant") == key and r.get("score_after") is not None
        ]
        neg = [
            r["score_after"]
            for r in rows
            if r.get("variant") == restamp_key and r.get("score_after") is not None
        ]
        out["post"][key] = _auc(pos, neg)
    return out


def _pareto_frontier(cands: list[dict]) -> list[dict]:
    """Non-dominated strategies over (robust_clear_rate ↑, sem_div ↓, human_like ↑).

    Strategies missing any of the three axes are dropped (they cannot be ranked by
    dominance). A strategy is dominated if another is at least as good on all three
    and strictly better on at least one.
    """
    valid = [
        c
        for c in cands
        if c.get("robust_clear_rate") is not None
        and c.get("sem_div") is not None
        and c.get("human_like") is not None
    ]
    front: list[dict] = []
    for c in valid:
        dominated = False
        for o in valid:
            if o is c:
                continue
            if (
                o["robust_clear_rate"] >= c["robust_clear_rate"]
                and o["sem_div"] <= c["sem_div"]
                and o["human_like"] >= c["human_like"]
                and (
                    o["robust_clear_rate"] > c["robust_clear_rate"]
                    or o["sem_div"] < c["sem_div"]
                    or o["human_like"] > c["human_like"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


def _weighted_score(axis: dict[str, Any], w: tuple[float, float, float]) -> float | None:
    """Scalarize an axis dict into a single score under weight vector w.

    A strategy is only scalarizable when all three axes are available; otherwise it
    cannot be ranked and None is returned.
    """
    removal = axis.get("robust_clear_rate")
    sem = axis.get("sem_div")
    human = axis.get("human_like")
    if removal is None or sem is None or human is None:
        return None
    return w[0] * removal + w[1] * (1.0 - sem) + w[2] * human


def _normalize_strategy(steps: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Enforce humanize-last ordering on a strategy.

    Moves any ``humanize`` step(s) to the final position (collapsing multiple
    humanize steps to one at the minimum intensity), so a strategy is always:
    removal/rewrite tactics first, then the optional humanize polish last.
    """
    removal = [(tactic, level) for tactic, level in steps if tactic != "humanize"]
    human = [level for tactic, level in steps if tactic == "humanize"]
    if human:
        removal.append(("humanize", min(human)))
    return removal


def _escalate_steps(
    steps: list[tuple[str, float]], step: float, cap: float
) -> list[tuple[str, float]]:
    """Raise every step's intensity by ``step`` (capped at ``cap``) for adaptivity."""
    return [(tactic, min(cap, round(level + step, 4))) for tactic, level in steps]


def _split_holdout(
    samples: list[dict[str, Any]], fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split samples into (search, holdout) sets by document id, deterministically.

    ``fraction`` is the share of documents kept for searching; the rest is the
    held-out validation set. ``fraction <= 0`` disables holdout (returns
    ``(samples, [])``).
    """
    if fraction <= 0:
        return samples, []
    docs = sorted({s["doc"] for s in samples})
    n_support = max(1, round(len(docs) * fraction))
    support = set(docs[:n_support])
    train = [s for s in samples if s["doc"] in support]
    holdout = [s for s in samples if s["doc"] not in support]
    return train, holdout


def _coverage_of(c: dict[str, Any], *, prefer_holdout: bool) -> float | None:
    """Population robust clear rate for a candidate, optionally preferring holdout."""
    if prefer_holdout and c.get("holdout_robust_clear_rate") is not None:
        return c["holdout_robust_clear_rate"]
    return c.get("robust_clear_rate")


def _best_on_frontier(
    frontier: list[dict],
    cands: list[dict],
    w: tuple[float, float, float],
    coverage_floor: float = 0.0,
    *,
    prefer_holdout: bool = False,
) -> dict | None:
    """Pick the Pareto-frontier strategy best under weight w.

    Only strategies whose population coverage is at least ``coverage_floor`` are
    eligible, so a strategy that clears only a handful of inputs is never
    recommended. Falls back to the best eligible candidate under w if the frontier
    is empty.
    """
    pool = frontier or cands
    best: dict | None = None
    best_sc: float | None = None
    for c in pool:
        cov = _coverage_of(c, prefer_holdout=prefer_holdout)
        if cov is None or cov < coverage_floor - 1e-9:
            continue
        sc = _weighted_score(c, w)
        if sc is None:
            continue
        if best is None or sc > best_sc:
            best, best_sc = c, sc
    return best


def _strategy_verdict(cands: list[dict]) -> str:
    """Honest verdict on whether the searched strategies clear the mark.

    'removable'    -> some strategy robustly cleared (best robust % == 1.0).
    'partial'      -> strategies cleared partially (0 < best robust % < 1.0).
    'resists'      -> nothing cleared (best robust % == 0.0).
    'undetermined' -> no strategy was evaluable (all three axes missing).
    """
    rates = [c.get("robust_clear_rate") for c in cands if c.get("robust_clear_rate") is not None]
    if not rates:
        return "undetermined"
    best_rate = max(rates)
    if best_rate <= 0.0:
        return "resists"
    if best_rate < 1.0:
        return "partial"
    return "removable"


def parse_weight_vec(spec: str) -> tuple[float, float, float]:
    """Parse a single weight triple like '0.5/0.3/0.2' summing to 1.0."""
    vecs = parse_weight_grid(spec)
    if len(vecs) != 1:
        raise SystemExit(f"error: expected exactly one weight vector, got {spec!r}")
    return vecs[0]


def aggregate(rows: list[dict[str, Any]], variants: list[tuple[str, int]]) -> dict[str, Any]:
    """Aggregate benchmark run statistics across variants."""
    by_variant: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = ["control", "layer-a"]
    order += [f"rewrite-{s}:{c}" for s, c in variants]
    if any(r["kind"] == "restamp" for r in rows):
        order += [f"restamp-{s}:{c}" for s, c in variants]
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    out: dict[str, Any] = {}
    for variant in order:
        group = by_variant.get(variant, [])
        if not group:
            continue
        before_pos = sum(1 for r in group if r.get("before_pos"))
        cleared = sum(1 for r in group if r.get("cleared"))
        robust = sum(1 for r in group if r.get("robust_cleared"))
        noop_n = sum(1 for r in group if r.get("noop"))
        after_pos = sum(1 for r in group if r.get("after_pos"))
        clear_rate = cleared / before_pos if before_pos else None
        robust_clear_rate = robust / before_pos if before_pos else None
        deltas = [
            (r["score_before"] - r["score_after"])
            for r in group
            if r.get("score_before") is not None and r.get("score_after") is not None
        ]
        quals = [r["quality"] for r in group if r.get("quality")]
        seconds = [r["seconds"] for r in group if isinstance(r.get("seconds"), (int, float))]
        attempts = [r["attempts"] for r in group if isinstance(r.get("attempts"), (int, float))]
        usd = sum(r.get("usd") or 0.0 for r in group)
        tokens_out = [q["tokens_out"] for q in quals]
        mean_tokens_out = _mean(tokens_out) if tokens_out else None
        scores_before = [r["score_before"] for r in group if r.get("score_before") is not None]
        scores_after = [r["score_after"] for r in group if r.get("score_after") is not None]
        margins = [r["margin"] for r in group if r.get("margin") is not None]
        human_scores = [r["ai_style_score"] for r in group if r.get("ai_style_score") is not None]
        entries: dict[str, Any] = {
            "n": len(group),
            "before_positive": before_pos,
            "after_positive": after_pos,
            "cleared": cleared,
            "clear_rate": round(clear_rate, 4) if clear_rate is not None else None,
            "robust_cleared": robust,
            "robust_clear_rate": round(robust_clear_rate, 4)
            if robust_clear_rate is not None
            else None,
            "noop_n": noop_n,
            "human_n": len(human_scores),
            "mean_ai_style_score": round(_mean(human_scores), 4) if human_scores else None,
            "mean_score_before": round(_mean(scores_before), 4) if scores_before else None,
            "mean_score_after": round(_mean(scores_after), 4) if scores_after else None,
            "mean_margin": round(_mean(margins), 4) if margins else None,
            "mean_score_delta": round(_mean(deltas), 4) if deltas else None,
            "median_score_delta": round(sorted(deltas)[len(deltas) // 2], 4) if deltas else None,
            "mean_lexical_divergence": round(_mean([q["lexical_divergence"] for q in quals]), 4)
            if quals
            else None,
            "semantic_n": sum(1 for q in quals if q.get("semantic_divergence") is not None),
            "mean_semantic_divergence": (
                round(
                    _mean(
                        [
                            q["semantic_divergence"]
                            for q in quals
                            if q.get("semantic_divergence") is not None
                        ]
                    ),
                    4,
                )
                if any(q.get("semantic_divergence") is not None for q in quals)
                else None
            ),
            "mean_length_ratio": round(_mean([q["length_ratio"] for q in quals]), 4)
            if quals
            else None,
            "mean_numbers_preserved": round(_mean([q["numbers_preserved"] for q in quals]), 4)
            if quals
            else None,
            "mean_tokens_in": round(_mean([q["tokens_in"] for q in quals])) if quals else None,
            "mean_tokens_out": round(mean_tokens_out) if mean_tokens_out else None,
            "mean_attempts": round(_mean([float(a) for a in attempts]), 2) if attempts else None,
            "mean_seconds": round(_mean(seconds), 2) if seconds else None,
            "est_usd": round(usd, 6),
            "clears_per_mtok_out": (
                round(clear_rate / (mean_tokens_out / 1e6), 2)
                if clear_rate is not None and mean_tokens_out
                else None
            ),
            "notes": sorted(
                {n for r in group for n in (r.get("notes") or []) if isinstance(n, str)}
            ),
        }
        out[variant] = entries
    return out


def aggregate_minimal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the minimal-rewrite-level search rows across samples."""
    evaluated = [r for r in rows if r.get("cleared") is not None]
    cleared = [r for r in evaluated if r["cleared"] is True]
    levels = [r["level"] for r in cleared if r.get("level") is not None]
    sems = [r["semantic_divergence"] for r in cleared if r.get("semantic_divergence") is not None]
    lexes = [r["lexical_divergence"] for r in cleared if r.get("lexical_divergence") is not None]
    margins = [r["margin"] for r in cleared if r.get("margin") is not None]
    usage: dict[float, int] = {}
    for r in cleared:
        if r.get("level") is not None:
            usage[r["level"]] = usage.get(r["level"], 0) + 1
    n = len(evaluated)
    # Excluded samples arrive as rows with no verdict and zero attempts; their
    # notes carry the excluded_reason (plus any sample-level notes such as
    # duplicate-generation warnings).
    excluded = [r for r in rows if r.get("cleared") is None and (r.get("attempts") or 0) == 0]
    excluded_reasons: dict[str, int] = {}
    for r in excluded:
        for note in r.get("notes") or []:
            excluded_reasons[str(note)] = excluded_reasons.get(str(note), 0) + 1
    n_duplicate_generations = sum(
        1
        for r in rows
        for note in (r.get("notes") or [])
        if str(note).startswith("identical watermarked generation")
    )
    return {
        "n_samples": n,
        "n_cleared": len(cleared),
        "clear_rate": round(len(cleared) / n, 4) if n else None,
        "n_excluded": len(excluded),
        "excluded_reasons": excluded_reasons or None,
        "n_duplicate_generations": n_duplicate_generations,
        "mean_min_level": round(_mean(levels), 4) if levels else None,
        "median_min_level": round(statistics.median(levels), 4) if levels else None,
        "mean_min_semantic_divergence": round(_mean(sems), 4) if sems else None,
        "median_min_semantic_divergence": round(statistics.median(sems), 4) if sems else None,
        "mean_min_lexical_divergence": round(_mean(lexes), 4) if lexes else None,
        "mean_min_margin": round(_mean(margins), 4) if margins else None,
        "level_usage": list(usage.items()),
    }


def _fmt(value: Any, default: str = "—") -> str:
    """Format numeric value for markdown table output."""
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def render_markdown(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    agg: dict[str, Any],
    auroc: dict[str, Any] | None = None,
) -> str:
    """Render benchmark results summary as markdown report."""
    L: list[str] = []
    L.append(f"# SynthID-text removal benchmark — {config['tag']}")
    L.append("")
    L.append(f"- Date: {config['timestamp']}")
    L.append(f"- watermarks-remover commit: {config.get('repo_commit') or 'unknown'}")
    L.append(f"- MarkLLM commit: {config.get('markllm_commit') or 'unknown'}")
    L.append(f"- Generator/detector model: {config['markllm_model']}")
    L.append(f"- Corpus: {config['corpus']} ({config['docs']} docs x {config['seeds']} seeds)")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append(
        "Watermarked and unwatermarked samples are generated with the MarkLLM "
        f"{config['scheme']} scheme (same config for generation and detection). "
        "Each sample must pass a sanity gate (watermarked detected, non-empty) before it "
        "counts. Rows: control (no removal), layer-a (Unicode scrub only), "
        "rewrite-<tactic>:<candidates> (Layer B rewrite), optional restamp-* "
        "(rewrite of the unwatermarked control to detect re-stamping)."
    )
    L.append("")
    L.append(
        "**Caveats:** MarkLLM's SynthID is an independent reimplementation under a "
        "config the benchmark controls — detection is only valid against the same "
        "config+keys, and it is **not** Google's production SynthID-Text keying. "
        "(Google retired text watermarking on its API in Aug 2026, so no vendor "
        "tier is available.) Rewriting with a watermarked model can re-stamp the "
        "text."
    )
    L.append("")
    L.append(
        "**Semantic divergence:** `1 - cosine(embed(original), embed(candidate))` "
        f"via sentence-transformers (model: {config.get('semantic_model') or 'not configured'}). "
        "Optional: when the package/model is unavailable the metric is None and the "
        "column renders '—'."
    )
    L.append("")
    L.append(
        "**Robust clear:** a rewrite counts as cleared when its after-score sits at "
        f"least `--target-margin` ({config.get('target_margin', 0.0)}) below the "
        "threshold. `robust %` is the share cleared by that margin; at the default "
        "0.0 it equals `clear %`, so a hair-thin crossing is still counted as clear."
    )
    L.append("")
    L.append(
        "**AI-likeness ↓:** the configured human-likeness backend "
        f"({config.get('human_backend', 'stylometry')}"
        + (
            f", using {config.get('human_backend_used')}"
            if config.get("human_backend_used")
            else ""
        )
        + (f" - {config.get('human_backend_reason')}" if config.get("human_backend_reason") else "")
        + "); lower = more human. "
        "**AUROC ↓:** post-removal AUROC (rewritten-watermarked vs rewritten-plain "
        "scores); closer to 0.5 = the rewritten population is less separable "
        "(more removal). **noop:** rewrites that returned ≈ their input (see "
        "backtranslate caveat below) and were excluded from the clear rate."
    )
    L.append("")
    L.append("## Results (per variant)")
    L.append("")
    post = (auroc or {}).get("post") or {}
    L.append(
        "| Variant | n | clear % | robust % | Δscore μ | margin μ | lex div | sem div | AI-likeness ↓ | AUROC ↓ | len ratio | nums keep | tok out | att | s/doc | clears/MTok | noop |"
    )
    L.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for variant, a in agg.items():
        L.append(
            "| {v} | {n} | {cr} | {rc} | {d} | {m} | {ld} | {sd} | {hu} | {au} | {lr} | {np} | {to} | {att} | {s} | {eff} | {nk} |".format(
                v=variant,
                n=a["n"],
                cr=_fmt(a["clear_rate"]),
                rc=_fmt(a.get("robust_clear_rate")),
                d=_fmt(a["mean_score_delta"]),
                m=_fmt(a.get("mean_margin")),
                ld=_fmt(a["mean_lexical_divergence"]),
                sd=_fmt(a.get("mean_semantic_divergence")),
                hu=_fmt(a.get("mean_ai_style_score")),
                au=_fmt(post.get(variant)),
                lr=_fmt(a["mean_length_ratio"]),
                np=_fmt(a["mean_numbers_preserved"]),
                to=_fmt(a["mean_tokens_out"]),
                att=_fmt(a.get("mean_attempts")),
                s=_fmt(a["mean_seconds"]),
                eff=_fmt(a["clears_per_mtok_out"]),
                nk=a.get("noop_n", 0),
            )
        )
    L.append("")
    L.append("## Controls")
    L.append("")
    excluded = [s for s in samples if s.get("excluded")]
    L.append(
        f"- Sanity-gate exclusions: {len(excluded)}/{len(samples)} "
        f"({'none' if not excluded else '; '.join(s.get('excluded_reason', '') for s in excluded[:5])})"
    )
    base_auroc = (auroc or {}).get("baseline_auroc")
    if base_auroc is not None:
        L.append(
            f"- Baseline detector AUROC (orig wm vs plain): {_fmt(base_auroc)} "
            "(≈1.0 = the same-config detector separates; a crash here means the "
            "same-config rig is not working)"
        )
    if "layer-a" in agg:
        L.append(
            f"- Layer A only clear rate: {_fmt(agg['layer-a']['clear_rate'])} "
            "(expect ≈0: statistical marks survive a Unicode scrub)"
        )
    if any(v.startswith("restamp-") for v in agg):
        for v, a in agg.items():
            if v.startswith("restamp-"):
                L.append(
                    f"- {v}: after-positive {a['after_positive']}/{a['n']} "
                    "(>0 ⇒ rewrite backend re-stamps the unwatermarked control)"
                )
    else:
        L.append("- Re-stamp control: not run (pass --restamp-control)")
    L.append("")
    L.append("## Reproduction")
    L.append("")
    L.append("    " + config["command"])
    L.append("")
    L.append("Full per-row data: results.json / results.csv in this directory.")
    L.append("")
    return "\n".join(L) + "\n"


def render_markdown_minimal(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    agg: dict[str, Any],
) -> str:
    """Render minimal-mode search results as markdown report."""
    L: list[str] = []
    L.append(f"# SynthID-text minimal-rewrite-level benchmark — {config['tag']}")
    L.append("")
    L.append(f"- Date: {config['timestamp']}")
    L.append(f"- watermarks-remover commit: {config.get('repo_commit') or 'unknown'}")
    L.append(f"- MarkLLM commit: {config.get('markllm_commit') or 'unknown'}")
    L.append(f"- Generator/detector model: {config['markllm_model']}")
    L.append(f"- Corpus: {config['corpus']} ({config['docs']} docs x {config['seeds']} seeds)")
    L.append(
        f"- Rewrite level range: {config['rewrite_level_start']} → {config['rewrite_level_max']} "
        f"by {config['rewrite_level_step']} ({config['level_attempts']} attempts/level)"
    )
    sem_config = config.get("semantic_model") or "not configured"
    sem_seen = any(r.get("semantic_divergence") is not None for r in rows)
    if config.get("semantic_available"):
        sem_status = "available" if sem_seen else "loaded; no semantic divergences recorded"
    else:
        sem_status = f"UNAVAILABLE -- {config.get('semantic_reason') or 'unknown'}"
    L.append(f"- Semantic model: {sem_config} ({sem_status})")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append(
        "For each watermarked sample the benchmark starts at the lowest rewrite "
        f"level and raises it by {config['rewrite_level_step']} per loop until a "
        "rewrite is no longer watermarked (same-config MarkLLM detection). "
        "The chosen rewrite is the "
        "one with the smallest semantic divergence among the clearing attempts at "
        "that level; that level's value and the resulting lexical/semantic "
        "divergence are recorded. Samples that never clear are excluded from the "
        "divergence average but counted in the clear rate."
    )
    if config.get("target_margin"):
        L.append("")
        L.append(
            f"A rewrite only counts as cleared when its after-score sits at least "
            f"{config['target_margin']:.4f} points below the detection threshold "
            "(--target-margin), so a hair-thin crossing is not a robust removal."
        )
    L.append("")
    L.append("## Results (across samples)")
    L.append("")
    L.append("| Metric | Value |")
    L.append("| --- | --- |")
    L.append(f"| Samples evaluated | {agg['n_samples']} |")
    L.append(f"| Samples excluded (sanity gate / generation) | {agg.get('n_excluded', 0)} |")
    L.append(f"| Identical generations across seeds | {agg.get('n_duplicate_generations', 0)} |")
    L.append(f"| Cleared | {agg['n_cleared']} |")
    L.append(f"| Clear rate | {_fmt(agg['clear_rate'])} |")
    L.append(f"| Mean minimal level | {_fmt(agg['mean_min_level'])} |")
    L.append(f"| Median minimal level | {_fmt(agg['median_min_level'])} |")
    L.append(f"| Mean minimal semantic divergence | {_fmt(agg['mean_min_semantic_divergence'])} |")
    L.append(
        f"| Median minimal semantic divergence | {_fmt(agg['median_min_semantic_divergence'])} |"
    )
    L.append(f"| Mean minimal lexical divergence | {_fmt(agg['mean_min_lexical_divergence'])} |")
    L.append(f"| Mean minimal margin (threshold-score) | {_fmt(agg['mean_min_margin'])} |")
    L.append("")
    L.append("### Level usage (minimal level per cleared sample)")
    L.append("")
    L.append("| Level | samples |")
    L.append("| --- | ---: |")
    for level, count in agg.get("level_usage") or []:
        L.append(f"| {_fmt(level)} | {count} |")
    L.append("")
    L.append("## Per-sample rows")
    L.append("")
    L.append("| doc | seed | cleared | level | margin | lex div | sem div | attempts |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        L.append(
            "| {doc} | {seed} | {cleared} | {lv} | {mg} | {ld} | {sd} | {att} |".format(
                doc=r["doc"],
                seed=r["seed"],
                cleared="yes" if r.get("cleared") else ("no" if r.get("cleared") is False else "—"),
                lv=_fmt(r.get("level")),
                mg=_fmt(r.get("margin")),
                ld=_fmt(r.get("lexical_divergence")),
                sd=_fmt(r.get("semantic_divergence")),
                att=r.get("attempts"),
            )
        )
    L.append("")
    L.append("## Reproduction")
    L.append("")
    L.append("    " + config["command"])
    L.append("")
    L.append("Full per-row data: results.json / results.csv in this directory.")
    L.append("")
    return "\n".join(L) + "\n"


def _steps_str(steps: list[tuple[str, float]]) -> str:
    """Format strategy steps into human-readable string."""
    return " → ".join(f"{s}@{level:g}" for s, level in steps)


def render_markdown_strategy(
    config: dict[str, Any], samples: list[dict[str, Any]], res: dict[str, Any]
) -> str:
    """Render strategy search results as markdown report."""
    L: list[str] = []
    L.append(f"# SynthID-text strategy search — {config['tag']}")
    L.append("")
    L.append(f"- Date: {config['timestamp']}")
    L.append(f"- watermarks-remover commit: {config.get('repo_commit') or 'unknown'}")
    L.append(f"- MarkLLM commit: {config.get('markllm_commit') or 'unknown'}")
    L.append(f"- Generator/detector model: {config['markllm_model']}")
    L.append(f"- Corpus: {config['corpus']} ({config['docs']} docs x {config['seeds']} seeds)")
    L.append(f"- Rewrite backend: {config['rewrite_backend']} ({config['rewrite_model']})")
    L.append(
        f"- Human-likeness backend: {config.get('human_backend', 'stylometry')}"
        + (
            f" (using {config.get('human_backend_used')})"
            if config.get("human_backend_used")
            else ""
        )
        + (f" - {config.get('human_backend_reason')}" if config.get("human_backend_reason") else "")
    )
    L.append(f"- Semantic model: {config.get('semantic_model') or 'not configured'}")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append(
        "Same-config MarkLLM SynthID samples. A strategy is an ordered list of "
        "`tactic@intensity` steps, each a Layer B rewrite with a numeric intensity "
        "modulating that tactic's prompt, applied sequentially (output feeds the "
        "next step). "
        + (
            "This run searched the strategy space: Phase 1 sweeps each tactic's "
            "intensity grid; Phase 2 runs a per-weight-vector beam search that "
            "combines an order of tactics with that weight's top intensities "
            "(`--phase2-levels-per-tactic`), so both step order and intensity are "
            "explored. The recommended strategy is the point on the weight-independent "
            "Pareto frontier that best matches the configured recommend weight "
            f"(`--recommend-weight`, default {config.get('recommend_weight', '0.5/0.3/0.2')}); "
            "the frontier below is unaffected by weights."
            if not config.get("strategies")
            else "This run composed and scored the explicitly requested strategy."
        )
    )
    L.append("")
    L.append(
        "Axes: **robust clear %** (removal, ↑; requires `--target-margin` below the "
        "threshold), **semantic divergence** (meaning drift, ↓), **human_like** "
        "(`1 - AI-likeness`, ↑ under the configured human-likeness backend)."
    )
    L.append("")
    rec = res.get("recommended")
    floor = config.get("coverage_floor", 0.5)
    L.append("## Recommended strategy")
    L.append("")
    if not rec:
        L.append(
            "No strategy in the searched space removed the mark on enough inputs to be "
            f"recommendable (population robust clear below the `--coverage-floor` {floor}). "
            "Nothing is recommended; the frontier below is diagnostics only."
        )
    else:
        L.append(f"- **{_steps_str(rec['steps'])}**")
        L.append(f"- robust clear %: {_fmt(rec.get('robust_clear_rate'))} (n={rec.get('n', 0)})")
        L.append(f"- semantic divergence: {_fmt(rec.get('sem_div'))}")
        L.append(f"- human_like: {_fmt(rec.get('human_like'))}")
        if rec.get("steps") and rec["steps"][-1][0] == "humanize":
            L.append("- final step: `humanize` style polish (runs last by design)")
    L.append("")
    adaptive = res.get("adaptive")
    if adaptive:
        L.append("## Adaptive escalation (escalate until removed)")
        L.append("")
        L.append(
            f"- default (no escalation): {_fmt(adaptive.get('base_clear_rate'))} "
            f"| after escalation: {_fmt(adaptive.get('adapt_clear_rate'))} clear (n={adaptive.get('n')})"
        )
        L.append(
            f"- escalation level to clear: mean {_fmt(adaptive.get('mean_level'))}, "
            f"median {_fmt(adaptive.get('median_level'))}, max {_fmt(adaptive.get('max_level'))}"
        )
        L.append("")
        L.append("| doc | seed | clear @ default | clear after escalation | level |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for r in adaptive.get("rows") or []:
            L.append(
                f"| {r.get('doc')} | {r.get('seed')} | "
                f"{'yes' if r.get('base_cleared') else 'no'} | "
                f"{'yes' if r.get('cleared') else 'no'} | {r.get('escalation_level')} |"
            )
        L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(_render_strategy_verdict(res))
    L.append("")
    L.append("## Pareto frontier (weight-independent)")
    L.append("")
    front = res.get("frontier") or []
    has_holdout = any(c.get("holdout_robust_clear_rate") is not None for c in front)
    if not front:
        L.append("No strategy had all three axes available; the frontier is empty.")
    else:
        L.append(
            "| strategy | robust % | sem div | human_like ↑ |"
            + (" holdout % |" if has_holdout else "")
        )
        L.append("| --- | ---: | ---: | ---: |" + (" ---: |" if has_holdout else ""))
        for c in front:
            row = (
                f"| {_steps_str(c['steps'])} | {_fmt(c.get('robust_clear_rate'))} | "
                f"{_fmt(c.get('sem_div'))} | {_fmt(c.get('human_like'))} |"
            )
            if has_holdout:
                row += f" {_fmt(c.get('holdout_robust_clear_rate'))} |"
            L.append(row)
    L.append("")
    L.append("## Per-tactic intensity curves (single-pass)")
    L.append("")
    curves = res.get("intensity_curves") or {}
    if not curves:
        L.append("Not produced (compose-run mode, or search did not sweep).")
    else:
        for tactic, rows in curves.items():
            L.append(f"### {tactic}")
            L.append("")
            L.append("| level | robust % | sem div | human_like ↑ |")
            L.append("| ---: | ---: | ---: | ---: |")
            for r in rows:
                L.append(
                    f"| {r['level']:g} | {_fmt(r.get('robust_clear_rate'))} | "
                    f"{_fmt(r.get('sem_div'))} | {_fmt(r.get('human_like'))} |"
                )
            L.append("")
    L.append("## Caveats")
    L.append("")
    L.append(
        "Same-config MarkLLM detection only; not Google's production SynthID-Text "
        "keying (retired from the API Aug 2026). Human-likeness and semantic "
        "divergence are gauges, not proof of human authorship. Strategies are "
        "search results on this corpus/backend; re-confirm on a larger powered run."
    )
    L.append("")
    L.append("## Reproduction")
    L.append("")
    L.append("    " + config["command"])
    L.append("")
    return "\n".join(L) + "\n"


def _render_strategy_verdict(res: dict[str, Any]) -> str:
    """Render the honest 'can the mark be removed?' verdict for the report."""
    cands = res.get("candidates") or []
    verdict = res.get("verdict") or _strategy_verdict(cands)
    rates = [c.get("robust_clear_rate") for c in cands if c.get("robust_clear_rate") is not None]
    best = _fmt(max(rates)) if rates else "n/a"
    if verdict == "removable":
        return (
            "The searched space contains a strategy that robustly clears the mark "
            f"(best robust % = {best}); treat the recommended strategy as the answer "
            "to whether the mark is removable."
        )
    if verdict == "partial":
        return (
            f"Strategies in the searched space partially clear the mark (best robust % = {best}), "
            "but none robustly clears every sample at the configured `--target-margin`."
        )
    if verdict == "resists":
        return (
            f"No strategy in the searched space cleared the mark (best robust % = {best}). "
            "At this token length the mark resists the searched attacks; a lower "
            "`--target-margin` or a higher-aggression strategy set (more steps / higher "
            "intensities via `--phase2-levels-per-tactic`) is the next lever."
        )
    return (
        "Verdict undetermined: no candidate strategy had all three axes evaluated "
        "(add `--require-semantic` / a detector)."
    )


def _strategy_csv(res: dict[str, Any]) -> list[str]:
    """Export strategy search results to CSV format."""
    import io

    front = {id(c) for c in (res.get("frontier") or [])}
    rec = res.get("recommended")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(
        [
            "strategy",
            "robust_clear_rate",
            "semantic_divergence",
            "human_like",
            "n",
            "holdout_robust_clear_rate",
            "humanize_last",
            "adaptive_base_clear",
            "adaptive_clear",
            "adaptive_median_level",
            "recommended",
            "in_frontier",
        ]
    )
    adaptive = res.get("adaptive") or {}
    for c in res.get("candidates") or []:
        steps = c.get("steps") or []
        is_rec = rec is not None and id(rec) == id(c)
        w.writerow(
            [
                _steps_str(steps),
                c.get("robust_clear_rate"),
                c.get("sem_div"),
                c.get("human_like"),
                c.get("n"),
                c.get("holdout_robust_clear_rate"),
                1 if steps and steps[-1][0] == "humanize" else 0,
                adaptive.get("base_clear_rate") if is_rec else "",
                adaptive.get("adapt_clear_rate") if is_rec else "",
                adaptive.get("median_level") if is_rec else "",
                1 if is_rec else 0,
                1 if id(c) in front else 0,
            ]
        )
    return out.getvalue().rstrip("\n").splitlines()


def _csv_cell(value: Any) -> Any:
    """Render a CSV cell: None -> empty field, True/False -> 1/0, else as-is.

    Prevents ``str(None)`` (the literal text "None") from leaking into numeric
    columns and lets ``csv.writer`` escape any commas in the notes field.
    """
    if value is None:
        return ""
    if value is True:
        return 1
    if value is False:
        return 0
    return value


def _minimal_csv(rows: list[dict[str, Any]]) -> list[str]:
    """minimal csv."""
    import io

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(
        [
            "doc",
            "seed",
            "variant",
            "kind",
            "cleared",
            "before_pos",
            "score_before",
            "level",
            "margin",
            "score_after",
            "lexical_divergence",
            "semantic_divergence",
            "attempts",
            "seconds",
            "notes",
        ]
    )
    for r in rows:
        w.writerow(
            [
                r["doc"],
                r["seed"],
                "minimal",
                "minimal",
                _csv_cell(r.get("cleared")),
                1 if r.get("before_pos") else 0,
                _csv_cell(r.get("score_before")),
                _csv_cell(r.get("level")),
                _csv_cell(r.get("margin")),
                _csv_cell(r.get("score_after")),
                _csv_cell(r.get("lexical_divergence")),
                _csv_cell(r.get("semantic_divergence")),
                _csv_cell(r.get("attempts")),
                _csv_cell(r.get("seconds")),
                "; ".join(str(n) for n in r.get("notes") or []),
            ]
        )
    return out.getvalue().rstrip("\n").splitlines()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for text rewrite tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--markllm-dir", default=os.environ.get("MARKLLM_DIR"))
    p.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS, help="Dir of .txt seeds or a single file"
    )
    p.add_argument("--docs", type=int, default=3, help="Max seed documents to use (default: 3)")
    p.add_argument(
        "--scheme",
        default=DEFAULT_SCHEME,
        choices=sorted(SCHEMES),
        help="MarkLLM watermark scheme (default: synthid); any key of the detector's scheme map",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Algorithm config JSON (default: <MarkLLM checkout>/config/<ALG>.json)",
    )
    p.add_argument("--seeds", type=int, default=1, help="Watermark seeds per doc (default: 1)")
    p.add_argument("--seed-base", type=int, default=1, help="First seed value (default: 1)")
    p.add_argument(
        "--max-new-tokens", type=int, default=300, help="Generation length (default: 300)"
    )
    p.add_argument(
        "--variants",
        default="paraphrase:3",
        help="Comma list of <tactic>:<candidates> (default: paraphrase:3). "
        "candidates = max rewrite attempts per input; the Layer B loop stops "
        "early when an attempt passes evaluation.",
    )
    p.add_argument(
        "--mode",
        choices=("variants", "minimal", "strategy"),
        default="variants",
        help="variants: run named-tactic rewrite variants (default). minimal: "
        "per sample, raise the numeric rewrite level (--rewrite-level-start, "
        "+--rewrite-level-step) until a rewrite is no longer watermarked, then "
        "report the average minimal level. strategy: search (or compose-run, with "
        "--strategies) for the best tactic@intensity combination and report the "
        "Pareto frontier.",
    )
    p.add_argument(
        "--restamp-control", action="store_true", help="Also rewrite the unwatermarked control"
    )
    p.add_argument("--out-dir", type=Path, default=Path("bench-synthid-text-results"))
    p.add_argument("--tag", default="", help="Short label for the report")
    p.add_argument(
        "--markllm-model",
        default=os.environ.get("MARKLLM_MODEL", DEFAULT_MARKLLM_MODEL),
    )
    p.add_argument(
        "--markllm-timeout",
        type=float,
        default=float(os.environ.get("WATERMARKS_MARKLLM_TIMEOUT", "600")),
    )
    p.add_argument(
        "--rewrite-backend",
        choices=("ollama", "openai-compatible"),
        default=os.environ.get("WATERMARKS_REWRITE_BACKEND", "ollama"),
    )
    p.add_argument("--rewrite-model", default=os.environ.get("WATERMARKS_REWRITE_MODEL"))
    p.add_argument(
        "--rewrite-base-url",
        default=os.environ.get("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--rewrite-api-key", default=None, help="API key (env-only in child; never argv)"
    )
    p.add_argument(
        "--rewrite-allow-remote",
        action="store_true",
        default=os.environ.get("WATERMARKS_REWRITE_ALLOW_REMOTE", "").strip().lower()
        in ("1", "true", "yes", "on"),
        help="Send content to non-loopback rewrite endpoints (default: $WATERMARKS_REWRITE_ALLOW_REMOTE)",
    )
    p.add_argument("--rewrite-temperature", type=float, default=0.9)
    p.add_argument(
        "--rewrite-loops",
        type=int,
        default=1,
        help="Max evaluation rounds per rewrite; each round generates "
        "--candidates variants and stops when one passes (default: 1)",
    )
    p.add_argument(
        "--rewrite-level-start",
        type=float,
        default=0.1,
        help="minimal mode: first rewrite level tried (default: 0.1)",
    )
    p.add_argument(
        "--rewrite-level-step",
        type=float,
        default=0.1,
        help="minimal mode: level increment per loop (default: 0.1, so 0.1, 0.2, ...)",
    )
    p.add_argument(
        "--rewrite-level-max",
        type=float,
        default=1.0,
        help="minimal mode: highest rewrite level tried (default: 1.0)",
    )
    p.add_argument(
        "--level-attempts",
        type=int,
        default=3,
        help="minimal mode: rewrite attempts per level before escalating (default: 3)",
    )
    p.add_argument(
        "--target-margin",
        type=float,
        default=0.0,
        help="Robust-removal margin: require the after-score to sit at least "
        "this many points below the detection threshold before a rewrite counts "
        "as cleared (default: 0.0). Higher is stricter — it survives a "
        "production detector but costs more content churn.",
    )
    p.add_argument(
        "--chars-per-token", type=float, default=4.0, help="Cost token estimate (default: 4.0)"
    )
    p.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic divergence (optional; "
        "requires sentence-transformers. Unavailable -> semantic_divergence is "
        "None and the report renders '—').",
    )
    p.add_argument(
        "--require-semantic",
        action="store_true",
        help="Fail fast (exit 2) when the semantic backend is unavailable "
        "instead of running a lexical-only benchmark.",
    )
    p.add_argument(
        "--cost-per-mtok-in", type=float, default=0.0, help="USD per million input tokens"
    )
    p.add_argument(
        "--cost-per-mtok-out", type=float, default=0.0, help="USD per million output tokens"
    )
    p.add_argument(
        "--no-worker",
        action="store_true",
        help="Do not use the persistent MarkLLM serve worker (one-shot subprocesses)",
    )
    p.add_argument(
        "--noop-lex-floor",
        type=float,
        default=0.05,
        help="Treat a rewrite that changed fewer than this fraction of bigrams as "
        "a no-op (cleared=None, so it never counts as a clear; reported as noop; "
        "default 0.05, 0 disables). Forwarded to rewrite_text.py. Mirrors the "
        "backtranslate no-op fix that keeps a near-verbatim output from being "
        "read as '0%% clear'.",
    )
    p.add_argument(
        "--human-backend",
        choices=("stylometry", "lastde", "binoculars", "pangram"),
        default="stylometry",
        help="Human-likeness axis. 'stylometry' (default) is the always-on "
        "stdlib-only score; 'lastde'/'binoculars' run an optional offline "
        "AI-text detector (probed at startup, degrade to stylometry if absent); "
        "'pangram' uses the Pangram Labs async **bulk** API (key in "
        "PANGRAM_API_KEY, model via --human-pangram-model). Score is AI-likeness "
        "(lower = more human); human_like = 1 - score.",
    )
    p.add_argument(
        "--human-detector-dir",
        default=os.environ.get("HUMAN_DETECTOR_DIR"),
        help="Checkout root for the Lastde/Binoculars detector (default: "
        "$HUMAN_DETECTOR_DIR). Required when --human-backend is a detector.",
    )
    p.add_argument(
        "--human-pangram-model",
        default=os.environ.get("PANGRAM_MODEL", "pangram-4"),
        help="Pangram model id (default: pangram-4; falls back to an allowed "
        "model discovered via GET /models). Key is read from PANGRAM_API_KEY.",
    )
    p.add_argument(
        "--intensity-grid",
        default="0.2,0.4,0.6,0.8,1.0",
        help="Comma list of intensities swept per tactic in strategy mode "
        "(default: 0.2,0.4,0.6,0.8,1.0)",
    )
    p.add_argument(
        "--weight-grid",
        default="0.8/0.1/0.1,0.5/0.3/0.2,0.2/0.6/0.2,0.2/0.2/0.6,0.34/0.33/0.33",
        help="Comma list of w_removal/w_semantic/w_human weight vectors driving the "
        "composition search (default: the 5-vector grid). The Pareto frontier is "
        "weight-independent.",
    )
    p.add_argument("--beam", type=int, default=4, help="Composition-search beam width (default: 4)")
    p.add_argument(
        "--max-passes", type=int, default=3, help="Max strategy steps in the search (default: 3)"
    )
    p.add_argument(
        "--phase2-levels-per-tactic",
        type=int,
        default=3,
        help="Per-weight top-k intensities considered in the phase 2 beam search "
        "(default: 3). Raising this broadens the search over intensities and "
        "aggressive multi-step strategies but costs more evals; lower it to stay "
        "inside a tight wall-clock budget.",
    )
    p.add_argument(
        "--recommend-weight",
        default="0.5/0.3/0.2",
        help="w_removal/w_semantic/w_human used to pick the recommended strategy "
        "from the weight-independent Pareto frontier (default: 0.5/0.3/0.2).",
    )
    p.add_argument(
        "--coverage-floor",
        type=float,
        default=0.5,
        help="Minimum population robust clear rate a strategy must reach before it is "
        "recommendable (default: 0.5). A strategy that clears only a fraction of the "
        "inputs is never put forward as 'the best'.",
    )
    p.add_argument(
        "--eval-split",
        type=float,
        default=0.0,
        help="Fraction of documents kept for the search; the rest is held out and used "
        "to validate that the recommended strategy generalizes across inputs "
        "(default: 0.0 = no holdout).",
    )
    p.add_argument(
        "--humanize-intensity",
        type=float,
        default=0.4,
        help="Intensity of the auto-appended final humanize polish step (default: 0.4).",
    )
    p.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Escalate the recommended strategy per-input until the mark is removed "
        "(runtime adaptivity). Inputs that resist the default are re-run with every "
        "step's intensity raised by --escalation-step.",
    )
    p.add_argument(
        "--escalation-step",
        type=float,
        default=0.1,
        help="Intensity increment per adaptive escalation round (default: 0.1).",
    )
    p.add_argument(
        "--escalation-max",
        type=float,
        default=1.0,
        help="Maximum intensity an adaptive escalation may reach (default: 1.0).",
    )
    p.add_argument(
        "--escalation-attempts",
        type=int,
        default=3,
        help="Max adaptive escalation rounds per input (default: 3).",
    )
    p.add_argument(
        "--write-strategy-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write each evaluated strategy candidate's per-sample input/output text "
        "under <out-dir>/work/strategies/<strategy>/ for inspection (default: on; "
        "disable with --no-write-strategy-outputs).",
    )
    p.add_argument(
        "--strategies",
        default=None,
        help="Explicit strategy to compose-run and score (not a search), e.g. "
        "'chunk@0.6,paraphrase@0.3,humanize@1.0'. Used with --mode strategy.",
    )
    p.add_argument(
        "--layer-a-after",
        action="store_true",
        default=False,
        help="Re-run the Unicode scrub (/clean) on the composed strategy's final "
        "output. Default OFF: the rewrite backend is assumed watermark-safe.",
    )
    return p


def _semantic_startup_probe(bench: Benchmark, semantic_model: str, require: bool) -> int | None:
    """Probe the semantic backend at startup; return an exit code or None.

    Semantic divergence is the quality axis of a default-level decision, so a
    missing backend must be loud: report the real cause immediately instead of
    emitting hours of '—' rows, and fail fast under --require-semantic.
    """
    if not semantic_model:
        return None
    if bench.semantic.available():
        eprint(f"semantic backend: ready ({semantic_model})")
        return None
    eprint(
        f"warning: semantic backend unavailable ({semantic_model}): "
        f"{bench.semantic.reason() or 'unknown'} - all semantic divergences will be None"
    )
    if require:
        eprint("error: --require-semantic passed but the semantic backend is unavailable")
        return 2
    return None


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()

    if not args.markllm_dir:
        eprint("error: --markllm-dir (or MARKLLM_DIR) is required")
        return 2
    upstream = Path(args.markllm_dir).expanduser().resolve()
    if not (upstream / "watermark").is_dir():
        eprint(f"error: MarkLLM checkout incomplete (no watermark/ dir): {upstream}")
        return 2
    if not args.rewrite_model:
        eprint("error: --rewrite-model is required (e.g. llama3.2 for ollama)")
        return 2
    if not _base_url_is_loopback(args.rewrite_base_url) and not args.rewrite_allow_remote:
        eprint(
            "error: rewrite base URL is not loopback; pass --rewrite-allow-remote "
            "(content will leave this machine)"
        )
        return 2

    # Fail fast on a bad strategy grid/spec and positive-value flags before
    # constructing Benchmark, which starts MarkLLMWorker and the semantic
    # backend. A misconfigured --intensity-grid / --weight-grid / --beam /
    # --max-passes would otherwise only be rejected after the expensive setup
    # (and before the worker-cleanup path that normally runs on early returns).
    if args.mode == "strategy":
        parse_float_grid(args.intensity_grid)
        parse_weight_grid(args.weight_grid)
        parse_weight_vec(args.recommend_weight)
        if args.phase2_levels_per_tactic < 1:
            eprint("error: --phase2-levels-per-tactic must be >= 1")
            return 1
        if args.beam < 1:
            eprint("error: --beam must be >= 1")
            return 1
        if args.max_passes < 1:
            eprint("error: --max-passes must be >= 1")
            return 1
        if args.escalation_step <= 0:
            eprint("error: --escalation-step must be positive")
            return 1
        if not (0 < args.escalation_max <= 1):
            eprint("error: --escalation-max must be in (0,1]")
            return 1
        if args.escalation_attempts < 1:
            eprint("error: --escalation-attempts must be >= 1")
            return 1
        if args.strategies:
            parse_strategy(args.strategies)

    bench = Benchmark(args, upstream)
    if not bench.corpus:
        eprint("error: empty corpus")
        return 2
    probe = _semantic_startup_probe(bench, args.semantic_model, args.require_semantic)
    if probe is not None:
        return probe

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"synthid-text-{time.strftime('%Y%m%d-%H%M%S')}"
    config = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tag": tag,
        "repo_commit": _repo_commit(),
        "markllm_commit": _markllm_commit(upstream),
        "markllm_dir": str(upstream),
        "markllm_model": args.markllm_model,
        "scheme": args.scheme,
        "config": str(args.config) if args.config else None,
        "variants": [f"{s}:{c}" for s, c in bench.variants],
        "corpus": str(args.corpus),
        "docs": args.docs,
        "seeds": args.seeds,
        "seed_base": args.seed_base,
        "max_new_tokens": args.max_new_tokens,
        "rewrite_backend": args.rewrite_backend,
        "rewrite_model": args.rewrite_model,
        "rewrite_base_url": args.rewrite_base_url,
        "rewrite_temperature": args.rewrite_temperature,
        "rewrite_loops": args.rewrite_loops,
        "restamp_control": args.restamp_control,
        "chars_per_token": args.chars_per_token,
        "cost_per_mtok_in": args.cost_per_mtok_in,
        "cost_per_mtok_out": args.cost_per_mtok_out,
        "mode": args.mode,
        "rewrite_level_start": args.rewrite_level_start,
        "rewrite_level_step": args.rewrite_level_step,
        "rewrite_level_max": args.rewrite_level_max,
        "level_attempts": args.level_attempts,
        "target_margin": args.target_margin,
        "semantic_model": args.semantic_model,
        "semantic_available": bool(bench.semantic.available()),
        "semantic_reason": bench.semantic.reason(),
        "noop_lex_floor": args.noop_lex_floor,
        "human_backend": args.human_backend,
        "human_detector_dir": args.human_detector_dir,
        "human_pangram_model": bench.human.pangram_model,
        "human_backend_used": bench.human.backend_used,
        "intensity_grid": args.intensity_grid,
        "weight_grid": args.weight_grid,
        "beam": args.beam,
        "max_passes": args.max_passes,
        "phase2_levels_per_tactic": args.phase2_levels_per_tactic,
        "recommend_weight": args.recommend_weight,
        "coverage_floor": args.coverage_floor,
        "eval_split": args.eval_split,
        "humanize_intensity": args.humanize_intensity,
        "adaptive": args.adaptive,
        "escalation_step": args.escalation_step,
        "escalation_max": args.escalation_max,
        "escalation_attempts": args.escalation_attempts,
        "strategies": args.strategies,
        "layer_a_after": args.layer_a_after,
        "write_strategy_outputs": args.write_strategy_outputs,
        "command": " ".join(
            [
                "python3 service/scripts/bench_synthid_text.py",
                f"--markllm-dir {args.markllm_dir}",
                f"--scheme {args.scheme}",
                *([f"--config {args.config}"] if args.config else []),
                f"--corpus {args.corpus}",
                f"--docs {args.docs} --seeds {args.seeds} --seed-base {args.seed_base}",
                f"--max-new-tokens {args.max_new_tokens}",
                f"--mode {args.mode}",
                f"--variants {args.variants}",
                f"--rewrite-backend {args.rewrite_backend}",
                f"--rewrite-model {args.rewrite_model}",
                f"--rewrite-base-url {args.rewrite_base_url}",
                f"--rewrite-temperature {args.rewrite_temperature}",
                f"--rewrite-loops {args.rewrite_loops}",
                f"--rewrite-level-start {args.rewrite_level_start}",
                f"--rewrite-level-step {args.rewrite_level_step}",
                f"--rewrite-level-max {args.rewrite_level_max}",
                f"--level-attempts {args.level_attempts}",
                *([f"--target-margin {args.target_margin}"] if args.target_margin else []),
                f"--semantic-model {args.semantic_model}",
                f"--noop-lex-floor {args.noop_lex_floor}",
                f"--human-backend {args.human_backend}",
                *(
                    [f"--human-detector-dir {args.human_detector_dir}"]
                    if args.human_detector_dir
                    else []
                ),
                *(
                    [f"--human-pangram-model {args.human_pangram_model}"]
                    if args.human_backend == "pangram"
                    else []
                ),
                f"--intensity-grid {args.intensity_grid}",
                f"--weight-grid {args.weight_grid}",
                f"--beam {args.beam}",
                f"--max-passes {args.max_passes}",
                f"--phase2-levels-per-tactic {args.phase2_levels_per_tactic}",
                f"--recommend-weight {args.recommend_weight}",
                f"--coverage-floor {args.coverage_floor}",
                *([f"--eval-split {args.eval_split}"] if args.eval_split else []),
                *(
                    [f"--humanize-intensity {args.humanize_intensity}"]
                    if args.humanize_intensity != 0.4
                    else []
                ),
                *(["--adaptive"] if args.adaptive else []),
                *(
                    [f"--escalation-step {args.escalation_step}"]
                    if args.escalation_step != 0.1
                    else []
                ),
                *(
                    [f"--escalation-max {args.escalation_max}"]
                    if args.escalation_max != 1.0
                    else []
                ),
                *(
                    [f"--escalation-attempts {args.escalation_attempts}"]
                    if args.escalation_attempts != 3
                    else []
                ),
                *([f"--strategies {args.strategies}"] if args.strategies else []),
                *(["--layer-a-after"] if args.layer_a_after else []),
                *(["--restamp-control"] if args.restamp_control else []),
                *(["--rewrite-allow-remote"] if args.rewrite_allow_remote else []),
                f"--out-dir {args.out_dir}",
                f"--tag {tag}",
            ]
        ),
    }

    workdir = out_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    eprint(f"corpus: {len(bench.corpus)} docs, {args.seeds} seed(s) each")
    eprint(f"variants: {', '.join(config['variants'])}")
    eprint(f"markllm via: {bench.python}")
    try:
        samples = bench.generate_samples(workdir)
        if args.mode == "minimal":
            rows = bench.minimal_search(samples, workdir)
        elif args.mode == "strategy":
            rows = []
            if args.strategies:
                strategy = _normalize_strategy(parse_strategy(args.strategies))
                candid = bench._eval_strategy(strategy, samples)
                if candid.get("n", 0) == 0:
                    eprint(
                        "error: strategy produced no evaluable samples (watermark not detected?)"
                    )
                    return 2
                candid["steps"] = strategy
                # Humanizer always runs last: append the finishing polish and re-measure.
                if not strategy or strategy[-1][0] != "humanize":
                    humanize_intensity = float(getattr(args, "humanize_intensity", 0.4))
                    final = bench._eval_strategy(
                        [*strategy, ("humanize", humanize_intensity)], samples
                    )
                    final["steps"] = [*strategy, ("humanize", humanize_intensity)]
                    candid = final
                strategy_outputs_written = bench._persist_strategy_outputs([candid], workdir)
                # Respect the coverage-floor contract: never publish a strategy that
                # does not clear enough inputs as the recommendation.
                floor = float(getattr(args, "coverage_floor", 0.5))
                rec = (
                    candid
                    if (
                        candid.get("robust_clear_rate") is not None
                        and candid["robust_clear_rate"] >= floor - 1e-9
                    )
                    else None
                )
                verdict = _strategy_verdict([candid]) if rec is not None else "resists"
                res = {
                    "candidates": [candid],
                    "recommended": rec,
                    "frontier": [candid],
                    "intensity_curves": {},
                    "verdict": verdict,
                    "strategy_outputs_written": strategy_outputs_written,
                }
            else:
                res = bench.strategy_search(samples, workdir)
            if getattr(args, "adaptive", False) and res.get("recommended") is not None:
                res["adaptive"] = bench._adaptive_apply_strategy(
                    res["recommended"]["steps"], samples
                )
        else:
            rows = bench.run_variants(samples, workdir)
    finally:
        bench.close_worker()

    # Scoring is done; reflect any backend fallback (e.g. Pangram -> stylometry)
    # that happened while measuring, so the report labels the scores correctly.
    config["human_backend_used"] = bench.human.backend_used
    config["human_backend_reason"] = bench.human.reason()

    if args.mode == "minimal":
        agg = aggregate_minimal(rows)
        report = render_markdown_minimal(config, samples, rows, agg)
        csv_lines = _minimal_csv(rows)
    elif args.mode == "strategy":
        report = render_markdown_strategy(config, samples, res)
        agg = {}
        csv_lines = _strategy_csv(res)
    else:
        # Attach USD cost using per-doc token estimates.
        for row in rows:
            q = row.get("quality") or {}
            if q:
                row["usd"] = (
                    q.get("tokens_in", 0) / 1e6 * args.cost_per_mtok_in
                    + q.get("tokens_out", 0) / 1e6 * args.cost_per_mtok_out
                )

        agg = aggregate(rows, bench.variants)

        auroc = compute_auroc(samples, rows, bench.variants)
        report = render_markdown(config, samples, rows, agg, auroc)
        import io

        _csv_buf = io.StringIO()
        _csv_w = csv.writer(_csv_buf)
        _csv_w.writerow(
            [
                "doc",
                "seed",
                "variant",
                "kind",
                "attempts",
                "evaluator",
                "passed",
                "before_pos",
                "after_pos",
                "cleared",
                "robust_cleared",
                "noop",
                "ai_style_score",
                "human_backend",
                "score_before",
                "score_after",
                "margin",
                "score_delta",
                "lexical_divergence",
                "semantic_divergence",
                "length_ratio",
                "numbers_preserved",
                "urls_preserved",
                "tokens_in",
                "tokens_out",
                "seconds",
                "usd",
                "notes",
            ]
        )
        for r in rows:
            q = r.get("quality") or {}
            delta = (
                round(r["score_before"] - r["score_after"], 4)
                if r.get("score_before") is not None and r.get("score_after") is not None
                else ""
            )
            _csv_w.writerow(
                [
                    r["doc"],
                    r["seed"],
                    r["variant"],
                    r.get("kind", ""),
                    _csv_cell(r.get("attempts")),
                    r.get("evaluator", ""),
                    _csv_cell(r.get("passed")),
                    1 if r.get("before_pos") else 0,
                    1 if r.get("after_pos") else 0,
                    _csv_cell(r.get("cleared")),
                    _csv_cell(r.get("robust_cleared")),
                    1 if r.get("noop") else 0,
                    _csv_cell(r.get("ai_style_score")),
                    r.get("human_backend", ""),
                    _csv_cell(r.get("score_before")),
                    _csv_cell(r.get("score_after")),
                    _csv_cell(r.get("margin")),
                    _csv_cell(delta),
                    _csv_cell(q.get("lexical_divergence")),
                    _csv_cell(q.get("semantic_divergence")),
                    _csv_cell(q.get("length_ratio")),
                    _csv_cell(q.get("numbers_preserved")),
                    _csv_cell(q.get("urls_preserved")),
                    _csv_cell(q.get("tokens_in")),
                    _csv_cell(q.get("tokens_out")),
                    _csv_cell(r.get("seconds")),
                    round(r.get("usd") or 0.0, 6),
                    "; ".join(str(n) for n in r.get("notes") or []),
                ]
            )
        csv_lines = _csv_buf.getvalue().rstrip("\n").splitlines()

    (out_dir / "report.md").write_text(report, encoding="utf-8")
    payload: dict[str, Any] = {"meta": config, "samples": samples, "rows": rows, "aggregates": agg}
    if args.mode == "strategy":
        payload["strategy"] = res
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    eprint("")
    eprint(f"results written to {out_dir}/")
    print("")
    if args.mode == "minimal":
        print("minimal rewrite level (across samples)")
        print("-" * 40)
        print(f"  samples evaluated : {agg['n_samples']}")
        print(f"  excluded          : {agg.get('n_excluded', 0)}")
        print(f"  duplicate gen     : {agg.get('n_duplicate_generations', 0)}")
        print(f"  cleared           : {agg['n_cleared']}")
        print(f"  clear rate        : {_fmt(agg['clear_rate'])}")
        print(f"  mean minimal level: {_fmt(agg['mean_min_level'])}")
        print(f"  mean min sem div  : {_fmt(agg['mean_min_semantic_divergence'])}")
        print(f"  mean min lex div  : {_fmt(agg['mean_min_lexical_divergence'])}")
        print(f"  mean min margin   : {_fmt(agg['mean_min_margin'])}")
    elif args.mode == "strategy":
        rec = res.get("recommended")
        print("best strategy (recommended)")
        print("-" * 40)
        if not rec:
            print("  none (no strategy removed the mark on enough inputs to be recommendable)")
        else:
            print(
                f"  steps         : {' → '.join(f'{s}@{level_i:g}' for s, level_i in rec['steps'])}"
            )
            print(f"  robust clear %: {_fmt(rec.get('robust_clear_rate'))}")
            print(f"  semantic div  : {_fmt(rec.get('sem_div'))}")
            print(f"  human_like    : {_fmt(rec.get('human_like'))}")
        adaptive = res.get("adaptive")
        if adaptive:
            print(
                f"  adaptive clear: default {_fmt(adaptive.get('base_clear_rate'))} -> "
                f"escalated {_fmt(adaptive.get('adapt_clear_rate'))} "
                f"(median level {_fmt(adaptive.get('median_level'))})"
            )
        print(
            f"  frontier size : {len(res.get('frontier') or [])} / candidates {len(res.get('candidates') or [])}"
        )
        n_outputs = res.get("strategy_outputs_written", 0)
        if n_outputs:
            print(
                f"  outputs       : {n_outputs} strategy dirs written to {workdir / 'strategies'} (--no-write-strategy-outputs to disable)"
            )
    else:
        print(
            "variant          n   clear%  dScore  margin  lexDiv  semDiv  lenR  nums  tokOut  att  s/doc  eff/MTok"
        )
        print("-" * 100)
        for variant, a in agg.items():
            print(
                f"{variant:<16} {a['n']:>3}  {_fmt(a['clear_rate']):>6}  "
                f"{_fmt(a['mean_score_delta']):>6}  {_fmt(a.get('mean_margin')):>6}  "
                f"{_fmt(a['mean_lexical_divergence']):>6}  "
                f"{_fmt(a.get('mean_semantic_divergence')):>6}  "
                f"{_fmt(a['mean_length_ratio']):>5}  {_fmt(a['mean_numbers_preserved']):>5}  "
                f"{_fmt(a['mean_tokens_out']):>6}  {_fmt(a.get('mean_attempts')):>4}  "
                f"{_fmt(a['mean_seconds']):>5}  {_fmt(a['clears_per_mtok_out']):>7}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
