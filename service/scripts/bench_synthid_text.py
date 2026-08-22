#!/usr/bin/env python3
"""Benchmark for MarkLLM text-watermark removal (Layer B rewrite).

Orchestrates the repo's existing machinery into a reproducible, shareable
benchmark:

  1. Generate a watermarked + unwatermarked corpus with a chosen MarkLLM
     scheme (same-config generation and detection; --scheme/--config).
  2. Run removal variants (Layer A only, Layer B rewrites at chosen
     strength x max-attempt counts; the rewrite loop stops early when an
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
import json
import os
import queue
import re
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

from common import eprint, subprocess_creationflags  # noqa: E402
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

    Each item is <strength>:<candidates>; strengths come from rewrite_text.py
    (paraphrase, backtranslate, structural, humanize, code). candidates is the
    max rewrite attempts per input — the Layer B loop stops early as soon as
    an attempt passes evaluation.
    """
    variants: list[tuple[str, int]] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise SystemExit(f"error: bad variant {item!r}; expected <strength>:<candidates>")
        strength, raw_c = parts
        try:
            c = int(raw_c)
        except ValueError:
            raise SystemExit(f"error: bad candidate count in variant {item!r}") from None
        if c < 1:
            raise SystemExit(f"error: candidate count must be >= 1 in variant {item!r}")
        variants.append((strength, c))
    if not variants:
        raise SystemExit("error: --variants must name at least one variant")
    return variants


def _base_url_is_loopback(base_url: str) -> bool:
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
    strength: str,
    candidates: int,
    max_loops: int,
    temperature: float,
    timeout: float,
    allow_remote: bool,
    api_key: str | None,
    markllm_model: str,
    markllm_timeout: float,
    markllm_scheme: str,
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
        "--strength",
        strength,
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
    a = set(re.findall(r"\d+", original))
    if not a:
        return 1.0
    b = set(re.findall(r"\d+", candidate))
    return len(a & b) / len(a)


def _urls_preserved(original: str, candidate: str) -> float:
    a = set(re.findall(r"https?://\S+", original))
    if not a:
        return 1.0
    b = set(re.findall(r"https?://\S+", candidate))
    return len(a & b) / len(a)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / max(chars_per_token, 1.0)))


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def _detect_positive(d: dict[str, Any] | None) -> bool:
    return bool(d and d.get("available") and d.get("is_watermarked"))


def _quality(original: str, candidate: str, chars_per_token: float) -> dict[str, Any]:
    return {
        "lexical_divergence": round(_lexical_divergence(original, candidate), 4),
        "length_ratio": round(len(candidate) / max(len(original), 1), 4),
        "numbers_preserved": round(_numbers_preserved(original, candidate), 4),
        "urls_preserved": round(_urls_preserved(original, candidate), 4),
        "tokens_in": estimate_tokens(original, chars_per_token),
        "tokens_out": estimate_tokens(candidate, chars_per_token),
    }


def _score_of(d: dict[str, Any] | None) -> float | None:
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
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 200:
                self._stderr_tail.pop(0)

    def _read_line(self, timeout: float) -> dict[str, Any] | None:
        q: queue.Queue[str] = queue.Queue()

        def _reader() -> None:
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
        resp = self._request({"op": "detect", "id": 0, "text": text})
        return {
            "available": True,
            "is_watermarked": resp["is_watermarked"],
            "score": resp.get("score"),
            "threshold": resp.get("threshold"),
        }

    def close(self) -> None:
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
        self.args = args
        self.upstream = upstream
        self.script = SCRIPTS_DIR / "detect_text_watermark.py"
        self.rewrite_script = SCRIPTS_DIR / "rewrite_text.py"
        self.python = str(_venv_python(upstream) or sys.executable)
        self.variants = parse_variants(args.variants)
        self.corpus = load_corpus(args.corpus, args.docs)
        self.chars_per_token = args.chars_per_token
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
        if self.worker is not None:
            with contextlib.suppress(Exception):
                self.worker.close()
        self.worker = None

    def close_worker(self) -> None:
        self._drop_worker()

    # -- step wrappers (monkeypatchable in tests) --------------------------

    def watermark_sample(self, prompt_path: Path, seed: int, out_dir: Path) -> dict[str, Any]:
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
        self, text: str, strength: str, candidates: int, max_loops: int = 1
    ) -> tuple[str, dict[str, Any]]:
        a = self.args
        return run_rewrite(
            self.python,
            self.rewrite_script,
            self.upstream,
            text,
            backend=a.rewrite_backend,
            model=a.rewrite_model,
            base_url=a.rewrite_base_url,
            strength=strength,
            candidates=candidates,
            max_loops=max_loops,
            temperature=a.rewrite_temperature,
            timeout=REWRITE_TIMEOUT,
            allow_remote=a.rewrite_allow_remote,
            api_key=a.rewrite_api_key,
            markllm_model=a.markllm_model,
            markllm_timeout=a.markllm_timeout,
            markllm_scheme=self.scheme,
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
            for strength, candidates in self.variants:
                variant = f"rewrite-{strength}:{candidates}"
                started = time.monotonic()
                try:
                    out_text, stats = self.rewrite(wm_text, strength, candidates)
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
                rows.append(row)

            # Optional re-stamp control: rewrite the UNwatermarked text; a
            # positive after-detection means the backend re-stamped it (or the
            # detector false-positives post-rewrite).
            if self.args.restamp_control:
                for strength, candidates in self.variants:
                    variant = f"restamp-{strength}:{candidates}"
                    try:
                        out_text, _stats = self.rewrite(
                            sample["unwatermarked"], strength, candidates
                        )
                    except RuntimeError as e:
                        rows.append(
                            {
                                **base,
                                "variant": variant,
                                "kind": "restamp",
                                "cleared": None,
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
                            "cleared": None,
                            "quality": _quality(
                                sample["unwatermarked"], out_text, self.chars_per_token
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
            "quality": _quality(original, candidate, self.chars_per_token),
            "seconds": seconds,
            "usd": 0.0,
            "notes": [],
        }
        if kind == "control":
            row["notes"].append("no removal applied (baseline)")
        elif kind == "layer-a":
            row["notes"].append("Layer A only; statistical marks are expected to survive")
        return row


# ---------------------------------------------------------------------------
# Aggregation and outputs
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(rows: list[dict[str, Any]], variants: list[tuple[str, int]]) -> dict[str, Any]:
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
        after_pos = sum(1 for r in group if r.get("after_pos"))
        clear_rate = cleared / before_pos if before_pos else None
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
        entries: dict[str, Any] = {
            "n": len(group),
            "before_positive": before_pos,
            "after_positive": after_pos,
            "cleared": cleared,
            "clear_rate": round(clear_rate, 4) if clear_rate is not None else None,
            "mean_score_before": round(_mean(scores_before), 4) if scores_before else None,
            "mean_score_after": round(_mean(scores_after), 4) if scores_after else None,
            "mean_score_delta": round(_mean(deltas), 4) if deltas else None,
            "median_score_delta": round(sorted(deltas)[len(deltas) // 2], 4) if deltas else None,
            "mean_lexical_divergence": round(_mean([q["lexical_divergence"] for q in quals]), 4)
            if quals
            else None,
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


def _fmt(value: Any, default: str = "—") -> str:
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
) -> str:
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
        "rewrite-<strength>:<candidates> (Layer B rewrite), optional restamp-* "
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
    L.append("## Results (per variant)")
    L.append("")
    L.append(
        "| Variant | n | clear % | Δscore μ | lex div | len ratio | nums keep | tok out | att | s/doc | clears/MTok |"
    )
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant, a in agg.items():
        L.append(
            "| {v} | {n} | {cr} | {d} | {ld} | {lr} | {np} | {to} | {att} | {s} | {eff} |".format(
                v=variant,
                n=a["n"],
                cr=_fmt(a["clear_rate"]),
                d=_fmt(a["mean_score_delta"]),
                ld=_fmt(a["mean_lexical_divergence"]),
                lr=_fmt(a["mean_length_ratio"]),
                np=_fmt(a["mean_numbers_preserved"]),
                to=_fmt(a["mean_tokens_out"]),
                att=_fmt(a.get("mean_attempts")),
                s=_fmt(a["mean_seconds"]),
                eff=_fmt(a["clears_per_mtok_out"]),
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


def build_parser() -> argparse.ArgumentParser:
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
        help="Comma list of <strength>:<candidates> (default: paraphrase:3). "
        "candidates = max rewrite attempts per input; the Layer B loop stops "
        "early when an attempt passes evaluation.",
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
        "--chars-per-token", type=float, default=4.0, help="Cost token estimate (default: 4.0)"
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
    return p


def main() -> int:
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

    bench = Benchmark(args, upstream)
    if not bench.corpus:
        eprint("error: empty corpus")
        return 2

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
        "command": " ".join(
            [
                "python3 service/scripts/bench_synthid_text.py",
                f"--markllm-dir {args.markllm_dir}",
                f"--scheme {args.scheme}",
                *([f"--config {args.config}"] if args.config else []),
                f"--corpus {args.corpus}",
                f"--docs {args.docs} --seeds {args.seeds} --seed-base {args.seed_base}",
                f"--max-new-tokens {args.max_new_tokens}",
                f"--variants {args.variants}",
                f"--rewrite-backend {args.rewrite_backend}",
                f"--rewrite-model {args.rewrite_model}",
                f"--rewrite-base-url {args.rewrite_base_url}",
                f"--rewrite-temperature {args.rewrite_temperature}",
                f"--rewrite-loops {args.rewrite_loops}",
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
        rows = bench.run_variants(samples, workdir)
    finally:
        bench.close_worker()

    # Attach USD cost using per-doc token estimates.
    for row in rows:
        q = row.get("quality") or {}
        if q:
            row["usd"] = (
                q.get("tokens_in", 0) / 1e6 * args.cost_per_mtok_in
                + q.get("tokens_out", 0) / 1e6 * args.cost_per_mtok_out
            )

    agg = aggregate(rows, bench.variants)

    report = render_markdown(config, samples, rows, agg)
    csv_lines = [
        "doc,seed,variant,kind,attempts,evaluator,passed,before_pos,after_pos,cleared,"
        "score_before,score_after,score_delta,lexical_divergence,length_ratio,"
        "numbers_preserved,urls_preserved,tokens_in,tokens_out,seconds,usd,notes"
    ]
    for r in rows:
        q = r.get("quality") or {}
        delta = (
            round(r["score_before"] - r["score_after"], 4)
            if r.get("score_before") is not None and r.get("score_after") is not None
            else ""
        )
        csv_lines.append(
            ",".join(
                str(v)
                for v in (
                    r["doc"],
                    r["seed"],
                    r["variant"],
                    r.get("kind", ""),
                    r.get("attempts", ""),
                    r.get("evaluator", ""),
                    "" if r.get("passed") is None else (1 if r["passed"] else 0),
                    1 if r.get("before_pos") else 0,
                    1 if r.get("after_pos") else 0,
                    "" if r.get("cleared") is None else (1 if r["cleared"] else 0),
                    r.get("score_before", ""),
                    r.get("score_after", ""),
                    delta,
                    q.get("lexical_divergence", ""),
                    q.get("length_ratio", ""),
                    q.get("numbers_preserved", ""),
                    q.get("urls_preserved", ""),
                    q.get("tokens_in", ""),
                    q.get("tokens_out", ""),
                    r.get("seconds", ""),
                    round(r.get("usd") or 0.0, 6),
                    "; ".join(str(n) for n in r.get("notes") or []),
                )
            )
        )

    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"meta": config, "samples": samples, "rows": rows, "aggregates": agg}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    eprint("")
    eprint(f"results written to {out_dir}/")
    print("")
    print("variant          n   clear%  dScore  lexDiv  lenR  nums  tokOut  att  s/doc  eff/MTok")
    print("-" * 82)
    for variant, a in agg.items():
        print(
            f"{variant:<16} {a['n']:>3}  {_fmt(a['clear_rate']):>6}  "
            f"{_fmt(a['mean_score_delta']):>6}  {_fmt(a['mean_lexical_divergence']):>6}  "
            f"{_fmt(a['mean_length_ratio']):>5}  {_fmt(a['mean_numbers_preserved']):>5}  "
            f"{_fmt(a['mean_tokens_out']):>6}  {_fmt(a.get('mean_attempts')):>4}  "
            f"{_fmt(a['mean_seconds']):>5}  {_fmt(a['clears_per_mtok_out']):>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
