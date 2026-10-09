#!/usr/bin/env python3
"""Exercise the text watermark sidecar and DeepSeek rewrite path, without scoring.

Uses only the stdlib and existing clients. See docs/deepseek-benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import rewrite_text
from text_watermark import parse_watermark_options

DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "benchmarks/providers/deepseek.json"


def load_fixture(path: Path) -> dict:
    """Read the small, provider-labelled input corpus before making any requests."""
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict) or fixture.get("provider") != "deepseek":
        raise ValueError("fixture must identify provider 'deepseek'")
    if not isinstance(fixture.get("model"), str) or not fixture["model"].strip():
        raise ValueError("fixture must include a model")
    samples = fixture.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 50:
        raise ValueError("fixture needs 1 to 50 samples")
    ids = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("sample must be an object")
        for field in ("id", "prompt"):
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"sample needs a non-empty {field}")
        if sample["id"] in ids:
            raise ValueError("sample IDs must be unique")
        ids.add(sample["id"])
    fixture["watermark_options"] = parse_watermark_options(fixture.get("watermark_options"))
    return fixture


def check_url(url: str, allow_remote: bool) -> None:
    """Keep credentials out of URLs and require explicit permission for remote IO."""
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("use an http(s) base URL without credentials, query, or fragment")
    rewrite_text._check_remote(url, allow_remote)


def failure(exc: Exception) -> dict:
    """Report failure categories, never remote error bodies or credential-bearing messages."""
    result = {"ok": False, "error": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        result["http_status"] = exc.code
    return result


def run(args: argparse.Namespace, fixture: dict) -> dict:
    """Generate one batch, then rewrite each successful sample once."""
    check_url(args.core_url, args.allow_remote)
    check_url(args.rewrite_base_url, args.allow_remote)
    api_key = os.environ.get("WATERMARKS_REWRITE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("set WATERMARKS_REWRITE_API_KEY")
    model = args.model or fixture["model"]
    if not model.strip():
        raise ValueError("model must not be empty")
    samples = fixture["samples"]
    report = {
        "provider": fixture["provider"],
        "purpose": "sidecar-rewrite-smoke",
        "started_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 — Python 3.10
        "requested_model": model,
        "rewrite_base_url": args.rewrite_base_url,
        "core_url": args.core_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": 0.9,
        "timeout_seconds": args.timeout,
        "watermark_options": fixture["watermark_options"],
        "watermark_verification": "not_performed",
        "results": [],
    }
    started = time.monotonic()
    core_key = os.environ.get("WATERMARKS_SERVER_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {core_key}"} if core_key else {}
    try:
        batch = rewrite_text._http_json(
            args.core_url.rstrip("/") + "/watermark/batch",
            {
                "files": [
                    {
                        "name": sample["id"],
                        "text": sample["prompt"],
                        "options": fixture["watermark_options"],
                    }
                    for sample in samples
                ]
            },
            headers,
            args.timeout,
        )
        if not isinstance(batch, dict) or batch.get("ok") is not True:
            raise ValueError("invalid batch response")
        entries = batch.get("results")
        if not isinstance(entries, list):
            raise ValueError("invalid batch response")
        if len(entries) != len(samples) or any(
            not isinstance(entry, dict) or entry.get("name") != sample["id"]
            for sample, entry in zip(samples, entries, strict=True)
        ):
            raise ValueError("batch response does not match sample IDs")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        report.update(failure(exc), stage="watermark")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report
    report["generation_seconds"] = round(time.monotonic() - started, 3)
    for sample, entry in zip(samples, entries, strict=True):
        row = {"id": sample["id"], "ok": False, "stage": "watermark"}
        text = entry.get("watermarked_text")
        if entry.get("ok") is not True or not isinstance(text, str) or not text.strip():
            row["error"] = "generation_failed_or_empty"
        else:
            row.update(stage="rewrite", watermarked_text=text)
            if isinstance(entry.get("report"), dict):
                row["generation_report"] = {
                    key: entry["report"][key]
                    for key in ("scheme_used", "model", "keys_used")
                    if key in entry["report"]
                }
            rewrite_started = time.monotonic()
            try:
                rewritten = rewrite_text.call_openai_compatible(
                    args.rewrite_base_url,
                    model,
                    rewrite_text.build_prompt("paraphrase", text, lang="en", original_lang="en"),
                    api_key,
                    args.timeout,
                    0.9,
                    None if args.reasoning_effort == "off" else args.reasoning_effort,
                )
                if not rewritten.strip():
                    raise ValueError("empty rewrite")
                row.update(ok=True, rewritten_text=rewritten, changed=rewritten != text)
            except (
                OSError,
                ValueError,
                TypeError,
                LookupError,
                AttributeError,
                RuntimeError,
            ) as exc:
                row.update(failure(exc))
            row["rewrite_seconds"] = round(time.monotonic() - rewrite_started, 3)
        report["results"].append(row)
    report["ok"] = all(row["ok"] for row in report["results"])
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run an explicitly enabled smoke test and write a credential-free JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--core-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--rewrite-base-url",
        default=os.environ.get("WATERMARKS_REWRITE_BASE_URL", "https://api.deepseek.com"),
        help="API origin without /v1; the existing rewrite client appends /v1/chat/completions",
    )
    parser.add_argument("--model", default=os.environ.get("WATERMARKS_REWRITE_MODEL"))
    parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "off"), default="none"
    )
    parser.add_argument(
        "--allow-remote", action="store_true", help="allow sending samples to remote endpoints"
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        report = run(args, load_fixture(args.fixture))
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        # Remote services may echo headers into response text; never persist credentials.
        for name in ("WATERMARKS_REWRITE_API_KEY", "WATERMARKS_SERVER_API_KEY"):
            secret = os.environ.get(name, "").strip()
            if secret:
                serialized = serialized.replace(json.dumps(secret)[1:-1], "[REDACTED]")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        print(f"smoke setup/output failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"smoke {'passed' if report['ok'] else 'failed'}; watermark verification not performed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
