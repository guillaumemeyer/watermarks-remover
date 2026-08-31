#!/usr/bin/env python3
"""Download a prompt corpus from a Hugging Face dataset via datasets-server.

Default: 30k C4 RealNewsLike passages into ./stealer/prompts/prompts.jsonl.

Stdlib only.  Uses the datasets-server ``rows`` API so it never depends on the
``datasets`` / ``pyarrow`` packages.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"
DEFAULT_COUNT = 30_000
DEFAULT_CONFIG = "realnewslike"
DEFAULT_DATASET = "allenai/c4"


def fetch_rows(base, dataset, config, split, offset, length, token=None):
    """Return one page of rows as parsed JSON.

    ``token`` (an ``hf_`` value from ``HF_TOKEN``) is sent as a bearer header; it
    lifts the unauthenticated datasets-server rate limit.  The token is only
    ever read from the environment and never printed.
    """

    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{base}/rows?{query}"
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"refusing non-http(s) datasets-server URL: {url!r}")
    headers = {"User-Agent": "watermarks-remover/stealer"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
        return json.load(resp)


def _retry_after(header):
    """Parse a Retry-After header into seconds (capped), or 0 if absent/invalid."""

    if not header:
        return 0.0
    try:
        return max(1.0, min(float(header), 120.0))
    except (TypeError, ValueError):
        return 0.0


def fetch_rows_retrying(
    base,
    dataset,
    config,
    split,
    offset,
    length,
    token=None,
    attempts=8,
    base_delay=2.0,
    max_delay=90.0,
):
    """Fetch rows, backoff on transient 5xx / 429 with jittered exponential delay.

    Auth failures (400/401/403/404) are fatal; rate limits and 5xx wait out the
    window (honoring ``Retry-After`` when present) and retry.
    """

    last = None
    for attempt in range(attempts):
        try:
            return fetch_rows(base, dataset, config, split, offset, length, token=token)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 401, 403, 404):
                raise
            if attempt + 1 >= attempts:
                break
            delay = _retry_after(exc.headers.get("Retry-After")) if exc.headers else 0.0
            if delay <= 0:
                delay = min(max_delay, base_delay * (2.0**attempt) + random.uniform(0, 0.5))  # noqa: S311
            print(f"  {exc.code}; retry in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if attempt + 1 >= attempts:
                break
            delay = min(max_delay, base_delay * (2.0**attempt) + random.uniform(0, 0.5))  # noqa: S311
            print(f"  {exc}; retry in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"could not fetch rows at offset {offset}: {last}")


def load_state(state_path):
    """Return the persisted resume cursor ``(next_offset, written)``, or ``(0, 0)``."""

    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return int(data.get("next_offset", 0)), int(data.get("written", 0))
        except (ValueError, json.JSONDecodeError):
            pass
    return 0, 0


def save_state(state_path, next_offset, written):
    """Persist the resume cursor as JSON."""

    state_path.write_text(
        json.dumps({"next_offset": next_offset, "written": written}), encoding="utf-8"
    )


def main(argv=None) -> int:
    """Fetch pages and append prompts until ``--count`` is reached, checkpointing per page."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="dataset config (subset)")
    p.add_argument("--split", default="train", help="split to read")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT, help="number of passages to keep")
    p.add_argument("--field", default="text", help="row field used as the prompt")
    p.add_argument("--out", default=None, help="output directory (default: ./stealer/prompts)")
    p.add_argument("--base-url", default=DATASETS_SERVER_BASE, help="datasets-server base URL")
    p.add_argument("--delay", type=float, default=1.5, help="seconds between page requests")
    p.add_argument("--min-chars", type=int, default=0, help="drop prompts shorter than this")
    p.add_argument(
        "--max-chars", type=int, default=0, help="truncate prompts longer than this (0 = keep)"
    )
    p.add_argument("--offset", type=int, default=0, help="row offset to start from")
    p.add_argument("--start-over", action="store_true", help="ignore any resume state")
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "prompts.jsonl"
    state_path = out_dir / ".download-state.json"

    token = os.environ.get("HF_TOKEN")

    offset, written = (0, 0) if args.start_over else load_state(state_path)
    if args.offset:
        offset = args.offset
    if args.start_over and prompts_path.exists():
        prompts_path.unlink()

    first = fetch_rows_retrying(
        args.base_url, args.dataset, args.config, args.split, offset, 1, token=token
    )
    page_size = int(first.get("num_rows_per_page", 100))
    page_size = max(1, min(page_size, 100))

    if written == 0 and not prompts_path.exists():
        prompts_path.write_text("", encoding="utf-8")

    print(
        f"downloading {args.dataset}:{args.config}:{args.split} "
        f"({args.count} prompts) -> {prompts_path}"
    )

    committed_offset, committed_written = offset, written
    committed_bytes = 0
    try:
        with prompts_path.open("a", encoding="utf-8") as fh:
            committed_bytes = fh.tell()
            while written < args.count:
                rows = fetch_rows_retrying(
                    args.base_url,
                    args.dataset,
                    args.config,
                    args.split,
                    offset,
                    page_size,
                    token=token,
                )
                items = rows.get("rows", [])
                if not items:
                    print("  no more rows", file=sys.stderr)
                    break
                batch: list[str] = []
                for item in items:
                    if written >= args.count:
                        break
                    text = (item.get("row") or {}).get(args.field)
                    if not isinstance(text, str):
                        continue
                    if args.min_chars and len(text) < args.min_chars:
                        continue
                    if args.max_chars and len(text) > args.max_chars:
                        text = text[: args.max_chars]
                    batch.append(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                for line in batch:
                    fh.write(line)
                fh.flush()
                offset += len(items)
                save_state(state_path, offset, written)
                committed_offset, committed_written, committed_bytes = offset, written, fh.tell()
                if written % 1000 < page_size or written >= args.count:
                    print(f"  {written}/{args.count} prompts")
                time.sleep(args.delay)
    except KeyboardInterrupt:
        # Roll back any page written this round so a resume cannot duplicate rows
        # while the persisted cursor still points at the previous page boundary.
        if committed_bytes:
            try:
                with prompts_path.open("r+b") as rf:
                    rf.truncate(committed_bytes)
            except OSError:
                pass
        offset, written = committed_offset, committed_written
        save_state(state_path, offset, written)
        print(
            "\ninterrupted; rolled back to the last complete page; resume with --start-over off",
            file=sys.stderr,
        )
        return 130

    if written >= args.count:
        # Completing the run makes the resume offset meaningless; reset it.
        save_state(state_path, offset, written)
        print(f"done: {written} prompts in {prompts_path}")
        return 0
    print(f"stopped early: {written} prompts (dataset exhausted?)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
