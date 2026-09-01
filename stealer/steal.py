#!/usr/bin/env python3
"""Black-box watermark stealing: query a model, build s*, and score text.

Subcommands:
  query   send the downloaded prompt corpus to a model, collecting replies
  build   derive the stolen scorer s*(token | context) from replies (+ baseline)
  detect  score a candidate text with an already-built s*
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import scorer as scorer_mod
from tokens import count_ngrams, default_tokenize

DRY_REPLY_HEAD = "This is a dry-run reply to the prompt: "


def read_corpus(path: Path, field: str = "text") -> list[str]:
    """Read a JSONL corpus (one object per line) or a plain newline-delimited file."""

    if not path.exists():
        raise SystemExit(f"not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            try:
                items.append(json.loads(stripped).get(field))
            except json.JSONDecodeError:
                items.append(stripped)
        else:
            items.append(stripped)
    return [it for it in items if isinstance(it, str) and it]


def read_replies(path: Path) -> list[str]:
    """Read a JSONL reply corpus (the ``reply`` field of each row)."""

    return read_corpus(path, field="reply")


def write_replies(path: Path, rows: list[dict]):
    """Append prompt/reply rows to a JSONL file."""

    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dry_reply(prompt: str) -> str:
    """Deterministic placeholder reply so the pipeline runs without a model."""

    return DRY_REPLY_HEAD + prompt[:200]


def _openai_reply(prompt: str, base_url: str, api_key: str, model: str, max_new_tokens: int) -> str:
    """Send one prompt to an OpenAI-compatible chat endpoint and return the text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 1.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "watermarks-remover/stealer",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def cmd_query(args) -> int:
    """Query the configured target (or baseline) model and collect replies."""

    prompts = read_corpus(Path(args.prompts))
    if args.backend == "dry-run":
        reply_fn = _dry_reply
    else:
        base_url = os.environ.get("WATERMARKS_STEAL_BASE_URL", args.base_url) or args.base_url
        api_key = os.environ.get("WATERMARKS_STEAL_API_KEY")
        model = os.environ.get("WATERMARKS_STEAL_MODEL", args.model) or args.model
        allow_remote = (
            os.environ.get("WATERMARKS_STEAL_ALLOW_REMOTE", "0") == "1" or args.allow_remote
        )
        host = urllib.parse.urlparse(base_url).hostname or "127.0.0.1"
        if host not in ("127.0.0.1", "localhost", "::1") and not allow_remote:
            raise SystemExit(
                "refusing to send content to a non-loopback endpoint; "
                "set --allow-remote or WATERMARKS_STEAL_ALLOW_REMOTE=1"
            )
        if not api_key:
            raise SystemExit("missing API key; set WATERMARKS_STEAL_API_KEY")
        if not model:
            raise SystemExit("missing model; set WATERMARKS_STEAL_MODEL or --model")
        reply_fn = lambda p: _openai_reply(p, base_url, api_key, model, args.max_new_tokens)  # noqa: E731

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    concurrency = max(1, args.concurrency)
    print(f"querying {len(prompts)} prompts -> {out} (backend={args.backend})")

    if concurrency == 1:
        for i, prompt in enumerate(prompts, 1):
            reply = reply_fn(prompt)
            write_replies(out, [{"prompt": prompt, "reply": reply}])
            if i % max(1, len(prompts) // 10) == 0 or i == len(prompts):
                print(f"  {i}/{len(prompts)}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = pool.map(reply_fn, prompts)
            for i, reply in enumerate(results, 1):
                write_replies(out, [{"prompt": prompts[i - 1], "reply": reply}])
                if i % max(1, len(prompts) // 10) == 0 or i == len(prompts):
                    print(f"  {i}/{len(prompts)}")
    print(f"done: {out}")
    return 0


def cmd_build(args) -> int:
    """Derive the stolen scorer ``s*`` from replies and an optional baseline."""

    replies = read_replies(Path(args.replies))
    baseline = read_replies(Path(args.baseline)) if args.baseline else []
    if not replies:
        raise SystemExit(f"no watermarked replies read from {args.replies}")
    print(f"building s* from {len(replies)} watermarked replies, {len(baseline)} baseline replies")

    wm = count_ngrams(replies, args.ctx, default_tokenize)
    if baseline:
        base = count_ngrams(baseline, args.ctx, default_tokenize)
    else:
        base = ({}, {}, {})
        print("  note: no baseline supplied -> unigram fallback only", file=sys.stderr)

    built = scorer_mod.build_scorer(
        wm, base, args.ctx, topk=args.topk, alpha=args.alpha, min_context=args.min_context
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(built, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(built['scorer'])} contexts, top-{args.topk})")
    return 0


def cmd_detect(args) -> int:
    """Score a candidate text against a saved ``s*`` table."""

    scorer = scorer_mod.load_scorer(Path(args.s_star))
    ctx = int(args.ctx) if args.ctx else int(scorer.get("config", {}).get("context_len") or 8)
    text = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")
    tokens = default_tokenize(text)
    result = scorer_mod.score_sequence(scorer, tokens, ctx)
    result["tokens"] = len(tokens)
    result["mean"] = round(result["score"] / result["applied"], 5) if result["applied"] else 0.0
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    """CLI entry point: dispatch query / build / detect subcommands."""

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="send prompts to a model and collect replies")
    q.add_argument("--prompts", required=True, help="prompt corpus (JSONL or plain lines)")
    q.add_argument("--out", required=True, help="JSONL of prompt/reply rows")
    q.add_argument("--backend", default="dry-run", choices=["dry-run", "openai-compatible"])
    q.add_argument("--base-url", default="https://api.openai.com/v1")
    q.add_argument("--model", default=None, help="model (env WATERMARKS_STEAL_MODEL preferred)")
    q.add_argument("--max-new-tokens", type=int, default=512)
    q.add_argument("--concurrency", type=int, default=1)
    q.add_argument("--allow-remote", action="store_true", help="allow non-loopback endpoints")
    q.set_defaults(func=cmd_query)

    b = sub.add_parser("build", help="derive s* from replies and a baseline")
    b.add_argument("--replies", required=True, help="watermarked replies (JSONL)")
    b.add_argument("--baseline", default=None, help="non-watermarked baseline replies (JSONL)")
    b.add_argument("--ctx", type=int, default=8, help="context length in tokens")
    b.add_argument("--topk", type=int, default=50, help="keep top-k tokens per context")
    b.add_argument("--alpha", type=float, default=0.4, help="add-alpha smoothing")
    b.add_argument("--min-context", type=int, default=1, help="min occurrences per context")
    b.add_argument("--out", required=True, help="output s*.json")
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("detect", help="score a candidate text with an s* table")
    d.add_argument("--text", default=None, help="text to score")
    d.add_argument("--file", default=None, help="read text from a file")
    d.add_argument("--s-star", required=True, help="s*.json")
    d.add_argument("--ctx", type=int, default=0, help="override context length")
    d.set_defaults(func=cmd_detect)

    args = p.parse_args(argv)
    if args.command == "detect" and args.text is None and args.file is None:
        print("detect requires --text or --file", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error {exc.code}: {exc.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
