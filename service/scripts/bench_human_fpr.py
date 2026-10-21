#!/usr/bin/env python3
"""False-positive rate of AI-text detectors on human-written text.

bench_synthid_text.py measures detectors on machine text only (watermarked vs
unwatermarked generations), so its human-likeness axis has no anchor: a
detector that flags 30% of real human writing as AI makes every "AI" verdict
on a rewrite much weaker evidence. This benchmark supplies that anchor.

It scores every file in a human corpus (default benchmarks/corpus-human/,
built by fetch_human_corpus.py) with one or more HumanLikeness backends and
reports, per backend:
  - false-positive rate (AI-likeness >= threshold), with a Wilson 95% interval
  - the same rate per word-length bin, since short texts are where detectors
    are least reliable
  - how many texts were too short to score at all (stylometry < 30 words)
  - the worst offenders, for manual inspection

A backend that fails to load or run degrades to stylometry inside
HumanLikeness. Those fallback scores would only duplicate the stylometry row,
so the backend is reported as unavailable (with the reason) and not scored.

Exit codes:
  0  report written
  2  usage error or empty corpus
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bench_synthid_text import HumanLikeness  # noqa: E402
from common import eprint  # noqa: E402
from score_stylometry import DEFAULT_THRESHOLD as STYLOMETRY_THRESHOLD  # noqa: E402

try:
    DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus-human"
except IndexError:
    DEFAULT_CORPUS = Path(__file__).resolve().parent / "benchmarks" / "corpus-human"

BACKENDS = ("stylometry", "pangram", "lastde", "binoculars")
# AI-likeness at or above which a backend's verdict counts as "AI". Stylometry
# uses its own CLI threshold; probability-style detectors default to 0.5.
DEFAULT_THRESHOLDS = {"stylometry": STYLOMETRY_THRESHOLD}
DEFAULT_BINS = (100, 300)
TOP_N = 5


def load_corpus(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """(file name, text) for every non-empty .txt in path, sorted by name."""
    docs = []
    for f in sorted(path.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            docs.append((f.name, text))
    return docs[:limit] if limit else docs


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def bin_label(words: int, edges: list[int]) -> str:
    """Label of the length bin a word count falls in, e.g. '<100', '100-299', '>=300'."""
    if words < edges[0]:
        return f"<{edges[0]}"
    for lo, hi in pairwise(edges):
        if words < hi:
            return f"{lo}-{hi - 1}"
    return f">={edges[-1]}"


def _rate(scores: list[float], threshold: float) -> dict[str, Any]:
    flagged = sum(1 for s in scores if s >= threshold)
    n = len(scores)
    return {
        "n": n,
        "flagged": flagged,
        "fpr": round(flagged / n, 4) if n else None,
        "ci95": wilson(flagged, n),
    }


def summarize(
    docs: list[tuple[str, str]],
    scores: list[float | None],
    threshold: float,
    edges: list[int],
) -> dict[str, Any]:
    """FPR overall and per length bin for one backend's scores over docs."""
    words = [len(text.split()) for _, text in docs]
    scored = [(d[0], w, s) for d, w, s in zip(docs, words, scores, strict=True) if s is not None]
    out = _rate([s for _, _, s in scored], threshold)
    out["unscored"] = len(docs) - len(scored)
    labels = [f"<{edges[0]}"] + [f"{lo}-{hi - 1}" for lo, hi in pairwise(edges)]
    labels.append(f">={edges[-1]}")
    out["bins"] = {
        label: _rate([s for _, w, s in scored if bin_label(w, edges) == label], threshold)
        for label in labels
    }
    worst = sorted(scored, key=lambda r: r[2], reverse=True)[:TOP_N]
    out["top_false_positives"] = [
        {"file": f, "words": w, "score": s} for f, w, s in worst if s >= threshold
    ]
    return out


def run(args: argparse.Namespace, docs: list[tuple[str, str]]) -> dict[str, Any]:
    """Score docs with each requested backend and build the report."""
    edges = sorted(set(args.bins))
    report: dict[str, Any] = {
        "corpus": str(args.corpus),
        "n_docs": len(docs),
        "bins": edges,
        "backends": [],
        "docs": [{"file": name, "words": len(text.split()), "scores": {}} for name, text in docs],
    }
    for backend in args.backend:
        human = HumanLikeness(
            backend, args.human_detector_dir, pangram_model=args.human_pangram_model
        )
        scores = human.score_many([text for _, text in docs])
        # Checked after scoring: a backend can load fine and still fail mid-run
        # (e.g. HTTP 402 on the bulk job), which also flips it to stylometry.
        if human.backend_used != backend:
            eprint(f"warning: {backend} unavailable ({human.reason()}); skipped")
            report["backends"].append(
                {"requested": backend, "available": False, "reason": human.reason()}
            )
            continue
        threshold = (
            args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS.get(backend, 0.5)
        )
        report["backends"].append(
            {
                "requested": backend,
                "available": True,
                "reason": human.reason(),
                "threshold": threshold,
                **summarize(docs, scores, threshold, edges),
            }
        )
        for row, s in zip(report["docs"], scores, strict=True):
            row["scores"][backend] = s
    return report


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable summary table of the report."""
    lines = [
        f"# Detector false positives on human text ({report['n_docs']} docs)",
        "",
        f"Corpus: `{report['corpus']}`",
        "",
    ]
    for b in report["backends"]:
        name = b["requested"]
        if not b["available"]:
            lines += [f"## {name}: unavailable", "", f"Not scored: {b['reason']}", ""]
            continue
        ci = b["ci95"]
        ci_txt = f" (95% CI {_pct(ci[0])}-{_pct(ci[1])})" if ci else ""
        lines += [
            f"## {name} (threshold {b['threshold']})",
            "",
            f"FPR {_pct(b['fpr'])}{ci_txt}: {b['flagged']}/{b['n']} flagged, "
            f"{b['unscored']} too short to score.",
            "",
            "| words | n | flagged | FPR | 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
        for label, r in b["bins"].items():
            ci = r["ci95"]
            ci_txt = f"{_pct(ci[0])}-{_pct(ci[1])}" if ci else "n/a"
            lines.append(f"| {label} | {r['n']} | {r['flagged']} | {_pct(r['fpr'])} | {ci_txt} |")
        if b["top_false_positives"]:
            lines += ["", "Worst false positives:"]
            lines += [
                f"- `{t['file']}` ({t['words']} words): {t['score']}"
                for t in b["top_false_positives"]
            ]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument(
        "--backend",
        action="append",
        choices=BACKENDS,
        help="detector to evaluate; repeat for several (default: stylometry)",
    )
    p.add_argument(
        "--human-detector-dir", default=None, help="dir with ai_human.py (lastde/binoculars)"
    )
    p.add_argument("--human-pangram-model", default=None)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"AI-likeness cutoff for every backend (default: {STYLOMETRY_THRESHOLD} "
        "for stylometry, 0.5 otherwise)",
    )
    p.add_argument(
        "--bins",
        default=",".join(map(str, DEFAULT_BINS)),
        help="comma-separated word-count bin edges (default: %(default)s)",
    )
    p.add_argument("--docs", type=int, default=None, help="score at most N docs")
    p.add_argument("--out-dir", type=Path, default=None, help="write results.json + report.md here")
    args = p.parse_args()
    args.backend = args.backend or ["stylometry"]
    try:
        args.bins = [int(x) for x in args.bins.split(",") if x.strip()]
    except ValueError:
        eprint(f"error: bad --bins {args.bins!r}")
        return 2
    if not args.bins or min(args.bins) < 1:
        eprint("error: --bins must be positive integers")
        return 2
    if not args.corpus.is_dir():
        eprint(f"error: corpus not found: {args.corpus} (run fetch_human_corpus.py first)")
        return 2
    docs = load_corpus(args.corpus, args.docs)
    if not docs:
        eprint(f"error: no .txt files in {args.corpus}")
        return 2

    report = run(args, docs)
    md = render_markdown(report)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.out_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
