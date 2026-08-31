"""Build and apply the stolen scorer ``s*(token | context)``.

`s*` estimates how much a token is boosted after a short context relative to a
non-watermarked baseline.  High positive values mean "likely green" — the
watermark favored that token.  A scrubber demotes green tokens (``apply_delta``
with positive ``delta`` subtracts ``delta * s*`` from their logits); spoofing
flips the sign.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _prob(count: int, total: int, vocab: int, alpha: float) -> float:
    """Add-alpha smoothed probability, falling back to a tiny floor."""

    return (count + alpha) / max(total + alpha * max(vocab, 1), 1.0)


def context_key(context) -> str:
    """Collision-free key for a context token tuple.

    A JSON array keeps token boundaries unambiguous even when a tokenizer emits
    whitespace-containing tokens: ``("a b", "c")`` and ``("a", "b c")`` no longer
    serialize to the same key.
    """

    return json.dumps(list(context), ensure_ascii=False, separators=(",", ":"))


def build_scorer(
    wm,
    base,
    context_len: int,
    topk: int = 50,
    alpha: float = 0.4,
    min_context: int = 1,
) -> dict:
    """Derive ``s*`` from watermarked and baseline (context, next-token) counts.

    ``wm`` and ``base`` are the ``(context_map, context_totals, unigrams)``
    tuples returned by :func:`stealer.tokens.count_ngrams`.  For each context
    seen in the watermarked corpus, the log-ratio
    ``log((p_wm(t|ctx) + eps) / (p_base(t|ctx) + eps))`` is computed per token;
    the top ``topk`` highest-ratio tokens are kept, sorted descending.

    The baseline's per-context distribution is used when that context was
    observed; otherwise the baseline unigram distribution is the fallback, so a
    baseline model different from the watermarked model still works.
    """

    wm_map, wm_totals, wm_unis = wm
    base_map, base_totals, base_unis = base
    wm_vocab = max(len(wm_unis), 1)
    base_vocab = max(len(base_unis), 1)
    base_uni_total = sum(base_unis.values()) or 1
    eps = 1e-6

    scorer: dict[str, list[dict[str, float]]] = {}
    for context, bucket in wm_map.items():
        if wm_totals.get(context, 0) < min_context:
            continue
        context_total = wm_totals[context]
        base_total = base_totals.get(context, 0)
        base_bucket = base_map.get(context)
        scored: list[tuple[str, float]] = []
        for tok, cnt in bucket.items():
            p_wm = _prob(cnt, context_total, wm_vocab, alpha)
            if base_bucket is not None:
                p_base = _prob(base_bucket.get(tok, 0), base_total, base_vocab, alpha)
            else:
                p_base = _prob(base_unis.get(tok, 0), base_uni_total, base_vocab, alpha)
            scored.append((tok, math.log((p_wm + eps) / (p_base + eps))))
        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:topk]
        if top:
            scorer[context_key(context)] = [
                {"token": tok, "score": round(score, 5)} for tok, score in top
            ]

    return {
        "config": {
            "context_len": context_len,
            "topk": topk,
            "alpha": alpha,
            "min_context": min_context,
            "baseline_fallback": "unigram",
        },
        "scorer": scorer,
    }


def load_scorer(path) -> dict:
    """Load a saved ``s*`` JSON table from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_sequence(scorer: dict, tokens: list[str], context_len: int) -> dict:
    """Aggregate ``s*`` over a token sequence (a candidate text).

    Returns ``{"score", "applied"}``: the summed ``s*`` for every (context,
    token) pair found in the table and how many lookups hit.  A higher mean
    means the text leans toward tokens the watermark boosts.
    """

    table = scorer.get("scorer", {})
    total = 0.0
    applied = 0
    for i in range(context_len, len(tokens)):
        context = context_key(tokens[i - context_len : i])
        tok = tokens[i]
        entry = table.get(context)
        if not entry:
            continue
        for item in entry:
            if item["token"] == tok:
                total += item["score"]
                applied += 1
                break
    return {"score": round(total, 5), "applied": applied}


def apply_delta(
    scorer: dict, context: tuple[str, ...], logits: dict[str, float], delta: float
) -> dict[str, float]:
    """Return ``logits`` with ``delta * s*`` subtracted for green tokens.

    ``delta > 0`` demotes the tokens the stolen scorer thinks are green
    (scrubbing); ``delta < 0`` promotes them (spoofing).  Only tokens present
    in the context's stored top-k have a nonzero ``s*``; the rest are unchanged.
    """

    entry = scorer.get("scorer", {}).get(context_key(context), [])
    by_token = {item["token"]: item["score"] for item in entry}
    adjusted = dict(logits)
    for tok in adjusted:
        score = by_token.get(tok)
        if score is not None:
            adjusted[tok] = adjusted[tok] - delta * score
    return adjusted
