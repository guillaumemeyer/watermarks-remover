"""Tokenization and (context, next-token) counting for watermark stealing."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def default_tokenize(text: str) -> list[str]:
    """Split into lowercase word / punctuation tokens (stdlib fallback).

    A real black-box attack uses the target model's tokenizer; the scorer's
    ``--ctx`` and lookups are keyed on whatever tokenizer produced the counts,
    so a caller can substitute its own tokenizer for both build and detect.
    """

    return [tok.lower() for tok in _WORD_RE.findall(text)]


def count_ngrams(
    texts: Iterable[str],
    context_len: int,
    tokenize: Callable[[str], list[str]] = default_tokenize,
) -> tuple[dict[tuple[str, ...], dict[str, int]], Counter, Counter]:
    """Count how often each next token follows each short context.

    Returns ``(context_map, context_totals, unigram_counts)`` where
    ``context_map[ctx][tok]`` is the number of times ``tok`` directly follows the
    token tuple ``ctx``, ``context_totals[ctx]`` counts every (ctx, next) pair,
    and ``unigram_counts[tok]`` is the raw token frequency.  Contexts shorter
    than ``context_len`` tokens (start of a reply) are skipped.
    """

    context_map: dict[tuple[str, ...], dict[str, int]] = {}
    context_totals: Counter = Counter()
    unigrams: Counter = Counter()
    for text in texts:
        toks = tokenize(text)
        for tok in toks:
            unigrams[tok] += 1
        for i in range(context_len, len(toks)):
            context = tuple(toks[i - context_len : i])
            tok = toks[i]
            bucket = context_map.get(context)
            if bucket is None:
                bucket = context_map[context] = {}
            bucket[tok] = bucket.get(tok, 0) + 1
            context_totals[context] += 1
    return context_map, context_totals, unigrams
