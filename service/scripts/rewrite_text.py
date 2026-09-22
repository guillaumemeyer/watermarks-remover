#!/usr/bin/env python3
"""Layer B optional rewrite hook for statistical (token-sampling) watermarks.

Backends:
  print-prompt       — emit prompt only (default; CI-safe, no model)
  ollama             — POST to Ollama /api/chat
  openai-compatible  — POST to OpenAI-style /v1/chat/completions

Env (optional):
  WATERMARKS_REWRITE_BACKEND
  WATERMARKS_REWRITE_BASE_URL
  WATERMARKS_REWRITE_MODEL
  WATERMARKS_REWRITE_API_KEY      (env-only; never pass keys on argv)
  WATERMARKS_REWRITE_ALLOW_REMOTE (set to 1 to allow non-loopback endpoints)
  WATERMARKS_REWRITE_REASONING_EFFORT (default none; see --reasoning-effort)
  WATERMARKS_REWRITE_TIMEOUT      (default 120; seconds per backend call)
  WATERMARKS_REWRITE_CANDIDATES   (default 1; variants generated per loop)
  WATERMARKS_REWRITE_LOOPS        (default 1; max evaluation rounds)
  WATERMARKS_PROTECT_LATEX        (auto|on|off; math/LaTeX protection, default auto)

Rewriting is iterative and evaluation-driven: each loop generates
--candidates (default 1) variants, evaluates each, and stops as soon as an
attempt passes watermark detection; --max-loops (default 1) caps how many
evaluation rounds run before the best-effort variant is returned
(WATERMARKS_REWRITE_LOOPS). The evaluator is chosen by priority: keyed-Gumbel
same-key replay (when --gumbel-key / WATERMARKS_GUMBEL_KEY is set), else
MarkLLM same-config detection (--markllm-scheme), else, when no detector is
configured, bigram-Jaccard lexical divergence (no pass/fail verdict — all
attempts are generated and the most diverged one is selected). A vendor-detector
seam (Google's retired SynthID-text detector) is reserved ahead of the
same-config detectors should a vendor endpoint return.

--protect-latex (auto|on|off, default auto) holds mathematics, LaTeX commands and
environments, citation keys, and verbatim/code spans out of the rewrite: they are
swapped for opaque placeholders before generation and restored after the
deterministic passes, so no model reformats an equation or drops a citation key
in the name of better prose. Restoration is reported (protected / restored /
missing / duplicated), never assumed. The print-prompt backend has no output to
restore from, so there the prompt carries a preserve-verbatim instruction instead.

The rewrite instruction comes from --tactic (a named prompt) and, when
--rewrite-level is set, that prompt is further modulated by a numeric rewrite
intensity in (0,1] that controls how many tokens change (0 — the unchanged
original — is excluded; 1 rewrites everything). The level is a request: output
lexical/semantic divergence is measured, not guaranteed. --style appends an
optional writing-style instruction (e.g. "write like Hemingway"), most useful
with --tactic humanize; it is a request, not a guarantee, and never overrides
the fact/voice rules. The humanize tactic additionally runs a deterministic
humanizer pass (humanize_pass.py) over each generated candidate — straight
quotes, no em/en dashes or double hyphens, filler-phrase collapses, and the
utilize->use swap — before evaluation, so the scored text is the text returned.

LLM output is stripped of model wrappers before anything else sees it (a
"Here is the rewritten text ...:" preamble, a trailing "I changed the
following:" / "Changes made:" / "Note:" section, a </think> reasoning block,
a code fence or quotes around the whole output), unless the input itself
carries the same construct. For length-preserving tactics
(LENGTH_PRESERVING_TACTICS; not structural) an attempt whose length drifted to
an extreme versus its input is a failed attempt: it is never selected, and when
every attempt drifted the rewrite fails rather than returning commentary.

Security notes:
  - Only http(s) endpoints are accepted; redirects are refused outright so an
    Authorization header (API key) can never be re-sent to an unvalidated host.
  - Non-loopback endpoints are denied unless WATERMARKS_REWRITE_ALLOW_REMOTE=1
    (or --allow-remote) is set explicitly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from academic_guard import academic_error, structure_error
from common import (
    cleaned_path,
    eprint,
    read_text_input,
    subprocess_creationflags,
    write_text_output,
)
from humanize_pass import humanize_pass
from latex_mask import LatexMask, looks_like_latex, mask_latex, placeholder_error, restore_latex
from text_detectors import GumbelTextDetector, MarkLLMTextDetector
from text_unicode import clean_text

DEFAULT_MARKLLM_MODEL = "facebook/opt-1.3b"
DEFAULT_CANDIDATES = 1
DEFAULT_MAX_LOOPS = 1
DEFAULT_NOOP_LEX_FLOOR = 0.05
# Longest text handed to the model in one strategy-step call; longer inputs are
# split at paragraph / sentence boundaries (WATERMARKS_REWRITE_CHUNK_CHARS).
DEFAULT_CHUNK_CHARS = 2500

PROMPTS = {
    "paraphrase": (
        "Rewrite the following text so that it uses substantially different wording at "
        "the token level. Change clause order, connectors, and transition words; vary "
        "sentence boundaries and length; and replace both content words and function "
        "words where meaning allows. Preserve all facts, numbers, names, and technical "
        "identifiers. Do not add or remove claims. Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "humanize": (
        "Rewrite the following text so it reads as if a human wrote it from scratch. "
        "Vary sentence rhythm and length unevenly — mix short and long sentences instead "
        "of a steady mid-length cadence — and merge or split paragraphs where a human "
        "would. Use plain, concrete wording and simple verbs (is/are/has); prefer active "
        'voice. Cut promotional language and inflated significance ("stands as a '
        'testament", "pivotal", "vibrant", "a rich tapestry"), superficial '
        'present-participle analyses ("reflecting", "showcasing", "underscoring"), '
        'vague attributions ("experts argue"), rule-of-three listing, filler ("in order '
        'to", "it is important to note"), empty hedging that bounds nothing (keep every hedge that limits a claim), and formulaic positive conclusions. '
        'Avoid AI vocabulary ("additionally", "delve", "crucial", "foster", '
        '"leverage", "utilize", "interplay", and abstract "landscape"). Do not add '
        "em dashes, bold text, emojis, or curly quotes. Preserve all facts, numbers, "
        "names, and technical identifiers. Do not add or remove claims. Output only the "
        "rewritten text.\n\n---\n{TEXT}"
    ),
    "academic": (
        "Rewrite the following academic prose so that the wording differs substantially "
        "at the token level while the argument survives intact. Write in the same "
        "language as the original — never translate. Vary connectives, clause order, "
        "and sentence boundaries, and let sentence length move with the subject. "
        "Keep every technical term, term of art, notation, symbol name, unit, and "
        "citation exactly as written: do not paraphrase terminology and never swap a "
        "technical term for an everyday synonym, even when the everyday word reads "
        "more smoothly. Keep the epistemic force of each statement — hedges, "
        "attributions, scope conditions, and stated limitations stay exactly as strong "
        "or as weak as in the original, and a conjecture must not become a result. "
        "Keep passive and impersonal constructions where they are the disciplinary "
        "norm. Do not simplify, summarize, explain, add examples, add transitions that "
        "announce structure, or add a concluding flourish. Preserve all facts, numbers, "
        "names, equations, and technical identifiers. Do not add or remove claims. "
        "Output only the rewritten text.\n\n---\n{TEXT}"
    ),
    "code": (
        "Rewrite the natural-language parts of this code — comments, docstrings, and "
        "string literals — using different wording. Rename local variables, function "
        "parameters, and private helper names to semantically equivalent names. Preserve "
        "program behavior, public API names, and all values that affect output. Output "
        "only the rewritten code.\n\n---\n{TEXT}"
    ),
    "backtranslate_out": (
        "Translate the following text to {LANG}. Output only the translation.\n\n---\n{TEXT}"
    ),
    "backtranslate_back": (
        "Translate the following text to {ORIGINAL_LANG}. Preserve meaning; use natural "
        "phrasing. Output only the translation.\n\n---\n{TEXT}"
    ),
    "structural_outline": (
        "Extract a bullet outline of all claims and structure from the text "
        "(no full sentences). Output only the outline.\n\n---\n{TEXT}"
    ),
    "structural_write": (
        "Write a complete document from this outline in natural, varied human prose. "
        "Avoid formulaic transitions. Do not omit any bullet. Output only the document."
        "\n\n---\n{TEXT}"
    ),
    "level": (
        "Rewrite the following text so that a fraction of the tokens close to "
        "{LEVEL:.2f} changes — 0 would mean the wording is kept unchanged, 1 means "
        "everything is rewritten. At low values keep the sentence structure, word "
        "order, and every token that can stay, changing only function words and a "
        "few non-essential content words. At high values change wording substantially "
        "at the token level. Preserve all facts, numbers, names, and technical "
        "identifiers. Do not add or remove claims. Output only the rewritten text."
        "\n\n---\n{TEXT}"
    ),
    "chunk_unit": (
        "Rewrite only this fragment to change a modest fraction of its tokens. "
        "At low intensity keep the sentence structure, word order, and every "
        "token that can stay, changing only function words and a few "
        "non-essential content words. Preserve all facts, numbers, names, and "
        "technical identifiers. Do not add or remove claims. Output only the "
        "rewritten fragment.\n\n---\n{TEXT}"
    ),
}


def _tokens(text: str) -> list[str]:
    """Extract lowercase alphanumeric/word tokens from text."""
    return re.findall(r"\w+", text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Extract consecutive token pairs as bigrams."""
    return set(itertools.pairwise(tokens))


def _lexical_divergence(original: str, candidate: str) -> float:
    """Bigram Jaccard distance: 0.0 identical, 1.0 fully different."""
    a = _tokens(original)
    b = _tokens(candidate)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    ba = _bigrams(a)
    bb = _bigrams(b)
    union = ba | bb
    if not union:
        return 0.0
    return 1.0 - len(ba & bb) / len(union)


def _below_noop_floor(divergence: float, floor: float) -> bool:
    """True when a rewrite's lexical divergence marks it a no-op (floor <= 0 disables)."""
    return floor > 0 and divergence < floor


# Length-preserving tactics keep every claim and rewrite at the token level, so an
# output far longer or shorter than its input is not a rewrite: it is usually model
# meta-commentary the wrapper stripper did not recognise, or a truncation. Such a
# candidate is rejected. `structural` is exempt because it rebuilds the text from
# an outline, so its length legitimately drifts. `chunk` is checked on the whole
# reassembled candidate (per-fragment ratios on single sentences are too noisy);
# its wrappers are still stripped fragment by fragment.
LENGTH_PRESERVING_TACTICS = frozenset(
    {"paraphrase", "humanize", "academic", "backtranslate", "code", "chunk", "mlm"}
)
LENGTH_DRIFT_MAX_RATIO = 2.0
LENGTH_DRIFT_MIN_RATIO = 0.5
# Absolute slack so short inputs are not flagged for small changes: a verbose
# paraphrase of a one-line sentence can double it ("Hi." -> "Hello there."
# triples), while commentary a model adds runs to hundreds of characters.
LENGTH_DRIFT_SLACK_CHARS = 80
_LENGTH_DRIFT_RANGE = f"{LENGTH_DRIFT_MIN_RATIO:.1f}-{LENGTH_DRIFT_MAX_RATIO:.1f}"


def _length_ratio(in_chars: int, out_chars: int) -> float | None:
    """Output/input character ratio, rounded for reports; None for empty input."""
    return round(out_chars / in_chars, 4) if in_chars else None


def _length_drift(in_chars: int, out_chars: int) -> bool:
    """True when a rewrite's length drifted to an extreme versus its input."""
    if in_chars <= 0:
        return False
    too_long = (
        out_chars > in_chars * LENGTH_DRIFT_MAX_RATIO
        and out_chars - in_chars > LENGTH_DRIFT_SLACK_CHARS
    )
    too_short = (
        out_chars < in_chars * LENGTH_DRIFT_MIN_RATIO
        and in_chars - out_chars > LENGTH_DRIFT_SLACK_CHARS
    )
    return too_long or too_short


def _select_candidate(original: str, candidates: list[str]) -> tuple[str, list[float]]:
    """Pick the most lexically diverged rewrite, skipping extreme length drift.

    A drifted candidate (see _length_drift) wins only when every candidate
    drifted: divergence alone favours a rewrite padded with model commentary.
    """
    scores = [_lexical_divergence(original, cand) for cand in candidates]
    kept = [i for i, cand in enumerate(candidates) if not _length_drift(len(original), len(cand))]
    best_idx = max(kept or range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], scores


# --- Model-wrapper stripping --------------------------------------------------
# Chat models often wrap a rewrite in meta-commentary ("Here is the rewritten
# text ...:", "I changed the following: ...", code fences, quotes) even when the
# prompt says "Output only the rewritten text". strip_model_wrappers() removes
# that wrapping from LLM output. Every rule is skipped when the step input carries
# the same construct, so content that was already there is kept.

# Every kind strip_model_wrappers() can report as removed.
WRAPPER_KINDS = ("think", "preamble", "separator", "trailer", "code_fence", "quotes")

_EMPH = r"[*_]*"  # markdown emphasis around a label ("**Note:**")
_INTERJECTION = (
    r"(?:sure|certainly|of\s+course|okay|ok|absolutely|alright|all\s+right|got\s+it"
    r"|no\s+problem)"
)
_COLON = r"[:\uff1a]"  # ASCII or fullwidth colon
_HERE = (
    rf"{_EMPH}(?:{_INTERJECTION}\s*[,.!]*\s*)?(?:here|below)"
    r"(?:'s|\u2019s|\s+is|\s+are)\b"
)
# Words tying a "Here is ...:" line to the rewrite itself; a bare "Here's what you
# need to know:" is content, not a preamble.
_TASK_RE = re.compile(
    r"\b(?:rewrit\w*|paraphras\w*|revis\w*|reword\w*|rephras\w*|translat\w*|humaniz\w*"
    r"|version|variant|fragment|outline|wording|tokens?)\b"
    r"|\b(?:the|your|this|my)\s+(?:\w+\s+){0,2}(?:text|passage|paragraph|document|code)\b",
    re.IGNORECASE,
)
_INTERJECTION_LINE_RE = re.compile(rf"{_EMPH}{_INTERJECTION}\s*[.!]*{_EMPH}", re.IGNORECASE)
_HERE_LINE_RE = re.compile(rf"{_HERE}[^\n]*{_COLON}{_EMPH}", re.IGNORECASE)
_LABEL_LINE_RE = re.compile(
    rf"{_EMPH}(?:(?:the|my)\s+)?(?:(?:rewritten|revised|paraphrased|humanized|reworded"
    r"|rephrased|translated|back-translated|final|updated|edited|new|improved)\s+)?"
    r"(?:text|version|document|fragment|passage|paragraph|output|code|draft|rewrite"
    r"|translation|paraphrase|result|answer|response)"
    rf"(?:\s*\([^)\n]{{0,30}}\))?\s*{_EMPH}{_COLON}{_EMPH}",
    re.IGNORECASE,
)
_FIRST_PERSON_PREAMBLE_RE = re.compile(
    rf"{_EMPH}i(?:'ve|\u2019ve|\s+have)?\s+(?:rewritten|rewrote|paraphrased|reworded"
    rf"|rephrased|humanized)\b[^\n]*{_COLON}{_EMPH}",
    re.IGNORECASE,
)
# "Here is the rewritten text: The atmosphere was ..." (content on the same line).
_INLINE_PREAMBLE_RE = re.compile(
    rf"{_HERE}([^\n:\uff1a]{{0,200}}){_COLON}[ \t*_]*(?=\S)", re.IGNORECASE
)
_TRAILER_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "changes",
        re.compile(
            rf"{_EMPH}(?:(?:key|main|major|summary\s+of(?:\s+the)?|list\s+of(?:\s+the)?"
            r"|explanation\s+of(?:\s+the)?)\s+)?(?:changes|modifications|edits|revisions)"
            rf"(?:\s+(?:made|applied|include|were\s+made))?\s*{_EMPH}\s*{_COLON}",
            re.IGNORECASE,
        ),
    ),
    (
        # "Here's a breakdown of the changes made to reach the intensity:"
        "changes",
        re.compile(
            rf"{_EMPH}here(?:'s|\u2019s|\s+is|\s+are)\s+(?:a\s+|the\s+)?(?:breakdown|summary"
            r"|list|rundown|recap)\s+of\s+(?:the\s+|my\s+)?(?:changes|modifications|edits"
            rf"|revisions)\b[^\n]*{_COLON}{_EMPH}\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "note",
        re.compile(
            rf"\(?{_EMPH}(?:notes?|n\.\s?b\.|explanation"
            rf"|(?:translator|editor)'?s?\s+notes?)\s*{_EMPH}\s*{_COLON}",
            re.IGNORECASE,
        ),
    ),
    (
        # Only honoured when the input is not first-person prose (see below).
        "first_person",
        re.compile(
            # "I changed the following:", "I replaced:", "In this version, I've
            # changed:" -- an edit verb naming the rewrite, or introducing a list.
            rf"{_EMPH}(?:in\s+(?:this|the|my)\s+(?:\w+\s+)?version,?\s+)?i(?:'ve|\u2019ve"
            r"|\s+have)?\s+(?:also\s+)?(?:changed|made|replaced|rephrased|reworded|rewrote"
            r"|rewritten|modified|altered|swapped|substituted|restructured|adjusted"
            r"|paraphrased|kept|preserved|maintained|retained|aimed|tried|varied|used"
            r"|removed|added|avoided|ensured)\b(?:[^\n]*\b(?:following|changes?|original"
            rf"|rewrit\w*|paraphras\w*|tokens?|token-level|wording|synonyms?|fraction)\b"
            rf"|[^\n]*{_COLON}{_EMPH}\s*$)",
            re.IGNORECASE,
        ),
    ),
    (
        # "This rewrite keeps ...", "In the rewritten version, ...". Only the
        # rewrite's own nouns: "The revised budget ..." is ordinary prose.
        "meta",
        re.compile(
            rf"{_EMPH}(?:in\s+)?(?:this|the|my)\s+(?:rewrite|rewritten\s+(?:text|version"
            r"|sentence|passage|paragraph)|paraphrased?\s+(?:text|version|sentence))\b",
            re.IGNORECASE,
        ),
    ),
    (
        # A sign-off offering more rewrites; "If you need changes to your order,
        # call us" is content, so the offer must be about further changes or
        # another version.
        "closing",
        re.compile(
            rf"{_EMPH}(?:let\s+me\s+know|feel\s+free|if\s+you(?:'d|\s+would)?\s+(?:like"
            r"|want|need)|i\s+hope\s+(?:this|that)|hope\s+this\s+helps)\b[^\n]*\b(?:(?:further"
            r"|other|more|additional)\s+(?:changes|adjustments|tweaks|edits|revisions"
            r"|modifications)|(?:another|a\s+different|an\s+alternative)\s+(?:version|rewrite"
            r"|variant|phrasing))\b",
            re.IGNORECASE,
        ),
    ),
    (
        # A label offering another variant: "Or, in a moderate intensity version:",
        # "Alternatively:", "Version 2:". Must end the line with a colon.
        "alternatives",
        re.compile(
            rf"{_EMPH}(?:(?:or|alternatively)\b[^\n]*|(?:another|an?\s+alternative"
            r"|alternative)\s+(?:version|variant|rewrite|phrasing|option)\b[^\n]*"
            rf"|(?:version|variant|option)\s+\d+){_EMPH}\s*{_COLON}{_EMPH}\s*$",
            re.IGNORECASE,
        ),
    ),
)
# Jargon a model echoes from the rewrite instructions when it comments on its own
# output ("changes approximately 0.32 tokens", "At low intensity:", "Function
# words: ..."). Kept to phrases ordinary prose does not use; a phrase counts only
# when the original never uses it.
_ECHO_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b\d+(?:\.\d+)?\s*%?\s+(?:of\s+(?:the\s+)?)?tokens\b|\btoken[- ]level\b"
        r"|\btokens?\s+changes?\b|\bfraction\s+of\s+(?:the\s+)?tokens\b",
        r"\bclause\s+order\b",
        r"\btransition\s+words?\b",
        r"\bsentence\s+(?:boundar(?:y|ies)|structure)\b",
        r"\bword\s+order\b",
        r"\b(?:function|content)\s+words?\b",
        r"\b(?:rewrite|requested|target|desired|change)\s+intensity\b"
        r"|\bintensity\s+of\s+(?:the\s+)?rewrite\b"
        rf"|^\W*(?:rewrite\s+|change\s+)?intensity(?:\s+level)?\s*{_COLON}\s*\d"
        r"|\b(?:low|moderate|medium|high|full)[- ]intensity\s+(?:version|rewrite|variant)\b"
        rf"|^\W*at\s+(?:a\s+)?(?:low|moderate|medium|high|full)[- ]intensity\s*{_COLON}",
        r"\bmodulation\s+(?:adjustment|level|factor)\b|\bmodulated\s+(?:version|to|at)\b"
        r"|\bmodulat\w*\s+(?:this\s+|the\s+)?rewrite\b",
        r"\b(?:change[sd]?|different)\s+(?:in\s+)?wording\b",
        r"\boriginal\s+(?:text|sentence|wording)\b",
    )
)
# Commonplace intensity wording counts only on a label line ending in a colon
# ("At low intensity, this might become:").
_LABEL_ECHO_RES = (
    *_ECHO_RES,
    re.compile(r"\b(?:low|moderate|medium|high|full)[- ]intensity\b", re.IGNORECASE),
)
# A "Here is ..." first line ending in a period is a preamble only when it names
# the rewrite itself.
_HERE_SENTENCE_RE = re.compile(rf"{_HERE}[^\n]*\.{_EMPH}", re.IGNORECASE)
# "Given the constraints, I'll change function words. Here's the rewritten text:"
_HERE_TAIL_RE = re.compile(rf"[^\n]*[.!?]\s+({_HERE}[^\n]*){_COLON}{_EMPH}", re.IGNORECASE)
_REWRITE_WORD_RE = re.compile(r"\b(?:rewrit|paraphras|reword|rephras)\w*", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_HR_RE = re.compile(r"-{3,}|\*{3,}|_{3,}|={3,}")  # matched against stripped lines
_FENCE_OPEN_RE = re.compile(r"(`{3,}|~{3,})[^`~\n]*")
_STANDALONE_I_RE = re.compile(r"\bI\b")
_QUOTE_PAIRS = {
    '"': '"',
    "\u201c": "\u201d",  # curly double quotes
    "\u00ab": "\u00bb",  # guillemets
    "\u201e": "\u201c",  # German low-high quotes
}


def _first_line(text: str) -> str:
    """The first non-blank line of *text*, stripped ('' when there is none)."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _inline_preamble(line: str) -> re.Match[str] | None:
    """Match a "Here is the rewritten text: <content>" prefix on *line*."""
    m = _INLINE_PREAMBLE_RE.match(line)
    return m if m and _TASK_RE.search(m.group(1)) else None


def _is_preamble_line(line: str, first_person_ok: bool) -> bool:
    """True when *line* is model chatter introducing the rewrite, not content."""
    s = line.strip()
    if _LABEL_LINE_RE.fullmatch(s):
        return True
    if _HERE_LINE_RE.fullmatch(s) and _TASK_RE.search(s):
        return True
    if _HERE_SENTENCE_RE.fullmatch(s) and _REWRITE_WORD_RE.search(s):
        return True
    tail = _HERE_TAIL_RE.fullmatch(s)
    if tail and _REWRITE_WORD_RE.search(tail.group(1)):
        return True
    return first_person_ok and bool(_FIRST_PERSON_PREAMBLE_RE.fullmatch(s))


def _last_block(lines: list[str]) -> str:
    """The last paragraph of *lines* (its trailing run of non-blank lines)."""
    block: list[str] = []
    for line in reversed(lines):
        if line.strip():
            block.append(line)
        elif block:
            break
    return "\n".join(reversed(block))


def _trailer_kind(line: str) -> str | None:
    """The kind of trailing commentary *line* opens, or None for content."""
    for kind, rx in _TRAILER_RES:
        if rx.match(line):
            return kind
    return None


def strip_model_wrappers(text: str, original: str = "") -> tuple[str, list[str]]:
    """Strip LLM meta-commentary wrapped around a rewrite.

    Returns (text, removed) where *removed* names what was stripped, in order:
    ``think`` (a reasoning block closed by ``</think>``), ``preamble`` (leading
    "Here is the rewritten text ...:" / "Sure!" / "Rewritten text:" lines),
    ``separator`` (leftover ``---`` rules), ``trailer`` (a trailing "I changed
    the following:" / "Changes made:" / "Note:" section, or paragraphs echoing
    the rewrite prompt's jargon such as "At low intensity:" or "0.32 tokens
    changed", and everything after it), ``code_fence`` and ``quotes`` (a fence
    or quote pair around the whole output).

    *original* is the text the model was asked to rewrite. A rule is skipped when
    the original carries the same construct (it starts with a "Here is ...:" line,
    has its own "Note:" paragraph, is fenced or quoted as a whole), and
    first-person trailer detection is off for first-person input, so wording
    that was already there is kept. A strip that would leave nothing is skipped,
    and text with nothing to strip is returned unchanged (whitespace included).
    """
    s = text.strip()
    orig = original.strip()
    removed: list[str] = []

    def mark(kind: str) -> None:
        if kind not in removed:
            removed.append(kind)

    if not _THINK_CLOSE_RE.search(orig):
        closes = list(_THINK_CLOSE_RE.finditer(s))
        if closes and s[closes[-1].end() :].strip():
            s = s[closes[-1].end() :].strip()
            mark("think")

    first_person_ok = not _STANDALONE_I_RE.search(orig)
    orig_first = _first_line(orig)
    if not (
        orig_first
        and (
            _is_preamble_line(orig_first, first_person_ok)
            or _INTERJECTION_LINE_RE.fullmatch(orig_first)
            or _INLINE_PREAMBLE_RE.match(orig_first)
        )
    ):
        for _ in range(3):  # e.g. "Sure!" then "Here is the rewritten text:"
            first, _sep, rest = s.partition("\n")
            if rest.strip() and _is_preamble_line(first, first_person_ok):
                s = rest.strip()
                mark("preamble")
                continue
            m = _inline_preamble(s)
            if m and s[m.end() :].strip():
                s = s[m.end() :].strip()
                mark("preamble")
                continue
            # A lone "Sure!" is chatter only when a preamble follows it; on its
            # own it may be a line of dialogue.
            nxt = _first_line(rest)
            if _INTERJECTION_LINE_RE.fullmatch(first.strip()) and (
                _is_preamble_line(nxt, first_person_ok) or _inline_preamble(nxt)
            ):
                s = rest.strip()
                mark("preamble")
                continue
            break

    orig_lines = orig.splitlines()
    if not (orig_lines and _HR_RE.fullmatch(orig_lines[0].strip())):
        lines = s.split("\n")
        while len(lines) > 1 and (_HR_RE.fullmatch(lines[0].strip()) or not lines[0].strip()):
            if lines[0].strip():
                mark("separator")
            lines.pop(0)
        s = "\n".join(lines)

    disabled = {kind for line in orig_lines if (kind := _trailer_kind(line.lstrip()))}
    if not first_person_ok:
        disabled.add("first_person")
    echoes = [rx for rx in _ECHO_RES if not rx.search(orig)]
    label_echoes = [rx for rx in _LABEL_ECHO_RES if not rx.search(orig)]
    lines = s.split("\n")
    ends_in_echo = any(rx.search(_last_block(lines)) for rx in echoes)

    def echoes_prompt(line: str) -> bool:
        """Whether a paragraph opening with *line* echoes the prompt's jargon.

        A label ending in a colon ("At low intensity, this might become:")
        needs one echo; any other paragraph needs the output to end in an
        echoing paragraph too.
        """
        if line.strip().rstrip("*_").endswith((":", "\uff1a")) and any(
            rx.search(line) for rx in label_echoes
        ):
            return True
        return ends_in_echo and any(rx.search(line) for rx in echoes)

    in_fence = False
    seen_content = False
    new_block = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            seen_content = True
            new_block = False
            continue
        if in_fence:
            continue
        if not line.strip() or _HR_RE.fullmatch(line.strip()):
            new_block = True
            continue
        if seen_content:
            kind = _trailer_kind(line)
            if (kind is not None and kind not in disabled) or (new_block and echoes_prompt(line)):
                kept = "\n".join(lines[:i]).rstrip()
                if kept.strip():
                    s = kept
                    mark("trailer")
                break
        seen_content = True
        new_block = False

    if not (orig_lines and _HR_RE.fullmatch(orig_lines[-1].strip())):
        lines = s.split("\n")
        while len(lines) > 1 and (_HR_RE.fullmatch(lines[-1].strip()) or not lines[-1].strip()):
            if lines[-1].strip():
                mark("separator")
            lines.pop()
        s = "\n".join(lines)

    lines = s.split("\n")
    fence = _FENCE_OPEN_RE.fullmatch(lines[0].strip()) if len(lines) >= 3 else None
    if fence and not orig.startswith(("```", "~~~")):
        marker = fence.group(1)
        close = lines[-1].strip()
        inner = lines[1:-1]
        if (
            close.startswith(marker)
            and set(close) == {marker[0]}
            and not any(line.lstrip().startswith(marker[0] * 3) for line in inner)
            and "\n".join(inner).strip()
        ):
            s = "\n".join(inner).strip("\r\n")  # keep the first line's indentation
            mark("code_fence")

    closing = _QUOTE_PAIRS.get(s[:1])
    if closing and len(s) >= 2 and s.endswith(closing):
        inner = s[1:-1]
        orig_quoted = len(orig) >= 2 and orig[0] == s[0] and orig.endswith(closing)
        if not orig_quoted and s[0] not in inner and closing not in inner and inner.strip():
            s = inner.strip()
            mark("quotes")

    return (s.rstrip(), removed) if removed else (text, [])


def _env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable with fallback."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _flag_env(name: str) -> bool:
    """Read an environment variable as a boolean flag."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Read an environment variable as an integer."""
    try:
        return int(_env(name, str(default)) or str(default))
    except ValueError:
        return default


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_remote(base_url: str, allow_remote: bool) -> None:
    """Enforce the rewrite-endpoint allowlist.

    Default-deny: only loopback endpoints are accepted. Anything else requires
    an explicit opt-in (--allow-remote / WATERMARKS_REWRITE_ALLOW_REMOTE=1),
    and non-http(s) schemes (e.g. file://) are always refused.
    """
    u = urlparse(base_url)
    if u.scheme not in ("http", "https"):
        raise SystemExit(
            f"error: rewrite base URL must be http(s), got scheme '{u.scheme}': {base_url}"
        )
    host = u.hostname or ""
    if host in _LOOPBACK_HOSTS:
        return
    if not allow_remote:
        raise SystemExit(
            "error: rewrite base URL host is not loopback "
            f"('{host}'); refusing to send content off-machine. "
            "Set WATERMARKS_REWRITE_ALLOW_REMOTE=1 or pass --allow-remote to override."
        )
    eprint(
        f"warning: rewrite base URL host is '{host}' (not localhost); "
        "content will leave this machine"
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects.

    urllib's default handler re-sends the request headers on 301/302/303,
    which would forward the Authorization header (API key) to an unvalidated
    host behind the localhost allowlist. Any 3xx now surfaces as HTTPError.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Custom redirect handler for URL requests."""
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _safe_detect(detector: object, text: str) -> dict:
    """Run a detector report defensively; a raising detector never fails the rewrite."""
    try:
        return detector.detect(text)  # type: ignore[attr-defined]
    except Exception as e:  # defensive: the detector contract is fail-soft
        return {"available": False, "error": f"evaluation failed: {e}"}


def _pick_evaluator(
    markllm_detector: MarkLLMTextDetector | None,
    gumbel_detector: GumbelTextDetector | None,
) -> tuple[str, object | None]:
    """Pick the evaluator that drives the iterative rewrite loop.

    Priority: keyed-Gumbel same-key replay (when the caller passed
    --gumbel-key) > MarkLLM same-config detection (--markllm-scheme) >
    bigram-Jaccard lexical divergence (fallback with no pass/fail verdict).
    A vendor-detector seam (Google's SynthID-text detector, retired Aug 2026)
    is reserved ahead of both should a vendor endpoint return; it only needs
    available()/detect()/name per the TextDetector protocol in
    text_detectors.py.
    """
    if gumbel_detector is not None:
        return "gumbel", gumbel_detector
    if markllm_detector is not None:
        return "markllm", markllm_detector
    return "lexical-divergence", None


def _generate_once(
    backend: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    timeout: float,
    temperature: float,
    reasoning_effort: str | None,
) -> str:
    """Generate a single rewrite variant through the configured backend."""
    if backend == "ollama":
        return call_ollama(base_url, model, prompt, timeout, temperature, reasoning_effort)
    if backend == "openai-compatible":
        return call_openai_compatible(
            base_url, model, prompt, api_key, timeout, temperature, reasoning_effort
        )
    raise SystemExit(f"unknown backend: {backend}")


def _generate_rewrite(
    backend: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    timeout: float,
    temperature: float,
    reasoning_effort: str | None,
    *,
    original: str,
) -> tuple[str, list[str]]:
    """Generate one LLM rewrite of *original* and strip model wrappers from it.

    Returns (text, removed) as strip_model_wrappers() does.
    """
    raw = _generate_once(
        backend, base_url, model, api_key, prompt, timeout, temperature, reasoning_effort
    )
    return strip_model_wrappers(raw, original)


def _tactic_prompt(tactic: str, text: str, lang: str, original_lang: str) -> str:
    """Build the prompt for a named rewrite tactic (no intensity modulation)."""
    if tactic == "paraphrase":
        return PROMPTS["paraphrase"].format(TEXT=text)
    if tactic == "humanize":
        return PROMPTS["humanize"].format(TEXT=text)
    if tactic == "academic":
        return PROMPTS["academic"].format(TEXT=text)
    if tactic == "code":
        return PROMPTS["code"].format(TEXT=text)
    # backtranslate / structural: a single combined instruction, used by
    # print-prompt and the rewrite() loop. apply_strategy does not send it: a
    # model can shortcut the combined form, so the strategy path runs the
    # TWO_STEP_PROMPTS pair as two generations instead (_two_step_generate).
    if tactic == "backtranslate":
        return (
            f"Translate the text to {lang}, then translate that result back to "
            f"{original_lang}. Preserve all facts, numbers, and names. "
            f"Output only the final {original_lang} text.\n\n---\n{text}"
        )
    if tactic == "structural":
        return (
            "First extract a bullet outline of all claims (no full sentences). "
            "Then write a complete document from that outline in natural, varied human "
            "prose without omitting any bullet. Output only the final document.\n\n---\n"
            f"{text}"
        )
    if tactic == "chunk":
        return PROMPTS["chunk_unit"].format(TEXT=text)
    if tactic == "mlm":
        # Local masked-LM edit: the prompt is informational only; generation runs
        # a non-autoregressive infill (see _mlm_infill) rather than the backend.
        return "Local masked-LM infill; no LLM prompt is used.\n\n---\n" + text
    raise ValueError(f"unknown tactic: {tactic}")


def _intensity_clause(level: float) -> str:
    """The intensity instruction appended to a tactic prompt.

    The level is a request, not a contract: measured lexical/semantic
    divergence is the real outcome, and a model may not hit the fraction exactly.
    """
    return (
        f"Modulate this rewrite so roughly a fraction {level:.2f} of tokens change: "
        "0 would keep the wording unchanged, 1 rewrites everything. At low intensity "
        "keep the sentence structure, word order, and every token that can stay, "
        "changing only function words and a few non-essential content words; at high "
        "intensity change wording substantially at the token level. Preserve all facts, "
        "numbers, names, and technical identifiers. Do not add or remove claims."
    )


# Function words / short / technical tokens we never hand to a masked LM.
_MLM_SKIP_WORDS = {
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "man",
    "new",
    "now",
    "old",
    "see",
    "two",
    "way",
    "who",
    "boy",
    "did",
    "its",
    "let",
    "put",
    "say",
    "she",
    "too",
    "use",
    "that",
    "with",
    "have",
    "this",
    "will",
    "your",
    "from",
    "they",
    "been",
    "were",
    "would",
    "there",
    "their",
    "what",
    "when",
    "which",
    "also",
    "into",
    "than",
    "then",
    "them",
    "these",
    "those",
    "such",
    "only",
    "very",
    "just",
    "about",
    "some",
    "more",
    "most",
    "other",
    "over",
    "under",
    "through",
    "between",
    "while",
    "where",
    "because",
}
_MLM_TOKEN_RE = re.compile(r"(\s+|[.,;:!?()\"'—-])")
MLM_MODEL = "roberta-large"
_MLM_MAX_TOKENS = 512  # roberta-large positional limit
_MLM_CACHE: dict[str, Any] = {}  # {"pipeline": ..., "mask_token": ...}
# Where the tactic's stack is declared; named in every "unavailable" error.
MLM_REQUIREMENTS = "service/scripts/requirements-mlm.txt"


def _cuda_available() -> bool:
    """True when a CUDA device is usable; False on CPU-only or no-torch hosts."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch absent; run on CPU/auto
        return False


def _get_mlm() -> tuple[Any, str]:
    """Return the process-cached roberta-large fill-mask pipeline + mask token.

    Built lazily on first use; a failed import or model load surfaces as
    RuntimeError (fail-soft optional dependency, declared in
    requirements-mlm.txt). The device is chosen at runtime so a CPU-only host
    still works (architecture selects the accelerator when available).
    """
    if "pipeline" not in _MLM_CACHE:
        try:
            from transformers import pipeline
        except Exception as e:  # fail-soft: optional dependency
            raise RuntimeError(f"mlm tactic unavailable: {e} (install {MLM_REQUIREMENTS})") from e
        kwargs: dict[str, Any] = {"model": MLM_MODEL}
        if _cuda_available():
            kwargs["device"] = 0
        try:
            fill_mask = pipeline("fill-mask", **kwargs)
        except Exception as e:  # e.g. no torch, or offline with no cached weights
            raise RuntimeError(f"mlm tactic unavailable: cannot load {MLM_MODEL}: {e}") from e
        _MLM_CACHE["pipeline"] = fill_mask
        _MLM_CACHE["mask_token"] = fill_mask.tokenizer.mask_token
    return _MLM_CACHE["pipeline"], _MLM_CACHE["mask_token"]


def load_mlm() -> None:
    """Build the mlm pipeline now (process-cached); RuntimeError when it can't.

    Lets a caller about to run a multi-step strategy reject up front when the
    mlm stack is broken, instead of after the LLM steps ahead of it have run.
    """
    _get_mlm()


# _get_mlm()'s imports, for mlm_import_error(): torch backs the pipeline and
# transformers.pipelines pulls in its vision/OCR modules. Keep them in sync.
_MLM_IMPORT_PROBE = (
    "import sys\n"
    "try:\n"
    "    import torch\n"
    "    from transformers import pipeline\n"
    "except Exception as e:\n"
    "    sys.stdout.write(f'{type(e).__name__}: {e}')\n"
    "    sys.exit(1)\n"
)


def mlm_import_error(timeout: float = 120.0) -> str | None:
    """Why the mlm tactic's stack can't be imported here, or None when it can.

    Runs _get_mlm()'s imports in a child of this interpreter, so the check
    never loads torch into the caller and a broken native wheel can't crash
    it. It stops short of the weights: roberta-large still downloads on first
    use unless cached. No rlimit preexec: torch maps large shared libraries
    (see text_detectors._markllm_preexec). The generous timeout covers a cold
    import (tens of seconds on a large site-packages).
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c", _MLM_IMPORT_PROBE],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=timeout,
            check=False,
            creationflags=subprocess_creationflags,
        )
    except subprocess.TimeoutExpired:
        return f"import probe timed out after {timeout:g}s"
    except OSError as e:
        return f"import probe could not start: {e}"
    if r.returncode == 0:
        return None
    # The probe prints "<Type>: <message>"; a hard crash leaves only stderr.
    lines = r.stdout.strip().splitlines() or r.stderr.strip().splitlines()[-1:]
    return " ".join(" ".join(lines).split()) or f"import probe exited with status {r.returncode}"


def _mlm_chunks(parts: list[str], tokenizer: Any, max_tokens: int = _MLM_MAX_TOKENS):
    """Split `parts` into contiguous chunks whose token length stays <= max_tokens.

    Chunk boundaries fall on separator/word edges, so each chunk joins to a clean
    substring, preserving ordering across chunks. Yields (global_start, chunk).
    """
    cur: list[str] = []
    cur_start = 0
    for i, part in enumerate(parts):
        trial = [*cur, part]
        if cur and len(tokenizer("".join(trial))["input_ids"]) > max_tokens:
            yield cur_start, cur
            cur = [part]
            cur_start = i
        else:
            cur = trial
    if cur:
        yield cur_start, cur


def _mlm_infill(text: str, level: float) -> str:
    """Mask `level` of content words and infill with roberta-large (local edit).

    Non-autoregressive: the output is a mix of the original token stream and
    masked-LM predictions, not fresh LLM-sampled prose. Uses a process-cached,
    runtime-device pipeline and splits inputs longer than roberta's positional
    limit into separately-infilled chunks.
    """
    mlm, mask_token = _get_mlm()
    tokenizer = mlm.tokenizer
    tokens = _MLM_TOKEN_RE.split(text)
    content = [
        i
        for i, t in enumerate(tokens)
        if t.strip()
        and t.isalpha()
        and len(t) > 3
        and t.lower() not in _MLM_SKIP_WORDS
        and not t[0].isupper()
    ]
    k = max(1, round(level * len(content))) if content else 0
    if k == 0:
        return text
    step = len(content) / k
    mset: set[int] = set()
    pos = 0.0
    for _ in range(k):
        idx = content[int(pos)]
        mset.add(idx)
        pos += step
    out_parts = [mask_token if i in mset else tokens[i] for i in range(len(tokens))]
    for chunk_start, chunk in _mlm_chunks(out_parts, tokenizer):
        positions = [chunk_start + j for j, p in enumerate(chunk) if p == mask_token]
        if not positions:
            continue
        preds = mlm("".join(chunk), top_k=1)
        picks = [(p if isinstance(p, dict) else p[0]) for p in preds]
        for k_i, global_idx in enumerate(positions):
            if k_i < len(picks):
                tokens[global_idx] = picks[k_i]["token_str"].strip()
    return "".join(tokens)


def _style_clause(style: str) -> str:
    """The style instruction appended to a rewrite prompt.

    Intended for the humanize / manual-polish tactics (e.g. "write like
    Hemingway"). A request, not a contract: the model may only approximate a
    style, and the fact/voice rules still apply.
    """
    return (
        f"Apply this writing style throughout the rewrite: {style}. Keep the "
        "style subordinate to the content — preserve all facts, numbers, names, "
        "and technical identifiers, and do not add or remove claims."
    )


def _latex_placeholder_clause(mask: LatexMask) -> str:
    """The instruction that keeps masked math/LaTeX placeholders intact.

    The placeholders are restored mechanically afterwards, so a model that drops
    one costs the document an equation; say so plainly in the prompt.
    """
    return (
        f"The text contains {mask.count} protected token(s) of the form "
        f"{mask.token(0)} standing for mathematics, LaTeX commands or verbatim "
        "spans. Reproduce every one of them exactly as written, in the same order, "
        "and keep them attached to the sentence that carries them. Never translate, "
        "renumber, reformat, merge, drop or invent such a token."
    )


def _latex_guard_clause() -> str:
    """The preserve-verbatim instruction used when spans cannot be masked.

    ``print-prompt`` hands the prompt to another agent and never sees the output,
    so there is no map to restore from; the instruction is the only protection
    available on that path, and it is a request rather than a guarantee.
    """
    return (
        "Reproduce all mathematics, LaTeX commands, environments, citation keys, "
        "labels, and verbatim/code spans exactly as they appear: symbols, indices, "
        "signs, delimiters, and spacing inside them are not part of the rewrite."
    )


def build_prompt(
    tactic: str | None,
    text: str,
    *,
    lang: str = "French",
    original_lang: str = "English",
    rewrite_level: float | None = None,
    style: str | None = None,
    mask: LatexMask | None = None,
    latex_guard: bool = False,
) -> str:
    """Construct the LLM rewrite prompt for a given tactic and intensity."""
    if tactic is None:
        if rewrite_level is not None:
            base = PROMPTS["level"].format(TEXT=text, LEVEL=rewrite_level)
        else:
            raise ValueError("unknown tactic: None")
    else:
        base = _tactic_prompt(tactic, text, lang, original_lang)
        # A (tactic, intensity) pair: modulate the named tactic prompt with the
        # level instead of replacing it with the generic level-only prompt. Code is
        # exempt — identifier/comment rewrites are not naturally intensity-modulated.
        if rewrite_level is not None and tactic != "code":
            base = base + "\n\n" + _intensity_clause(rewrite_level)
    if style:
        base = base + "\n\n" + _style_clause(style)
    if mask is not None and mask.count:
        base = base + "\n\n" + _latex_placeholder_clause(mask)
    elif latex_guard:
        base = base + "\n\n" + _latex_guard_clause()
    return base


# Tactics that are two dependent transformations, as (first, final) PROMPTS
# keys. Sent as one combined prompt, a model can shortcut them: a backtranslate
# came back byte-identical to its input, and a structural rewrite returned the
# bullet outline followed by the prose. apply_strategy runs each as two
# generations, the final one fed only the first one's output.
TWO_STEP_PROMPTS: dict[str, tuple[str, str]] = {
    "backtranslate": ("backtranslate_out", "backtranslate_back"),
    "structural": ("structural_outline", "structural_write"),
}


def _two_step_generate(
    tactic: str,
    text: str,
    generate: Callable[[str], tuple[str, list[str]]],
    *,
    lang: str,
    original_lang: str,
    style: str | None = None,
    mask: LatexMask | None = None,
) -> tuple[str, list[str]]:
    """Run a TWO_STEP_PROMPTS tactic as two generations, in order.

    The first generation sees only *text* and returns the intermediate (the
    pivot-language translation, or the bullet outline); the final one sees
    only that intermediate, so it cannot copy the original token stream
    through. The style clause joins the final prompt only: it shapes the prose
    that is returned, and would not survive a pivot translation or an outline.
    The intensity clause joins neither. "Change roughly this fraction of the
    tokens, keeping every token that can stay" has no meaning for a
    translation or an outline extraction, and at low levels it asks for the
    very copy-through (the input, or the outline bullets) that the split exists
    to prevent. When *mask* holds protected spans, both prompts carry the
    placeholder-preservation clause so the spans survive the pivot.

    *generate* makes one backend call and returns (text, stripped wrapper
    kinds); the kinds of both calls are merged in the result.
    """
    first_key, final_key = TWO_STEP_PROMPTS[tactic]
    langs = {"LANG": lang, "ORIGINAL_LANG": original_lang}
    clause = "\n\n" + _latex_placeholder_clause(mask) if mask is not None and mask.count else ""
    intermediate, removed = generate(PROMPTS[first_key].format(TEXT=text, **langs) + clause)
    if not intermediate.strip():
        raise RuntimeError(f"{tactic}: the first generation returned empty output")
    prompt = PROMPTS[final_key].format(TEXT=intermediate, **langs)
    if style:
        prompt += "\n\n" + _style_clause(style)
    prompt += clause
    final, removed_final = generate(prompt)
    return final, removed + [k for k in removed_final if k not in removed]


# Words that end in a period without ending a sentence. A split there hands the
# model a fragment such as "Eq.~\eqref{eq:weyl})." on its own. Lower-case, the
# final period removed; multi-part forms ("e.g", "et al") are listed as written.
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "cf",
        "vs",
        "viz",
        "etc",
        "et al",
        "al",
        "ca",
        "approx",
        "resp",
        "sect",
        "sec",
        "secs",
        "eq",
        "eqs",
        "eqn",
        "eqns",
        "fig",
        "figs",
        "ref",
        "refs",
        "tab",
        "thm",
        "lem",
        "prop",
        "cor",
        "def",
        "rem",
        "ex",
        "ch",
        "chap",
        "vol",
        "no",
        "nos",
        "pp",
        "p",
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "st",
        "jr",
        "sr",
        "ed",
        "eds",
        "rev",
        "univ",
        "dept",
        "ph.d",
        "b.sc",
        "m.sc",
        "ibid",
        "op. cit",
    }
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _is_abbreviation(before: str) -> bool:
    """True when *before* ends in an abbreviation, so the period is not a sentence end."""
    m = re.search(r"(\S+)$", before)
    if not m:
        return False
    word = m.group(1)
    if not word.endswith("."):
        return False
    core = word.rstrip(".").lstrip("([{\"'")
    low = core.lower()
    if low in _ABBREVIATIONS:
        return True
    # "et al." reaches here as "al."; check the two-word form as well.
    two = re.search(r"(\S+\s+\S+)$", before)
    if two and two.group(1).rstrip(".").lower() in _ABBREVIATIONS:
        return True
    # A single capital letter is an initial ("J. Doe"), not a sentence.
    return len(core) == 1 and core.isalpha() and core.isupper()


def _split_units(text: str) -> list[tuple[str, str]]:
    """Split a document into (unit, separator) pairs.

    Breaks after sentence punctuation or on any blank-line / newline run, so
    each fragment is rewritten independently (a fresh context per fragment ⇒
    new per-token watermark keys). A period that closes an abbreviation
    ("e.g.", "Sect.", "et al.", an initial) does not break. Punctuation is kept
    with its fragment. The separator is the whitespace/blank-line run that
    follows a unit ('' for the last); unshuffled chunk mode reassembles with it
    so paragraph/line layout is preserved, while shuffled mode drops it.
    """
    parts = re.split(r"((?<=[.!?])\s+|\n+)", text)
    raw: list[tuple[str, str]] = []
    for i in range(0, len(parts), 2):
        raw.append((parts[i], parts[i + 1] if i + 1 < len(parts) else ""))
    merged: list[tuple[str, str]] = []
    for unit, sep in raw:
        if merged:
            prev_unit, prev_sep = merged[-1]
            if prev_sep and "\n" not in prev_sep and _is_abbreviation(prev_unit):
                merged[-1] = (prev_unit + prev_sep + unit, sep)
                continue
        merged.append((unit, sep))
    return [(unit.strip(), sep) for unit, sep in merged if unit.strip() or "\n" in sep]


def _prose_chunks(text: str, limit: int) -> list[str]:
    """Split *text* into pieces of at most *limit* characters that join back exactly.

    A cut falls, in order of preference, after a blank line, after a sentence
    end (abbreviations excluded), after a line break, or after a space inside
    the window, never earlier than a third of the way in so the pieces stay
    substantial; a window with none of those is cut at *limit*. Placeholders
    contain no whitespace, so a cut never splits one. ``limit <= 0`` disables
    chunking.
    """
    if limit <= 0 or len(text) <= limit:
        return [text]
    floor = max(1, limit // 3)
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = -1
        for m in re.finditer(r"\n[ \t]*\n", window):
            cut = max(cut, m.end())
        if cut < floor:
            cut = -1
            for m in _SENTENCE_END_RE.finditer(window):
                if not _is_abbreviation(window[: m.start()]):
                    cut = max(cut, m.end())
        if cut < floor:
            cut = max((m.end() for m in re.finditer(r"\n", window)), default=-1)
        if cut < floor:
            cut = max((m.end() for m in re.finditer(r" ", window)), default=-1)
        if cut < floor:
            cut = limit
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def _http_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    """Perform an HTTP POST request and return JSON response."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"refusing non-http(s) rewrite endpoint: {url}")
    body = json.dumps(payload).encode("utf-8")
    # S310: URL scheme is restricted to http/https just above.
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ollama_budget(prompt: str) -> dict[str, int]:
    """Ollama sampling options sized to *prompt*.

    Ollama silently drops the oldest tokens when a prompt outgrows ``num_ctx``,
    which for a rewrite means losing the instruction, and stops at
    ``num_predict`` tokens, which truncates the output. Both grow with the
    prompt here (about one token per three characters, conservative for LaTeX
    and non-English text); WATERMARKS_OLLAMA_CONTEXT / WATERMARKS_OLLAMA_MAX_TOKENS
    are floors, not caps. WATERMARKS_OLLAMA_THREADS adds ``num_thread`` when set.
    """
    est_prompt = len(prompt) // 3 + 64
    predict = max(256, _env_int("WATERMARKS_OLLAMA_MAX_TOKENS", 2048), est_prompt)
    ctx = max(1024, _env_int("WATERMARKS_OLLAMA_CONTEXT", 8192), est_prompt + predict + 256)
    options = {"num_ctx": ctx, "num_predict": predict}
    threads = _env_int("WATERMARKS_OLLAMA_THREADS", 0)
    if threads > 0:
        options["num_thread"] = threads
    return options


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
    reasoning_effort: str | None = None,
) -> str:
    """Call Ollama API endpoint for text rewrite.

    reasoning_effort "none" sends ``think: false``. Ollama runs a thinking
    model's reasoning by default, and gemma4:12b spent 1,848 tokens (~200 s on
    a laptop GPU) thinking before a two-word paraphrase. Models without a
    thinking mode accept ``think: false``; other effort values leave the
    model's default, because ``think: true`` is an error on those models.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload: dict = {
        "model": model,
        "stream": False,
        "keep_alive": "5m",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, **_ollama_budget(prompt)},
    }
    if reasoning_effort == "none":
        payload["think"] = False
    data = _http_json(url, payload, {}, timeout)
    msg = data.get("message") or {}
    if data.get("done_reason") == "length" or data.get("done") is False:
        raise RuntimeError("Ollama response was truncated; use a shorter input section")
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"ollama empty response: {data!r}"[:500])
    return str(content).strip()


def call_openai_compatible(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    temperature: float,
    reasoning_effort: str | None = None,
) -> str:
    """Call OpenAI-compatible chat completions API."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    data = _http_json(
        url,
        payload,
        headers,
        timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openai-compatible empty choices: {data!r}"[:500])
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError(f"openai-compatible empty content: {data!r}"[:500])
    return str(content).strip()


def _candidate_pass(
    evaluation: dict, target_margin: float
) -> tuple[bool | None, float | None, float | None]:
    """Judge a detection report against a score-margin objective.

    Returns (passed, margin, raw_margin). ``passed`` is True when the detector
    reports the text not-watermarked AND its score sits at least ``target_margin``
    below the threshold; False when still watermarked; None when no verdict is
    available (fail-soft) or when the margin floor is not met. ``margin`` =
    round(threshold - score, 4), or None when either field is missing OR the
    threshold is not a score-scale cutoff. ``raw_margin`` is the unrounded
    margin, or None in the same cases where ``margin`` is None; it lets ranking
    compare candidates without the loss from rounding to four decimals.

    Some detectors (keyed-Gumbel) report a *p-value* threshold that does not
    scale with their ``score``, so ``threshold - score`` is meaningless; a
    not-watermarked report whose score is above such a threshold is treated as
    a clear pass rather than a gated margin.
    """
    verdict = evaluation.get("is_watermarked")
    if verdict is None:
        return None, None, None
    score = evaluation.get("score")
    threshold = evaluation.get("threshold")
    margin = None
    raw_margin = None
    if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
        raw_margin = float(threshold) - float(score)
        margin = round(raw_margin, 4)
    if verdict is True:
        return False, margin, raw_margin
    # Not watermarked. If the threshold is not a score-scale cutoff it cannot
    # express a margin, so fall back to a clean pass (no margin floor).
    if raw_margin is not None and raw_margin < 0:
        margin = None
        raw_margin = None
    met = raw_margin is not None and raw_margin >= target_margin - 1e-9
    if raw_margin is None or met:
        return True, margin, raw_margin
    return None, margin, raw_margin


def _margin_of(rec: dict) -> tuple[float, float, float]:
    # Rank by the unrounded margin first: the telemetry margin is rounded to
    # four decimals, so two candidates can share that rounded value while
    # differing in the raw margin (e.g. 0.12343 vs 0.12344 both round to
    # 0.1234). Preserve the p-value and lexical-divergence tie-breakers.
    """Compute rankable candidate margin score."""
    raw = rec.get("raw_margin")
    raw_val = float(raw) if raw is not None else -float("inf")
    # For evaluators that report p_value (e.g. Keyed-Gumbel), lower p_value indicates a safer pass
    eval_rec = rec.get("evaluation") or {}
    pval = eval_rec.get("p_value")
    neg_pval = -float(pval) if isinstance(pval, (int, float)) else -float("inf")
    # Secondary tiebreaker: prefer less divergence
    div = -float(rec.get("lexical_divergence", 0.0))
    return (raw_val, neg_pval, div)


def rewrite(
    text: str,
    *,
    backend: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    tactic: str,
    lang: str,
    original_lang: str,
    timeout: float,
    layer_a_after: bool,
    temperature: float,
    candidates: int,
    max_loops: int = 1,
    allow_remote: bool = False,
    reasoning_effort: str | None = None,
    markllm_scheme: str | None = None,
    markllm_dir: str | None = None,
    markllm_model: str | None = None,
    markllm_timeout: float = 180.0,
    gumbel_key: str | None = None,
    rewrite_level: float | None = None,
    style: str | None = None,
    target_margin: float = 0.0,
    selection: str = "min-divergence",
    chunk_shuffle: bool = False,
    noop_lex_floor: float = 0.05,
    protect_latex: str = "auto",
) -> tuple[str, dict]:
    """Execute text rewrite pass across candidates and select best candidate."""
    # Math, LaTeX control structures and verbatim spans carry no token-sampling
    # watermark worth attacking, so they leave the text as placeholders and come
    # back after the deterministic passes have run. print-prompt has no output to
    # restore from, so that path asks the prompt to preserve them instead.
    is_print_prompt = backend == "print-prompt"
    if is_print_prompt:
        work_text, mask = text, LatexMask()
        guard = protect_latex == "on" or (protect_latex == "auto" and looks_like_latex(text))
    else:
        work_text, mask = mask_latex(text, mode=protect_latex)
        guard = False
    prompt = build_prompt(
        tactic,
        work_text,
        lang=lang,
        original_lang=original_lang,
        rewrite_level=rewrite_level,
        style=style,
        mask=mask,
        latex_guard=guard,
    )
    info: dict = {
        "backend": backend,
        "tactic": tactic,
        "protect_latex": protect_latex,
        "latex_protection": (
            "prompt-guard" if guard else ("placeholders" if mask.count else "none")
        ),
        "latex_protected": mask.count,
        "rewrite_level": rewrite_level,
        "style": style,
        "target_margin": target_margin,
        "selection": selection,
        "noop_lex_floor": noop_lex_floor,
        "noop": False,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "prompt_chars": len(prompt),
        "input_chars": len(text),
    }
    if reasoning_effort:
        info["reasoning_effort"] = reasoning_effort

    markllm: dict | None = None
    markllm_detector: MarkLLMTextDetector | None = None
    if markllm_scheme:
        markllm_detector = MarkLLMTextDetector(
            scheme=markllm_scheme,
            upstream_dir=markllm_dir,
            model=markllm_model or DEFAULT_MARKLLM_MODEL,
            timeout=markllm_timeout,
        )
        markllm = {
            "scheme": markllm_scheme,
            "before": _safe_detect(markllm_detector, text),
        }
        if not markllm["before"]["available"]:
            eprint(f"markllm verification unavailable: {markllm['before']['error']}")
        info["markllm"] = markllm

    gumbel: dict | None = None
    gumbel_detector: GumbelTextDetector | None = None
    if gumbel_key:
        gumbel_detector = GumbelTextDetector(key=gumbel_key)
        gumbel = {"before": _safe_detect(gumbel_detector, text)}
        if not gumbel["before"]["available"]:
            eprint(f"gumbel verification unavailable: {gumbel['before']['error']}")
        info["gumbel"] = gumbel

    if backend == "print-prompt":
        info["mode"] = "print-prompt"
        if candidates > 1:
            eprint("note: --candidates ignored in print-prompt mode")
        return prompt, info

    if not model and tactic != "mlm":
        raise SystemExit("error: --model required for ollama/openai-compatible backends")
    if not base_url and tactic != "mlm":
        raise SystemExit("error: --base-url required for ollama/openai-compatible backends")

    # The mlm tactic never talks to a remote endpoint; base_url may be None.
    if base_url is not None:
        _check_remote(base_url, allow_remote)

    n_cands = max(1, candidates)
    n_loops = max(1, max_loops)
    info["candidates"] = n_cands
    info["max_loops"] = n_loops
    evaluator_name, evaluator = _pick_evaluator(markllm_detector, gumbel_detector)
    info["evaluator"] = evaluator_name

    is_chunk = tactic == "chunk"
    is_mlm = tactic == "mlm"
    info["chunked"] = is_chunk
    info["chunk_shuffle"] = bool(chunk_shuffle)
    info["mlm"] = is_mlm
    length_guard = tactic in LENGTH_PRESERVING_TACTICS
    info["length_guard"] = length_guard

    def _rewrite_unit(unit: str, removed: list[str]) -> str:
        """Rewrite a single text unit, recording any stripped model wrappers."""
        local = len(mask.token_re.findall(unit)) if mask.count else 0
        unit_mask = LatexMask(spans=("",) * local, prefix=mask.prefix) if local else None
        out, unit_removed = _generate_rewrite(
            backend,
            base_url,
            model,
            api_key,
            build_prompt(
                tactic,
                unit,
                lang=lang,
                original_lang=original_lang,
                rewrite_level=rewrite_level,
                style=style,
                mask=unit_mask,
            ),
            timeout,
            temperature,
            reasoning_effort,
            original=unit,
        )
        removed.extend(kind for kind in unit_removed if kind not in removed)
        return out

    def _generate_candidate() -> tuple[str, list[str]]:
        """Generate one rewrite candidate; returns (text, stripped wrapper kinds)."""
        if is_mlm:
            return _mlm_infill(work_text, rewrite_level or 0.3), []
        if is_chunk:
            removed: list[str] = []
            pairs = _split_units(work_text)
            if chunk_shuffle:
                units = [unit for unit, _ in pairs if unit]
                random.shuffle(units)
                return " ".join(_rewrite_unit(unit, removed) for unit in units), removed
            # Skip rewriting empty leading units (blank lines at the top) but
            # keep their separators so the reassembled document keeps the layout.
            joined = "".join(
                (_rewrite_unit(unit, removed) if unit else "") + sep for unit, sep in pairs
            )
            return joined, removed
        return _generate_rewrite(
            backend,
            base_url,
            model,
            api_key,
            prompt,
            timeout,
            temperature,
            reasoning_effort,
            original=work_text,
        )

    # Iterative rewrite: each loop generates --candidates variants and
    # evaluates them ALL (so the best is chosen, not the first to squeak under
    # the threshold). A variant "passes" when the detector reports it
    # not-watermarked AND its score sits at least --target-margin below the
    # threshold; --max-loops caps the evaluation rounds before the best-effort
    # variant is returned. When no detector is configured the evaluator is
    # lexical divergence, which has no pass/fail verdict, so every attempt is
    # generated and the most diverged one is selected (an unguided best-effort).
    # For a length-preserving tactic, a variant whose length drifted to an
    # extreme is a failed attempt: it is not evaluated, cannot pass, and is only
    # ever returned if no other attempt exists (then the rewrite fails instead).
    attempts: list[tuple[str, dict]] = []
    passed: bool | None = None
    for loop in range(n_loops):
        loop_passed = False
        for _ in range(n_cands):
            cand, stripped = _generate_candidate()
            if tactic == "humanize":
                cand = humanize_pass(cand)
            cand_stats: dict | None = None
            if layer_a_after:
                cand, cand_stats = clean_text(cand)
            # Restore last: the deterministic passes above must not see the
            # protected spans (humanize_pass would rewrite a LaTeX en dash, and a
            # Layer A scrub has no business inside an equation). Everything below
            # — divergence, detection, telemetry — scores the real text.
            integrity_error = placeholder_error(cand, mask)
            if tactic == "academic":
                integrity_error = integrity_error or academic_error(work_text, cand)
            cand, latex_stats = restore_latex(cand, mask)
            divergence = _lexical_divergence(text, cand)
            ratio = _length_ratio(len(text), len(cand))
            drift = length_guard and _length_drift(len(text), len(cand))
            if drift or integrity_error:
                evaluation: dict = {
                    "evaluator": evaluator_name,
                    "available": False,
                    "error": integrity_error
                    or f"not evaluated: output length drifted (ratio {ratio})",
                }
            elif evaluator is None:
                evaluation = {
                    "evaluator": "lexical-divergence",
                    "score": round(divergence, 4),
                }
            else:
                evaluation = _safe_detect(evaluator, cand)
            passed_i, margin, raw_margin = _candidate_pass(evaluation, target_margin)
            if drift or integrity_error:
                passed_i = False
            score = evaluation.get("score")
            threshold = evaluation.get("threshold")
            attempts.append(
                (
                    cand,
                    {
                        "loop": loop,
                        "lexical_divergence": round(divergence, 4),
                        "selection_score": round(divergence, 4),
                        "score_after": round(float(score), 4)
                        if isinstance(score, (int, float))
                        else None,
                        "threshold": round(float(threshold), 4)
                        if isinstance(threshold, (int, float))
                        else None,
                        "margin": margin,
                        "raw_margin": raw_margin,
                        "selected": False,
                        "passed": passed_i,
                        "evaluation": evaluation,
                        "layer_a_after": cand_stats,
                        "length_ratio": ratio,
                        "length_drift": drift,
                        "wrappers_stripped": stripped,
                        "latex": latex_stats if mask.count else None,
                        "integrity_error": integrity_error,
                    },
                )
            )
            if passed_i is True:
                loop_passed = True
        if loop_passed:
            passed = True
            break

    if evaluator is not None and passed is None:
        passed = False
    info["attempts_made"] = len(attempts)
    info["passed"] = passed
    drifted = sum(1 for _c, r in attempts if r["length_drift"])
    info["length_drift_rejected"] = drifted

    # Best-effort selection: among the candidates that passed (met the margin
    # objective) pick the one that changed the least (min-divergence, the
    # content-preserving default) or the one with the largest margin
    # (--select max-margin, robustness-first). When none passed, fall back to
    # the lowest watermark score (detector evaluator) or the most diverged
    # variant (unguided fallback).
    # Length-drifted attempts are never selected.
    selected_idx: int
    best_score: float | None = None
    best_score_idx: int | None = None
    best_div = -1.0
    best_div_idx: int | None = None
    for i, (_cand, rec) in enumerate(attempts):
        if rec["length_drift"] or rec["integrity_error"]:
            continue
        if rec["lexical_divergence"] > best_div:
            best_div = rec["lexical_divergence"]
            best_div_idx = i
        if evaluator is not None:
            s = rec["evaluation"].get("score")
            if isinstance(s, (int, float)) and (best_score is None or s < best_score):
                best_score = float(s)
                best_score_idx = i
    passed_idxs = [i for i, (_c, r) in enumerate(attempts) if r["passed"] is True]
    if passed_idxs:
        if selection == "max-margin":
            selected_idx = max(passed_idxs, key=lambda i: _margin_of(attempts[i][1]))
        else:
            selected_idx = min(passed_idxs, key=lambda i: attempts[i][1]["lexical_divergence"])
    elif best_score_idx is not None:
        selected_idx = best_score_idx
    elif best_div_idx is not None:
        selected_idx = best_div_idx
    else:
        if any(r["integrity_error"] for _c, r in attempts):
            reason = next(r["integrity_error"] for _c, r in attempts if r["integrity_error"])
            raise RuntimeError(f"no acceptable rewrite: {reason}")
        ratios = ", ".join(str(r["length_ratio"]) for _c, r in attempts)
        raise RuntimeError(
            f"no acceptable rewrite: all {len(attempts)} attempt(s) drifted in length "
            f"(output/input ratio {ratios}; accepted {_LENGTH_DRIFT_RANGE}), likely "
            "model meta-commentary. Raise --candidates/--max-loops or use another model."
        )
    if drifted:
        eprint(f"warning: rejected {drifted} rewrite attempt(s) whose length drifted")

    out, rec = attempts[selected_idx]
    rec["selected"] = True
    info["wrappers_stripped"] = rec["wrappers_stripped"]
    if rec["wrappers_stripped"]:
        eprint(
            "note: stripped model meta-commentary from the rewrite "
            f"({', '.join(rec['wrappers_stripped'])})"
        )
    info["candidate_scores"] = [r for _c, r in attempts]
    if layer_a_after:
        info["layer_a_after"] = rec["layer_a_after"]
    if mask.count:
        info["latex"] = rec["latex"]
        if rec["latex"]["missing"]:
            eprint(
                f"warning: {rec['latex']['missing']} of {mask.count} protected "
                "math/LaTeX span(s) were dropped by the rewrite model and are "
                "absent from the output"
            )
    info["output_chars"] = len(out)
    info["mode"] = "rewritten"

    # No-op guard: a rewrite that changed almost nothing is not a removal
    # attempt. Report it so a benchmark never counts a near-verbatim output as
    # "0% clear" (the misleading backtranslate row). Disabled with floor <= 0.
    _out_div = _lexical_divergence(text, out)
    _is_noop = _below_noop_floor(_out_div, noop_lex_floor)
    info["noop"] = bool(_is_noop)
    if _is_noop:
        eprint(
            f"warning: rewrite returned ≈ input (lex divergence {_out_div:.4f} < "
            f"{noop_lex_floor:.4f}); treating as no-op"
        )
    note = (
        "Layer B is best-effort against statistical token-sampling watermarks; "
        "cannot certify removal against a vendor detector."
    )
    if evaluator is not None and passed is not True:
        margin_suffix = f" (target margin {target_margin:.2f})" if target_margin else ""
        note += (
            f" Exhausted {len(attempts)} attempt(s) without passing "
            f"{evaluator_name} evaluation{margin_suffix}; returned the best-effort variant."
        )
    if markllm_scheme:
        note += (
            " Cross-model hygiene: rewrite with a model that is neither the "
            "generator nor itself watermarked, or the rewritten text can be "
            "re-stamped."
        )
    info["note"] = note

    if markllm:
        assert markllm_detector is not None  # set together with markllm above
        if evaluator_name == "markllm":
            # The loop already scored the selected attempt; reuse the verdict
            # instead of paying another MarkLLM detection.
            after = rec["evaluation"]
        else:
            after = _safe_detect(markllm_detector, out)
        markllm["after"] = after
        before = markllm["before"]
        if before.get("available") and after.get("available"):
            markllm["cleared"] = bool(
                before.get("is_watermarked") and not after.get("is_watermarked")
            )
        else:
            markllm["cleared"] = None
        a_score = after.get("score")
        a_thr = after.get("threshold")
        markllm["score_after"] = (
            round(float(a_score), 4) if isinstance(a_score, (int, float)) else None
        )
        markllm["margin"] = (
            round(float(a_thr) - float(a_score), 4)
            if isinstance(a_score, (int, float)) and isinstance(a_thr, (int, float))
            else None
        )
        markllm["note"] = (
            "MarkLLM detection is only valid against the SAME scheme config + "
            "keys used at generation; it does not certify a vendor detector."
        )

    if gumbel:
        assert gumbel_detector is not None  # set together with gumbel above
        if evaluator_name == "gumbel":
            # The loop already scored the selected attempt; reuse the verdict
            # instead of paying another replay.
            after = rec["evaluation"]
        else:
            after = _safe_detect(gumbel_detector, out)
        gumbel["after"] = after
        before = gumbel["before"]
        if before.get("available") and after.get("available"):
            gumbel["cleared"] = bool(
                before.get("is_watermarked") and not after.get("is_watermarked")
            )
        else:
            gumbel["cleared"] = None
        gumbel["note"] = (
            "Keyed-Gumbel detection is a same-key replay: valid only with the "
            "same key, tokenizer, and PRF layout used at generation; it does "
            "not certify a vendor detector."
        )

    eprint(
        f"note: evaluator={evaluator_name} attempts={len(attempts)}/"
        f"{n_cands * n_loops} loops={n_loops} passed={passed}"
    )
    return out, info


# Tactics accepted by a strategy spec. `mlm` is a local masked-LM edit; the
# rest go through the configured rewrite backend.
KNOWN_TACTICS = frozenset(
    {"paraphrase", "backtranslate", "structural", "humanize", "academic", "code", "chunk", "mlm"}
)
LLM_TACTICS = frozenset(KNOWN_TACTICS - {"mlm"})


def parse_strategy(spec: str) -> list[tuple[str, float]]:
    """Parse a strategy like ``"paraphrase@0.8,mlm@0.2"`` -> [(tactic, intensity)].

    Validates tactic names and that intensity lies in (0,1]. Raises ValueError on
    malformed input (callers treat a bad strategy as a request error).
    """
    if not spec or not spec.strip():
        raise ValueError("strategy must be a non-empty list of tactic@intensity steps")
    steps: list[tuple[str, float]] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            raise ValueError(f"bad strategy step {item!r}; expected tactic@intensity")
        if "@" not in item:
            raise ValueError(f"bad strategy step {item!r}; expected tactic@intensity")
        tactic, raw_level = item.rsplit("@", 1)
        tactic = tactic.strip()
        if tactic not in KNOWN_TACTICS:
            raise ValueError(f"unknown strategy tactic {tactic!r}")
        try:
            level = float(raw_level)
        except ValueError:
            raise ValueError(f"bad intensity in strategy step {item!r}") from None
        if not (0 < level <= 1):
            raise ValueError(f"strategy intensity must be in (0,1], got {level} in {item!r}")
        steps.append((tactic, level))
    return steps


def _has_prose(text: str, mask: LatexMask) -> bool:
    """True when *text* holds letters outside the protected placeholders."""
    return re.search(r"[^\W\d_]", mask.token_re.sub("", text)) is not None


def _piece_placeholder_error(piece: str, out: str, mask: LatexMask) -> str | None:
    """Reject an output whose placeholder sequence differs from its input piece's."""
    if not mask.count:
        return None
    if mask.token_re.findall(out) != mask.token_re.findall(piece):
        return "protected math/LaTeX placeholders were lost, duplicated, invented or reordered"
    return None


def apply_strategy(
    text: str,
    steps: list[tuple[str, float]],
    *,
    backend: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    timeout: float = 120.0,
    temperature: float = 0.3,
    reasoning_effort: str | None = None,
    lang: str = "French",
    original_lang: str = "English",
    style: str | None = None,
    layer_a_after: bool = False,
    candidates: int = 1,
    max_loops: int = 1,
    protect_latex: str = "auto",
    noop_lex_floor: float = DEFAULT_NOOP_LEX_FLOOR,
    chunk_chars: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply a strategy's steps sequentially to *text* (best-effort rewrite).

    Unlike ``rewrite()`` this does not run a detection/evaluation loop — each
    step's accepted output feeds the next, so it suits an operation that wants
    the rewrite regardless of a removal verdict. ``mlm`` steps use a local
    masked-LM edit; ``backtranslate`` and ``structural`` make two backend
    generations per piece (``_two_step_generate``); every other tactic makes
    one via ``build_prompt``/``_generate_rewrite``, which strips model wrappers
    (preamble, trailing commentary, fences, quotes) from the output.

    Math, LaTeX commands/environments, citation keys and code spans are masked
    once for the whole strategy (``protect_latex``) and restored at the end.
    An LLM step works piece by piece: the text is cut at paragraph or sentence
    boundaries into pieces of at most *chunk_chars* characters
    (WATERMARKS_REWRITE_CHUNK_CHARS, default DEFAULT_CHUNK_CHARS; 0 disables),
    each rewritten in its own model call with its own checks, so a whole
    section never has to fit one context window and a slip in one paragraph
    never spoils the rest.

    A piece whose rewrite drifted to an extreme length (LENGTH_PRESERVING_TACTICS),
    lost or reordered a protected span, or — for ``academic`` — changed a number,
    the language or the LaTeX structure, is a failed attempt. It is regenerated
    until an attempt is acceptable or the ``candidates * max_loops`` budget is
    spent (the first acceptable attempt wins; the default budget of 1 allows
    no retry). When no attempt is acceptable that piece passes through
    unchanged, so damaged text never reaches the result, and the failure is
    reported (the step is ``ok: false``).

    The no-op guard of ``rewrite()`` applies: a result whose bigram divergence
    from *text* is below *noop_lex_floor* (0 disables) is reported as ``noop``
    with an entry in ``warnings``, and so is any single step that returned ≈
    its input, since neither is a removal attempt.

    Returns (final_text, stats). stats carries per-step tactic/intensity/
    input-output lengths plus ok/attempts/generations/chunks/length_ratio/
    length_checked/wrappers_stripped/rejected/lexical_divergence/noop (and
    error on a failed step), and top-level ok/noop/lexical_divergence/
    warnings/errors/attempt_budget/chunk_chars plus the LaTeX restoration
    report when spans were protected.
    """
    needs_llm = any(t in LLM_TACTICS for t, _ in steps)
    if needs_llm:
        if backend not in ("openai-compatible", "ollama"):
            raise RuntimeError(f"strategy needs an LLM backend, got {backend!r}")
        if not model or not base_url:
            raise RuntimeError("strategy needs --model and --base-url for LLM steps")
    if chunk_chars is None:
        chunk_chars = _env_int("WATERMARKS_REWRITE_CHUNK_CHARS", DEFAULT_CHUNK_CHARS)

    budget = max(1, candidates) * max(1, max_loops)
    # Mask once for the whole strategy: every step then feeds the next with the
    # protected spans already out of reach, and one restore closes the pass.
    cur, mask = mask_latex(text, mode=protect_latex)

    def generate(prompt: str, original: str) -> tuple[str, list[str]]:
        """One backend generation with the strategy's endpoint and sampling."""
        return _generate_rewrite(
            backend,
            base_url,
            model,
            api_key,
            prompt,
            timeout,
            temperature,
            reasoning_effort,
            original=original,
        )

    step_stats: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    for n, (tactic, intensity) in enumerate(steps, start=1):
        step_in = cur
        in_chars = len(cur)
        checked = tactic in LENGTH_PRESERVING_TACTICS
        step_label = f"step {n} ({tactic}@{intensity:g})"
        step_rejected: list[dict[str, Any]] = []
        step_stripped: list[str] = []
        chunk_errors: list[str] = []
        attempts = 0
        generations = 0
        notes: list[str] = []

        if tactic == "mlm":
            # An mlm edit is deterministic: a retry would reproduce the same output.
            pieces = [cur]
            attempts = 1
            out = _mlm_infill(cur, intensity)
            problem = placeholder_error(out, mask)
            if problem:
                step_rejected.append(
                    {
                        "chunk": 1,
                        "out_chars": len(out),
                        "length_ratio": _length_ratio(in_chars, len(out)),
                        "wrappers_stripped": [],
                        "integrity_error": problem,
                    }
                )
                chunk_errors.append(f"{step_label}: {problem}; input kept unchanged.")
            else:
                cur = out
        else:
            pieces = _prose_chunks(cur, chunk_chars)
            outputs: list[str] = []
            for k, piece in enumerate(pieces, start=1):
                core = piece.strip()
                if not core or not _has_prose(core, mask):
                    outputs.append(piece)
                    continue
                # Interior layout survives (each piece keeps the whitespace around
                # it); the very first/last edge follows the single-call contract,
                # where the model's stripped output is the result.
                lead = "" if k == 1 else piece[: len(piece) - len(piece.lstrip())]
                trail = "" if k == len(pieces) else piece[len(piece.rstrip()) :]
                local = len(mask.token_re.findall(core)) if mask.count else 0
                piece_mask = LatexMask(spans=("",) * local, prefix=mask.prefix) if local else None
                label = step_label + (f", chunk {k}/{len(pieces)}" if len(pieces) > 1 else "")
                accepted: tuple[str, list[str]] | None = None
                rejected: list[dict[str, Any]] = []
                for _ in range(budget):
                    attempts += 1
                    if tactic in TWO_STEP_PROMPTS:
                        out, stripped = _two_step_generate(
                            tactic,
                            core,
                            lambda p, _core=core: generate(p, _core),
                            lang=lang,
                            original_lang=original_lang,
                            style=style,
                            mask=piece_mask,
                        )
                        generations += 2
                    else:
                        prompt = build_prompt(
                            tactic,
                            core,
                            lang=lang,
                            original_lang=original_lang,
                            rewrite_level=intensity,
                            style=style,
                            mask=piece_mask,
                        )
                        out, stripped = generate(prompt, core)
                        generations += 1
                    problem = _piece_placeholder_error(core, out, mask)
                    if tactic == "academic":
                        problem = problem or academic_error(core, out)
                    elif mask.count:
                        problem = problem or structure_error(core, out)
                    if problem or (checked and _length_drift(len(core), len(out))):
                        rejected.append(
                            {
                                "chunk": k,
                                "out_chars": len(out),
                                "length_ratio": _length_ratio(len(core), len(out)),
                                "wrappers_stripped": stripped,
                                "integrity_error": problem,
                            }
                        )
                        continue
                    accepted = (out, stripped)
                    break
                step_rejected.extend(rejected)
                if accepted is None:
                    ratios = ", ".join(str(r["length_ratio"]) for r in rejected)
                    error = (
                        f"{label}: no acceptable rewrite in {len(rejected)} attempt(s); output "
                        f"length drifted (output/input ratio {ratios}; accepted "
                        f"{_LENGTH_DRIFT_RANGE}), likely model meta-commentary. The step's "
                        "input was passed through unchanged."
                    )
                    if any(r.get("integrity_error") for r in rejected):
                        reason = next(
                            r["integrity_error"] for r in rejected if r.get("integrity_error")
                        )
                        error = f"{label}: {reason}; input kept unchanged."
                    chunk_errors.append(error)
                    outputs.append(piece)
                    continue
                out, stripped = accepted
                outputs.append(lead + out + trail)
                for kind in stripped:
                    if kind not in step_stripped:
                        step_stripped.append(kind)
                if stripped:
                    notes.append(
                        f"{label}: stripped model meta-commentary from the rewrite "
                        f"({', '.join(stripped)})"
                    )
                if rejected:
                    notes.append(
                        f"{label}: regenerated after {len(rejected)} attempt(s) whose length "
                        "drifted or whose protected content changed"
                    )
            cur = "".join(outputs)

        for error in chunk_errors:
            eprint(f"error: {error}")
        for note in notes:
            eprint(f"warning: {note}")
        errors.extend(chunk_errors)
        warnings.extend(notes)
        step_div = _lexical_divergence(step_in, cur)
        stat: dict[str, Any] = {
            "tactic": tactic,
            "intensity": round(intensity, 4),
            "in_chars": in_chars,
            "out_chars": len(cur),
            "ok": not chunk_errors,
            "attempts": attempts,
            "generations": generations,
            "chunks": len(pieces),
            "length_ratio": _length_ratio(in_chars, len(cur)),
            "length_checked": checked,
            "wrappers_stripped": step_stripped,
            "rejected": step_rejected,
            "lexical_divergence": round(step_div, 4),
            "noop": _below_noop_floor(step_div, noop_lex_floor),
        }
        if chunk_errors:
            stat["error"] = " | ".join(chunk_errors)
        step_stats.append(stat)

    # Layer A scrub once, on the complete strategy output (not per step).
    if layer_a_after and cur:
        cur = clean_text(cur)[0]
    cur, latex_stats = restore_latex(cur, mask)
    if latex_stats["missing"]:
        eprint(
            f"warning: {latex_stats['missing']} of {mask.count} protected "
            "math/LaTeX span(s) were dropped by the rewrite model"
        )

    # No-op guard, judged on the returned text as in rewrite(). When the whole
    # strategy is not a no-op, still name any step that was: a shortcut
    # backtranslate hides behind a later step that did change the text.
    out_div = _lexical_divergence(text, cur)
    noop = _below_noop_floor(out_div, noop_lex_floor)
    noop_notes: list[str] = []
    if noop:
        noop_notes.append(
            f"strategy returned ≈ its input (lexical divergence {out_div:.4f} < "
            f"{noop_lex_floor:.4f}); treated as a no-op, not a rewrite"
        )
    else:
        noop_notes.extend(
            f"step {i} ({st['tactic']}@{st['intensity']:g}) returned ≈ its input "
            f"(lexical divergence {st['lexical_divergence']:.4f} < {noop_lex_floor:.4f})"
            for i, st in enumerate(step_stats, 1)
            if st["noop"]
        )
    for w in noop_notes:
        eprint(f"warning: {w}")
    warnings.extend(noop_notes)

    stats: dict[str, Any] = {
        "backend": backend,
        "tactic": "strategy",
        "mode": "strategy",
        "strategy": [f"{t}@{i:g}" for t, i in steps],
        "steps": step_stats,
        "input_chars": len(text),
        "output_chars": len(cur),
        "ok": not errors,
        "warnings": warnings,
        "errors": errors,
        "attempt_budget": budget,
        "chunk_chars": chunk_chars,
        "lexical_divergence": round(out_div, 4),
        "noop_lex_floor": noop_lex_floor,
        "noop": noop,
        "protect_latex": protect_latex,
        "latex_protected": mask.count,
    }
    if mask.count:
        stats["latex"] = latex_stats
    return cur, stats


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for text rewrite tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", default="-", help="Input text file, or - for stdin")
    p.add_argument("-o", "--output", help="Output path (default: stdout or *.rewritten.*)")
    p.add_argument(
        "--backend",
        choices=("print-prompt", "ollama", "openai-compatible"),
        default=_env("WATERMARKS_REWRITE_BACKEND", "print-prompt"),
    )
    p.add_argument("--model", default=_env("WATERMARKS_REWRITE_MODEL"))
    p.add_argument(
        "--base-url",
        default=_env("WATERMARKS_REWRITE_BASE_URL", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        default=None,
        help="Allow non-loopback rewrite endpoints (default: deny; "
        "WATERMARKS_REWRITE_ALLOW_REMOTE=1 has the same effect)",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "off"),
        default=_env("WATERMARKS_REWRITE_REASONING_EFFORT", "none"),
        help="Reasoning control: sent as reasoning_effort to openai-compatible "
        "backends; for ollama, 'none' sends think=false. 'none' skips "
        "chain-of-thought (reasoning models like deepseek-v4-flash or gemma4 "
        "otherwise burn minutes on a rewrite). 'off' omits the parameter entirely.",
    )
    # NOTE: no --api-key flag on purpose — keys on argv are visible in `ps`
    # and shell history. Set WATERMARKS_REWRITE_API_KEY instead.
    p.add_argument(
        "--tactic",
        choices=(
            "paraphrase",
            "backtranslate",
            "structural",
            "humanize",
            "academic",
            "code",
            "chunk",
            "mlm",
        ),
        default="paraphrase",
    )
    p.add_argument(
        "--protect-latex",
        choices=("auto", "on", "off"),
        default=_env("WATERMARKS_PROTECT_LATEX", "auto"),
        help="Hold mathematics, LaTeX commands/environments and verbatim spans out "
        "of the rewrite by replacing them with placeholders that are restored "
        "afterwards ('auto': whenever the text looks like LaTeX/Markdown math). "
        "In print-prompt mode there is no map to restore, so the prompt carries a "
        "preserve-verbatim instruction instead.",
    )
    p.add_argument(
        "--strategy",
        default=None,
        help="Ordered tactic@intensity strategy to apply (e.g. "
        "'paraphrase@0.8,mlm@0.2'). When set, applies the whole strategy "
        "sequentially (each step feeds the next) instead of a single --tactic; "
        "no detection/evaluation loop. A step whose output length drifts to an "
        "extreme is regenerated within --candidates x --max-loops attempts, "
        "else its input passes through unchanged and the step is reported failed.",
    )
    p.add_argument(
        "--style",
        default=None,
        help="Optional writing-style instruction appended to the rewrite prompt "
        "(e.g. 'write like Hemingway'). Most meaningful with --tactic humanize; "
        "a request, not a guarantee.",
    )
    p.add_argument(
        "--noop-lex-floor",
        type=float,
        default=DEFAULT_NOOP_LEX_FLOOR,
        help="Treat a rewrite that changed fewer than this fraction of bigrams as "
        "a no-op (emitted as noop:true in --json-stats, with a warning; default "
        f"{DEFAULT_NOOP_LEX_FLOOR}, 0 disables). A no-op is not a removal attempt. "
        "Applies to --tactic and --strategy.",
    )
    p.add_argument(
        "--chunk-chars",
        type=int,
        default=_env_int("WATERMARKS_REWRITE_CHUNK_CHARS", DEFAULT_CHUNK_CHARS),
        help="With --strategy, the longest text sent to the model in one call; "
        "longer inputs are split at paragraph or sentence boundaries and each "
        f"piece is rewritten and checked on its own (default {DEFAULT_CHUNK_CHARS}; "
        "WATERMARKS_REWRITE_CHUNK_CHARS; 0 disables chunking).",
    )
    p.add_argument(
        "--rewrite-level",
        type=float,
        default=None,
        help="Numeric rewrite intensity in (0,1]; 0 (the unchanged original) is "
        "excluded. When set alongside --tactic it modulates that tactic's "
        "prompt with an intensity clause (change roughly this fraction of tokens). "
        "Omit to use the plain --tactic prompt. Planned nominal default 0.5, to "
        "be tuned from benchmark output.",
    )
    p.add_argument(
        "--target-margin",
        type=float,
        default=0.0,
        help="Require a detection to sit at least this far below the threshold "
        "to count as a pass (robustness floor; default 0.0 = any not-watermarked "
        "verdict). Margin = threshold - score.",
    )
    p.add_argument(
        "--select",
        choices=("min-divergence", "max-margin"),
        default="min-divergence",
        help="Among candidates that pass --target-margin, select the one that "
        "changed the least (min-divergence, the content-preserving default) or "
        "the one with the largest score margin (max-margin, robustness-first).",
    )
    p.add_argument(
        "--chunk-shuffle",
        action="store_true",
        help="With --tactic chunk, shuffle the rewritten fragments (breaks "
        "cross-fragment context ordering; destroys document coherence, so "
        "opt-in)",
    )
    p.add_argument("--lang", default="French", help="Pivot language for backtranslate")
    p.add_argument("--original-lang", default="English")
    p.add_argument(
        "--timeout",
        type=float,
        default=float(_env("WATERMARKS_REWRITE_TIMEOUT", "120.0")),
        help="Seconds to wait for each backend call (default 120; WATERMARKS_REWRITE_TIMEOUT)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the rewrite backend",
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=_env_int("WATERMARKS_REWRITE_CANDIDATES", DEFAULT_CANDIDATES),
        help="Variants generated per loop iteration; each variant is one "
        "rewrite + one evaluation, and the loop stops as soon as an attempt "
        f"passes (default: {DEFAULT_CANDIDATES}; WATERMARKS_REWRITE_CANDIDATES)",
    )
    p.add_argument(
        "--max-loops",
        type=int,
        default=_env_int("WATERMARKS_REWRITE_LOOPS", DEFAULT_MAX_LOOPS),
        help="Max evaluation rounds; each round generates --candidates "
        "variants and stops when one passes. Raising this retries new "
        f"variants until an evaluation passes (default: {DEFAULT_MAX_LOOPS}; "
        "WATERMARKS_REWRITE_LOOPS)",
    )
    p.add_argument(
        "--no-layer-a-after",
        action="store_true",
        help="Skip Layer A scrub on model output",
    )
    p.add_argument("--json-stats", action="store_true", help="Stats JSON on stderr")
    p.add_argument(
        "--markllm-scheme",
        # Keep in sync with detect_text_watermark.SCHEMES.
        choices=("kgw", "synthid", "synthid-text", "exp", "unigram", "sir"),
        default=None,
        help="Run MarkLLM before/after detection around the rewrite AND drive "
        "the iterative rewrite loop with it (the evaluator, when configured; "
        "otherwise lexical divergence selects the best variant). "
        "Scheme = any key of detect_text_watermark.SCHEMES.",
    )
    p.add_argument(
        "--markllm-dir",
        default=_env("MARKLLM_DIR"),
        help="MarkLLM checkout root (default: $MARKLLM_DIR)",
    )
    p.add_argument(
        "--markllm-model",
        default=_env("MARKLLM_MODEL", DEFAULT_MARKLLM_MODEL),
        help=f"Scoring model for MarkLLM detection (default: $MARKLLM_MODEL or {DEFAULT_MARKLLM_MODEL})",
    )
    p.add_argument(
        "--markllm-timeout",
        type=float,
        default=float(_env("WATERMARKS_MARKLLM_TIMEOUT", "180.0")),
        help="Timeout per MarkLLM detection call (default: 180.0)",
    )
    p.add_argument(
        "--gumbel-key",
        default=_env("WATERMARKS_GUMBEL_KEY"),
        help="Secret key for keyed-Gumbel (Aaronson EXP) same-key replay "
        "detection; drives the iterative rewrite loop as the evaluator when "
        "set (default: $WATERMARKS_GUMBEL_KEY). Preferred via env — keys on "
        "argv are visible in ps/history; never logged.",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Rewrite even when the input looks like a binary container",
    )
    return p


def main() -> int:
    """CLI entry point."""
    args = build_parser().parse_args()

    if args.rewrite_level is not None and not (0 < args.rewrite_level <= 1):
        eprint(f"error: --rewrite-level must be in (0,1], got {args.rewrite_level}")
        return 2

    text = read_text_input(args.path, allow_binary=args.force_text)
    allow_remote = (
        args.allow_remote
        if args.allow_remote is not None
        else _flag_env("WATERMARKS_REWRITE_ALLOW_REMOTE")
    )
    steps: list[tuple[str, float]] | None = None
    if args.strategy:
        try:
            steps = parse_strategy(args.strategy)
        except ValueError as e:
            eprint(f"error: {e}")
            return 2
    try:
        if steps is not None:
            # Enforce the remote-endpoint policy for a strategy, same as the
            # single-tactic rewrite path.
            if any(t in LLM_TACTICS for t, _ in steps) and args.base_url:
                _check_remote(args.base_url, allow_remote)
            result, info = apply_strategy(
                text,
                steps,
                backend=args.backend,
                model=args.model,
                base_url=args.base_url,
                api_key=_env("WATERMARKS_REWRITE_API_KEY"),
                timeout=args.timeout,
                temperature=args.temperature,
                reasoning_effort=(
                    None if args.reasoning_effort == "off" else args.reasoning_effort
                ),
                lang=args.lang,
                original_lang=args.original_lang,
                style=args.style,
                layer_a_after=not args.no_layer_a_after,
                candidates=args.candidates,
                max_loops=args.max_loops,
                protect_latex=args.protect_latex,
                noop_lex_floor=args.noop_lex_floor,
                chunk_chars=args.chunk_chars,
            )
        else:
            result, info = rewrite(
                text,
                backend=args.backend,
                model=args.model,
                base_url=args.base_url,
                api_key=_env("WATERMARKS_REWRITE_API_KEY"),
                tactic=args.tactic,
                style=args.style,
                lang=args.lang,
                original_lang=args.original_lang,
                timeout=args.timeout,
                layer_a_after=not args.no_layer_a_after,
                temperature=args.temperature,
                candidates=args.candidates,
                max_loops=args.max_loops,
                allow_remote=allow_remote,
                reasoning_effort=(
                    None if args.reasoning_effort == "off" else args.reasoning_effort
                ),
                markllm_scheme=args.markllm_scheme,
                markllm_dir=args.markllm_dir,
                markllm_model=args.markllm_model,
                markllm_timeout=args.markllm_timeout,
                gumbel_key=args.gumbel_key,
                rewrite_level=args.rewrite_level,
                target_margin=args.target_margin,
                selection=args.select,
                chunk_shuffle=args.chunk_shuffle,
                noop_lex_floor=args.noop_lex_floor,
                protect_latex=args.protect_latex,
            )
    except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
        eprint(f"rewrite failed: {e}")
        return 1

    out = args.output
    if out is None and args.path not in (None, "-") and args.backend != "print-prompt":
        out = str(cleaned_path(Path(args.path), suffix=".rewritten"))
    elif out is None and args.backend == "print-prompt":
        out = "-"

    write_text_output(result, out)
    if args.json_stats:
        eprint(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        eprint(
            f"backend={info['backend']} tactic={info['tactic']} "
            f"mode={info.get('mode')} evaluator={info.get('evaluator', '-')} "
            f"attempts={info.get('attempts_made', '-')} passed={info.get('passed', '-')} "
            f"chars {info['input_chars']}->{info.get('output_chars', len(result))}"
        )
    return 0 if info.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
