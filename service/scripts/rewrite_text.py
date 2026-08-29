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
  WATERMARKS_REWRITE_CANDIDATES   (default 1; variants generated per loop)
  WATERMARKS_REWRITE_LOOPS        (default 1; max evaluation rounds)

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

The rewrite instruction comes from --strength (a named prompt) or, when
--rewrite-level is set, a numeric rewrite intensity in (0,1] that controls how
many tokens change (0 — the unchanged original — is excluded; 1 rewrites
everything). The level is a request: output lexical/semantic divergence is
measured, not guaranteed.

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
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cleaned_path, eprint, read_text_input, write_text_output
from text_detectors import GumbelTextDetector, MarkLLMTextDetector
from text_unicode import clean_text

DEFAULT_MARKLLM_MODEL = "facebook/opt-1.3b"
DEFAULT_CANDIDATES = 1
DEFAULT_MAX_LOOPS = 1

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
        "Vary sentence rhythm and length, replace formulaic AI-style transitions and "
        "filler with concrete natural phrasing, and use plain, varied wording. Preserve "
        "all facts, numbers, names, and technical identifiers. Do not add or remove "
        "claims. Output only the rewritten text.\n\n---\n{TEXT}"
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
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
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


def _select_candidate(original: str, candidates: list[str]) -> tuple[str, list[float]]:
    """Pick the most lexically diverged rewrite, gently guarding extreme length drift."""
    scores: list[float] = []
    for cand in candidates:
        score = _lexical_divergence(original, cand)
        if original:
            ratio = len(cand) / len(original)
            if ratio > 2.0 or ratio < 0.5:
                score -= 0.15
        scores.append(score)
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], scores


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _flag_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
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
        return call_ollama(base_url, model, prompt, timeout, temperature)
    if backend == "openai-compatible":
        return call_openai_compatible(
            base_url, model, prompt, api_key, timeout, temperature, reasoning_effort
        )
    raise SystemExit(f"unknown backend: {backend}")


def build_prompt(
    strength: str,
    text: str,
    *,
    lang: str,
    original_lang: str,
    rewrite_level: float | None = None,
) -> str:
    if rewrite_level is not None:
        return PROMPTS["level"].format(TEXT=text, LEVEL=rewrite_level)
    if strength == "paraphrase":
        return PROMPTS["paraphrase"].format(TEXT=text)
    if strength == "humanize":
        return PROMPTS["humanize"].format(TEXT=text)
    if strength == "code":
        return PROMPTS["code"].format(TEXT=text)
    if strength == "backtranslate":
        # single combined instruction for print-prompt / one-shot backends
        return (
            f"Translate the text to {lang}, then translate that result back to "
            f"{original_lang}. Preserve all facts, numbers, and names. "
            f"Output only the final {original_lang} text.\n\n---\n{text}"
        )
    if strength == "structural":
        return (
            "First extract a bullet outline of all claims (no full sentences). "
            "Then write a complete document from that outline in natural, varied human "
            "prose without omitting any bullet. Output only the final document.\n\n---\n"
            f"{text}"
        )
    if strength == "chunk":
        return PROMPTS["chunk_unit"].format(TEXT=text)
    raise ValueError(f"unknown strength: {strength}")


def _split_units(text: str) -> list[tuple[str, str]]:
    """Split a document into (unit, separator) pairs.

    Breaks after sentence punctuation or on any blank-line / newline run, so
    each fragment is rewritten independently (a fresh context per fragment ⇒
    new per-token watermark keys). Punctuation is kept with its fragment. The
    separator is the whitespace/blank-line run that follows a unit ('' for the
    last); unshuffled chunk mode reassembles with it so paragraph/line layout
    is preserved, while shuffled mode drops it.
    """
    parts = re.split(r"((?<=[.!?])\s+|\n+)", text)
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        unit = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if unit.strip() or "\n" in sep:
            pairs.append((unit.strip(), sep))
        i += 2
    return pairs


def _http_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
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


def call_ollama(base_url: str, model: str, prompt: str, timeout: float, temperature: float) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    data = _http_json(
        url,
        {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature},
        },
        {},
        timeout,
    )
    msg = data.get("message") or {}
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
    strength: str,
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
    target_margin: float = 0.0,
    selection: str = "min-divergence",
    chunk_shuffle: bool = False,
) -> tuple[str, dict]:
    prompt = build_prompt(
        strength, text, lang=lang, original_lang=original_lang, rewrite_level=rewrite_level
    )
    info: dict = {
        "backend": backend,
        "strength": strength,
        "rewrite_level": rewrite_level,
        "target_margin": target_margin,
        "selection": selection,
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

    if not model:
        raise SystemExit("error: --model required for ollama/openai-compatible backends")
    if not base_url:
        raise SystemExit("error: --base-url required for ollama/openai-compatible backends")

    _check_remote(base_url, allow_remote)

    n_cands = max(1, candidates)
    n_loops = max(1, max_loops)
    info["candidates"] = n_cands
    info["max_loops"] = n_loops
    evaluator_name, evaluator = _pick_evaluator(markllm_detector, gumbel_detector)
    info["evaluator"] = evaluator_name

    is_chunk = strength == "chunk"
    info["chunked"] = is_chunk
    info["chunk_shuffle"] = bool(chunk_shuffle)

    def _rewrite_unit(unit: str) -> str:
        return _generate_once(
            backend,
            base_url,
            model,
            api_key,
            build_prompt(
                strength,
                unit,
                lang=lang,
                original_lang=original_lang,
                rewrite_level=rewrite_level,
            ),
            timeout,
            temperature,
            reasoning_effort,
        )

    def _generate_candidate() -> str:
        if is_chunk:
            pairs = _split_units(text)
            if chunk_shuffle:
                units = [unit for unit, _ in pairs if unit]
                random.shuffle(units)
                return " ".join(_rewrite_unit(unit) for unit in units)
            # Skip rewriting empty leading units (blank lines at the top) but
            # keep their separators so the reassembled document keeps the layout.
            return "".join((_rewrite_unit(unit) if unit else "") + sep for unit, sep in pairs)
        return _generate_once(
            backend, base_url, model, api_key, prompt, timeout, temperature, reasoning_effort
        )

    # Iterative rewrite: each loop generates --candidates variants and
    # evaluates them ALL (so the best is chosen, not the first to squeak under
    # the threshold). A variant "passes" when the detector reports it
    # not-watermarked AND its score sits at least --target-margin below the
    # threshold; --max-loops caps the evaluation rounds before the best-effort
    # variant is returned. When no detector is configured the evaluator is
    # lexical divergence, which has no pass/fail verdict, so every attempt is
    # generated and the most diverged one is selected (an unguided best-effort).
    attempts: list[tuple[str, dict]] = []
    passed: bool | None = None
    for loop in range(n_loops):
        loop_passed = False
        for _ in range(n_cands):
            cand = _generate_candidate()
            cand_stats: dict | None = None
            if layer_a_after:
                cand, cand_stats = clean_text(cand)
            divergence = _lexical_divergence(text, cand)
            if evaluator is None:
                evaluation: dict = {
                    "evaluator": "lexical-divergence",
                    "score": round(divergence, 4),
                }
            else:
                evaluation = _safe_detect(evaluator, cand)
            passed_i, margin, raw_margin = _candidate_pass(evaluation, target_margin)
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

    # Best-effort selection: among the candidates that passed (met the margin
    # objective) pick the one that changed the least (min-divergence, the
    # content-preserving default) or the one with the largest margin
    # (--select max-margin, robustness-first). When none passed, fall back to
    # the lowest watermark score (detector evaluator) or the most diverged
    # variant (unguided fallback).
    selected_idx: int
    best_score: float | None = None
    best_score_idx: int | None = None
    best_div = -1.0
    best_div_idx = 0
    for i, (_cand, rec) in enumerate(attempts):
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
    else:
        selected_idx = best_div_idx

    out, rec = attempts[selected_idx]
    rec["selected"] = True
    info["candidate_scores"] = [r for _c, r in attempts]
    if layer_a_after:
        info["layer_a_after"] = rec["layer_a_after"]
    info["output_chars"] = len(out)
    info["mode"] = "rewritten"
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


def build_parser() -> argparse.ArgumentParser:
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
        help="OpenAI-compatible reasoning_effort; 'none' skips chain-of-thought "
        "(reasoning models like deepseek-v4-flash otherwise burn minutes on a "
        "rewrite). 'off' omits the parameter entirely.",
    )
    # NOTE: no --api-key flag on purpose — keys on argv are visible in `ps`
    # and shell history. Set WATERMARKS_REWRITE_API_KEY instead.
    p.add_argument(
        "--strength",
        choices=("paraphrase", "backtranslate", "structural", "humanize", "code", "chunk"),
        default="paraphrase",
    )
    p.add_argument(
        "--rewrite-level",
        type=float,
        default=None,
        help="Numeric rewrite intensity in (0,1]; 0 (the unchanged original) is "
        "excluded. When set, overrides --strength and builds a level-based prompt "
        "that changes a fraction of tokens close to this value. Omit to use "
        "--strength (the benchmark's minimal mode drives this explicitly). "
        "Planned nominal default 0.5, to be tuned from benchmark output.",
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
        help="With --strength chunk, shuffle the rewritten fragments (breaks "
        "cross-fragment context ordering; destroys document coherence, so "
        "opt-in)",
    )
    p.add_argument("--lang", default="French", help="Pivot language for backtranslate")
    p.add_argument("--original-lang", default="English")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.9,
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
    try:
        result, info = rewrite(
            text,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            api_key=_env("WATERMARKS_REWRITE_API_KEY"),
            strength=args.strength,
            lang=args.lang,
            original_lang=args.original_lang,
            timeout=args.timeout,
            layer_a_after=not args.no_layer_a_after,
            temperature=args.temperature,
            candidates=args.candidates,
            max_loops=args.max_loops,
            allow_remote=allow_remote,
            reasoning_effort=(None if args.reasoning_effort == "off" else args.reasoning_effort),
            markllm_scheme=args.markllm_scheme,
            markllm_dir=args.markllm_dir,
            markllm_model=args.markllm_model,
            markllm_timeout=args.markllm_timeout,
            gumbel_key=args.gumbel_key,
            rewrite_level=args.rewrite_level,
            target_margin=args.target_margin,
            selection=args.select,
            chunk_shuffle=args.chunk_shuffle,
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
            f"backend={info['backend']} strength={info['strength']} "
            f"mode={info.get('mode')} evaluator={info.get('evaluator', '-')} "
            f"attempts={info.get('attempts_made', '-')} passed={info.get('passed', '-')} "
            f"chars {info['input_chars']}->{info.get('output_chars', len(result))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
