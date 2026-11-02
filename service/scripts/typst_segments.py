"""Typst source support: markup/code segmenter, document metadata, compile check.

Typst files are UTF-8 text with a markup mode, a code mode (entered with ``#``),
a math mode (``$...$``), raw blocks (backticks) and comments. Layer A (invisible
Unicode) is syntax-blind and runs on the whole file like any other text format.
Layer B (statistical rewrite) must never touch syntax, so it only ever sees the
*prose* segments returned here.

Design rule: when the scanner is unsure about a region it is **protected**. An
unterminated construct protects everything to the end of the file.

Invariant (tested): ``"".join(s.text for s in segment(text)) == text``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from common import subprocess_creationflags

TYPST_EXTS = frozenset({".typ"})

PROSE = "prose"
PROTECTED = "protected"

_IDENT_START = re.compile(r"[A-Za-z_]")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_LABEL = re.compile(r"<[A-Za-z_][A-Za-z0-9_.:-]*>")
_REF = re.compile(r"@[A-Za-z_][A-Za-z0-9_.:-]*")
_URL = re.compile(r"https?://\S*")
_UNICODE_ESC = re.compile(r"\\u\{[0-9A-Fa-f]+\}")
_LIST_MARKER = re.compile(r"[ \t]*(?:=+|-|\+|/|[0-9]+\.)(?=[ \t])")

#: Keywords whose ``#`` expression is a statement running to the end of its line
#: (extended while brackets are open).
_STATEMENT_KW = frozenset({"let", "set", "show", "import", "include"})
#: Keywords that chain blocks: ``#if c [..] else [..]``.
_BLOCK_KW = frozenset({"if", "for", "while", "context", "else"})

#: Characters that carry markup meaning. A rewrite that *introduces* one of
#: these into a prose span is rejected (see ``rewrite_prose``).
MARKUP_SPECIALS = frozenset("*_~\\#$`<>@[]{}")
#: First characters that would turn a rewritten line start into a list/heading.
_LINE_START_UNSAFE = frozenset("=-+/#0123456789")


@dataclass(frozen=True)
class Segment:
    """A contiguous slice of the source, either prose or protected."""

    kind: str
    text: str


# --------------------------------------------------------------------------- #
# Low-level scanners. Each takes (text, i) and returns the index just past the
# construct, or len(text) when unterminated (=> protect to end of file).
# --------------------------------------------------------------------------- #


def _skip_line_comment(text: str, i: int) -> int:
    end = text.find("\n", i)
    return len(text) if end == -1 else end


def _skip_block_comment(text: str, i: int) -> int:
    """Skip a (nestable) ``/* ... */`` comment starting at *i*."""
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith("/*", i):
            depth += 1
            i += 2
        elif text.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return n


def _skip_string(text: str, i: int) -> int:
    """Skip a ``"..."`` string literal starting at the opening quote."""
    n = len(text)
    i += 1
    while i < n:
        if text[i] == "\\":
            i += 2
        elif text[i] == '"':
            return i + 1
        else:
            i += 1
    return n


def _skip_raw(text: str, i: int) -> int:
    """Skip a raw block opened by a run of backticks at *i*."""
    n = len(text)
    j = i
    while j < n and text[j] == "`":
        j += 1
    run = j - i
    if run == 2:  # empty raw: `` is an empty raw element
        return j
    while j < n:
        if text[j] == "`":
            k = j
            while k < n and text[k] == "`":
                k += 1
            if k - j == run:
                return k
            j = k
        else:
            j += 1
    return n


def _skip_math(text: str, i: int) -> int:
    """Skip ``$...$`` math starting at the opening dollar."""
    n = len(text)
    i += 1
    while i < n:
        if text[i] == "\\":
            i += 2
        elif text[i] == "$":
            return i + 1
        else:
            i += 1
    return n


_CLOSERS = {"(": ")", "[": "]", "{": "}"}


def _skip_group(text: str, i: int) -> int:
    """Skip a balanced ``(..)``, ``[..]`` or ``{..}`` group starting at *i*.

    Strings, raw blocks, math, and comments inside the group are skipped so
    brackets in them do not count. Unbalanced input returns ``len(text)``.
    """
    n = len(text)
    stack: list[str] = []
    while i < n:
        ch = text[i]
        if ch in _CLOSERS:
            stack.append(_CLOSERS[ch])
            i += 1
        elif ch in ")]}":
            if not stack or stack[-1] != ch:
                return n  # mismatched: uncertain, protect the rest
            stack.pop()
            i += 1
            if not stack:
                return i
        elif ch == '"':
            i = _skip_string(text, i)
        elif ch == "`":
            i = _skip_raw(text, i)
        elif ch == "$":
            i = _skip_math(text, i)
        elif text.startswith("//", i):
            i = _skip_line_comment(text, i)
        elif text.startswith("/*", i):
            i = _skip_block_comment(text, i)
        elif ch == "\\":
            i += 2
        else:
            i += 1
    return n


def _skip_to_statement_end(text: str, i: int) -> int:
    """From *i*, consume to end of line, continuing while brackets are open."""
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            return i
        if ch in _CLOSERS:
            i = _skip_group(text, i)
        elif ch == '"':
            i = _skip_string(text, i)
        elif ch == "`":
            i = _skip_raw(text, i)
        elif ch == "$":
            i = _skip_math(text, i)
        elif text.startswith("//", i):
            return _skip_line_comment(text, i)
        elif text.startswith("/*", i):
            i = _skip_block_comment(text, i)
        else:
            i += 1
    return n


def _skip_hash_expr(text: str, i: int) -> int:
    """Skip the code expression introduced by the ``#`` at *i*.

    Content blocks (``#strong[...]``) are protected whole: their prose is not
    rewritten. That loses some rewritable text but can never corrupt syntax.
    """
    n = len(text)
    j = i + 1
    if j >= n:
        return n
    if text[j] in "({[":
        return _skip_chain(text, j)
    m = _IDENT.match(text, j)
    if not m:
        return j  # lone '#': protect only the hash
    word = m.group(0)
    j = m.end()
    if word in _STATEMENT_KW:
        return _skip_to_statement_end(text, j)
    if word in _BLOCK_KW:
        return _skip_block_expr(text, j)
    return _skip_chain(text, j)


_WS = re.compile(r"[ \t]*(?:\n[ \t]*)?")


def _skip_block_head(text: str, j: int) -> int:
    """Skip a condition/iterator head up to (not into) its ``{``/``[`` block.

    If the head runs to a newline first the scanner is unsure, so everything up
    to that newline is protected and no block is entered.
    """
    n = len(text)
    while j < n:
        ch = text[j]
        if ch in "{[":
            return j
        if ch == "\n":
            return j
        if ch == "(":
            j = _skip_group(text, j)
        elif ch == '"':
            j = _skip_string(text, j)
        else:
            j += 1
    return n


def _skip_block_expr(text: str, j: int) -> int:
    """Skip ``#if c [..] else if d {..} else [..]`` style chains from after the keyword."""
    n = len(text)
    while True:
        j = _skip_block_head(text, j)
        if j >= n or text[j] not in "{[":
            return j
        j = _skip_group(text, j)
        k = _WS.match(text, j).end()
        if text.startswith("else", k) and not _IDENT.match(text, k + 4):
            j = k + 4
            continue
        return j


def _skip_chain(text: str, j: int) -> int:
    """Consume ``(..)``/``[..]``/``{..}``/``.field`` postfix pieces from *j*."""
    n = len(text)
    while j < n:
        ch = text[j]
        if ch in _CLOSERS:
            j = _skip_group(text, j)
        elif ch == "." and j + 1 < n and _IDENT_START.match(text[j + 1]):
            j = _IDENT.match(text, j + 1).end()
        else:
            break
    return j


# --------------------------------------------------------------------------- #
# Segmenter
# --------------------------------------------------------------------------- #


def _protected_at(text: str, i: int, at_line_start: bool) -> int:
    """Return the end of a protected construct starting at *i*, or ``i``."""
    ch = text[i]
    if text.startswith("//", i):
        return _skip_line_comment(text, i)
    if text.startswith("/*", i):
        return _skip_block_comment(text, i)
    if ch == "#":
        return _skip_hash_expr(text, i)
    if ch == "$":
        return _skip_math(text, i)
    if ch == "`":
        return _skip_raw(text, i)
    if ch == "\\":
        m = _UNICODE_ESC.match(text, i)
        return m.end() if m else min(i + 2, len(text))
    if ch == "<":
        m = _LABEL.match(text, i)
        return m.end() if m else i
    if ch == "@":
        m = _REF.match(text, i)
        # An '@' glued to a word is an e-mail address, not a reference.
        if m and (i == 0 or not text[i - 1].isalnum()):
            return m.end()
        return i
    if ch in "hH":
        m = _URL.match(text, i)
        if m and (i == 0 or not text[i - 1].isalnum()):
            return m.end()
        return i
    if ch in "*_~[]{}":
        return i + 1
    if text.startswith("-?", i):
        return i + 2
    if at_line_start:
        m = _LIST_MARKER.match(text, i)
        if m:
            return m.end()
    return i


def segment(text: str) -> list[Segment]:
    """Split Typst source into protected and prose segments.

    Prose segments never contain a newline, so a rewrite cannot merge lines or
    paragraphs. Concatenating every segment reproduces *text* exactly.
    """
    segs: list[Segment] = []
    n = len(text)
    i = 0
    prose_start = 0
    line_start = True

    def flush(upto: int) -> None:
        nonlocal prose_start
        if upto > prose_start:
            segs.append(Segment(PROSE, text[prose_start:upto]))
        prose_start = upto

    while i < n:
        ch = text[i]
        if ch == "\n":
            flush(i)
            segs.append(Segment(PROTECTED, "\n"))
            i += 1
            prose_start = i
            line_start = True
            continue
        if line_start and ch in " \t":
            j = i
            while j < n and text[j] in " \t":
                j += 1
            m = _LIST_MARKER.match(text, i)
            if not m:
                i = j
                continue  # indentation stays inside prose; no marker here
        end = _protected_at(text, i, line_start)
        if end > i:
            flush(i)
            # A protected region may span newlines (comments, raw, code).
            segs.append(Segment(PROTECTED, text[i:end]))
            i = end
            prose_start = i
            line_start = text[i - 1] == "\n"
            continue
        line_start = False
        i += 1
    flush(n)
    return _merge(segs)


def _merge(segs: list[Segment]) -> list[Segment]:
    """Merge adjacent protected segments so callers see maximal regions."""
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].kind == s.kind == PROTECTED:
            out[-1] = Segment(PROTECTED, out[-1].text + s.text)
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Layer B adapter
# --------------------------------------------------------------------------- #


def _rewrite_is_safe(original: str, rewritten: str, at_line_start: bool) -> bool:
    """Reject rewrites that could change Typst structure (uncertain => keep)."""
    if "\n" in rewritten or "\r" in rewritten:
        return False
    if not rewritten.strip():
        return False
    introduced = (set(rewritten) & MARKUP_SPECIALS) - set(original)
    if introduced:
        return False
    return not (at_line_start and rewritten[0] in _LINE_START_UNSAFE)


def rewrite_prose(
    text: str, rewrite_fn: Callable[[str], str], *, segments: list[Segment] | None = None
) -> tuple[str, dict]:
    """Apply *rewrite_fn* to prose segments only; return (new_text, stats).

    Leading/trailing whitespace of each prose segment is kept verbatim and only
    the core is passed to *rewrite_fn*. Segments without a letter are skipped.
    A rewrite that fails ``_rewrite_is_safe`` is discarded and the original
    segment is kept.
    """
    segs = segments if segments is not None else segment(text)
    out: list[str] = []
    rewritten = rejected = 0
    at_line_start = True
    for s in segs:
        if s.kind == PROTECTED:
            out.append(s.text)
            at_line_start = s.text.endswith("\n")
            continue
        core = s.text.strip()
        if not core or not any(c.isalpha() for c in core):
            out.append(s.text)
            at_line_start = False
            continue
        lead = s.text[: len(s.text) - len(s.text.lstrip())]
        trail = s.text[len(s.text.rstrip()) :]
        new = rewrite_fn(core)
        if new != core and _rewrite_is_safe(core, new, at_line_start and not lead.strip()):
            out.append(lead + new + trail)
            rewritten += 1
        else:
            out.append(s.text)
            rejected += new != core
        at_line_start = False
    return "".join(out), {"prose_rewritten": rewritten, "prose_rejected": rejected}


def prose_text(text: str) -> str:
    """All rewritable prose joined by newlines (used to build a print-prompt)."""
    return "\n".join(
        s.text.strip()
        for s in segment(text)
        if s.kind == PROSE and any(c.isalpha() for c in s.text)
    )


# --------------------------------------------------------------------------- #
# Document metadata (#set document(...))
# --------------------------------------------------------------------------- #

_DOC_SET = re.compile(r"#set\s+document\s*\(")
_ARG = re.compile(r'(\w+)\s*:\s*("(?:[^"\\]|\\.)*"|\([^()]*\)|[^,)\s]+)')
_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')
META_FIELDS = ("title", "author", "keywords", "description", "date")


def typst_metadata(text: str) -> dict:
    """Report author/title data from ``#set document(...)`` rules (read-only).

    Only rules in real markup are considered: matches inside comments, raw
    blocks, or strings are ignored because they come from protected segments
    that do not start with ``#set``.
    """
    found: list[dict] = []
    for seg in segment(text):
        if seg.kind != PROTECTED or not _DOC_SET.match(seg.text):
            continue
        m = _DOC_SET.match(seg.text)
        end = _skip_group(seg.text, m.end() - 1)
        args = seg.text[m.end() : end - 1]
        fields: dict[str, object] = {}
        for key, value in _ARG.findall(args):
            if key not in META_FIELDS:
                continue
            strings = _STR.findall(value)
            fields[key] = (
                strings
                if len(strings) > 1 or value.startswith("(")
                else (strings[0] if strings else value)
            )
        found.append(fields)
    merged: dict[str, object] = {}
    for f in found:
        merged.update(f)
    return {"present": bool(found), "rules": len(found), "fields": merged}


def format_metadata(meta: dict) -> str:
    """Human-readable lines for ``typst_metadata`` output."""
    if not meta["present"]:
        return "Typst document metadata: none (no #set document(...) rule)"
    lines = ["Typst document metadata (#set document, not removed):"]
    for key in META_FIELDS:
        if key in meta["fields"]:
            lines.append(f"  {key}: {meta['fields'][key]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional compile check
# --------------------------------------------------------------------------- #

COMPILE_TIMEOUT_S = 60


def _compiles(source: str, directory: Path) -> tuple[bool, str]:
    """Compile *source* with the typst binary from a scratch file in *directory*.

    The scratch file sits next to the original so relative ``#import`` /
    ``#include`` paths resolve identically to the real file. Removed afterwards.
    """
    typst = shutil.which("typst")
    if typst is None:
        return False, "typst not found"
    handle, name = tempfile.mkstemp(prefix=".wm-typst-check-", suffix=".typ", dir=directory)
    scratch = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write(source)
        with tempfile.TemporaryDirectory(prefix="wm-typst-out-") as out_dir:
            proc = subprocess.run(
                [typst, "compile", str(scratch), str(Path(out_dir) / "out.pdf")],
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
                check=False,
                creationflags=subprocess_creationflags,
            )
        return proc.returncode == 0, proc.stderr.strip()[:500]
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"typst could not run: {e}"
    finally:
        scratch.unlink(missing_ok=True)


def compile_check(original: str, cleaned: str, directory: Path) -> dict:
    """Verify *cleaned* still compiles when *original* did.

    Never a hard dependency: without ``typst`` on PATH the check is skipped.
    Statuses: ``skipped``, ``unchanged`` (nothing to verify), ``ok``,
    ``original_does_not_compile`` (cannot judge), ``regression``.
    """
    if shutil.which("typst") is None:
        return {"status": "skipped", "reason": "typst binary not found on PATH"}
    if original == cleaned:
        return {"status": "unchanged"}
    ok_after, err_after = _compiles(cleaned, directory)
    if ok_after:
        return {"status": "ok"}
    ok_before, _ = _compiles(original, directory)
    if not ok_before:
        return {"status": "original_does_not_compile", "detail": err_after}
    return {"status": "regression", "detail": err_after}
