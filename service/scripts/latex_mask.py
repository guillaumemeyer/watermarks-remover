#!/usr/bin/env python3
"""Protect mathematics, LaTeX control structures and verbatim spans from Layer B.

The Layer B rewrite hands prose to a generative model, which is free to reword
anything it is given. In academic sources that freedom is a hazard rather than a
feature: an index, a sign, a citation key or an environment delimiter carries no
token-sampling watermark worth attacking, and a model that "improves" one
corrupts the document silently.

This module lifts those spans out of the text before the rewrite and puts them
back afterwards. Each protected span becomes an opaque placeholder
(``[[WMX0001]]``) that survives a paraphrase, a masked-LM infill (the token
carries digits, so ``_mlm_infill`` never selects it) and the deterministic
humanizer pass (which would otherwise turn a LaTeX en dash ``--`` into a comma).

What is protected: display and inline math (``$...$``, ``$$...$$``, ``\\[...\\]``,
``\\(...\\)``), the math/verbatim/layout environments listed in
``PROTECTED_ENVIRONMENTS``, reference-like commands (``\\cite``, ``\\ref``,
``\\label``, ``\\input``, ...), Markdown code fences and inline code spans, YAML front matter and the
URL part of Markdown links and images.
Prose environments (``abstract``, ``itemize``, ``theorem``, ...) are deliberately
left alone so the rewrite still reaches the text inside them.

Restoration is reported, never assumed: a model can drop or duplicate a
placeholder, and ``restore_latex`` counts both instead of pretending the round
trip was lossless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Environments whose body is notation, layout or verbatim content rather than
#: prose. A trailing ``*`` variant is matched for every entry.
PROTECTED_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        # amsmath and friends
        "equation",
        "align",
        "alignat",
        "gather",
        "multline",
        "flalign",
        "eqnarray",
        "displaymath",
        "math",
        "split",
        "subequations",
        "cases",
        "array",
        "matrix",
        "pmatrix",
        "bmatrix",
        "Bmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
        "IEEEeqnarray",
        "dmath",
        # verbatim / code
        "verbatim",
        "Verbatim",
        "lstlisting",
        "minted",
        "alltt",
        # drawing and tabular layout: rewriting an alignment breaks the document
        "tikzpicture",
        "pgfpicture",
        "tabular",
        "tabularx",
        "longtable",
        "algorithmic",
    }
)

#: Commands whose arguments are identifiers (keys, labels, paths), not prose.
PROTECTED_COMMANDS: tuple[str, ...] = (
    r"cite[A-Za-z]*",
    r"[Cc]ref",
    r"autoref",
    r"eqref",
    r"pageref",
    r"nameref",
    r"ref",
    r"label",
    r"bibliographystyle",
    r"bibliography",
    r"addbibresource",
    r"input",
    r"include(?:graphics)?",
    r"usepackage",
    r"documentclass",
    r"(?:re)?newcommand",
    r"DeclareMathOperator\*?",
    r"url",
    r"href",
)

#: Accepted values for the protection mode (CLI ``--protect-latex``, `/clean`
#: option ``protect_latex``, env ``WATERMARKS_PROTECT_LATEX``).
PROTECT_LATEX_MODES = frozenset({"auto", "on", "off"})

_DEFAULT_PREFIX = "WMX"

# A cheap "is there anything here worth protecting" probe for mode="auto".
_LATEX_HINT_RE = re.compile(
    r"\\begin\{[A-Za-z@*]+\}"
    r"|\\(?:cite[A-Za-z]*|ref|eqref|label|input|usepackage|documentclass)\s*[\[{]"
    r"|\\\[|\\\(|\$\$"
    r"|(?<!\\)\$(?:\\.|[^$\\\n])+?(?<!\\)\$"
    r"|\\(?:newcommand|renewcommand|DeclareMathOperator|href|url)\b"
    r"|^```|^~~~|`[^`\n]+`"
    r"|\A---[ \t]*\n|\]\([^()\s]+\)",
    re.M,
)


@dataclass(frozen=True)
class LatexMask:
    """The spans lifted out of a text, and the token naming used for them."""

    spans: tuple[str, ...] = ()
    prefix: str = _DEFAULT_PREFIX

    @property
    def count(self) -> int:
        """Number of protected spans."""
        return len(self.spans)

    def token(self, index: int) -> str:
        """The placeholder standing for span *index*."""
        return f"[[{self.prefix}{index:04d}]]"

    @property
    def token_re(self) -> re.Pattern[str]:
        """Pattern matching this mask's placeholders."""
        return re.compile(rf"\[\[{re.escape(self.prefix)}(\d{{4,}})\]\]")


def _environment_alternation(extra_envs: tuple[str, ...]) -> str:
    names = sorted(PROTECTED_ENVIRONMENTS | set(extra_envs), key=len, reverse=True)
    return "|".join(re.escape(n) + r"\*?" for n in names)


def _build_pattern(extra_envs: tuple[str, ...]) -> re.Pattern[str]:
    envs = _environment_alternation(extra_envs)
    commands = "|".join(PROTECTED_COMMANDS)
    return re.compile(
        r"(?P<preamble>\A(?:\s|%[^\n]*\n)*\\documentclass\b[\s\S]*?\\begin\{document\})"
        # Markdown/YAML front matter: keys and values, not prose.
        r"|(?P<frontmatter>\A---[ \t]*\n[\s\S]*?\n(?:---|\.\.\.)[ \t]*(?:\n|\Z))"
        # Markdown fenced code first: a fence can contain anything, including $.
        r"|(?P<fence>^(?P<tick>```|~~~)[^\n]*\n[\s\S]*?^(?P=tick)[ \t\r]*$)"
        # Protected environments, matched by name so prose environments stay open
        # to the rewrite. The backreference keeps \begin/\end paired.
        rf"|(?P<env>\\begin\{{(?P<envname>{envs})\}}[\s\S]*?\\end\{{(?P=envname)\}})"
        r"|(?P<display>\\\[[\s\S]*?\\\])"
        r"|(?P<inline>\\\([\s\S]*?\\\))"
        r"|(?P<ddollar>(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$)"
        # Inline math may wrap one line but never spans a blank line: a lone `$`
        # in prose (a price, a shell prompt) must not swallow a paragraph.
        r"|(?P<dollar>(?<!\\)\$(?:\\.|[^$\\\n]|\n(?!\s*\n))+?(?<!\\)\$)"
        rf"|(?P<command>\\(?:{commands})(?![A-Za-z])\*?(?=\s*[\[{{]))"
        r"|(?P<code>`[^`\n]+`)"
        # Markdown link/image targets: the URL part of [text](url) / ![alt](src).
        r"|(?P<link>\]\([^()\s]*(?:\([^()\s]*\)[^()\s]*)*\))",
        re.M,
    )


def _command_end(text: str, end: int) -> int:
    """Consume balanced arguments, including nested macro definitions."""
    while end < len(text):
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text) or text[start] not in "{[":
            break
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth, pos = 1, start + 1
        while pos < len(text) and depth:
            char = text[pos]
            if char == "\\":
                pos += 2
                continue
            if char == "%":
                newline = text.find("\n", pos)
                pos = len(text) if newline == -1 else newline + 1
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
            pos += 1
        if depth:
            raise ValueError("Unbalanced protected LaTeX command; review the source first")
        end = pos
    return end


def looks_like_latex(text: str) -> bool:
    """True when *text* carries math, LaTeX markup or fenced code worth protecting."""
    return bool(_LATEX_HINT_RE.search(text))


def _pick_prefix(text: str) -> str:
    """A placeholder prefix that does not already occur in *text*."""
    prefix = _DEFAULT_PREFIX
    while re.search(rf"\[\[{re.escape(prefix)}\d", text):
        prefix += "X"
    return prefix


def mask_latex(
    text: str,
    *,
    mode: str = "auto",
    extra_envs: tuple[str, ...] = (),
) -> tuple[str, LatexMask]:
    """Replace protected spans in *text* with placeholders.

    ``mode`` is ``"auto"`` (protect when the text looks like LaTeX/Markdown math),
    ``"on"`` (always) or ``"off"`` (never). Returns the masked text and the mask
    needed to restore it; an unprotected text comes back unchanged with an empty
    mask, so callers run the same code path either way.
    """
    if mode not in PROTECT_LATEX_MODES:
        raise ValueError(f"unknown protect mode {mode!r}; expected auto|on|off")
    if mode == "off" or not text:
        return text, LatexMask()
    if mode == "auto" and not looks_like_latex(text):
        return text, LatexMask()

    prefix = _pick_prefix(text)
    spans: list[str] = []
    naming = LatexMask(prefix=prefix)

    parts: list[str] = []
    cursor = 0
    for match in _build_pattern(extra_envs).finditer(text):
        if match.start() < cursor:
            continue
        end = _command_end(text, match.end()) if match.lastgroup == "command" else match.end()
        parts.append(text[cursor : match.start()])
        spans.append(text[match.start() : end])
        parts.append(naming.token(len(spans) - 1))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), LatexMask(spans=tuple(spans), prefix=prefix)


def restore_latex(text: str, mask: LatexMask) -> tuple[str, dict[str, int]]:
    """Put the protected spans back, reporting what the round trip actually did.

    A rewrite model may drop a placeholder or emit it twice. Both are recorded
    (``missing`` / ``duplicated``) rather than silently absorbed, because a
    dropped placeholder means the output lost an equation.
    """
    stats = {"protected": mask.count, "restored": 0, "missing": 0, "duplicated": 0}
    if not mask.count:
        return text, stats

    seen: dict[int, int] = {}

    def _put_back(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if not 0 <= index < mask.count:
            # A token the model invented; leave it visible instead of guessing.
            return match.group(0)
        seen[index] = seen.get(index, 0) + 1
        return mask.spans[index]

    out = mask.token_re.sub(_put_back, text)
    stats["restored"] = len(seen)
    stats["missing"] = mask.count - len(seen)
    stats["duplicated"] = sum(n - 1 for n in seen.values())
    return out, stats


def placeholder_error(text: str, mask: LatexMask) -> str | None:
    """Reject missing, duplicate, invented or reordered protected spans."""
    if not mask.count:
        return None
    indices = [int(m.group(1)) for m in mask.token_re.finditer(text)]
    if indices != list(range(mask.count)):
        return "protected math/LaTeX placeholders were lost, duplicated, invented or reordered"
    return None
