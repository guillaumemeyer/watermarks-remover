"""Conservative mechanical guards; these do not prove scientific equivalence."""

import re
from collections import Counter


def language_hint(text: str) -> str | None:
    """Detect only clear Portuguese/English evidence; return None when uncertain."""
    words = Counter(re.findall(r"[^\W\d_]+", text.lower()))
    portuguese = set(
        [
            "o",
            "os",
            "uma",
            "um",
            "de",
            "da",
            "das",
            "dos",
            "em",
            "não",
            "para",
            "com",
            "por",
            "é",
            "são",
            "mas",
            "neste",
            "esta",
            "essa",
            "que",
            "ao",
            "no",
            "na",
            "do",
            "permanece",
            "estudo",
            "análise",
            "relação",
            "hipóteses",
            "parâmetro",
            "indeterminado",
        ]
    )
    english = set(
        [
            "the",
            "this",
            "that",
            "these",
            "those",
            "which",
            "with",
            "within",
            "from",
            "their",
            "is",
            "are",
            "was",
            "were",
            "an",
            "but",
            "and",
            "remains",
            "study",
            "analysis",
            "relationship",
            "assumptions",
            "parameter",
            "indeterminate",
        ]
    )
    pt = sum(words[word] for word in portuguese)
    en = sum(words[word] for word in english)
    if pt >= 3 and pt >= 2 * en + 1:
        return "pt"
    if en >= 3 and en >= 2 * pt + 1:
        return "en"
    return None


_CONTROLS = r"\\[A-Za-z@]+\*?|\\[^A-Za-z]|[{}]"


def structure_error(original: str, candidate: str) -> str | None:
    """Reject a rewrite whose LaTeX control sequences or braces differ from the input's.

    Both texts are expected with their math spans already replaced by
    placeholders; what is compared is the command and grouping skeleton left in
    the prose (``\\section``, ``\\emph``, ``{``, ``}``), in order.
    """
    if re.findall(_CONTROLS, original) != re.findall(_CONTROLS, candidate):
        return "LaTeX structure changed"
    return None


def academic_error(original: str, candidate: str) -> str | None:
    """Compare text with mathematical spans already replaced by placeholders."""
    source_language, target_language = language_hint(original), language_hint(candidate)
    if source_language and target_language and source_language != target_language:
        return "language changed"
    structure = structure_error(original, candidate)
    if structure:
        return structure
    numbers = r"(?<!\w)\d+(?:[.,]\d+)*"
    if Counter(re.findall(numbers, original)) != Counter(re.findall(numbers, candidate)):
        return "numbers changed"
    return None
