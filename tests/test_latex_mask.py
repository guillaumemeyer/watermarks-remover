"""Tests for the Layer B math/LaTeX protection and the `academic` tactic.

The rewrite model never sees a protected span, and every span it was given back
is accounted for: an academic source that loses an equation to a paraphrase is a
corrupted document, not a cleaned one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
from latex_mask import LatexMask, looks_like_latex, mask_latex, restore_latex
from rewrite_text import apply_strategy, build_prompt, parse_strategy, rewrite

PHYSICS = r"""O propagador de gluons no gauge de Landau, $D(p^2)$, satisfaz a
condicao de horizonte de Gribov \cite{Gribov:1977wm}; ver \eqref{eq:horizon} e
as paginas 12--18 de \cite{Zwanziger:1989mf}.

\begin{equation}
\label{eq:horizon}
\gamma^4 \int d^4x\, f^{abc} A^b_\mu (M^{-1})^{ad} f^{dec} A^e_\mu = 4(N^2-1)V\gamma^4 ,
\end{equation}

O parametro de massa permanece indeterminado nesse regime.
"""


def _rewrite_kwargs(**overrides):
    kwargs = dict(
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="academic",
        lang="French",
        original_lang="English",
        timeout=10,
        layer_a_after=False,
        temperature=0.9,
        candidates=1,
    )
    kwargs.update(overrides)
    return kwargs


# --- masking ---------------------------------------------------------------


def test_roundtrip_restores_source_exactly():
    masked, mask = mask_latex(PHYSICS)
    assert mask.count == 5
    restored, stats = restore_latex(masked, mask)
    assert restored == PHYSICS
    assert stats == {"protected": 5, "restored": 5, "missing": 0, "duplicated": 0}


def test_math_and_citations_leave_the_masked_text():
    masked, _mask = mask_latex(PHYSICS)
    for gone in ("$D(p^2)$", r"\cite{Gribov:1977wm}", r"\begin{equation}", r"\eqref{eq:horizon}"):
        assert gone not in masked
    # Prose survives untouched so the rewrite still has something to work on.
    assert "propagador de gluons" in masked
    assert "O parametro de massa permanece indeterminado" in masked


def test_prose_environment_is_not_protected_but_its_math_is():
    text = r"\begin{abstract}Estudamos o limite $g \to 0$ do modelo.\end{abstract}"
    masked, mask = mask_latex(text)
    assert mask.count == 1
    assert r"\begin{abstract}" in masked
    assert "Estudamos o limite" in masked
    assert r"$g \to 0$" not in masked


def test_lone_dollar_does_not_swallow_a_paragraph():
    text = "O custo foi de $200 no semestre.\n\nOutro paragrafo com $x$ real.\n"
    masked, mask = mask_latex(text)
    assert mask.count == 1
    assert masked.startswith("O custo foi de $200 no semestre.")
    assert "$x$" not in masked


def test_escaped_dollar_is_not_math():
    text = r"O preco \$5 e o total \$9 ficam no texto."
    _masked, mask = mask_latex(text)
    assert mask.count == 0


def test_markdown_fence_and_inline_code_protected():
    text = "Veja o trecho:\n\n```python\nx = 2 * 3\n```\n\nE o valor `alpha_s` medido.\n"
    masked, mask = mask_latex(text)
    assert mask.count == 2
    assert "x = 2 * 3" not in masked
    assert "`alpha_s`" not in masked
    assert restore_latex(masked, mask)[0] == text


def test_auto_mode_skips_plain_prose():
    plain = "Uma frase comum sem qualquer notacao matematica."
    assert looks_like_latex(plain) is False
    masked, mask = mask_latex(plain, mode="auto")
    assert (masked, mask.count) == (plain, 0)


def test_on_mode_protects_even_without_hints_and_off_never_does():
    text = "Custa $5 hoje e `talvez` amanha."
    assert mask_latex(text, mode="on")[1].count >= 1
    assert mask_latex(text, mode="off")[1].count == 0
    assert mask_latex(PHYSICS, mode="off")[0] == PHYSICS


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match=r"expected auto\|on\|off"):
        mask_latex(PHYSICS, mode="sometimes")


def test_prefix_avoids_collision_with_existing_tokens():
    text = "Um token literal [[WMX0000]] ja no texto com $x$ real."
    masked, mask = mask_latex(text)
    assert mask.prefix != "WMX"
    assert "[[WMX0000]]" in masked  # the pre-existing literal is left alone
    assert restore_latex(masked, mask)[0] == text


def test_dropped_placeholder_is_reported_not_hidden():
    masked, mask = mask_latex(PHYSICS)
    lost = masked.replace(mask.token(0), "")
    _out, stats = restore_latex(lost, mask)
    assert stats["missing"] == 1
    assert stats["restored"] == mask.count - 1


def test_duplicated_placeholder_is_reported():
    masked, mask = mask_latex(PHYSICS)
    doubled = masked.replace(mask.token(1), mask.token(1) + " " + mask.token(1))
    out, stats = restore_latex(doubled, mask)
    assert stats["duplicated"] == 1
    assert out.count(r"\cite{Gribov:1977wm}") == 2


def test_invented_token_is_left_visible():
    masked, mask = mask_latex(PHYSICS)
    invented = masked + f"\n[[{mask.prefix}9999]]\n"
    out, _stats = restore_latex(invented, mask)
    assert f"[[{mask.prefix}9999]]" in out


# --- prompt wiring ---------------------------------------------------------


def test_prompt_carries_placeholder_instruction():
    mask = LatexMask(spans=("$x$", "$y$"))
    prompt = build_prompt("academic", "texto [[WMX0000]]", mask=mask)
    assert "2 protected token(s)" in prompt
    assert "[[WMX0000]]" in prompt
    assert "never translate, renumber" in prompt.lower()


def test_prompt_guard_used_when_nothing_can_be_restored():
    prompt = build_prompt("academic", PHYSICS, latex_guard=True)
    assert "Reproduce all mathematics" in prompt


def test_no_latex_clause_without_protection():
    prompt = build_prompt("academic", "texto simples")
    assert "protected token" not in prompt
    assert "Reproduce all mathematics" not in prompt


# --- academic tactic -------------------------------------------------------


def test_academic_prompt_keeps_terminology_and_language():
    prompt = build_prompt("academic", "O horizonte de Gribov limita o dominio.")
    assert "O horizonte de Gribov limita o dominio." in prompt
    assert "same language as the original" in prompt
    assert "epistemic force" in prompt
    assert "term of art" in prompt


def test_academic_is_a_strategy_tactic():
    assert parse_strategy("academic@0.6,mlm@0.2") == [("academic", 0.6), ("mlm", 0.2)]


def test_academic_prompt_modulated_by_intensity():
    prompt = build_prompt("academic", "ABC", rewrite_level=0.4)
    assert "fraction 0.40 of tokens change" in prompt


# --- end-to-end through rewrite() ------------------------------------------


def test_rewrite_rejects_lost_math_instead_of_returning_corrupted_source(monkeypatch):
    seen: list[str] = []

    def _fake(base_url, model, prompt, *args, **kwargs):
        seen.append(prompt)
        return "Prosa reescrita com [[WMX0000]] e a equacao [[WMX0004]] no fim."

    monkeypatch.setattr(rewrite_text, "call_ollama", _fake)
    with pytest.raises(RuntimeError, match="protected math/LaTeX"):
        rewrite(PHYSICS, **_rewrite_kwargs())

    assert "$D(p^2)$" not in seen[0]


def test_rewrite_restores_all_protected_spans(monkeypatch):
    masked, _ = mask_latex(PHYSICS)
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: masked)
    out, info = rewrite(PHYSICS, **_rewrite_kwargs())
    assert out == PHYSICS
    assert info["latex"] == {"protected": 5, "restored": 5, "missing": 0, "duplicated": 0}


@pytest.mark.parametrize("bad", ["", "[[WMX0000]] [[WMX0000]]", "[[WMX9999]]"])
def test_strategy_keeps_original_on_placeholder_corruption(monkeypatch, bad):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: "Texto " + bad)
    source = "Texto $x+y$."
    out, info = apply_strategy(
        source,
        [("academic", 0.2)],
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
    )
    assert out == source
    assert info["ok"] is False
    assert "protected math/LaTeX" in info["errors"][0]


def test_humanize_pass_never_reaches_protected_spans(monkeypatch):
    """`--` inside protected math stays; the humanizer only sees the prose."""
    text = r"Intervalo $a--b$ discutido no texto\cite{K}."
    monkeypatch.setattr(
        rewrite_text,
        "call_ollama",
        lambda *a, **k: "Intervalo [[WMX0000]] tratado—de novo—em [[WMX0001]].",
    )
    out, _info = rewrite(text, **_rewrite_kwargs(tactic="humanize"))
    assert "$a--b$" in out
    assert r"\cite{K}" in out
    assert "tratado, de novo, em" in out  # the prose em dashes were collapsed


def test_print_prompt_backend_guards_instead_of_masking():
    out, info = rewrite(PHYSICS, **_rewrite_kwargs(backend="print-prompt", model=None))
    assert info["latex_protection"] == "prompt-guard"
    assert info["latex_protected"] == 0
    assert "$D(p^2)$" in out  # the prompt still carries the real source
    assert "Reproduce all mathematics" in out


def test_protect_off_sends_math_to_the_model(monkeypatch):
    monkeypatch.setattr(rewrite_text, "call_ollama", lambda *a, **k: PHYSICS)
    _out, info = rewrite(PHYSICS, **_rewrite_kwargs(protect_latex="off"))
    assert info["latex_protection"] == "none"
    assert info["latex_protected"] == 0


def test_apply_strategy_protects_across_steps(monkeypatch):
    prompts: list[str] = []
    masked, _mask = mask_latex(PHYSICS)

    def _fake(base_url, model, prompt, *args, **kwargs):
        prompts.append(prompt)
        return masked  # a model that changed nothing but kept every placeholder

    monkeypatch.setattr(rewrite_text, "call_ollama", _fake)
    out, stats = apply_strategy(
        PHYSICS,
        [("academic", 0.6), ("paraphrase", 0.4)],
        backend="ollama",
        model="m",
        base_url="http://127.0.0.1:11434",
        api_key=None,
    )
    assert stats["latex_protected"] == 5
    assert stats["latex"]["missing"] == 0
    assert all("$D(p^2)$" not in p for p in prompts)
    assert out == PHYSICS
