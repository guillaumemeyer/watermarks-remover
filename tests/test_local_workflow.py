"""Checks for local review, preservation and bounded model requests."""

import json

import local_workflow
import pytest
import rewrite_text
from latex_mask import mask_latex, restore_latex


def test_chunks_reassemble_exactly_without_breaking_placeholders():
    source = ("Texto [[WMX0000]] com espaços.\n\n" * 100) + "Fim."
    parts = local_workflow.chunks(source, limit=100)
    assert "".join(parts) == source
    assert all(len(part) <= 100 for part in parts)
    assert sum(part.count("[[WMX0000]]") for part in parts) == 100


def test_nested_macros_and_full_preamble_roundtrip():
    source = (
        "\\documentclass{article}\n\\newcommand{\\mass}[1]{\\frac{#1}{2}}\n"
        "\\begin{document}\nTexto $x$.\n\\end{document}"
    )
    masked, mask = mask_latex(source)
    assert "documentclass" not in masked
    assert "newcommand" not in masked
    assert restore_latex(masked, mask)[0] == source
    macro = r"\newcommand{\mass}[1]{\frac{#1}{2}} Texto."
    masked, mask = mask_latex(macro)
    assert mask.spans == (r"\newcommand{\mass}[1]{\frac{#1}{2}}",)
    assert restore_latex(masked, mask)[0] == macro


def test_academic_review_preserves_math_source_and_separators(monkeypatch):
    seen = []

    def generate(*args, original):
        seen.append(args[4])
        return original.replace("Estudamos", "Examinamos"), []

    monkeypatch.setattr(local_workflow, "_generate_rewrite", generate)
    source = "  Estudamos $D(p^2)$ em 4 dimensões \\cite{key}.\n\n"
    out, report = local_workflow.review_academic(source, "fake")
    assert out == source.replace("Estudamos", "Examinamos")
    assert all("$D(p^2)$" not in prompt for prompt in seen)
    assert report["latex"]["missing"] == 0
    assert report["scientific_review_required"] is True


@pytest.mark.parametrize(
    ("before", "after", "error"),
    [
        ("[[WMX0000]]", "", "protected notation"),
        ("4 dimensões", "5 dimensões", "numbers"),
        (r"\emph{Texto}", "Texto", "LaTeX structure"),
    ],
)
def test_review_rejects_content_damage(monkeypatch, before, after, error):
    monkeypatch.setattr(
        local_workflow,
        "_generate_rewrite",
        lambda *a, original: (original.replace(before, after), []),
    )
    with pytest.raises(RuntimeError, match=error):
        local_workflow.review_academic(r"\emph{Texto} $x$ em 4 dimensões.", "fake")


def test_academic_output_is_new_and_has_provenance_report(tmp_path, monkeypatch):
    source = tmp_path / "input.tex"
    source.write_text("Texto $x$.", encoding="utf-8")
    monkeypatch.setattr(local_workflow, "_generate_rewrite", lambda *a, original: (original, []))
    assert local_workflow.main(["academic", str(source)]) == 0
    output = tmp_path / "input.assisted.tex"
    assert output.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    report = json.loads((tmp_path / "input.assisted.tex.review.json").read_text(encoding="utf-8"))
    assert report["assisted"] is True
    with pytest.raises(SystemExit):
        local_workflow.main(["academic", str(source)])
    with pytest.raises(SystemExit):
        local_workflow.main(["academic", str(source), "-o", str(source)])


def test_failed_review_does_not_write_output(tmp_path, monkeypatch):
    source = tmp_path / "input.tex"
    source.write_text("Texto $x$.", encoding="utf-8")
    monkeypatch.setattr(local_workflow, "_generate_rewrite", lambda *a, **k: ("Texto.", []))
    assert local_workflow.main(["academic", str(source)]) == 1
    assert not (tmp_path / "input.assisted.tex").exists()
    assert source.read_text(encoding="utf-8") == "Texto $x$."


def test_ollama_has_bounded_runtime_and_rejects_truncation(monkeypatch):
    seen = {}

    def response(url, payload, headers, timeout):
        seen.update(payload)
        return {"done_reason": "length", "message": {"content": "incomplete"}}

    for key in ("CONTEXT", "MAX_TOKENS", "THREADS"):
        monkeypatch.delenv("WATERMARKS_OLLAMA_" + key, raising=False)
    monkeypatch.setattr(rewrite_text, "_http_json", response)
    with pytest.raises(RuntimeError, match="truncated"):
        rewrite_text.call_ollama("http://localhost:11434", "fake", "source", 30, 0.2, "none")
    # The budget is a floor that grows with the prompt; think is off only for
    # reasoning_effort "none" (think: true is an error on non-thinking models).
    assert seen["options"]["num_ctx"] >= 8192
    assert seen["options"]["num_predict"] >= 2048
    assert "num_thread" not in seen["options"]
    assert seen["think"] is False
    assert seen["keep_alive"] == "5m"

    seen.clear()
    monkeypatch.setenv("WATERMARKS_OLLAMA_CONTEXT", "16384")
    monkeypatch.setenv("WATERMARKS_OLLAMA_MAX_TOKENS", "4096")
    monkeypatch.setenv("WATERMARKS_OLLAMA_THREADS", "6")
    monkeypatch.setattr(
        rewrite_text, "_http_json", lambda *a: seen.update(a[1]) or {"message": {"content": "ok"}}
    )
    assert rewrite_text.call_ollama("http://localhost:11434", "fake", "source", 30, 0.2) == "ok"
    assert seen["options"]["num_ctx"] == 16384
    assert seen["options"]["num_predict"] == 4096
    assert seen["options"]["num_thread"] == 6
    assert "think" not in seen


def test_review_rejects_clear_portuguese_to_english_translation(monkeypatch):
    monkeypatch.setattr(
        local_workflow,
        "_generate_rewrite",
        lambda *a, **k: ("The study examines the model and the parameter is indeterminate.", []),
    )
    with pytest.raises(RuntimeError, match="language changed"):
        local_workflow.review_academic(
            "O estudo examina o modelo e o parâmetro permanece indeterminado.", "fake"
        )
