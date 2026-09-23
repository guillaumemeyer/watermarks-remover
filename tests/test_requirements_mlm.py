"""The `mlm` tactic's dependencies are declared, pinned, and installable.

The default /clean strategy ("paraphrase@0.8,mlm@0.2") runs the mlm tactic
inside the service, yet no dependency file declared what it imports: a host
whose transformers could not import its pipelines without Pillow answered every
default-strategy text /clean with a 400, and nothing in the repo said what to
install. requirements-mlm.txt now does. These tests keep it honest: exact pins,
in lockstep with the MarkLLM stack it shares, installed by `make
bootstrap-mlm`, and resolved by CI.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
MLM = SCRIPTS / "requirements-mlm.txt"


def _pins(path: Path) -> dict[str, str]:
    """Requirement name (PEP 503-normalized) -> version spec, comments skipped."""
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\S+)", line)
        assert match, f"{path.name}: unexpected requirement line {raw!r}"
        pins[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return pins


def test_mlm_requirements_declare_what_the_tactic_imports():
    pins = _pins(MLM)
    for name in ("torch", "transformers", "pillow"):
        assert name in pins, f"requirements-mlm.txt must declare {name}"


def test_every_mlm_requirement_is_an_exact_pin():
    for name, spec in _pins(MLM).items():
        assert re.fullmatch(r"==\d+(\.\d+)*(\.\*)?", spec), f"{name}{spec} is not an exact pin"


def test_mlm_pins_match_the_markllm_stack():
    mlm = _pins(MLM)
    markllm = _pins(SCRIPTS / "requirements-markllm.txt")
    shared = mlm.keys() & markllm.keys()
    assert {"torch", "transformers", "tokenizers", "huggingface-hub", "pillow"} <= shared
    drift = {name: (mlm[name], markllm[name]) for name in shared if mlm[name] != markllm[name]}
    assert not drift, f"requirements-mlm.txt drifted from requirements-markllm.txt: {drift}"


def test_pillow_is_the_audited_pin():
    # requirements-synthid-scorer.txt is the backend file CI pip-audits.
    scorer = _pins(SCRIPTS / "requirements-synthid-scorer.txt")
    assert _pins(MLM)["pillow"] == scorer["pillow"]


def _makefile() -> str:
    return (ROOT / "Makefile").read_text(encoding="utf-8")


def _recipe(target: str) -> str:
    match = re.search(rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)+)", _makefile(), re.M)
    assert match, f"Makefile has no recipe for {target}"
    return match.group(1)


def test_bootstrap_mlm_installs_the_declared_stack():
    recipe = _recipe("bootstrap-mlm")
    assert "-r $(SCRIPTS)/requirements-mlm.txt" in recipe
    # torch comes from TORCH_INDEX_URL at the version read from the file, so
    # the target can never install a torch the requirements don't pin, and
    # alone: that index's copies of torch's dependencies are stale.
    assert '--no-deps --index-url "$(TORCH_INDEX_URL)"' in recipe
    assert "$(SCRIPTS)/requirements-mlm.txt)" in recipe
    assert not re.search(r"torch\s*[=<>~!]=?\s*\d", recipe), "hard-coded torch version"
    assert "$(MLM_SMOKE)" in recipe


def test_mlm_smoke_runs_the_tactic_without_an_llm_backend():
    smoke = re.search(r"^MLM_SMOKE\s*=\s*(.+)$", _makefile(), re.M)
    assert smoke, "Makefile defines no MLM_SMOKE"
    assert "--strategy mlm@" in smoke.group(1)
    assert "--backend print-prompt" in smoke.group(1)
    assert "$(MLM_SMOKE)" in _recipe("smoke-mlm")


def test_ci_resolves_the_mlm_requirements():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "-r service/scripts/requirements-mlm.txt" in ci


def test_docs_install_the_pinned_torch():
    """README's manual (no-make) install must name the torch the file pins.

    Only install commands count: changelog entries quoting an older pin are
    history, not instructions.
    """
    torch_pin = "torch" + _pins(MLM)["torch"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installs = [
        re.search(r"torch==[0-9][0-9.*]*", line).group(0)
        for line in readme.splitlines()
        if "--index-url" in line and "torch==" in line
    ]
    assert installs, "README shows no manual torch install for requirements-mlm.txt"
    assert set(installs) == {torch_pin}, f"README installs {installs}, file pins {torch_pin}"
