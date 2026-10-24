"""Tests for the attack chain coordinator."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def _write_random_png(path: Path, w: int, h: int) -> None:
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    img.save(path)


def test_valid_schemes_cover_all_known():
    from attack_chain import VALID_SCHEMES

    expected = {
        "tree_ring",
        "gaussian_shading",
        "stegastamp",
        "stable_signature",
        "synthid",
        "hidden",
        "trustmark",
        "unknown",
    }
    assert expected == VALID_SCHEMES


def test_synthid_bypass_pipeline_stages_are_valid():
    """All pipeline args in ATTACK_CHAINS that invoke clean_synthid_bypass.py
    must use stage names that clean_synthid_bypass.py actually accepts."""
    from attack_chain import ATTACK_CHAINS
    from clean_synthid_bypass import STAGES as VALID_STAGES

    for scheme, chain in ATTACK_CHAINS.items():
        for step in chain:
            if step["engine"] != "clean_synthid_bypass.py":
                continue
            args = step["args"]
            if "--pipeline" not in args:
                continue
            pipeline_str = args[args.index("--pipeline") + 1]
            stages = [s.strip().lower() for s in pipeline_str.replace("+", ",").split(",")]
            for s in stages:
                assert s in VALID_STAGES, (
                    f"scheme {scheme!r}: stage {s!r} not in {VALID_STAGES}"
                )


def test_chain_result_structure():
    from attack_chain import ChainResult

    r = ChainResult(
        scheme="unknown",
        steps_run=1,
        steps_total=2,
        stealth_score=0.15,
        elapsed_s=1.5,
        output_path="out.png",
    )
    assert r.scheme == "unknown"
    assert r.steps_run <= r.steps_total


def test_bad_input_exits_2():
    result = _run("attack_chain.py", "/nonexistent.png")
    assert result.returncode == 2


def test_invalid_scheme_exits_2(tmp_path):
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--scheme", "nonexistent")
    assert result.returncode == 2


def test_json_output(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--scheme", "unknown", "--json")
    assert result.returncode in (0, 1), f"unexpected exit {result.returncode}: {result.stderr[:200]}"
    report = json.loads(result.stdout)
    assert "scheme" in report
    assert "steps_run" in report
    assert "stealth_score" in report
    assert "complete" in report


def test_auto_classify_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    img = tmp_path / "test.png"
    _write_random_png(img, 64, 64)
    result = _run("attack_chain.py", str(img), "--json")
    assert result.returncode in (0, 1)


def test_flux3_step_is_reached_after_fft(tmp_path):
    """Verify that the Flux3 step is NOT skipped by the forensic early stop."""
    import shutil as _shutil

    from attack_chain import ATTACK_CHAINS, run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 128, 128)
    out = tmp_path / "output.png"

    engines_called: list[str] = []

    def mock_run(cmd, **kwargs):
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        engines_called.append(script_name)
        out_idx = cmd.index("-o") + 1 if "-o" in cmd else -1
        step_out = Path(cmd[out_idx]) if out_idx > 0 else out
        _shutil.copy2(str(img), str(step_out))
        if script_name == "forensic_check.py":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"stealth_score":0.05}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        result = run_chain(img, "stegastamp", output_path=out, stealth_threshold=0.7)

    expected = len(ATTACK_CHAINS["stegastamp"])
    assert result.steps_run == expected, f"expected {expected} steps to run, got {result.steps_run}"
    assert "clean_flux3.py" in engines_called, f"Flux3 was never called: {engines_called}"


def test_pick_flux3_resolution_selects_cost_effective_tier():
    from attack_chain import _pick_flux3_resolution

    assert _pick_flux3_resolution(2304, 1728) == "2k"
    assert _pick_flux3_resolution(480, 480) == "768"
    assert _pick_flux3_resolution(1024, 1536) == "1.5k"
    assert _pick_flux3_resolution(4096, 4096) == "4k"


def test_resolve_api_key_from_env():
    from attack_chain import _resolve_api_key

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"}):
        assert _resolve_api_key(None) == "test-key-123"
    assert _resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_unreadable_file(tmp_path):
    from attack_chain import _resolve_api_key

    def raise_permission_error(*_args, **_kwargs):
        raise PermissionError("mocked unreadable")

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False), \
         patch("attack_chain.Path.home", return_value=tmp_path):
        env_file = tmp_path / ".claude" / "openrouter.env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("OPENROUTER_API_KEY=file-key")
        with patch.object(type(env_file), "read_text", raise_permission_error):
            key = _resolve_api_key(None)
        assert key == ""


def test_api_key_not_in_process_args(tmp_path):
    """Verify API key is passed via env, not as a CLI argument."""
    from attack_chain import _FLUX3_ENGINES, run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 64, 64)
    out = tmp_path / "output.png"

    captured_cmds: list[list[str]] = []
    captured_envs: list[dict | None] = []

    def mock_run(cmd, **kwargs):
        captured_cmds.append(cmd)
        captured_envs.append(kwargs.get("env"))
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script_name in _FLUX3_ENGINES:
            import shutil

            shutil.copy2(str(img), str(out))
            return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="unavailable")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        run_chain(img, "unknown", output_path=out, api_key="secret-key-123")

    for cmd in captured_cmds:
        assert "secret-key-123" not in cmd, "API key must not appear in process arguments"

    flux3_envs = [e for cmd, e in zip(captured_cmds, captured_envs, strict=True)
                  if Path(cmd[1]).name in _FLUX3_ENGINES]
    assert any(e and e.get("OPENROUTER_API_KEY") == "secret-key-123" for e in flux3_envs), \
        "API key must be passed via OPENROUTER_API_KEY env var"


def test_non_flux3_timeout_marks_incomplete(tmp_path):
    """A timeout in a pre-Flux3 step (with steps_run>0) must set flux3_failed."""
    import shutil as _shutil

    from attack_chain import _FLUX3_ENGINES, run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 64, 64)
    out = tmp_path / "output.png"

    engines_called: list[str] = []

    def mock_run(cmd, **kwargs):
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        engines_called.append(script_name)
        out_idx = cmd.index("-o") + 1 if "-o" in cmd else -1
        step_out = Path(cmd[out_idx]) if out_idx > 0 else out
        if script_name == "clean_fmdiffwa.py":
            _shutil.copy2(str(img), str(step_out))
            raise subprocess.TimeoutExpired(cmd, 600)
        if script_name == "clean_marknull.py":
            _shutil.copy2(str(img), str(step_out))
            return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")
        if script_name == "forensic_check.py":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"stealth_score":0.9}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        result = run_chain(img, "gaussian_shading", output_path=out, stealth_threshold=0.7)

    assert result.steps_run == 1, f"expected 1 step to succeed, got {result.steps_run}"
    assert not result.complete, "chain should be incomplete when pre-Flux3 step times out"
    assert not any(Path(e).name in _FLUX3_ENGINES for e in engines_called), \
        f"Flux3 should not have been called: {engines_called}"


def test_flux3_provider_forwarded(tmp_path):
    """Verify --provider is forwarded to Flux3 engine."""
    import shutil as _shutil

    from attack_chain import _FLUX3_ENGINES, run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 64, 64)
    out = tmp_path / "output.png"

    captured_cmds: list[list[str]] = []

    def mock_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        out_idx = cmd.index("-o") + 1 if "-o" in cmd else -1
        step_out = Path(cmd[out_idx]) if out_idx > 0 else out
        _shutil.copy2(str(img), str(step_out))
        if script_name == "forensic_check.py":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"stealth_score":0.05}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        run_chain(img, "unknown", output_path=out, flux3_provider="bfl")

    flux3_cmds = [c for c in captured_cmds if Path(c[1]).name in _FLUX3_ENGINES]
    assert flux3_cmds, "Flux3 was never called"
    assert any("--provider" in c and "bfl" in c for c in flux3_cmds), \
        f"--provider bfl not found in Flux3 commands: {flux3_cmds}"


def test_flux3_timeout_applies_to_flux3_step(tmp_path):
    """Verify flux3_timeout is used for Flux3 step, not the general timeout."""
    import shutil as _shutil

    from attack_chain import _FLUX3_ENGINES, run_chain

    img = tmp_path / "input.png"
    _write_random_png(img, 64, 64)
    out = tmp_path / "output.png"

    captured_timeouts: dict[str, int] = {}

    def mock_run(cmd, **kwargs):
        script_name = Path(cmd[1]).name if len(cmd) > 1 else ""
        captured_timeouts[script_name] = kwargs.get("timeout", 0)
        out_idx = cmd.index("-o") + 1 if "-o" in cmd else -1
        step_out = Path(cmd[out_idx]) if out_idx > 0 else out
        _shutil.copy2(str(img), str(step_out))
        if script_name == "forensic_check.py":
            return subprocess.CompletedProcess(cmd, 0, stdout='{"stealth_score":0.05}', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok":true}', stderr="")

    with patch("attack_chain.subprocess.run", side_effect=mock_run):
        run_chain(img, "unknown", output_path=out, timeout=300, flux3_timeout=120)

    flux3_called = [e for e in _FLUX3_ENGINES if e in captured_timeouts]
    assert flux3_called, f"Flux3 was never called; engines: {list(captured_timeouts)}"
    for engine_name in flux3_called:
        assert captured_timeouts[engine_name] == 120, \
            f"Flux3 step got timeout {captured_timeouts[engine_name]}, expected 120"

    non_flux3 = {k: v for k, v in captured_timeouts.items()
                 if k not in _FLUX3_ENGINES and k != "forensic_check.py"}
    for engine_name, t in non_flux3.items():
        assert t == 300, f"{engine_name} got timeout {t}, expected 300"


import json as _json

import pytest

FIXTURES = ROOT / "tests" / "fixtures"
BENCHMARKS = ROOT / "benchmarks"
_SYNTHID_WATERMARKED = FIXTURES / "real_synthid_watermarked.png"
_SYNTHID_ATTACKED = FIXTURES / "real_synthid_attacked_unknown_chain.png"
_SYNTHID_ATTACKED_SYNTHID_CHAIN = FIXTURES / "real_synthid_attacked_synthid_chain.png"
_TREERING_WATERMARKED = FIXTURES / "real_treering_watermarked.png"
_TREERING_ATTACKED = FIXTURES / "real_treering_attacked.png"
_TREERING_REPEAT = FIXTURES / "real_treering_attacked_repeat.png"
_REVERSE_SYNTHID = Path(
    os.environ.get("REVERSE_SYNTHID_DIR", "G:/GitHub/reverse-SynthID")
)


def _skip_unless_fixture(*paths: Path):
    for p in paths:
        if not p.is_file():
            pytest.skip(f"fixture not found: {p.name}")


def _run_benchmark(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARKS / script), *args],
        text=True, capture_output=True, check=False, timeout=180,
    )


def test_synthid_fixture_detected_before_attack():
    """The real SynthID watermarked fixture must score as watermarked."""
    _skip_unless_fixture(_SYNTHID_WATERMARKED)
    if not _REVERSE_SYNTHID.is_dir():
        pytest.skip("reverse-SynthID checkout not found")
    r = _run("score_synthid.py", str(_SYNTHID_WATERMARKED),
             "--upstream-dir", str(_REVERSE_SYNTHID), "--json")
    if r.returncode == 3:
        pytest.skip("SynthID scorer unavailable")
    assert r.returncode == 0, r.stderr
    data = _json.loads(r.stdout)
    assert data["is_watermarked"] is True, f"confidence={data.get('confidence')}"


def test_synthid_unknown_chain_removes_watermark():
    """The unknown chain (Flux3 regeneration) output must not be detected as SynthID."""
    _skip_unless_fixture(_SYNTHID_ATTACKED)
    if not _REVERSE_SYNTHID.is_dir():
        pytest.skip("reverse-SynthID checkout not found")
    r = _run("score_synthid.py", str(_SYNTHID_ATTACKED),
             "--upstream-dir", str(_REVERSE_SYNTHID), "--json")
    if r.returncode == 3:
        pytest.skip("SynthID scorer unavailable")
    assert r.returncode == 0, r.stderr
    data = _json.loads(r.stdout)
    assert data["is_watermarked"] is False, \
        f"SynthID still detected after unknown chain: confidence={data.get('confidence')}"


def _run_treering_decode(image_path: Path) -> dict:
    """Run decode_treering.py; skip only on missing dependencies, fail on other errors."""
    if not (BENCHMARKS / "decode_treering.py").is_file():
        pytest.skip("decode_treering.py not found in benchmarks/")
    r = _run_benchmark("decode_treering.py", str(image_path), "--json")
    if r.returncode != 0:
        _dep_kw = ("ModuleNotFoundError", "No module named")
        if any(kw in r.stderr for kw in _dep_kw):
            pytest.skip(f"decode_treering missing dependency: {r.stderr[:120]}")
        try:
            err_data = _json.loads(r.stdout)
            err_msg = err_data.get("error", "")
            if any(kw in err_msg for kw in _dep_kw):
                pytest.skip(f"decode_treering missing dependency: {err_msg[:120]}")
        except (ValueError, KeyError):
            pass
        raise AssertionError(f"decode_treering failed (exit {r.returncode}): {r.stderr[:200]} {r.stdout[:200]}")
    return _json.loads(r.stdout)


def test_synthid_chain_fixture_removes_watermark():
    """The synthid chain fixture must not be detected as SynthID watermarked."""
    _skip_unless_fixture(_SYNTHID_ATTACKED_SYNTHID_CHAIN)
    if not _REVERSE_SYNTHID.is_dir():
        pytest.skip("reverse-SynthID checkout not found")
    r = _run("score_synthid.py", str(_SYNTHID_ATTACKED_SYNTHID_CHAIN),
             "--upstream-dir", str(_REVERSE_SYNTHID), "--json")
    if r.returncode == 3:
        pytest.skip("SynthID scorer unavailable")
    assert r.returncode == 0, r.stderr
    data = _json.loads(r.stdout)
    assert data["is_watermarked"] is False, \
        f"SynthID still detected after synthid chain: confidence={data.get('confidence')}"


def test_treering_fixture_detected_before_attack():
    """The real Tree-Ring watermarked fixture must decode as detected."""
    _skip_unless_fixture(_TREERING_WATERMARKED)
    data = _run_treering_decode(_TREERING_WATERMARKED)
    assert data["detected"] is True, f"L1={data.get('l1_dist')}"


def test_treering_attacked_evades_detection():
    """The attacked Tree-Ring fixture must evade detection (L1 > 78)."""
    _skip_unless_fixture(_TREERING_ATTACKED)
    data = _run_treering_decode(_TREERING_ATTACKED)
    assert data["detected"] is False, f"L1={data.get('l1_dist')} (threshold 78)"
    assert data["l1_dist"] > 78.0


def test_treering_repeat_also_evades():
    """A second independent run also evades Tree-Ring detection."""
    _skip_unless_fixture(_TREERING_REPEAT)
    data = _run_treering_decode(_TREERING_REPEAT)
    assert data["detected"] is False, f"L1={data.get('l1_dist')} (threshold 78)"
    assert data["l1_dist"] > 78.0


def _grayscale(path: Path):
    """Load an image as a float grayscale array, or skip when numpy/Pillow are absent."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        pytest.skip(f"quality checks need numpy+Pillow: {exc}")
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=float)


def _highfreq_energy(gray) -> float:
    """Laplacian residual std: additive grain and noise raise this sharply."""
    lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(lap.std())


def _residual_stripe_coherence(source_path: Path, attacked_path: Path) -> float:
    """How column-coherent the attack's residual is, relative to incoherent change.

    Vertical banding survives averaging down the columns; unstructured change
    cancels, shrinking by sqrt(height).  Dividing the measured column-profile
    energy by that expectation puts honest regeneration near 1 whatever its
    magnitude, which a raw energy cannot do -- regeneration that moves pixels
    further would otherwise look like banding.

    The normalisation cuts both ways: because the denominator grows with the
    residual, a given stripe amplitude is easier to detect on a near-identity
    output than on a heavily redrawn one.  Measured on these fixtures, an
    injected 6px-period stripe trips the limit at +-0.5 gray levels on the
    SynthID output but needs +-4 on the Tree-Ring output.
    """
    import numpy as np
    from PIL import Image

    with Image.open(source_path) as im:
        size = im.size
    src = _grayscale(source_path)
    with Image.open(attacked_path) as im:
        att = im.convert("L")
        if att.size != size:
            att = att.resize(size, Image.Resampling.LANCZOS)
        att = np.asarray(att, dtype=float)

    residual = att - src
    col = residual.mean(axis=0)
    smooth = np.convolve(col, np.ones(31) / 31, mode="same")
    measured = float((col - smooth)[20:-20].std())
    incoherent = float(residual.std() / np.sqrt(residual.shape[0]))
    return measured / max(incoherent, 1e-9)


# An attacked image may not add grain beyond this multiple of the source's own
# high-frequency energy.  PGD-style additive noise measured 2.6-2.7x and looked
# obviously filtered; diffusion resampling stays near 1.0x.
_MAX_GRAIN_RATIO = 1.5
# Column coherence of the residual.  The regression this guards against is a
# chain with the full-image FFT stage back in it: the previously committed
# FFT+Flux3 fixtures measure 9.5 and 12.8 (the FFT stage alone, 13.8).  Every
# current regeneration output measures 1.6-2.7, so 5.0 sits between them.
_MAX_STRIPE_COHERENCE = 5.0


@pytest.mark.parametrize("attacked", [_TREERING_ATTACKED, _TREERING_REPEAT])
def test_treering_attacked_has_no_added_grain(attacked):
    """The Tree-Ring attack must not leave visible noise texture (user-facing quality)."""
    _skip_unless_fixture(_TREERING_WATERMARKED, attacked)
    src = _highfreq_energy(_grayscale(_TREERING_WATERMARKED))
    out = _highfreq_energy(_grayscale(attacked))
    assert out <= src * _MAX_GRAIN_RATIO, (
        f"{attacked.name} added grain: {out:.2f} vs source {src:.2f} "
        f"(limit {src * _MAX_GRAIN_RATIO:.2f})"
    )


@pytest.mark.parametrize(
    ("source", "attacked", "label"),
    [
        (_SYNTHID_WATERMARKED, _SYNTHID_ATTACKED_SYNTHID_CHAIN, "synthid chain"),
        (_SYNTHID_WATERMARKED, _SYNTHID_ATTACKED, "unknown chain"),
        (_TREERING_WATERMARKED, _TREERING_ATTACKED, "tree_ring chain"),
    ],
)
def test_attacked_output_has_no_vertical_banding(source, attacked, label):
    """No chain may leave FFT-style vertical stripes (user-facing quality)."""
    _skip_unless_fixture(source, attacked)
    coherence = _residual_stripe_coherence(source, attacked)
    assert coherence <= _MAX_STRIPE_COHERENCE, (
        f"{label} output shows vertical banding: column coherence {coherence:.2f} "
        f"(limit {_MAX_STRIPE_COHERENCE})"
    )


def test_every_chain_engine_script_exists():
    """Each engine named in ATTACK_CHAINS must be a real script in service/scripts/."""
    from attack_chain import ATTACK_CHAINS

    for scheme, chain in ATTACK_CHAINS.items():
        for step in chain:
            engine = step["engine"]
            assert (SCRIPTS / engine).is_file(), (
                f"scheme {scheme!r} names missing engine {engine!r}"
            )
