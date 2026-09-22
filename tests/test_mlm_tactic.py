"""The `mlm` tactic reports why it can't run, and a probe can tell in advance.

The default /clean strategy ("paraphrase@0.8,mlm@0.2") failed with "mlm tactic
unavailable: Failed to import transformers.pipelines ... No module named 'PIL'"
on a host whose transformers imported pytesseract, and so PIL, from its
pipelines. The server's `find_spec("transformers")` gate passed there, because
the package was installed; only a real import shows the failure.
`mlm_import_error()` runs that import in a child interpreter. These tests drive
it with stand-in packages on PYTHONPATH, so they never import real torch and
pass the same with or without the stack installed.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text

PIL_FAILURE = (
    "Failed to import transformers.pipelines because of the following error "
    "(look up to see its traceback):\nNo module named 'PIL'"
)


@pytest.fixture
def fresh_mlm(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty pipeline cache and no CUDA probe, so _get_mlm() builds anew."""
    monkeypatch.setattr(rewrite_text, "_MLM_CACHE", {})
    monkeypatch.setattr(rewrite_text, "_cuda_available", lambda: False)


class _BrokenTransformers(types.ModuleType):
    """transformers whose lazy `pipeline` attribute fails like the reported host."""

    def __getattr__(self, name: str):
        if name == "pipeline":
            raise RuntimeError(PIL_FAILURE)
        raise AttributeError(name)  # e.g. the import system probing __path__


# --- _get_mlm / load_mlm ------------------------------------------------------


def test_import_failure_names_the_requirements_file(monkeypatch, fresh_mlm):
    monkeypatch.setitem(sys.modules, "transformers", _BrokenTransformers("transformers"))
    with pytest.raises(RuntimeError) as exc:
        rewrite_text.load_mlm()
    message = str(exc.value)
    assert message.startswith("mlm tactic unavailable: Failed to import transformers.pipelines")
    assert "No module named 'PIL'" in message
    assert "service/scripts/requirements-mlm.txt" in message


def test_model_load_failure_is_a_runtime_error_not_a_crash(monkeypatch, fresh_mlm):
    """An unreachable Hub with no cached weights raises OSError inside pipeline().

    Only RuntimeError maps to a 400 in the server; anything else became a bare
    500 "internal error" with the reason lost.
    """
    fake = types.ModuleType("transformers")

    def pipeline(task, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    fake.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake)
    with pytest.raises(RuntimeError, match=r"cannot load roberta-large: We couldn't connect"):
        rewrite_text.load_mlm()


def test_load_mlm_builds_the_pipeline_once(monkeypatch, fresh_mlm):
    built: list[tuple[str, dict]] = []
    fake = types.ModuleType("transformers")

    def pipeline(task, **kwargs):
        built.append((task, kwargs))
        tokenizer = types.SimpleNamespace(mask_token="<mask>")  # noqa: S106 - not a credential
        return types.SimpleNamespace(tokenizer=tokenizer)

    fake.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake)
    rewrite_text.load_mlm()
    rewrite_text.load_mlm()
    assert built == [("fill-mask", {"model": "roberta-large"})]
    assert rewrite_text._get_mlm()[1] == "<mask>"


def test_probe_imports_what_the_tactic_imports():
    # The probe is only a prediction if it attempts the tactic's own import.
    assert "from transformers import pipeline" in inspect.getsource(rewrite_text._get_mlm)
    assert "from transformers import pipeline" in rewrite_text._MLM_IMPORT_PROBE
    assert "import torch" in rewrite_text._MLM_IMPORT_PROBE


# --- mlm_import_error (real child interpreter, stand-in packages) ------------


def _stand_ins(root: Path, monkeypatch, *, torch: str = "", transformers: str) -> None:
    """Put fake torch/transformers packages ahead of site-packages for the child."""
    for name, body in (("torch", torch), ("transformers", transformers)):
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(body, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(root))


def test_probe_reports_the_reported_pil_failure(tmp_path, monkeypatch):
    _stand_ins(tmp_path, monkeypatch, transformers=f"raise RuntimeError({PIL_FAILURE!r})\n")
    assert rewrite_text.mlm_import_error() == (
        "RuntimeError: Failed to import transformers.pipelines because of the following "
        "error (look up to see its traceback): No module named 'PIL'"
    )


def test_probe_passes_when_the_stack_imports(tmp_path, monkeypatch):
    # Importable is all the probe checks: it must never build the pipeline
    # (that loads 1.4 GB of weights).
    _stand_ins(
        tmp_path,
        monkeypatch,
        transformers="def pipeline(*args, **kwargs):\n    raise SystemExit(7)\n",
    )
    assert rewrite_text.mlm_import_error() is None


def test_probe_requires_torch(tmp_path, monkeypatch):
    # transformers can import without torch and only fail when the pipeline
    # is built, so the probe checks torch itself.
    _stand_ins(
        tmp_path,
        monkeypatch,
        torch="raise ModuleNotFoundError(\"No module named 'torch'\")\n",
        transformers="def pipeline(*args, **kwargs):\n    pass\n",
    )
    assert rewrite_text.mlm_import_error() == "ModuleNotFoundError: No module named 'torch'"


def test_probe_reports_a_crash_without_output(tmp_path, monkeypatch):
    _stand_ins(tmp_path, monkeypatch, transformers="import os\nos._exit(3)\n")
    assert rewrite_text.mlm_import_error() == "import probe exited with status 3"


def test_probe_reports_the_last_stderr_line_of_a_hard_crash(tmp_path, monkeypatch):
    _stand_ins(
        tmp_path,
        monkeypatch,
        transformers=(
            "import os, sys\n"
            "sys.stderr.write('noise\\nFatal Python error: Segmentation fault\\n')\n"
            "sys.stderr.flush()\n"
            "os._exit(139)\n"
        ),
    )
    assert rewrite_text.mlm_import_error() == "Fatal Python error: Segmentation fault"


def test_probe_times_out_instead_of_hanging(tmp_path, monkeypatch):
    _stand_ins(tmp_path, monkeypatch, transformers="import time\ntime.sleep(60)\n")
    assert rewrite_text.mlm_import_error(timeout=0.5) == "import probe timed out after 0.5s"
