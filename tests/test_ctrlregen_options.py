"""Validate image quality controls without invoking diffusion models."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "scripts"))
import server


@pytest.mark.parametrize("value", [0.1, 0.25, 1])
def test_intensity_accepts_valid_numbers(value):
    assert (
        server._parse_clean_options({"ctrlregen_intensity": value})["ctrlregen_intensity"] == value
    )


@pytest.mark.parametrize(
    "value", [True, False, 0, -0.1, 1.1, "0.1", None, float("nan"), float("inf")]
)
def test_intensity_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        server._parse_clean_options({"ctrlregen_intensity": value})


def test_face_protection_requires_ctrlregen():
    with pytest.raises(ValueError):
        server._parse_clean_options({"protect_faces": True})
    assert server._parse_clean_options({"protect_faces": True, "remove_pixel": "ctrlregen"})[
        "protect_faces"
    ]


def test_image_options_reach_cleaner(monkeypatch):
    captured = {}
    monkeypatch.setattr(server, "classify_bytes", lambda *args: "image")

    def clean(src, dest, **kwargs):
        captured.update(kwargs)
        dest.write_bytes(b"output")
        return {"format": "png"}

    monkeypatch.setattr(server, "clean_image", clean)
    server._clean_payload(
        b"input",
        "sample.png",
        {"ctrlregen_intensity": 0.1, "protect_faces": True, "remove_pixel": "ctrlregen"},
    )
    assert captured["ctrlregen_intensity"] == 0.1
    assert captured["protect_faces"] is True


def test_openapi_describes_numeric_intensity():
    schema = server._clean_request_schema()["properties"]["options"]["properties"]
    assert schema["ctrlregen_intensity"]["type"] == "number"
    assert schema["ctrlregen_intensity"]["exclusiveMinimum"] is True
    assert schema["protect_faces"]["type"] == "boolean"


def test_face_protection_flag_reaches_adapter(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import image_meta

    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout=json.dumps({"available": True}), stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", run)
    result = image_meta.run_ctrlregen_clean(
        tmp_path / "input.png",
        tmp_path / "output.png",
        upstream_dir=str(tmp_path),
        protect_faces=True,
        intensity=0.1,
    )
    assert result["available"] is True
    assert "--protect-faces" in captured["command"]
    index = captured["command"].index("--intensity")
    assert captured["command"][index + 1] == "0.1"
