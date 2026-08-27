"""Tests for the optional reverse-SynthID scorer adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import image_meta
from image_meta import ImageInspectReport, run_synthid_score

SCORE_SCRIPT = SCRIPTS / "score_synthid.py"


def test_score_synthid_cli_unavailable_without_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("REVERSE_SYNTHID_DIR", raising=False)
    dummy = tmp_path / "img.png"
    dummy.write_bytes(b"not really an image")

    r = subprocess.run(
        [sys.executable, str(SCORE_SCRIPT), str(dummy)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert r.returncode == 3
    assert "REVERSE_SYNTHID_DIR" in (r.stderr or "")


def test_run_synthid_score_unconfigured_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("REVERSE_SYNTHID_DIR", raising=False)
    assert run_synthid_score(Path("x.png")) is None


def test_run_synthid_score_unavailable_returns_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=3, stdout="", stderr="unavailable")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_synthid_score(Path("x.png"), upstream_dir=str(tmp_path / "upstream"))

    assert result is not None
    assert result.get("available") is False
    assert "unavailable" in result.get("error", "")


def test_run_synthid_score_prefers_checkout_venv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    upstream = tmp_path / "upstream"
    venv_python = upstream / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\n")
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    run_synthid_score(Path("img.png"), upstream_dir=str(upstream))

    assert captured["cmd"][0] == str(venv_python)


def test_run_synthid_score_falls_back_to_sys_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    upstream = tmp_path / "upstream"  # no .venv present
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    run_synthid_score(Path("img.png"), upstream_dir=str(upstream))

    assert captured["cmd"][0] == sys.executable


def test_run_synthid_score_parses_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "available": True,
        "is_watermarked": True,
        "confidence": 0.91,
        "phase_match": 0.65,
    }
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_synthid_score(Path("img.png"), upstream_dir=str(tmp_path / "upstream"))

    assert result == payload
    assert "--json" in captured["cmd"]
    assert "--upstream-dir" in captured["cmd"]
    assert str(tmp_path / "upstream") in captured["cmd"]


def test_run_synthid_score_runtime_error_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    result = run_synthid_score(Path("img.png"), upstream_dir=str(tmp_path / "upstream"))

    assert result is not None
    assert result.get("available") is False
    assert "boom" in result.get("error", "")


def test_inspect_image_cli_prints_synthid_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "inspect_image_cli", str(SCRIPTS / "inspect_image.py")
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    report = ImageInspectReport(
        path="shot.png",
        format="png",
        has_c2pa=False,
        has_ai_metadata=False,
        synthid={
            "available": True,
            "is_watermarked": True,
            "confidence": 0.91,
        },
    )
    img = tmp_path / "shot.png"
    img.write_bytes(b"not really an image")
    monkeypatch.setattr(cli, "inspect_image", lambda path, synthid_dir=None: report)
    monkeypatch.setattr(sys, "argv", ["inspect_image.py", str(img)])

    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "SynthID score: confidence 0.910 (watermarked: yes)" in out


def test_inspect_report_to_dict_includes_synthid():
    report = ImageInspectReport(
        path="x.png",
        format="png",
        has_c2pa=False,
        has_ai_metadata=False,
        synthid={"available": True, "confidence": 0.8},
    )
    assert report.to_dict()["synthid"]["confidence"] == 0.8

    empty = ImageInspectReport(
        path="x.png",
        format="png",
        has_c2pa=False,
        has_ai_metadata=False,
    )
    assert empty.to_dict()["synthid"] is None


def test_synthid_score_http_blocks_redirect(tmp_path: Path):
    """Ensure _synthid_score_http refuses 302 redirects to prevent SSRF and key leakage."""
    import http.server
    import threading

    state: dict = {"collector_port": None}
    captured: dict = {}

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{state['collector_port']}/leak",
            )
            self.end_headers()

        def log_message(self, format, *args):
            pass

    class Collector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            captured["hit"] = True
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            captured["hit"] = True
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    collector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    redirector = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    state["collector_port"] = collector.server_address[1]
    t1 = threading.Thread(target=collector.serve_forever, daemon=True)
    t2 = threading.Thread(target=redirector.serve_forever, daemon=True)
    t1.start()
    t2.start()

    img = tmp_path / "target.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    try:
        res = image_meta._synthid_score_http(
            img,
            f"http://127.0.0.1:{redirector.server_address[1]}",
            api_key="secret-key",
            timeout=2.0,
        )
        assert res is not None
        assert res.get("available") is False
        assert (
            "unreachable" in res.get("error", "").lower()
            or "httperror" in res.get("error", "").lower()
        )
        assert captured == {}, "redirect target must not receive any forwarded request"
    finally:
        collector.shutdown()
        redirector.shutdown()
