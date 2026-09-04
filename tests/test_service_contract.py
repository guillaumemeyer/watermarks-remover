"""Tests for the fail-closed service contract (#196, #316).

Verifies:
- Preflight refusal contract documentation in SKILL.md
- install_skill.py reachability diagnostics (online, offline, suppression)
- check_staged.py and hook_written_file.py fail-closed behavior on service_unavailable
"""

from __future__ import annotations

import http.server
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import audit_lib
import check_staged
import hook_written_file
import install_skill


class _MockHealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "version": "0.7.0-test"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_service():
    server = http.server.HTTPServer(("127.0.0.1", 0), _MockHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    yield url
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_reachability_diagnostic_when_online(mock_service):
    reachable, msg = install_skill.check_service_reachability(url=mock_service)
    assert reachable is True
    assert "service reachable" in msg
    assert "v0.7.0-test" in msg


def test_reachability_diagnostic_when_offline():
    reachable, msg = install_skill.check_service_reachability(url="http://127.0.0.1:59998", timeout=0.2)
    assert reachable is False
    assert "service unreachable" in msg
    assert "make serve" in msg


def test_skill_markdown_contains_refusal_contract():
    skill_file = ROOT / "skills" / "remove-ai-marks" / "SKILL.md"
    assert skill_file.is_file()
    content = skill_file.read_text(encoding="utf-8")

    expected_refusal = (
        "The watermarks-remover service is offline or unreachable at $WATERMARKS_SERVICE_URL. "
        "No files or text have been modified. Start the service with `make serve` or `docker compose up -d` before retrying."
    )
    assert expected_refusal in content

    assert "Mandatory Preflight Check" in content
    assert "curl -sf \"$WM/health\"" in content
    assert "PROHIBITION (FAIL CLOSED)" in content
    assert "#196" in content


def test_clean_user_facing_text_contains_fail_closed_notice():
    skill_file = ROOT / "skills" / "clean-user-facing-text" / "SKILL.md"
    assert skill_file.is_file()
    content = skill_file.read_text(encoding="utf-8")
    assert "Fail-closed execution notice" in content
    assert "#196" in content


def test_audit_lib_treats_service_unavailable_as_actionable():
    item = {
        "path": "test.txt",
        "kind": "text",
        "status": "service_unavailable",
        "has_c2pa": False,
        "confidence": [],
    }
    assert audit_lib.is_actionable(item) is True

    item_error = {
        "path": "test.txt",
        "kind": "text",
        "error": "service_unavailable",
        "has_c2pa": False,
        "confidence": [],
    }
    assert audit_lib.is_actionable(item_error) is True


def test_check_staged_exits_nonzero_on_service_unavailable(tmp_path, monkeypatch, capsys):
    test_file = tmp_path / "staged.txt"
    test_file.write_text("sample content", encoding="utf-8")

    def mock_scan(path, check_stylometry=False):
        return {
            "path": str(path),
            "kind": "text",
            "status": "service_unavailable",
            "detail": "connection refused on 127.0.0.1:8765",
            "has_c2pa": False,
            "findings": [],
            "confidence": [],
        }

    monkeypatch.setattr(check_staged, "scan_file", mock_scan)
    with patch("sys.argv", ["check_staged.py", str(test_file)]):
        exit_code = check_staged.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "service_unavailable" in captured.err


def test_hook_written_file_check_exits_error_on_service_unavailable(tmp_path, monkeypatch):
    test_file = tmp_path / "written.txt"
    test_file.write_text("sample content", encoding="utf-8")

    def mock_scan(path):
        return {
            "path": str(path),
            "name": path.name,
            "kind": "text",
            "status": "service_unavailable",
            "detail": "service offline",
        }

    monkeypatch.setattr(hook_written_file, "scan_file", mock_scan)
    exit_code = hook_written_file.run_check(test_file)
    assert exit_code == hook_written_file.EXIT_HOOK_ERROR
