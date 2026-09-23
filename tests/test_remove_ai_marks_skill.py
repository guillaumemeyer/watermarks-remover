"""The remove-ai-marks skill ships no code, only instructions an agent runs.

So what can drift is the instructions themselves: the option names, the auth
variable, and the shell helpers. These tests pin the option list against the
server and run the helpers from SKILL.md against a live server that requires a
key.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = ROOT / "skills" / "remove-ai-marks" / "SKILL.md"
sys.path.insert(0, str(ROOT / "service" / "scripts"))

import server

API_KEY = "sekret"
HEALTH_CHECK = "wm /health -o /dev/null -w '%{http_code}\\n'"


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_skill_documents_every_clean_option():
    text = _skill_text()
    missing = sorted(opt for opt in server.ALLOWED_CLEAN_OPTIONS if f"`{opt}`" not in text)
    assert not missing


def test_skill_never_puts_a_base64_payload_on_the_command_line():
    # `curl -d "{\"file\": \"$(base64 < f)\"}"` passes the whole file as one
    # argument, which fails with "Argument list too long" above ~24 KB on
    # Windows and ~96 KB on Linux -- most PDFs and images.
    assert "$(base64" not in _skill_text()


def test_skill_health_check_is_the_documented_one():
    assert HEALTH_CHECK in _skill_text()


def _helpers() -> str:
    blocks = re.findall(r"```bash\n(.*?)```", _skill_text(), flags=re.S)
    matching = [block for block in blocks if "wm_post()" in block]
    assert len(matching) == 1, "SKILL.md must define the helpers in one bash block"
    return matching[0]


def _posix_shell() -> str | None:
    bash = shutil.which("bash")
    # On Windows, System32\bash.exe launches WSL, whose VM has its own loopback
    # and cannot reach a server bound to 127.0.0.1 here.
    if bash is None or (sys.platform == "win32" and "system32" in bash.lower()):
        return None
    probe = subprocess.run(
        [bash, "-c", "command -v curl && command -v base64"], capture_output=True, check=False
    )
    return bash if probe.returncode == 0 else None


BASH = _posix_shell()
needs_bash = pytest.mark.skipif(
    BASH is None, reason="needs a bash with curl and base64 on this machine's loopback"
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def service_url(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", API_KEY)
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _run(script: str, url: str, cwd: Path, key: str | None = API_KEY):
    env = {**os.environ, "WATERMARKS_SERVICE_URL": url}
    env.update(NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    env.pop("WATERMARKS_SERVICE_API_KEY", None)
    if key is not None:
        env["WATERMARKS_SERVICE_API_KEY"] = key
    return subprocess.run(
        [BASH, "-c", _helpers() + "\n" + script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


@needs_bash
def test_health_check_tells_a_missing_key_from_a_down_service(service_url, tmp_path):
    assert _run(HEALTH_CHECK, service_url, tmp_path).stdout.strip() == "200"
    # The skill used to probe /health without the key, and read the 401 as
    # "service down".
    assert _run(HEALTH_CHECK, service_url, tmp_path, key=None).stdout.strip() == "401"

    down = _run(HEALTH_CHECK, f"http://127.0.0.1:{_free_port()}", tmp_path)
    assert down.stdout.strip() == "000"


@needs_bash
def test_helpers_move_files_past_the_argument_limit(service_url, tmp_path):
    # ~120 KB, so its base64 (~160 KB) is past both the Windows (~32 KB) and the
    # Linux (128 KiB) limit on one command-line argument. A single carrier keeps
    # the server's work, and so the idle wait on the connection, short.
    body = "Olá\u200b mundo. ".encode() + b"The quick brown fox jumps over the lazy dog. " * 2_700
    (tmp_path / "big.txt").write_bytes(body)
    (tmp_path / "big.md").write_bytes(b"# T\n\n" + body)

    inspected = _run("wm_post inspect big.txt", service_url, tmp_path)
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["report"]["suspicious_total"] == 1

    cleaned = _run(
        'R=$(mktemp); wm_post clean big.md > "$R" && wm_save "$R" big.cleaned.md',
        service_url,
        tmp_path,
    )
    assert cleaned.returncode == 0, cleaned.stderr
    assert (tmp_path / "big.cleaned.md").read_bytes() == b"# T\n\n" + body.replace(
        "\u200b".encode(), b""
    )
    printed = json.loads(cleaned.stdout)
    assert "cleaned" not in printed  # the report, without the base64 payload
    assert printed["report"]["changed"] is True


@needs_bash
def test_helpers_forward_options_and_surface_errors(service_url, tmp_path):
    (tmp_path / "a.md").write_bytes(b"# T\n")
    bad = _run(
        'R=$(mktemp); wm_post clean a.md \'{"deep_images": "nope"}\' > "$R" && wm_save "$R" out.md',
        service_url,
        tmp_path,
    )
    assert bad.returncode != 0
    assert "deep_images" in bad.stderr
    assert not (tmp_path / "out.md").exists()  # no empty output file on an error
