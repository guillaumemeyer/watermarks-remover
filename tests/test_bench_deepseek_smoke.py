"""Offline protocol tests; no model downloads or provider calls."""

from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service/scripts"))

import bench_deepseek_smoke as smoke
import server


@contextmanager
def serving(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@contextmanager
def endpoint(callback):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, dict(self.headers)))
            status, payload = callback(self.path, body)
            raw = json.dumps(payload).encode()
            self.send_response(status)
            if status == 307:
                self.send_header("Location", payload["location"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    with serving(Handler) as url:
        yield url, requests


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_API_KEY", "fake-provider-secret")
    monkeypatch.setenv("WATERMARKS_SERVER_API_KEY", "fake-core-secret")
    monkeypatch.delenv("WATERMARKS_REWRITE_MODEL", raising=False)
    monkeypatch.delenv("WATERMARKS_REWRITE_BASE_URL", raising=False)


def batch_response(_path, body):
    return 200, {
        "ok": True,
        "results": [
            {"name": item["name"], "ok": True, "watermarked_text": "Generated " + item["text"]}
            for item in body["files"]
        ],
    }


def rewrite_response(_path, _body):
    return 200, {"choices": [{"message": {"content": "Rewritten text."}}]}


def invoke(tmp_path, core_url, rewrite_url, *extra):
    out = tmp_path / "run.json"
    code = smoke.main(
        ["--core-url", core_url, "--rewrite-base-url", rewrite_url, "--out", str(out), *extra]
    )
    return code, json.loads(out.read_text()) if out.exists() else None


def test_real_core_dispatch_to_sidecar_and_existing_rewrite_client(tmp_path, monkeypatch):
    def generate(path, body):
        assert path == "/watermark"
        return 200, {
            "ok": True,
            "watermarked_text": "Generated " + body["text"],
            "report": {"scheme_used": "synthid"},
        }

    with endpoint(generate) as (sidecar_url, sidecar_calls):
        monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", sidecar_url)
        monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_API_KEY", "fake-sidecar-secret")
        monkeypatch.setattr(server, "API_KEY", "fake-core-secret")
        with serving(server.Handler) as core_url, endpoint(rewrite_response) as (ds_url, ds_calls):
            code, report = invoke(tmp_path, core_url, ds_url, "--model", "test-model")

    assert code == 0
    assert report["ok"] is True
    assert report["provider"] == "deepseek"
    assert report["requested_model"] == "test-model"
    assert report["watermark_verification"] == "not_performed"
    assert len(report["results"]) == len(sidecar_calls) == len(ds_calls) == 2
    for row, generation, rewrite in zip(report["results"], sidecar_calls, ds_calls, strict=True):
        assert generation[2]["Authorization"] == "Bearer fake-sidecar-secret"
        assert generation[1]["options"]["seed"] == 42
        assert rewrite[0] == "/v1/chat/completions"
        assert rewrite[2]["Authorization"] == "Bearer fake-provider-secret"
        assert rewrite[1]["model"] == "test-model"
        assert rewrite[1]["reasoning_effort"] == "none"
        assert row["watermarked_text"] in rewrite[1]["messages"][0]["content"]
        assert row["rewritten_text"] == "Rewritten text."
        assert row["generation_report"]["scheme_used"] == "synthid"
    serialized = json.dumps(report)
    assert "secret" not in serialized
    assert "cleared" not in serialized


def test_failed_generation_preserved_and_not_sent_to_provider(tmp_path):
    def partial(path, body):
        status, result = batch_response(path, body)
        result["results"][0] = {"name": "library", "ok": False, "error": "private server detail"}
        return status, result

    with endpoint(partial) as (core, _), endpoint(rewrite_response) as (ds, calls):
        code, report = invoke(tmp_path, core, ds)
    assert code == 1
    assert len(calls) == 1
    assert report["results"][0]["stage"] == "watermark"
    assert report["results"][0]["ok"] is False
    assert report["results"][1]["ok"] is True
    assert "private server detail" not in json.dumps(report)


@pytest.mark.parametrize("response", [None, {}, {"ok": True, "results": []}])
def test_bad_batch_never_calls_provider(tmp_path, response):
    with (
        endpoint(lambda *_: (200, response)) as (core, _),
        endpoint(rewrite_response) as (ds, calls),
    ):
        code, report = invoke(tmp_path, core, ds)
    assert code == 1
    assert report["stage"] == "watermark"
    assert not calls


def test_reordered_batch_is_rejected(tmp_path):
    def reordered(path, body):
        status, result = batch_response(path, body)
        result["results"].reverse()
        return status, result

    with endpoint(reordered) as (core, _), endpoint(rewrite_response) as (ds, calls):
        code, report = invoke(tmp_path, core, ds)
    assert code == 1
    assert report["stage"] == "watermark"
    assert not calls


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (401, {"error": "fake-provider-secret"}),
        (200, {"choices": []}),
        (200, {"choices": [{"message": {"content": " "}}]}),
    ],
)
def test_rewrite_failures_are_not_success(tmp_path, status, body, capsys):
    with endpoint(batch_response) as (core, _), endpoint(lambda *_: (status, body)) as (ds, _):
        code, report = invoke(tmp_path, core, ds)
    assert code == 1
    assert all(row["stage"] == "rewrite" and not row["ok"] for row in report["results"])
    assert "fake-provider-secret" not in json.dumps(report) + str(capsys.readouterr())
    if status == 401:
        assert report["results"][0]["http_status"] == 401


def test_redirect_never_forwards_credentials(tmp_path):
    with (
        endpoint(rewrite_response) as (destination, destination_calls),
        endpoint(batch_response) as (core, _),
        endpoint(lambda *_: (307, {"location": destination})) as (redirect, _),
    ):
        code, report = invoke(tmp_path, core, redirect)
    assert code == 1
    assert report["results"][0]["http_status"] == 307
    assert not destination_calls


def test_timeout_is_failure(tmp_path, monkeypatch):
    def timeout(*_args, **_kwargs):
        raise TimeoutError("fake-provider-secret")

    monkeypatch.setattr(smoke.rewrite_text, "call_openai_compatible", timeout)
    with endpoint(batch_response) as (core, _):
        code, report = invoke(tmp_path, core, "http://localhost:1")
    assert code == 1
    assert report["results"][0]["error"] == "TimeoutError"
    assert "fake-provider-secret" not in json.dumps(report)


def test_env_model_and_reasoning_off_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("WATERMARKS_REWRITE_MODEL", "env-model")
    with (
        endpoint(batch_response) as (core, core_calls),
        endpoint(
            lambda *_: (200, {"choices": [{"message": {"content": "fake-provider-secret"}}]})
        ) as (ds, calls),
    ):
        code, report = invoke(tmp_path, core, ds, "--reasoning-effort", "off")
    assert code == 0
    assert core_calls[0][2]["Authorization"] == "Bearer fake-core-secret"
    assert calls[0][1]["model"] == "env-model"
    assert "reasoning_effort" not in calls[0][1]
    assert report["results"][0]["rewritten_text"] == "[REDACTED]"


def test_remote_requires_opt_in_before_any_io(tmp_path):
    with (
        endpoint(batch_response) as (core, calls),
        pytest.raises(SystemExit, match="refusing to send content"),
    ):
        invoke(tmp_path, core, "https://api.deepseek.com")
    assert not calls


def test_missing_key_before_any_io(tmp_path, monkeypatch):
    monkeypatch.delenv("WATERMARKS_REWRITE_API_KEY")
    with endpoint(batch_response) as (core, calls):
        code, report = invoke(tmp_path, core, "http://localhost:1")
    assert code == 1
    assert report is None
    assert not calls


@pytest.mark.parametrize(
    "url", ["file:///tmp/data", "https://user:secret@example.com", "https://example.com?key=secret"]
)
def test_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        smoke.check_url(url, True)


def test_duplicate_fixture_ids_rejected(tmp_path):
    data = smoke.load_fixture(smoke.DEFAULT_FIXTURE)
    data["samples"][1]["id"] = data["samples"][0]["id"]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unique"):
        smoke.load_fixture(fixture)
