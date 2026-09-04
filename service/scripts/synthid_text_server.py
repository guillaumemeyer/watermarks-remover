#!/usr/bin/env python3
"""Tiny stdlib HTTP sidecar exposing SynthID and research text watermarking.

Runs inside the wr-markllm harness / sidecar image so the published core
image never bundles heavy ML dependencies (PyTorch, Transformers, MarkLLM).
The core service calls this sidecar for text watermarking when
WATERMARKS_SYNTHID_TEXT_URL is set (see compose.yaml / .env.example).

Endpoints:
    GET  /health          -> {"ok": true, "version": ...}
    POST /watermark       -> {"text": str, "keys": list[int], "options": dict}
                          -> {"ok": true, "kind": "text", "watermarked_text": str, "report": dict}
    POST /watermark/batch -> {"files": [...]} -> {"ok": true, "results": [...]}

Hardening mirrors synthid_score_server.py and server.py: optional bearer key,
input size caps, unprivileged user, read-only rootfs with a /tmp tmpfs.
Intended for the compose network or loopback/trusted network only.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from text_watermark import normalize_keys, parse_watermark_options  # noqa: E402

VERSION = os.environ.get("WATERMARKS_SYNTHID_TEXT_SERVER_VERSION", "dev")

MAX_INPUT_BYTES = int(os.environ.get("WATERMARKS_MAX_INPUT_BYTES", str(256 << 20)))
MAX_BODY_BYTES = MAX_INPUT_BYTES + (MAX_INPUT_BYTES >> 1)
MAX_BATCH_FILES = int(os.environ.get("WATERMARKS_MAX_BATCH_FILES", "50"))

API_KEY = os.environ.get("WATERMARKS_SYNTHID_TEXT_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("WATERMARKS_SYNTHID_TEXT_MODEL", "facebook/opt-1.3b").strip()
MARKLLM_DIR = os.environ.get("MARKLLM_DIR", "/opt/markllm").strip()

MAX_CACHED_MODELS = int(os.environ.get("WATERMARKS_MAX_CACHED_MODELS", "2"))
_MODEL_CACHE: OrderedDict[str, Any] = OrderedDict()
_MODEL_LOCK = threading.Lock()


def _json_ok(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _generate_watermarked_sample(
    text: str,
    keys: list[int] | None,
    options: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Generate watermarked text using MarkLLM."""
    opts = options or {}
    scheme = opts.get("scheme", "synthid").lower()
    model_name = opts.get("model", DEFAULT_MODEL)
    seed = opts.get("seed")
    max_new_tokens = int(opts.get("max_new_tokens", 200))
    min_length = int(opts.get("min_length", 0))
    temperature = opts.get("temperature")
    top_p = opts.get("top_p")

    upstream = Path(MARKLLM_DIR).expanduser().resolve()
    if not upstream.is_dir():
        raise RuntimeError(f"MarkLLM upstream directory not found: {MARKLLM_DIR}")

    from detect_text_watermark import (
        SCHEMES,
        _generate,
        _load_algorithm,
        _resolve_config,
        resolve_device,
    )

    alg = SCHEMES.get(scheme, "SynthID")
    device = resolve_device(opts.get("device"))
    config_path = _resolve_config(upstream, alg, opts.get("config"))
    offline = bool(opts.get("offline", False))
    cache_key = f"{alg}:{model_name}:{device}:{config_path}:{temperature}:{top_p}:{offline}"
    keys_tag = ",".join(str(k) for k in keys) if keys is not None else "<default>"
    cache_key = f"{cache_key}:{keys_tag}"

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            wm = _MODEL_CACHE[cache_key]
            _MODEL_CACHE.move_to_end(cache_key)
        else:
            wm = _load_algorithm(
                upstream,
                alg,
                config_path,
                model_name,
                device,
                offline=offline,
                temperature=temperature,
                top_p=top_p,
                watermark_keys=keys,
            )
            if len(_MODEL_CACHE) >= MAX_CACHED_MODELS:
                _MODEL_CACHE.popitem(last=False)
            _MODEL_CACHE[cache_key] = wm

        watermarked, _ = _generate(
            wm,
            text,
            seed=int(seed) if seed is not None else None,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            need_unwatermarked=False,
        )

    report = {
        "scheme_used": scheme,
        "model": model_name,
        "watermarked_chars": len(watermarked),
        "keys_used": keys,
    }
    return watermarked, report


class Handler(BaseHTTPRequestHandler):
    server_version = f"watermarks-remover-synthid-text/{VERSION}"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _read_json(self) -> dict[str, Any] | None:
        raw_len = self.headers.get("Content-Length")
        if not raw_len or not raw_len.isdigit():
            return None
        length = int(raw_len)
        if length > MAX_BODY_BYTES:
            return None
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            return None
        return body if isinstance(body, dict) else None

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        data = _json_ok(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return
        if urlparse(self.path).path == "/health":
            self._respond(HTTPStatus.OK, {"ok": True, "version": VERSION})
        else:
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._respond(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": "unauthorized"})
            return

        path = urlparse(self.path).path
        if path not in ("/watermark", "/watermark/batch"):
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        body = self._read_json()
        if body is None:
            raw_len = self.headers.get("Content-Length")
            oversized = raw_len is not None and raw_len.isdigit() and int(raw_len) > MAX_BODY_BYTES
            self._respond(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE if oversized else HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "invalid request body"},
            )
            return

        if path == "/watermark/batch":
            self._handle_batch(body)
        else:
            self._handle_single(body)

    def _parse_entry(self, entry: dict[str, Any]) -> tuple[str, list[int] | None, dict[str, Any]]:
        text = entry.get("text")
        if text is None and "file" in entry:
            raw_file = entry["file"]
            if not isinstance(raw_file, str):
                raise ValueError("'file' must be a base64-encoded string")
            try:
                decoded = base64.b64decode(raw_file, validate=True)
            except (binascii.Error, UnicodeDecodeError) as e:
                raise ValueError(f"failed to decode base64 'file': {e}") from e
            if len(decoded) > MAX_INPUT_BYTES:
                raise ValueError(f"decoded file exceeds input size cap ({MAX_INPUT_BYTES} bytes)")
            try:
                text = decoded.decode("utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"decoded file is not valid UTF-8: {e}") from e

        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' must be a non-empty string")
        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            raise ValueError(f"'text' exceeds input size cap ({MAX_INPUT_BYTES} bytes)")

        keys = normalize_keys(entry.get("keys"))
        options = parse_watermark_options(entry.get("options"))
        return text, keys, options

    def _handle_single(self, body: dict[str, Any]) -> None:
        try:
            text, keys, options = self._parse_entry(body)
            watermarked_text, report = _generate_watermarked_sample(text, keys, options)
            self._respond(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "kind": "text",
                    "watermarked_text": watermarked_text,
                    "report": report,
                },
            )
        except ValueError as e:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
        except Exception as e:
            self.log_error("watermark error: %r", e)
            self._respond(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "watermark generation failed"},
            )

    def _handle_batch(self, body: dict[str, Any]) -> None:
        files = body.get("files")
        if not isinstance(files, list):
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "'files' must be a list"})
            return
        if not files:
            self._respond(
                HTTPStatus.BAD_REQUEST, {"ok": False, "error": "'files' must not be empty"}
            )
            return
        if len(files) > MAX_BATCH_FILES:
            self._respond(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"'files' exceeds the {MAX_BATCH_FILES}-file batch limit"},
            )
            return

        results: list[dict[str, Any]] = []
        for entry in files:
            if not isinstance(entry, dict):
                results.append(
                    {"name": "", "ok": False, "error": "each entry in 'files' must be an object"}
                )
                continue
            name = entry.get("name") if isinstance(entry.get("name"), str) else ""
            try:
                text, keys, options = self._parse_entry(entry)
                watermarked_text, report = _generate_watermarked_sample(text, keys, options)
                results.append(
                    {
                        "name": name,
                        "ok": True,
                        "kind": "text",
                        "watermarked_text": watermarked_text,
                        "report": report,
                    }
                )
            except ValueError as e:
                results.append({"name": name, "ok": False, "error": str(e)})
            except Exception as e:
                self.log_error("batch watermark error for %r: %r", name, e)
                results.append({"name": name, "ok": False, "error": "generation error"})

        self._respond(HTTPStatus.OK, {"ok": True, "results": results})


def main() -> int:
    global API_KEY  # noqa: PLW0603 — CLI overrides env
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--host", default=os.environ.get("WATERMARKS_SYNTHID_TEXT_SERVER_HOST", "127.0.0.1")
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WATERMARKS_SYNTHID_TEXT_SERVER_PORT", "8767")),
    )
    p.add_argument("--api-key", default=API_KEY, help="require this bearer token (default: none)")
    args = p.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding {args.host} — intended for a trusted network only",
            file=sys.stderr,
        )
    API_KEY = args.api_key
    print(
        f"synthid text sidecar {VERSION} on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
