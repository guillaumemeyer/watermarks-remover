#!/usr/bin/env python3
"""HTTP sidecar exposing the MarkDiffusion DiffusionPurification remover.

Runs inside the wr-markdiffusion image so the published core image never
bundles the heavy ML deps. The core service calls this sidecar for pixel
watermark removal when WATERMARKS_MARKDIFFUSION_CLEAN_URL is set (see
compose.yaml / .env.example).

Endpoints:
    GET  /health  -> {"ok": true, "version": ...}
    POST /clean   -> {"file": <base64>, "name": "img.png", "options": {...}}
               -> {"ok": true, "cleaned": <base64>, "report": {...}}
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from markdiffusion_harness import (
    DEFAULT_MODEL,
    _Unavailable,
    _import_markdiffusion,
    _load_diffusion,
    resolve_device,
    resolve_upstream,
)

VERSION = os.environ.get("WATERMARKS_SERVER_VERSION", "dev")
MAX_INPUT_BYTES = int(os.environ.get("WATERMARKS_MAX_INPUT_BYTES", str(256 << 20)))
MAX_BODY_BYTES = MAX_INPUT_BYTES + (MAX_INPUT_BYTES >> 1)
API_KEY = os.environ.get("WATERMARKS_MARKDIFFUSION_API_KEY", "").strip()


def _json_ok(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _run_clean(data: bytes, name: str, options: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    upstream_raw = os.environ.get("MARKDIFFUSION_DIR") or None
    upstream = resolve_upstream(upstream_raw)

    intensity = float(options.get("intensity", 0.3))
    model = str(options.get("model") or os.environ.get("MARKDIFFUSION_MODEL") or DEFAULT_MODEL)
    size = int(options.get("size", 512))
    steps = int(options.get("steps", 50))
    device_hint = str(options.get("device", "auto"))
    guidance = float(options.get("guidance", 7.5))
    prompt = str(options.get("prompt") or "")

    suffix = Path(name).suffix.lower() or ".png"
    with tempfile.TemporaryDirectory(prefix="wm-markdiffusion-") as tmp:
        src_path = Path(tmp) / f"input{suffix}"
        out_path = Path(tmp) / f"output.png"
        src_path.write_bytes(data)

        device = resolve_device(device_hint)

        try:
            _import_markdiffusion(upstream)
            from markdiffusion.evaluation.tools.image_editor import DiffusionPurification
            from markdiffusion.utils import DiffusionConfig
            from PIL import Image

            pipe, scheduler = _load_diffusion(model, device, False, size)
            diffusion_config = DiffusionConfig(
                scheduler=scheduler,
                pipe=pipe,
                device=device,
                image_size=(size, size),
                num_inference_steps=steps,
                guidance_scale=guidance,
                inversion_type="ddim",
            )
            purifier = DiffusionPurification(
                diffusion_config,
                purification_strength=intensity,
                prompt=prompt,
            )
            image = Image.open(src_path).convert("RGB")
            purified = purifier.edit(image)
        except _Unavailable as e:
            return HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": str(e)}
        except Exception as e:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "ok": False,
                "error": f"purification error: {e}",
            }

        try:
            import io as _io
            buf = _io.BytesIO()
            purified.save(buf, format="PNG")
            cleaned_bytes = buf.getvalue()
            out_path.write_bytes(cleaned_bytes)
        except (OSError, ValueError) as e:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "ok": False,
                "error": f"cannot write output: {e}",
            }

        report = {
            "available": True,
            "upstream_dir": str(upstream) if upstream else None,
            "model": model,
            "device": device,
            "purification_intensity": intensity,
            "size": size,
            "steps": steps,
            "bytes_out": len(cleaned_bytes),
        }
        cleaned_b64 = base64.b64encode(cleaned_bytes).decode("ascii")
        return HTTPStatus.OK, {"ok": True, "cleaned": cleaned_b64, "report": report}


class Handler(BaseHTTPRequestHandler):
    server_version = f"watermarks-remover-markdiffusion/{VERSION}"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _read_json(self) -> dict[str, Any] | None:
        raw = self.headers.get("Content-Length")
        if raw is None or not raw.isdigit():
            return None
        length = int(raw)
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
        if urlparse(self.path).path != "/clean":
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        body = self._read_json()
        if body is None:
            raw_len = self.headers.get("Content-Length")
            oversized = (
                raw_len is not None and raw_len.isdigit() and int(raw_len) > MAX_BODY_BYTES
            )
            self._respond(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE if oversized else HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "invalid request body"},
            )
            return

        raw = body.get("file")
        if not isinstance(raw, str):
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing 'file' field"})
            return
        try:
            image_data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            self._respond(
                HTTPStatus.BAD_REQUEST, {"ok": False, "error": "'file' is not valid base64"}
            )
            return
        if len(image_data) > MAX_INPUT_BYTES:
            self._respond(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "file too large"}
            )
            return

        name = body.get("name", "input.png")
        if not isinstance(name, str):
            name = "input.png"
        options = body.get("options") or {}
        if not isinstance(options, dict):
            options = {}

        status, response = _run_clean(image_data, name, options)
        self._respond(status, response)


def main() -> int:
    global API_KEY  # noqa: PLW0603
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--host",
        default=os.environ.get("WATERMARKS_MARKDIFFUSION_SERVER_HOST", "127.0.0.1"),
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WATERMARKS_MARKDIFFUSION_SERVER_PORT", "8769")),
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
        f"markdiffusion clean sidecar {VERSION} on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
