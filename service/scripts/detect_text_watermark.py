#!/usr/bin/env python3
"""Optional MarkLLM text-watermark harness backed by an external THU-BPM/MarkLLM checkout.

This script does NOT vendor upstream code. It imports ``AutoWatermark`` from a
user-provided checkout (https://github.com/THU-BPM/MarkLLM) at runtime, using
that environment's optional dependencies (torch, transformers, datasets, ...).

MarkLLM is Apache-2.0. It is an independent verification harness: detection is
only valid against the SAME scheme config + keys used at generation. It cannot
certify that a vendor detector will fail on the given text.

Subcommands:
  detect    run detection on a text file with a known scheme/config
  watermark generate watermarked (and optionally unwatermarked) sample text
            from a prompt, for controlled before/after experiments

Exit codes:
  0  success
  1  runtime error (model load, detection/generation failure)
  2  bad input (missing/unreadable file, binary input, bad args)
  3  unavailable (not configured / missing checkout / missing deps)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import socketserver
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import emit_json, eprint, read_text_input, safe_write_text  # noqa: E402

# Scheme name as the user types it -> MarkLLM algorithm name (config/{ALG}.json).
# All schemes ship configs in the MarkLLM checkout; SIR additionally needs
# its transform/embedding models downloaded per the MarkLLM README.
SCHEMES = {
    "kgw": "KGW",
    "synthid": "SynthID",
    "synthid-text": "SynthID",
    "exp": "EXP",
    "unigram": "Unigram",
    "sir": "SIR",
}

DEFAULT_MODEL = "facebook/opt-1.3b"

# Algorithm configs are ~200 B (KGW/SynthID). Cap well above that so a crafted
# or accidental huge file is refused before either this script or upstream
# reads it into memory.
MAX_CONFIG_BYTES = 1 << 20


class _Unavailable(RuntimeError):
    """Backend present but unusable (unconfigured checkout, missing deps)."""


def resolve_upstream(raw: str | None) -> Path | None:
    if not raw:
        return None
    upstream = Path(raw).expanduser().resolve()
    if not upstream.is_dir():
        return None
    return upstream


def resolve_device(raw: str | None) -> str:
    """Resolve the ``auto`` device hint to a concrete torch device."""
    if raw and raw != "auto":
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Never auto-select mps: SynthID/KGW build torch.Generator(device=...),
        # which supports only cpu/cuda and raises RuntimeError on 'mps' (Apple
        # Silicon). Fall through to cpu. Pass --device mps explicitly to override.
    except Exception:  # noqa: S110 - optional torch device detection
        pass
    return "cpu"


def _load_algorithm(
    upstream: Path,
    alg: str,
    config: Path,
    model: str,
    device: str,
    offline: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    watermark_keys: list[int] | None = None,
):
    """Import the checkout and build an ``AutoWatermark`` instance.

    ``temperature``/``top_p`` (when not None) are folded into the generation
    kwargs so callers can control sampling. ``watermark_keys`` (when not None)
    overrides the algorithm's watermark keys in the config *before* the model
    is constructed, so the logits processor bakes in the requested keys rather
    than the config file's defaults.
    """
    gen_kwargs_extra: dict[str, float] = {}
    if temperature is not None:
        gen_kwargs_extra["temperature"] = temperature
    if top_p is not None:
        gen_kwargs_extra["top_p"] = top_p
    sys.path.insert(0, str(upstream))
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import AutoWatermark
    except ImportError as e:
        raise _Unavailable(f"MarkLLM dependencies missing: {e}") from e

    # --offline: never contact the HF hub. local_files_only makes transformers
    # fail fast instead of hanging, and HF_HUB_OFFLINE covers the lower-level
    # hub calls. Custom-code execution is not possible either way: transformers
    # only honors auto_map/trust_remote_code when explicitly enabled, which is
    # never done here.
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    load_kwargs = {"local_files_only": True} if offline else {}

    tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
    lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs).to(device)
    transformers_config = TransformersConfig(
        model=lm,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        min_length=0,
        do_sample=True,
        no_repeat_ngram_size=4,
        **gen_kwargs_extra,
    )

    # Apply the requested watermark keys before construction so the logits
    # processor is built with them, instead of the config file's defaults.
    tmp: str | None = None
    if watermark_keys is not None:
        data = json.loads(config.read_text("utf-8"))
        data["keys"] = [int(k) for k in watermark_keys]
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="watermarks-keys-", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            config = Path(tmp)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    try:
        return AutoWatermark.load(
            alg,
            algorithm_config=str(config),
            transformers_config=transformers_config,
        )
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _threshold_from_config(config: Path) -> float | None:
    try:
        data = json.loads(config.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("threshold", "z_threshold"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _resolve_config(upstream: Path, alg: str, config: str | None) -> Path:
    path = Path(config).expanduser().resolve() if config else upstream / "config" / f"{alg}.json"
    if not path.is_file():
        raise _Unavailable(f"MarkLLM config not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as e:
        raise _Unavailable(f"cannot stat MarkLLM config {path}: {e}") from e
    if size > MAX_CONFIG_BYTES:
        raise _Unavailable(f"MarkLLM config too large ({size} bytes > {MAX_CONFIG_BYTES}): {path}")
    return path


def _generate(
    wm: Any,
    prompt: str,
    seed: int | None,
    max_new_tokens: int,
    min_length: int = 0,
    need_unwatermarked: bool = True,
) -> tuple[str, str | None]:
    """Generate watermarked (and optionally unwatermarked) text for *prompt*."""
    if seed is not None:
        import torch

        torch.manual_seed(seed)
    wm.config.gen_kwargs["max_new_tokens"] = max_new_tokens
    wm.config.gen_kwargs["min_length"] = min_length
    watermarked = wm.generate_watermarked_text(prompt)
    unwatermarked = wm.generate_unwatermarked_text(prompt) if need_unwatermarked else None
    return watermarked, unwatermarked


def _detect_payload(wm: Any, text: str, threshold: float | None) -> dict[str, Any]:
    """Same-config detection payload (is_watermarked/score/threshold)."""
    result = wm.detect_watermark(text, return_dict=True)
    is_watermarked = bool(result.get("is_watermarked", False))
    score = result.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    return {
        "is_watermarked": is_watermarked,
        "score": score,
        "threshold": threshold,
    }


def _cmd_detect(args: argparse.Namespace, upstream: Path, alg: str) -> int:
    if args.path != "-" and not Path(args.path).is_file():
        eprint(f"not a file: {args.path}")
        return 2
    text = read_text_input(args.path, allow_binary=args.force_text)

    device = resolve_device(args.device)

    try:
        config = _resolve_config(upstream, alg, args.config)
        threshold = _threshold_from_config(config)
        wm = _load_algorithm(
            upstream,
            alg,
            config,
            args.model,
            device,
            offline=args.offline,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        det = _detect_payload(wm, text, threshold)
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"detection error: {e}")
        return 1

    is_watermarked = det["is_watermarked"]
    score = det["score"]

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "scheme": alg,
        "config": str(config),
        "model": args.model,
        "device": device,
        "is_watermarked": is_watermarked,
        "score": score,
        "threshold": threshold,
    }

    if args.json:
        emit_json(payload)
    else:
        label = "watermarked" if is_watermarked else "not watermarked"
        score_txt = f"{score:.4f}" if score is not None else "n/a"
        thresh_txt = f"{threshold:.4f}" if threshold is not None else "n/a"
        print(f"{alg}: {label} (score {score_txt}, threshold {thresh_txt})")

    return 0


def _cmd_watermark(args: argparse.Namespace, upstream: Path, alg: str) -> int:
    prompt = read_text_input(args.prompt, allow_binary=args.force_text)

    device = resolve_device(args.device)

    try:
        config = _resolve_config(upstream, alg, args.config)
        wm = _load_algorithm(
            upstream,
            alg,
            config,
            args.model,
            device,
            offline=args.offline,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        watermarked, unwatermarked = _generate(
            wm,
            prompt,
            args.seed,
            args.max_new_tokens,
            args.min_length,
            need_unwatermarked=bool(args.unwatermarked_output),
        )
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"generation error: {e}")
        return 1

    wm_out = "-" if args.watermarked_output is None else args.watermarked_output
    safe_write_text(wm_out, watermarked)
    if unwatermarked is not None:
        safe_write_text(args.unwatermarked_output, unwatermarked)

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "scheme": alg,
        "config": str(config),
        "model": args.model,
        "device": device,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "watermarked_output": wm_out,
        "unwatermarked_output": args.unwatermarked_output,
        "watermarked_chars": len(watermarked),
        "unwatermarked_chars": len(unwatermarked) if unwatermarked is not None else None,
    }

    if args.json:
        emit_json(payload)
    else:
        print(f"{alg}: watermarked sample ({payload['watermarked_chars']} chars) -> {wm_out}")
        if unwatermarked is not None:
            print(
                f"      unwatermarked sample ({payload['unwatermarked_chars']} chars) -> {args.unwatermarked_output}"
            )


def _handle_serve_request(wm: Any, req: dict[str, Any], threshold: float | None) -> dict[str, Any]:
    """Handle one JSON-lines request; never raises (responds with ok:false)."""
    rid = req.get("id")
    op = req.get("op")
    if op == "exit":
        return {"ok": True, "id": rid}
    try:
        if op == "watermark":
            prompt = req.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("'prompt' must be a non-empty string")
            # Per-request generation knobs.
            for key in ("temperature", "top_p"):
                value = req.get(key)
                if isinstance(value, (int, float)):
                    wm.config.gen_kwargs[key] = float(value)
            watermarked, unwatermarked = _generate(
                wm,
                prompt,
                req.get("seed"),
                req.get("max_new_tokens", 200),
                req.get("min_length", 0),
                need_unwatermarked=True,
            )
            return {
                "ok": True,
                "id": rid,
                "watermarked": watermarked,
                "unwatermarked": unwatermarked,
                "watermarked_chars": len(watermarked),
                "unwatermarked_chars": len(unwatermarked),
            }
        if op == "detect":
            text = req.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("'text' must be a non-empty string")
            det = _detect_payload(wm, text, threshold)
            return {"ok": True, "id": rid, **det}
        return {"ok": False, "id": rid, "error": f"unknown op {op!r}"}
    except Exception as e:  # a bad request must not kill the worker
        return {"ok": False, "id": rid, "error": str(e)}


def _cmd_serve(args: argparse.Namespace, upstream: Path, alg: str) -> int:
    """Serve watermark/detect requests over JSON-lines stdin/stdout.

    Loads the MarkLLM model once and keeps it resident so callers (e.g. the
    SynthID-text benchmark) can run many operations without paying the
    torch + model load cost per call. Protocol:

      request:  {"op": "watermark", "id": N, "prompt": str, "seed": int|None,
                 "max_new_tokens": int, "min_length": int,
                 "temperature": float|None, "top_p": float|None}
                {"op": "detect", "id": N, "text": str}
                {"op": "exit", "id": N}

    Per-request "temperature"/"top_p" override the worker's generation kwargs;
    omitted keys keep the current value.
      response: {"ok": true, "id": N, ...} | {"ok": false, "id": N, "error": str}

    The first stdout line is a {"ready": true, ...} handshake emitted after
    model load. Errors on one request never kill the worker.
    """
    device = resolve_device(args.device)
    try:
        config = _resolve_config(upstream, alg, args.config)
        threshold = _threshold_from_config(config)
        wm = _load_algorithm(
            upstream,
            alg,
            config,
            args.model,
            device,
            offline=args.offline,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except _Unavailable as e:
        eprint(str(e))
        return 3
    except Exception as e:
        eprint(f"serve load error: {e}")
        return 1

    def respond(payload: dict[str, Any]) -> None:
        print(json.dumps(payload), flush=True)

    ready: dict[str, Any] = {
        "ready": True,
        "scheme": alg,
        "model": args.model,
        "device": device,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    lock = threading.Lock()
    server: socketserver.ThreadingTCPServer | None = None
    if args.port >= 0:
        # Loopback TCP listener so other processes (e.g. the rewrite
        # subprocess's MarkLLM detector) can reuse this resident model
        # instead of cold-starting their own. Port 0 = ephemeral.
        server = _serve_socket_server(wm, threshold, args.port, lock)
        ready["port"] = server.server_address[1]
    respond(ready)

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                respond({"ok": False, "error": "invalid JSON request"})
                continue
            if not isinstance(req, dict):
                respond({"ok": False, "error": "request must be a JSON object"})
                continue
            with lock:
                resp = _handle_serve_request(wm, req, threshold)
            respond(resp)
            if req.get("op") == "exit":
                return 0
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    return 0


def _serve_socket_server(
    wm: Any, threshold: float | None, port: int, lock: threading.Lock
) -> socketserver.ThreadingTCPServer:
    """A loopback JSON-lines TCP server sharing this process's model."""

    class _Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            f = self.request.makefile("r", encoding="utf-8")
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    if not isinstance(req, dict):
                        raise ValueError("request must be a JSON object")
                except (json.JSONDecodeError, ValueError):
                    resp: dict[str, Any] = {"ok": False, "error": "invalid JSON request"}
                else:
                    with lock:
                        resp = _handle_serve_request(wm, req, threshold)
                try:
                    self.request.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except OSError:
                    return
                if isinstance(req, dict) and req.get("op") == "exit":
                    return

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    _Server.address_family = socket.AF_INET
    srv = _Server(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="MarkLLM checkout root (default: $MARKLLM_DIR)",
    )
    p.add_argument(
        "--scheme",
        required=True,
        choices=sorted(SCHEMES),
        help="Watermark scheme to use (kgw, synthid)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Algorithm config JSON (default: <checkout>/config/<ALG>.json)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (default: unset -> MarkLLM/HF default)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Generation nucleus-sampling top-p (default: unset -> MarkLLM/HF default)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MARKLLM_MODEL", DEFAULT_MODEL),
        help=f"HF causal LM for scoring (default: $MARKLLM_MODEL or {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|cuda|mps (default: auto)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Never contact the HF hub: load the scoring model from the local "
        "cache only (fails fast if not cached)",
    )
    p.add_argument(
        "--force-text",
        action="store_true",
        help="Process input even when it looks like a binary container",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect", help="Detect a scheme watermark in text")
    detect.add_argument("path", help="Text file to detect on, or - for stdin")
    _add_common(detect)
    detect.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    detect.set_defaults(handler=_cmd_detect)

    wm = sub.add_parser("watermark", help="Generate watermarked sample text")
    wm.add_argument("prompt", help="Prompt file, or - for stdin")
    wm.add_argument(
        "-o",
        "--watermarked-output",
        default=None,
        help="Output path for the watermarked sample (default: stdout)",
    )
    wm.add_argument(
        "-o2",
        "--unwatermarked-output",
        default=None,
        help="Also write an unwatermarked sample to this path",
    )
    wm.add_argument("--max-new-tokens", type=int, default=200)
    wm.add_argument("--min-length", type=int, default=0)
    wm.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    _add_common(wm)
    wm.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    wm.set_defaults(handler=_cmd_watermark)

    serve = sub.add_parser(
        "serve", help="Serve watermark/detect over JSON-lines stdin (persistent worker)"
    )
    _add_common(serve)
    serve.add_argument(
        "--port",
        type=int,
        default=-1,
        help="Also listen on 127.0.0.1:PORT (JSON-lines; 0 = ephemeral) for "
        "other processes to reuse this resident model (default: no listener)",
    )
    serve.set_defaults(handler=_cmd_serve)

    args = p.parse_args()

    if args.cmd == "detect" and args.path != "-" and not Path(args.path).is_file():
        eprint(f"not a file: {args.path}")
        return 2

    raw_upstream = args.upstream_dir or os.environ.get("MARKLLM_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)
    if upstream is None:
        eprint(
            "MarkLLM not configured: set MARKLLM_DIR or pass --upstream-dir",
        )
        return 3

    if not (upstream / "watermark").is_dir():
        eprint(f"MarkLLM checkout incomplete (no watermark/ dir): {upstream}")
        return 3

    alg = SCHEMES[args.scheme]
    return args.handler(args, upstream, alg)


if __name__ == "__main__":
    raise SystemExit(main())
