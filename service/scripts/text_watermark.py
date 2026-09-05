#!/usr/bin/env python3
"""Client and dispatch module for text watermark generation.

Routes watermark generation requests to the HTTP sidecar (synthid_text_server.py)
when WATERMARKS_SYNTHID_TEXT_URL is set, or falls back to a local MarkLLM
checkout when MARKLLM_DIR is set. When neither is configured, returns a
fail-soft explanatory error matching the repo's optional-harness conventions.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_SYNTHID_TEXT_TIMEOUT = 120.0

ALLOWED_WATERMARK_OPTIONS: frozenset[str] = frozenset(
    {
        "scheme",
        "seed",
        "max_new_tokens",
        "min_length",
        "temperature",
        "top_p",
        "model",
        "device",
        "config",
        "offline",
    }
)


def parse_watermark_options(raw: Any) -> dict[str, Any]:
    """Validate and sanitize text watermarking options against the documented contract."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("'options' must be a dictionary")
    unknown = set(raw) - ALLOWED_WATERMARK_OPTIONS
    if unknown:
        raise ValueError(f"unsupported option(s): {', '.join(sorted(unknown))}")

    parsed: dict[str, Any] = {}
    if "scheme" in raw:
        scheme = raw["scheme"]
        if not isinstance(scheme, str) or not scheme.strip():
            raise ValueError("'scheme' must be a non-empty string")
        parsed["scheme"] = scheme.strip()

    if "model" in raw:
        model = raw["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("'model' must be a non-empty string")
        parsed["model"] = model.strip()

    if "seed" in raw:
        val = raw["seed"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError("'seed' must be an integer")
        parsed["seed"] = val

    if "max_new_tokens" in raw:
        val = raw["max_new_tokens"]
        if isinstance(val, bool):
            raise ValueError("'max_new_tokens' must be an integer")
        try:
            tokens = int(val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"'max_new_tokens' must be an integer: {e}") from e
        if tokens < 1 or tokens > 8192:
            raise ValueError("'max_new_tokens' must be between 1 and 8192")
        parsed["max_new_tokens"] = tokens

    if "min_length" in raw:
        val = raw["min_length"]
        if isinstance(val, bool):
            raise ValueError("'min_length' must be an integer")
        try:
            length = int(val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"'min_length' must be an integer: {e}") from e
        if length < 0 or length > 8192:
            raise ValueError("'min_length' must be between 0 and 8192")
        parsed["min_length"] = length

    if "temperature" in raw:
        val = raw["temperature"]
        if isinstance(val, bool):
            raise ValueError("'temperature' must be a number")
        try:
            temp = float(val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"'temperature' must be a number: {e}") from e
        if temp <= 0:
            raise ValueError("'temperature' must be greater than 0")
        parsed["temperature"] = temp

    if "top_p" in raw:
        val = raw["top_p"]
        if isinstance(val, bool):
            raise ValueError("'top_p' must be a number")
        try:
            p = float(val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"'top_p' must be a number: {e}") from e
        if p <= 0 or p > 1:
            raise ValueError("'top_p' must be between 0 and 1")
        parsed["top_p"] = p

    if "device" in raw:
        device = raw["device"]
        if not isinstance(device, str) or not device.strip():
            raise ValueError("'device' must be a non-empty string")
        parsed["device"] = device.strip()

    if "config" in raw:
        config = raw["config"]
        if not isinstance(config, str) or not config.strip():
            raise ValueError("'config' must be a non-empty string")
        parsed["config"] = config.strip()

    if "offline" in raw:
        val = raw["offline"]
        if not isinstance(val, bool):
            raise ValueError("'offline' must be a boolean")
        parsed["offline"] = val

    return parsed


def resolve_timeout(
    explicit: float | None,
    *,
    env_var: str = "WATERMARKS_SYNTHID_TEXT_TIMEOUT",
    default: float = DEFAULT_SYNTHID_TEXT_TIMEOUT,
    floor: float = 1.0,
    ceiling: float = 600.0,
) -> float:
    """Resolve timeout from an explicit value or environment variable.

    Returns *default* when neither is set. Clamps the result to
    [floor, ceiling] to prevent unbounded waits or nonsensical sub-second
    deadlines.
    """
    if explicit is not None:
        t = float(explicit)
    else:
        raw = os.environ.get(env_var, "").strip()
        t = float(raw) if raw else default
    return max(floor, min(t, ceiling))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects to prevent SSRF and credential forwarding."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect())


# Module-level opener seam for testing/injection; None defaults to _default_opener().
OPENER: Any = None


def normalize_keys(raw: Any) -> list[int] | None:
    """Normalize keys input into a list of integer keys.

    Accepts:
        - list of ints: [118, 504, 421, 521]
        - comma-separated string: "118,504,421,521"
        - JSON array string: "[118, 504, 421, 521]"
        - None: returns None (uses default scheme keys)
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("unsupported keys format: boolean is not a valid key")
    if isinstance(raw, list):
        try:
            keys = []
            for x in raw:
                if isinstance(x, bool):
                    raise ValueError(f"boolean {x!r} is not an integer key")
                keys.append(int(x))
            return keys
        except (ValueError, TypeError) as e:
            raise ValueError(f"all keys in list must be integers: {e}") from e
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON array for keys: {e}") from e
            if not isinstance(parsed, list):
                raise ValueError("keys JSON must decode to a list of integers")
            keys = []
            try:
                for x in parsed:
                    if isinstance(x, bool):
                        raise ValueError(f"boolean {x!r} is not an integer key")
                    keys.append(int(x))
                return keys
            except (ValueError, TypeError) as e:
                raise ValueError(f"all keys in list must be integers: {e}") from e
        try:
            return [int(part.strip()) for part in s.split(",") if part.strip()]
        except ValueError as e:
            raise ValueError(f"comma-separated keys must all be integers: {e}") from e
    raise ValueError(f"unsupported keys format: {type(raw).__name__}")


def _watermark_http(
    text: str,
    base_url: str,
    *,
    api_key: str | None = None,
    timeout: float = DEFAULT_SYNTHID_TEXT_TIMEOUT,
    keys: list[int] | None = None,
    options: dict[str, Any] | None = None,
    opener: Any = None,
) -> dict[str, Any]:
    """Call the HTTP sidecar (synthid_text_server.py) to watermark text."""
    if urlparse(base_url).scheme not in ("http", "https"):
        return {
            "ok": False,
            "kind": "text",
            "error": f"refusing non-http(s) watermark endpoint: {base_url}",
            "error_code": "backend_error",
        }

    body_dict: dict[str, Any] = {
        "text": text,
        "options": options or {},
    }
    if keys is not None:
        body_dict["keys"] = keys

    payload_bytes = json.dumps(body_dict).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # S310: URL scheme is restricted to http/https just above.
    req = urllib.request.Request(  # noqa: S310
        base_url.rstrip("/") + "/watermark",
        data=payload_bytes,
        headers=headers,
        method="POST",
    )

    active_opener = opener if opener is not None else (OPENER or _default_opener())
    try:
        with active_opener.open(req, timeout=timeout) as resp:
            data = resp.read()
        parsed = json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return {
                "ok": False,
                "kind": "text",
                "error": f"SynthID text sidecar redirect refused (SSRF protection): {e}",
                "error_code": "ssrf",
            }
        # Distinguish client-side (4xx) from server-side (5xx) sidecar errors
        # so the gateway can propagate the correct HTTP status. A 401 with a
        # configured API key means the bearer token was missing/invalid, which
        # is a gateway config problem (502), not a request-validation issue (400).
        if e.code == 401:
            return {
                "ok": False,
                "kind": "text",
                "error": f"SynthID text sidecar authentication error (401): {e}",
                "error_code": "auth",
            }
        if 400 <= e.code < 500:
            return {
                "ok": False,
                "kind": "text",
                "error": f"SynthID text sidecar client error ({e.code}): {e}",
                "error_code": "client_error",
            }
        return {
            "ok": False,
            "kind": "text",
            "error": f"SynthID text sidecar HTTP error ({e.code}): {e}",
            "error_code": "backend_error",
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ) as e:
        return {
            "ok": False,
            "kind": "text",
            "error": f"SynthID text sidecar unreachable: {e}",
            "error_code": "unreachable",
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "kind": "text",
            "error": "bad watermark sidecar response",
            "error_code": "backend_error",
        }
    return parsed


_LOCAL_MODEL_CACHE: OrderedDict[str, Any] = OrderedDict()
_LOCAL_MODEL_LOCK = threading.Lock()
MAX_LOCAL_CACHED_MODELS = int(os.environ.get("WATERMARKS_MAX_CACHED_MODELS", "2"))

# Long-lived single-worker executor for local generation.  Unlike a
# ``with ThreadPoolExecutor() as ex:`` context manager whose __exit__
# calls ``shutdown(wait=True)`` (blocking until the worker finishes even
# after a TimeoutError), a long-lived pool lets us ``future.cancel()``
# and return immediately when a deadline expires.
_LOCAL_GEN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="wm-local",
)


def _watermark_local_markllm(
    text: str,
    upstream_dir: str,
    *,
    keys: list[int] | None = None,
    options: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Generate watermarked text locally using detect_text_watermark.py / MarkLLM."""
    if timeout is not None and timeout <= 0:
        return {
            "ok": False,
            "kind": "text",
            "error": "watermark generation timed out",
            "error_code": "timeout",
        }

    from detect_text_watermark import (
        _load_algorithm,
        _resolve_config,
        resolve_device,
        resolve_upstream,
    )

    opts = options or {}
    alg = opts.get("scheme", "synthid").lower()
    from detect_text_watermark import SCHEMES

    scheme_alg = SCHEMES.get(alg, "SynthID")

    upstream_path = resolve_upstream(upstream_dir)
    if not upstream_path:
        return {
            "ok": False,
            "kind": "text",
            "error": f"invalid MARKLLM_DIR: {upstream_dir}",
            "error_code": "backend_error",
        }

    device = resolve_device(opts.get("device"))
    model = opts.get("model", "facebook/opt-1.3b")

    try:
        config_path = _resolve_config(upstream_path, scheme_alg, opts.get("config"))
        offline = bool(opts.get("offline", False))
        temperature = opts.get("temperature")
        top_p = opts.get("top_p")
        keys_tag = "<default>" if keys is None else ",".join(str(k) for k in keys)
        cache_key = (
            f"{scheme_alg}:{model}:{device}:{config_path}"
            f":{temperature}:{top_p}:{offline}:{keys_tag}"
        )

        with _LOCAL_MODEL_LOCK:
            if cache_key in _LOCAL_MODEL_CACHE:
                wm = _LOCAL_MODEL_CACHE[cache_key]
                _LOCAL_MODEL_CACHE.move_to_end(cache_key)
            else:
                wm = _load_algorithm(
                    upstream_path,
                    scheme_alg,
                    config_path,
                    model,
                    device,
                    offline=offline,
                    temperature=temperature,
                    top_p=top_p,
                    watermark_keys=keys,
                )
                if len(_LOCAL_MODEL_CACHE) >= MAX_LOCAL_CACHED_MODELS:
                    _LOCAL_MODEL_CACHE.popitem(last=False)
                _LOCAL_MODEL_CACHE[cache_key] = wm

            # All mutable config is set inside the lock so concurrent
            # requests on the same cached model don't race on gen_kwargs.
            seed = opts.get("seed")
            if seed is not None:
                try:
                    import torch  # type: ignore

                    torch.manual_seed(int(seed))
                except (ImportError, ModuleNotFoundError):
                    pass

            wm.config.gen_kwargs["max_new_tokens"] = int(opts.get("max_new_tokens", 200))
            wm.config.gen_kwargs["min_length"] = int(opts.get("min_length", 0))

            if timeout is not None:
                future = _LOCAL_GEN_EXECUTOR.submit(wm.generate_watermarked_text, text)
                try:
                    watermarked = future.result(timeout=timeout)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    future.cancel()
                    raise
            else:
                watermarked = wm.generate_watermarked_text(text)

        return {
            "ok": True,
            "kind": "text",
            "watermarked_text": watermarked,
            "report": {
                "scheme_used": scheme_alg.lower(),
                "model": model,
                "watermarked_chars": len(watermarked),
                "keys_used": keys,
            },
        }
    except (TimeoutError, concurrent.futures.TimeoutError):
        return {
            "ok": False,
            "kind": "text",
            "error": "watermark generation timed out",
            "error_code": "timeout",
        }
    except Exception as e:
        return {
            "ok": False,
            "kind": "text",
            "error": f"MarkLLM generation failed: {e}",
            "error_code": "backend_error",
        }


def generate_watermark_text(
    text: str,
    *,
    keys: list[int] | str | None = None,
    options: dict[str, Any] | None = None,
    sidecar_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    opener: Any = None,
) -> dict[str, Any]:
    """Generate watermarked text via the configured sidecar or local fallback.

    Priority:
      1. HTTP sidecar via WATERMARKS_SYNTHID_TEXT_URL (or sidecar_url argument)
      2. Local MarkLLM checkout via MARKLLM_DIR
      3. Fail-soft error when neither is configured
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "ok": False,
            "kind": "text",
            "error": "'text' must be a non-empty string",
            "error_code": "client_error",
        }

    try:
        normalized_keys = normalize_keys(keys)
    except ValueError as e:
        return {"ok": False, "kind": "text", "error": str(e), "error_code": "client_error"}

    # 1. HTTP Sidecar
    url = (sidecar_url or os.environ.get("WATERMARKS_SYNTHID_TEXT_URL", "")).strip()
    if url:
        key = (
            api_key
            if api_key is not None
            else os.environ.get("WATERMARKS_SYNTHID_TEXT_API_KEY", "").strip()
        )
        t = resolve_timeout(timeout)
        return _watermark_http(
            text,
            url,
            api_key=key,
            timeout=t,
            keys=normalized_keys,
            options=options,
            opener=opener,
        )

    # 2. Local MarkLLM checkout
    upstream = os.environ.get("MARKLLM_DIR", "").strip()
    if upstream:
        return _watermark_local_markllm(
            text, upstream, keys=normalized_keys, options=options, timeout=timeout
        )

    # 3. Unconfigured fail-soft
    return {
        "ok": False,
        "kind": "text",
        "error": (
            "no text watermark generator configured (set WATERMARKS_SYNTHID_TEXT_URL "
            "for the sidecar or MARKLLM_DIR for local execution)"
        ),
        "error_code": "unconfigured",
    }
