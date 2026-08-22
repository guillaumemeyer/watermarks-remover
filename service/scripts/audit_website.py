#!/usr/bin/env python3
"""Aggregate AI-provenance audit over the URLs listed in a sitemap.

Stdlib-only: downloads each URL, classifies it by content type/suffix/magic,
and runs the same deterministic text/image/container inspections used by the
local audit. Optional external tools (c2patool/exiftool) are not invoked for
remote URLs; download the assets and run audit_dir.py locally for those.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import io
import ipaddress
import socket
import ssl
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib

from audit_lib import aggregate, print_human_report, scan_file
from common import EXIT_PARTIAL, emit_json, eprint
from container_meta import detect_container_format

DEFAULT_MAX_BYTES = 4 << 20
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_PAGES = 200
MAX_SITEMAP_DECOMPRESSED_BYTES = 64 << 20
MAX_REDIRECTS = 5
USER_AGENT = "remove-ai-marks-audit/1.0"

_EXT_FOR_KIND = {
    "webp": ".webp",
    "avif": ".avif",
    "heic": ".heic",
    "gif": ".gif",
    "bmp": ".bmp",
    "tiff": ".tiff",
    "xlsx": ".xlsx",
    "pptx": ".pptx",
    "epub": ".epub",
    "mp4": ".mp4",
    "wav": ".wav",
    "mp3": ".mp3",
    "png": ".png",
    "jpeg": ".jpg",
    "svg": ".svg",
    "pdf": ".pdf",
    "docx": ".docx",
    "odt": ".odt",
    "html": ".html",
    "markdown": ".md",
    "text": ".txt",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(data: bytes) -> tuple[str, list[str]]:
    """Parse a (possibly gzip-compressed) sitemap into (kind, urls)."""
    if data[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            data = stream.read(MAX_SITEMAP_DECOMPRESSED_BYTES + 1)
        if len(data) > MAX_SITEMAP_DECOMPRESSED_BYTES:
            raise ValueError(
                f"sitemap decompressed size exceeds cap ({MAX_SITEMAP_DECOMPRESSED_BYTES} bytes)"
            )
    elif len(data) > MAX_SITEMAP_DECOMPRESSED_BYTES:
        raise ValueError(f"sitemap size exceeds cap ({MAX_SITEMAP_DECOMPRESSED_BYTES} bytes)")

    # This repo is stdlib-only by policy, so sitemaps use ElementTree. stdlib
    # ET does not expand external entities (no XXE), but internal entity
    # expansion (billion-laughs) is still possible, so DTDs/entities are
    # rejected outright before parsing. If the policy ever changes, swap in
    # defusedxml.ElementTree with the same DTD rejection as defense in depth.
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("sitemap declares a DTD / entities; refusing to parse")

    root = ET.fromstring(data)  # noqa: S314 - DTD/entity declarations rejected above
    kind = _local(root.tag)
    urls = []
    for el in root.iter():
        if _local(el.tag) == "loc" and el.text:
            urls.append(el.text.strip())
    return kind, urls


def guess_kind(url: str, data: bytes, content_type: str | None = None) -> str:
    """Classify a downloaded URL from headers, suffix, then magic bytes."""
    ct = (content_type or "").lower().split(";")[0].strip()
    if "html" in ct:
        return "html"
    if ct == "image/png":
        return "png"
    if ct == "image/jpeg":
        return "jpeg"
    if "svg" in ct:
        return "svg"
    if ct == "application/pdf":
        return "pdf"
    if "wordprocessingml" in ct:
        return "docx"
    if "opendocument.text" in ct:
        return "odt"
    if "markdown" in ct:
        return "markdown"
    if ct == "text/plain":
        return "text"
    # The formats the local audit (format_dispatch) handles arrived over HTTP
    # as "text" before this table knew them: the bytes fell to the Unicode
    # scanner and binary assets reported clean (#166).
    if ct in (
        "image/webp",
        "image/avif",
        "image/heic",
        "image/heif",
        "image/gif",
        "image/bmp",
        "image/tiff",
    ):
        return {
            "image/webp": "webp",
            "image/avif": "avif",
            "image/heic": "heic",
            "image/heif": "heic",
            "image/gif": "gif",
            "image/bmp": "bmp",
            "image/tiff": "tiff",
        }[ct]
    if ct == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return "xlsx"
    if ct == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return "pptx"
    if ct == "application/epub+zip":
        return "epub"
    if ct in ("video/mp4", "video/quicktime", "audio/x-m4a", "audio/mp4"):
        return "mp4"
    if ct in ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"):
        return {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}[
            ct
        ]

    path = urllib.parse.urlparse(url).path.lower()
    for ext, kind in (
        (".png", "png"),
        (".jpg", "jpeg"),
        (".jpeg", "jpeg"),
        (".webp", "webp"),
        (".avif", "avif"),
        (".heic", "heic"),
        (".heif", "heic"),
        (".gif", "gif"),
        (".bmp", "bmp"),
        (".tiff", "tiff"),
        (".tif", "tiff"),
        (".svg", "svg"),
        (".pdf", "pdf"),
        (".docx", "docx"),
        (".xlsx", "xlsx"),
        (".pptx", "pptx"),
        (".odt", "odt"),
        (".epub", "epub"),
        (".mp4", "mp4"),
        (".mov", "mp4"),
        (".m4a", "mp4"),
        (".wav", "wav"),
        (".mp3", "mp3"),
        (".html", "html"),
        (".htm", "html"),
        (".md", "markdown"),
        (".markdown", "markdown"),
        (".txt", "text"),
    ):
        if path.endswith(ext):
            return kind

    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"GIF8"):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "tiff"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return "avif"
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"):
            return "heic"
        return "mp4"
    if (
        data.startswith(b"PK\x03\x04")
        or data.startswith(b"PK\x05\x06")
        or data.startswith(b"PK\x07\x08")
    ):
        ckind = detect_container_format(Path(path), data)
        if ckind != "unknown":
            return ckind
    if data[:100].lstrip().startswith(b"<") and b"svg" in data[:500].lower():
        return "svg"
    if b"<html" in data[:2000].lower() or data[:100].lstrip().lower().startswith(b"<"):
        return "html"
    return "text"


UrlOrigin = tuple[str, str, int]


def _url_origin(url: str) -> UrlOrigin:
    """Return a normalized HTTP(S) origin or reject unsafe URL forms."""
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()

    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {scheme or '(missing)'}")

    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials in URLs are not allowed")

    host = parsed.hostname
    if not host:
        raise ValueError("URL has no hostname")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid URL port: {exc}") from exc

    if port is None:
        port = 443 if scheme == "https" else 80

    return scheme, host.rstrip(".").lower(), port


def _origin_allowed(candidate: UrlOrigin, expected: UrlOrigin) -> bool:
    """Allow same-origin URLs plus a normal HTTP-to-HTTPS upgrade."""
    if candidate == expected:
        return True
    return (
        expected[0] == "http"
        and expected[2] == 80
        and candidate[0] == "https"
        and candidate[2] == 443
        and candidate[1] == expected[1]
    )


def _resolve_public_addresses(origin: UrlOrigin) -> tuple[str, ...]:
    """Resolve an origin to validated public numeric addresses."""
    _scheme, host, port = origin
    host_for_ip = host.split("%", 1)[0]
    raw_addresses: list[str] = []

    try:
        raw_addresses.append(str(ipaddress.ip_address(host_for_ip)))
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"cannot resolve hostname {host}: {exc}") from exc

        for info in infos:
            raw_addresses.append(str(info[4][0]).split("%", 1)[0])

    addresses: list[str] = []
    for raw in raw_addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue

        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped

        if not address.is_global:
            raise ValueError(f"refusing non-public address for {host}: {address}")

        canonical = str(address)
        if canonical not in addresses:
            addresses.append(canonical)

    if not addresses:
        raise ValueError(f"hostname resolved to no IP addresses: {host}")

    return tuple(addresses)


def _validated_target(
    url: str,
    *,
    expected_origin: UrlOrigin | None = None,
) -> tuple[UrlOrigin, tuple[str, ...]]:
    """Validate URL policy and bind it to numeric public IPs."""
    origin = _url_origin(url)

    if expected_origin is not None and not _origin_allowed(origin, expected_origin):
        raise ValueError(f"cross-origin URL is not allowed: {origin[0]}://{origin[1]}:{origin[2]}")

    return origin, _resolve_public_addresses(origin)


def _validate_public_http_url(
    url: str,
    *,
    expected_origin: UrlOrigin | None = None,
) -> UrlOrigin:
    """Validate a public HTTP(S) URL and return its origin."""
    origin, _addresses = _validated_target(url, expected_origin=expected_origin)
    return origin


def _open_pinned_connection(
    origin: UrlOrigin,
    addresses: tuple[str, ...],
    timeout: int,
) -> http.client.HTTPConnection:
    """Connect to a validated IP while preserving Host and TLS SNI."""
    scheme, host, port = origin
    last_error: OSError | None = None

    for address in addresses:
        raw = None
        try:
            raw = socket.create_connection((address, port), timeout=timeout)

            if scheme == "https":
                context = ssl.create_default_context()
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                with contextlib.suppress(NotImplementedError):
                    context.set_alpn_protocols(["http/1.1"])

                wrapped = context.wrap_socket(raw, server_hostname=host)
                raw = None

                conn = http.client.HTTPSConnection(
                    host,
                    port,
                    timeout=timeout,
                    context=context,
                )
                conn.sock = wrapped
                return conn

            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conn.sock = raw
            raw = None
            return conn

        except OSError as exc:
            last_error = exc
            if raw is not None:
                with contextlib.suppress(OSError):
                    raw.close()

    if last_error is not None:
        raise last_error
    raise OSError(f"no validated address available for {host}")


def _request_target(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return target


def fetch(
    url: str,
    timeout: int,
    max_bytes: int,
    *,
    allowed_origin: UrlOrigin | None = None,
) -> tuple[bytes, str | None]:
    """Fetch with IP pinning, redirect validation and a byte cap."""
    current_url = url
    expected_origin = allowed_origin

    for redirect_count in range(MAX_REDIRECTS + 1):
        origin, addresses = _validated_target(
            current_url,
            expected_origin=expected_origin,
        )

        if expected_origin is None:
            expected_origin = origin

        conn = _open_pinned_connection(origin, addresses, timeout)
        response = None

        try:
            conn.request(
                "GET",
                _request_target(current_url),
                headers={
                    "User-Agent": USER_AGENT,
                    "Connection": "close",
                },
            )
            response = conn.getresponse()

            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if location:
                    if redirect_count >= MAX_REDIRECTS:
                        raise ValueError(f"too many redirects (>{MAX_REDIRECTS})")
                    current_url = urllib.parse.urljoin(current_url, location)
                    continue

            content_type = response.getheader("Content-Type")
            chunks = []
            total = 0

            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"exceeds {max_bytes} bytes")
                chunks.append(chunk)

            return b"".join(chunks), content_type

        finally:
            if response is not None:
                response.close()
            conn.close()

    raise ValueError(f"too many redirects (>{MAX_REDIRECTS})")


def inspect_remote(url: str, data: bytes, content_type: str | None = None) -> dict:
    """Inspect downloaded bytes using the local scan_file pipeline."""
    kind = guess_kind(url, data, content_type)
    ext = _EXT_FOR_KIND.get(kind, ".bin")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"asset{ext}"
        tmp.write_bytes(data)
        result = scan_file(tmp, display_name=url)
    result["kind"] = kind
    return result


def discover_sitemap(base_url: str, timeout: int) -> str | None:
    """Find a same-site sitemap via standard paths then robots.txt."""
    origin = _validate_public_http_url(base_url)
    base = base_url.rstrip("/")

    for candidate in (
        f"{base}/sitemap.xml",
        f"{base}/sitemap_index.xml",
    ):
        try:
            data, _ = fetch(
                candidate,
                timeout,
                DEFAULT_MAX_BYTES,
                allowed_origin=origin,
            )
            parse_sitemap(data)
            return candidate
        except Exception:  # noqa: S112 - try next candidate sitemap on any failure
            continue

    try:
        data, _ = fetch(
            f"{base}/robots.txt",
            timeout,
            1 << 20,
            allowed_origin=origin,
        )

        text = data.decode("utf-8", errors="replace")

        for line in text.splitlines():
            if line.lower().startswith("sitemap:"):
                candidate = line.split(":", 1)[1].strip()
                candidate_origin = _url_origin(candidate)

                if not _origin_allowed(candidate_origin, origin):
                    raise ValueError(f"cross-origin sitemap is not allowed: {candidate}")

                return candidate

    except Exception:  # noqa: S110 - robots.txt is optional; fall through to None
        pass

    return None


def collect_urls(
    sitemap_url: str,
    timeout: int,
    max_pages: int,
) -> list[str]:
    """Collect same-site URLs while following nested sitemap indexes."""
    urls: list[str] = []
    seen: set[str] = set()
    origin = _validate_public_http_url(sitemap_url)

    def _check_loc(loc: str) -> None:
        candidate_origin = _url_origin(loc)

        if not _origin_allowed(candidate_origin, origin):
            raise ValueError(f"cross-origin sitemap URL is not allowed: {loc}")

    def _recurse(url: str, depth: int = 0) -> None:
        if len(urls) >= max_pages or depth > 3:
            return

        data, _ = fetch(
            url,
            timeout,
            DEFAULT_MAX_BYTES,
            allowed_origin=origin,
        )

        kind, locs = parse_sitemap(data)

        if kind == "sitemapindex":
            for loc in locs:
                _check_loc(loc)

                if loc not in seen:
                    seen.add(loc)
                    _recurse(loc, depth + 1)

        else:
            for loc in locs:
                _check_loc(loc)

                if loc not in seen:
                    seen.add(loc)
                    urls.append(loc)

                    if len(urls) >= max_pages:
                        break

    _recurse(sitemap_url)
    return urls


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sitemap", help="Sitemap URL to audit")
    p.add_argument("--base", help="Base URL; discover the sitemap automatically")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not args.sitemap and not args.base:
        eprint("provide --sitemap URL or --base URL")
        return 2

    sitemap_url = args.sitemap
    if not sitemap_url:
        try:
            sitemap_url = discover_sitemap(args.base, args.timeout)
        except ValueError as e:
            eprint(f"invalid base URL: {e}")
            return 2
        if not sitemap_url:
            eprint(f"no sitemap found for {args.base}")
            return 2

    try:
        urls = collect_urls(sitemap_url, args.timeout, args.max_pages)
    except Exception as e:
        eprint(f"could not collect URLs from {sitemap_url}: {e}")
        return 2
    if not urls:
        eprint("no URLs collected from sitemap")
        return 2

    files = []
    failures = []
    for url in urls[: args.max_pages]:
        try:
            data, content_type = fetch(url, args.timeout, args.max_bytes)
        except Exception as e:
            failures.append({"url": url, "error": str(e)})
            continue
        try:
            files.append(inspect_remote(url, data, content_type))
        except Exception as e:
            failures.append({"url": url, "error": f"inspect failed: {e}"})

    summary = aggregate(files)
    report = {
        "sitemap": sitemap_url,
        "base": args.base,
        "urls_collected": len(urls),
        "urls_scanned": len(files),
        "urls_failed": failures,
        "summary": summary,
        "files": files,
    }

    if args.json:
        emit_json(report)
    else:
        print_human_report(
            files,
            summary,
            extra_header={
                "Sitemap": sitemap_url,
                "URLs collected": str(len(urls)),
                "URLs scanned": str(len(files)),
                "URLs failed": str(len(failures)),
            },
        )
        for failure in failures:
            print(f"  [error] {failure['url']}: {failure['error']}")

    # A partial scan (one or more URLs failed to fetch/inspect) gets a
    # distinct code: an incomplete audit outranks actionable findings.
    return EXIT_PARTIAL if failures else (1 if summary["actionable_files"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
