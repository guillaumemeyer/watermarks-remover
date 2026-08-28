"""Shared helpers for remove-ai-marks scripts."""

from __future__ import annotations

import contextlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Hard caps on attacker-influenced input sizes. Whole-file in-memory
# processing means a 1 GiB default is a host-memory DoS; keep defaults low.
# The env overrides remain as an explicit escape hatch.
MAX_INPUT_BYTES = int(os.environ.get("WATERMARKS_MAX_INPUT_BYTES", str(256 << 20)))
MAX_STDIN_BYTES = int(os.environ.get("WATERMARKS_MAX_STDIN_BYTES", str(64 << 20)))

# Exit codes shared by the audit CLIs. 0 = clean, 1 = actionable findings,
# 2 = usage/refusal error, 3 = partial scan (some files/URLs failed to
# scan). A partial scan takes precedence over actionable findings: an
# incomplete audit is the more important CI signal.
EXIT_PARTIAL = 3

# Child-process resource limits (address space / output file size). Applied
# via preexec_fn so a crafted file cannot make exiftool/c2patool/OpenCV
# exhaust host memory or fill the disk.
_CHILD_RLIMIT_AS = int(os.environ.get("WATERMARKS_CHILD_RLIMIT_AS", str(4 << 30)))
_CHILD_RLIMIT_FSIZE = int(os.environ.get("WATERMARKS_CHILD_RLIMIT_FSIZE", str(2 << 30)))


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _reconfigure_stream(stream: Any, errors: str) -> None:
    """Switch a std stream to UTF-8 when it supports reconfiguration.

    On Windows, redirected stdin/stdout/stderr default to the ANSI codepage
    (e.g. cp1252), which cannot encode/decode the invisible Unicode
    characters this tool exists to remove. UTF-8 covers every codepoint, so
    text writes stop raising and piped input matches the file-path handling.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors=errors)


def _configure_stdio() -> None:
    _reconfigure_stream(sys.stdin, "surrogateescape")
    _reconfigure_stream(sys.stdout, "backslashreplace")
    _reconfigure_stream(sys.stderr, "backslashreplace")


_configure_stdio()


# Containers that get mistaken for text on the command line. Decoding one as
# text walks compressed bytes and reports whatever codepoints fall out of them:
# noise that tracks the compression, not the content. Worse, cleaning such a
# "text" writes the mangled bytes back and destroys the file.
BINARY_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "a ZIP container (DOCX, ODT, XLSX, PPTX, EPUB, JAR)"),
    (b"PK\x05\x06", "an empty ZIP container"),
    (b"PK\x07\x08", "a spanned ZIP container"),
    (b"%PDF-", "a PDF"),
    (b"\x89PNG\r\n\x1a\n", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF87a", "a GIF image"),
    (b"GIF89a", "a GIF image"),
    (b"BM", "a BMP image"),
    (b"II*\x00", "a TIFF image"),
    (b"MM\x00*", "a TIFF image"),
    (b"RIFF", "a RIFF container (WEBP, WAV, AVI)"),
    (b"OggS", "an Ogg media file"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"BZh", "a bzip2 archive"),
    (b"\xfd7zXZ\x00", "an xz archive"),
    (b"7z\xbc\xaf\x27\x1c", "a 7-Zip archive"),
    (b"Rar!\x1a\x07", "a RAR archive"),
    (b"\x7fELF", "an ELF binary"),
    (b"\xca\xfe\xba\xbe", "a Java class or Mach-O fat binary"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "a legacy Office document (.doc, .xls, .ppt)"),
    (b"SQLite format 3\x00", "a SQLite database"),
    (b"8BPS", "a Photoshop document"),
    (b"wOFF", "a WOFF font"),
    (b"wOF2", "a WOFF2 font"),
    (b"\x00\x01\x00\x00\x00", "a TrueType font"),
    (b"OTTO", "an OpenType font"),
)

BINARY_SNIFF_BYTES = 8192

# Real text runs ~0% control bytes; compressed and executable data runs far
# above this. Tab, LF, CR, FF and ESC are excluded as legitimate in text.
_CONTROL_RATIO_LIMIT = 0.05
_ALLOWED_CONTROLS = frozenset({0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1B})


def looks_binary(data: bytes) -> str | None:
    """Describe why *data* is not plausibly text, or None when it looks like text.

    Deliberately conservative: encodings other than UTF-8 must keep working, so
    undecodable bytes alone are not proof. Every caller offers an override.
    """
    if not data:
        return None
    for magic, label in BINARY_MAGIC:
        if data.startswith(magic):
            return label
    head = data[:BINARY_SNIFF_BYTES]
    if b"\x00" in head:
        return "binary data (contains NUL bytes)"
    controls = sum(1 for b in head if b < 0x20 and b not in _ALLOWED_CONTROLS)
    if controls / len(head) > _CONTROL_RATIO_LIMIT:
        return "binary data (dense in control bytes)"
    return None


# Advice for the text-only scripts: another tool in this repo handles the file.
TEXT_TOOL_ADVICE = (
    "Use inspect_file.py / clean_file.py, which route by format,",
    "or pass --force-text to scan the raw bytes anyway.",
)

# Advice for the routers themselves. They *are* inspect_file.py / clean_file.py,
# and classify() has already ruled out every known container, so pointing back
# at them would be circular.
ROUTER_ADVICE = (
    "These bytes match no supported text, image or container format.",
    "Pass --force-text to handle them as text anyway, or --as to force a format.",
)


def guard_binary(
    data: bytes,
    origin: str,
    *,
    allow_binary: bool = False,
    advice: tuple[str, ...] | None = None,
) -> None:
    """Refuse binary input for the text-only tools unless explicitly overridden."""
    if allow_binary:
        return
    kind = looks_binary(data)
    if kind is None:
        return
    eprint(f"refusing to treat {origin} as text: it looks like {kind}.")
    for line in advice or TEXT_TOOL_ADVICE:
        eprint(line)
    raise SystemExit(2)


def read_text_input(
    path: str | None,
    *,
    allow_binary: bool = False,
    advice: tuple[str, ...] | None = None,
) -> str:
    if path is None or path == "-":
        return _read_stdin_capped(allow_binary=allow_binary, advice=advice)
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        eprint(f"cannot read {path}: {e}")
        raise SystemExit(2) from None
    # Reject non-regular files (FIFOs, devices, sockets): st_size is 0 for a
    # FIFO, so opening one would block or stream until EOF with no cap.
    if not stat.S_ISREG(st.st_mode):
        eprint(f"refusing non-regular input (not a regular file): {path}")
        raise SystemExit(2)
    if st.st_size > MAX_INPUT_BYTES:
        eprint(f"refusing input larger than {MAX_INPUT_BYTES} bytes: {path}")
        raise SystemExit(2)
    try:
        with p.open("rb") as f:
            # Read at most MAX_INPUT_BYTES + 1: a file can grow after stat().
            # The extra byte lets the size check below refuse rather than the
            # read silently truncating an oversized file.
            data = f.read(MAX_INPUT_BYTES + 1)
    except OSError as e:
        eprint(f"cannot read {path}: {e}")
        raise SystemExit(2) from None
    if len(data) > MAX_INPUT_BYTES:
        eprint(f"refusing input larger than {MAX_INPUT_BYTES} bytes: {path}")
        raise SystemExit(2)
    guard_binary(data, str(path), allow_binary=allow_binary, advice=advice)
    return data.decode("utf-8", errors="surrogateescape")


def _read_stdin_capped(
    *,
    allow_binary: bool = False,
    advice: tuple[str, ...] | None = None,
) -> str:
    """Read stdin with a hard cap (uncapped stdin was a memory-DoS hole).

    Read the raw byte stream rather than the decoded text, so the binary sniff
    sees the real octets. Going through the text layer first makes detection
    depend on the console codec: under cp1252 a PNG's leading 0x89 comes back
    as 0xe2 0x80 0xb0 and the magic number is gone before we look. That the
    decode is UTF-8 today is only true while _configure_stdio() succeeds, and
    its reconfigure() is deliberately best-effort.
    """
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        # A replaced or non-binary stdin (pytest capture, custom harness).
        # Fall back to the text layer; the sniff is then codec-dependent.
        text = sys.stdin.read()
        if len(text.encode("utf-8", errors="surrogateescape")) > MAX_STDIN_BYTES:
            eprint(f"refusing stdin input larger than {MAX_STDIN_BYTES} bytes")
            raise SystemExit(2)
        guard_binary(
            text[:BINARY_SNIFF_BYTES].encode("utf-8", errors="surrogateescape"),
            "stdin",
            allow_binary=allow_binary,
            advice=advice,
        )
        return text

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(1 << 20)
        if not chunk:
            break
        if not chunks:
            guard_binary(
                chunk[:BINARY_SNIFF_BYTES],
                "stdin",
                allow_binary=allow_binary,
                advice=advice,
            )
        total += len(chunk)
        if total > MAX_STDIN_BYTES:
            eprint(f"refusing stdin input larger than {MAX_STDIN_BYTES} bytes")
            raise SystemExit(2)
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="surrogateescape")


def write_text_output(text: str, path: str | None) -> None:
    if path is None or path == "-":
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    safe_write_text(path, text)


def _default_file_mode() -> int:
    """0o666 & ~umask — the mode a plain open() would produce."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def safe_write_bytes(path: str | Path, data: bytes) -> None:
    """Atomically write bytes to *path* without following symlinks.

    Writes to a temp file in the destination directory and ``os.replace``s it
    into place. ``os.replace`` replaces a symlink rather than following it, and
    the explicit symlink check gives a clear error instead of surprising
    behavior. This defeats pre-placed symlinks (e.g. in /tmp or download dirs)
    redirecting a clean write onto an arbitrary victim file.
    """
    dest = Path(path)
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        raise OSError(f"refusing to write through symlink: {dest}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(parent))
    try:
        # mkstemp creates 0600; restore the umask-default mode so outputs
        # keep normal permissions. Windows has no fchmod and no POSIX mode
        # bits to restore, so the call is skipped there.
        if hasattr(os, "fchmod"):
            os.fchmod(fd, _default_file_mode())
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def safe_write_text(path: str | Path, text: str) -> None:
    safe_write_bytes(path, text.encode("utf-8", errors="surrogateescape"))


def backup_path(src: Path) -> tuple[Path, bool]:
    """Create a ``.bak`` copy of *src* no-clobber; return (backup, created).

    Used by ``--in-place`` flows so the original is never partially lost: the
    original file stays untouched until the cleaned output is atomically
    renamed over it.

    A pre-existing ``.bak`` is preserved, not overwritten — the second run of
    an auto-fix hook (clean → commit-blocked → re-stage → clean again) used to
    back up the first run's output over the original, destroying the only
    pristine copy (#172). The path is reserved atomically with ``O_EXCL``: only
    one process/run creates the ``.bak``, and any later call that loses the
    creation race keeps the already-existing pristine copy instead of backing
    up possibly already-cleaned text. The returned tuple's second element
    reports whether this call created the backup or kept an existing one.
    """
    bak = src.with_suffix(src.suffix + ".bak")
    try:
        fd = os.open(bak, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _default_file_mode())
    except FileExistsError:
        # Another process or a previous run reserved the backup first; it holds
        # the pristine copy, so write nothing through the existing path.
        return bak, False
    except OSError as e:
        eprint(f"cannot create backup {bak}: {e}")
        raise SystemExit(2) from None
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(src.read_bytes())
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        with contextlib.suppress(OSError):
            os.unlink(bak)
        eprint(f"cannot create backup {bak}: {e}")
        raise SystemExit(2) from None
    return bak, True


def subprocess_rlimits() -> None:
    """Apply conservative resource limits in a subprocess (preexec_fn).

    The scripts are single-threaded, so preexec_fn's fork-time caveats do not
    apply here. No-op on platforms without the resource module.
    """
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (_CHILD_RLIMIT_AS, _CHILD_RLIMIT_AS))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_CHILD_RLIMIT_FSIZE, _CHILD_RLIMIT_FSIZE))
    except (ImportError, OSError, ValueError):
        pass


# subprocess.run(preexec_fn=...) is POSIX-only; on Windows the argument
# itself raises ValueError before the callable runs. Windows resource
# limiting would need a Job Object (pywin32), which is out of scope.
subprocess_preexec_fn = subprocess_rlimits if os.name == "posix" else None

# The mirror image on Windows: a parent without a console of its own -- pythonw,
# a service, anything GUI-hosted -- makes Windows allocate a fresh console
# window for every child it starts. The window blinks and takes keyboard focus,
# several times per cleaned file once exiftool, qpdf, ghostscript and c2patool
# have each had a turn. CREATE_NO_WINDOW suppresses it; the flag does not exist
# on POSIX, where 0 is the no-op value subprocess already expects.
subprocess_creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def emit_json(data: Any) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


CONFIDENCE_LEVELS = (
    "confirmed",
    "probable",
    "informational",
    "likely_false_positive",
)


def classify_finding_confidence(finding: str) -> str:
    """Classify a scanner finding by confidence.

    The four buckets are a heuristic mapping of *how strong* a finding is:

    - confirmed: a recognized provenance structure (C2PA/JUMBF manifest, or a
      parsed field such as digitalSourceType / trainedAlgorithmicMedia).
    - probable: an AI/vendor marker found inside a recognized metadata
      structure, but not a fully parsed provenance claim.
    - informational: context-only notes (CMS generators, presence of an XMP
      packet or customXml parts, unsupported/partial inspection).
    - likely_false_positive: raw whole-file byte scans that can collide with
      compressed image/stream data.

    The mapping is intentionally conservative; a scanner finding is a signal,
    not a verdict.
    """
    t = finding.lower()

    if any(
        s in t
        for s in (
            "c2patool reports",
            "c2pa-related manifest",
            "png chunk c2",
            "png chunk cabx",
            "png chunk jumb",
            "png chunk jumd",
            "jpeg app11 segment",
            "digital_source_type",
            "digitalsourcetype",
            "trainedalgorithmicmedia",
            "compositewithtrainedalgorithmicmedia",
            "softwareagent",
        )
    ):
        return "confirmed"

    if t.startswith("info:") or any(
        s in t
        for s in (
            "cms generator",
            "customxml parts",
            "xmp packet present",
            "unsupported",
            "not fully inspected",
            "format not",
            "svg <metadata> present",
            "not a valid",
            "truncated chunk",
            "bad segment length",
            "svg decode note",
        )
    ):
        return "informational"

    if "byte-scan" in t:
        return "likely_false_positive"

    if any(
        s in t
        for s in (
            "ai:",
            "marker:",
            "meta:",
            "frontmatter",
            "json-ld",
            "attr:",
            "png ",
            "jpeg app",
            "exif",
            "xmp",
            "interesting",
            "pdf-structured",
            "layer-a",
        )
    ):
        return "probable"

    return "informational"


def c2patool_probe_note(tools: dict[str, Any]) -> str | None:
    """Describe an inconclusive c2patool run, or None when it answered.

    c2patool exits non-zero both for an asset with no manifest and for a
    binary that failed to run, so a caller that only reads `has_manifest`
    cannot tell "this asset is clean" from "the probe never ran". Reporting
    the second as the first is the dangerous direction: `has_c2pa: False`
    beside a dead probe reads as a clean bill of health on the one check a
    user would trust.

    The wording deliberately avoids the substring "c2patool reports", which
    classify_finding_confidence() maps to `confirmed`; "not fully inspected"
    puts it in the `informational` bucket instead.
    """
    ct = tools.get("c2patool") or {}
    # An explicitly recorded unavailable entry (available is False) means the
    # probe was attempted but could not run; report it so callers do not read
    # "not checked" as "clean". A tool with no entry at all is simply absent,
    # which the `tools` dict already reflects and is not a finding.
    if ct.get("available") is False:
        detail = ct.get("error") or "not found"
        return f"c2patool unavailable ({detail}); C2PA not fully inspected by this tool"
    if not ct.get("ok", True):
        detail = ct.get("error") or "no usable verdict"
        return f"c2patool probe inconclusive ({detail}); C2PA not fully inspected by this tool"
    return None


def cleaned_path(src: Path, suffix: str = ".cleaned") -> Path:
    """path/to/file.ext -> path/to/file.cleaned.ext"""
    return src.with_name(f"{src.stem}{suffix}{src.suffix}")


def which(cmd: str) -> str | None:
    from shutil import which as _which

    return _which(cmd)


def safe_arg(path: str) -> str:
    """Guard paths passed to option-parsing CLIs (exiftool, c2patool).

    A filename starting with '-' would otherwise be interpreted as an option
    (e.g. exiftool's -@argfile), turning a crafted filename into argv injection.
    """
    if path.startswith("-"):
        # './' also resolves correctly on Windows (Win32 accepts '/' as a
        # path separator), so no platform branch is needed here.
        return "./" + path
    return path
