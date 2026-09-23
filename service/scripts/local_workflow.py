#!/usr/bin/env python3
"""Local inspect, clean and assisted academic review with bounded Ollama calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from academic_guard import academic_error, language_hint
from common import MAX_INPUT_BYTES, eprint, looks_binary, subprocess_creationflags
from latex_mask import LatexMask, mask_latex, placeholder_error, restore_latex
from rewrite_text import _generate_rewrite, _length_drift, build_prompt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_MODEL = "llama3.2:latest"
CHUNK_CHARS = 2200


def configure(model: str | None = None) -> None:
    """Configure only this process; Ollama manages GPU placement independently."""
    os.environ.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "WATERMARKS_REWRITE_BACKEND": "ollama",
            "WATERMARKS_REWRITE_BASE_URL": "http://127.0.0.1:11434",
            "WATERMARKS_REWRITE_MODEL": model or DEFAULT_MODEL,
            "WATERMARKS_REWRITE_TEMPERATURE": "0.2",
            "WATERMARKS_PROTECT_LATEX": "on",
            "WATERMARKS_REWRITE_CANDIDATES": "1",
            "WATERMARKS_REWRITE_LOOPS": "1",
            "WATERMARKS_OLLAMA_CONTEXT": "4096",
            "WATERMARKS_OLLAMA_MAX_TOKENS": "1536",
            "WATERMARKS_OLLAMA_THREADS": "6",
        }
    )


def chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split already masked text at whitespace without losing any bytes."""
    result = []
    while len(text) > limit:
        stop = text.rfind("\n\n", 0, limit)
        if stop < limit // 3:
            stop = text.rfind("\n", 0, limit)
        if stop < limit // 3:
            stop = text.rfind(" ", 0, limit)
        if stop <= 0:
            raise ValueError("A prose token exceeds the chunk limit; review this section manually")
        result.append(text[: stop + 1])
        text = text[stop + 1 :]
    if text:
        result.append(text)
    return result


def review_academic(text: str, model: str) -> tuple[str, dict]:
    """Review sequential chunks; reject damaged notation before writing a result.

    Mechanical checks cannot certify scientific meaning or the force of a claim.
    """
    masked, mask = mask_latex(text, mode="on")
    pieces = chunks(masked)
    results = []
    reports = []
    language = language_hint(mask.token_re.sub("", masked))
    for number, piece in enumerate(pieces, 1):
        # Keep separators and indentation exactly, even when the model trims them.
        core = piece.strip()
        if not core or not mask.token_re.sub("", core).strip():
            results.append(piece)
            continue
        leading = piece[: len(piece) - len(piece.lstrip())]
        trailing = piece[len(piece.rstrip()) :]
        expected = mask.token_re.findall(core)
        local_mask = LatexMask(spans=("",) * len(expected), prefix=mask.prefix)
        prompt = build_prompt(
            "academic",
            core,
            rewrite_level=0.2,
            mask=local_mask,
            style="Keep the original language. Make only necessary, light editorial changes.",
        )
        eprint(f"Academic review: section {number}/{len(pieces)}")
        out, wrappers = _generate_rewrite(
            "ollama",
            "http://127.0.0.1:11434",
            model,
            None,
            prompt,
            float(os.environ.get("WATERMARKS_REWRITE_TIMEOUT", "900.0")),
            0.2,
            None,
            original=core,
        )
        if not out.strip() or _length_drift(len(core), len(out)):
            raise RuntimeError(
                f"Section {number}: empty or excessive-length rewrite; no output saved"
            )
        output_language = language_hint(mask.token_re.sub("", out))
        if language and output_language and language != output_language:
            raise RuntimeError(f"Section {number}: language changed; no output saved")
        if mask.token_re.findall(out) != expected:
            raise RuntimeError(f"Section {number}: protected notation changed; no output saved")
        integrity_error = academic_error(core, out)
        if integrity_error:
            raise RuntimeError(f"Section {number}: {integrity_error}; no output saved")
        results.append(leading + out + trailing)
        reports.append(
            {
                "section": number,
                "input_chars": len(core),
                "output_chars": len(out),
                "wrappers_stripped": wrappers,
            }
        )
    combined = "".join(results)
    error = placeholder_error(combined, mask)
    if error:
        raise RuntimeError(error)
    output, restored = restore_latex(combined, mask)
    return output, {
        "ok": True,
        "assisted": True,
        "scientific_review_required": True,
        "model": model,
        "language_hint": language,
        "strategy": "academic@0.2",
        "latex": restored,
        "sections": reports,
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "note": "Mechanical checks preserve recognized spans, control sequences and numbers; "
        "they do not establish semantic equivalence or watermark removal.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "clean", "academic", "serve", "check"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    configure(args.model)
    if args.mode in ("serve", "check"):
        import server

        server._DEFAULT_STRATEGY = server._load_default_strategy(
            ROOT / "config/clean_strategy.json"
        )
        if args.mode == "check":
            print(
                json.dumps(
                    {"python": sys.executable, "model": args.model, **server.capabilities()},
                    indent=2,
                )
            )
            return 0
        sys.argv = [
            "server.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--strategy-config",
            str(ROOT / "config/clean_strategy.json"),
        ]
        return server.main()
    if args.path is None or not args.path.is_file():
        parser.error("Provide an existing input file")
    source = args.path.resolve()
    if args.mode == "inspect":
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "inspect_file.py"), str(source), "--json"],
            check=False,
            creationflags=subprocess_creationflags,
        ).returncode
    suffix = ".assisted" if args.mode == "academic" else ".cleaned"
    destination = (args.output or source.with_name(source.stem + suffix + source.suffix)).resolve()
    report_path = destination.with_name(destination.name + ".review.json")
    if (
        destination == source
        or destination.exists()
        or (args.mode == "academic" and report_path.exists())
    ):
        parser.error("Output must be a new file; the original and previous results are preserved")
    if not destination.parent.is_dir():
        parser.error("Output directory does not exist")
    if args.mode == "clean":
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "clean_file.py"),
                str(source),
                "-o",
                str(destination),
                "--json",
            ],
            check=False,
            creationflags=subprocess_creationflags,
        ).returncode
    if source.suffix.lower() not in {".txt", ".tex", ".md", ".rst"}:
        parser.error("Academic review accepts UTF-8 TXT, TEX, Markdown or RST source files")
    if source.stat().st_size > MAX_INPUT_BYTES:
        parser.error("Input exceeds the service file-size limit")
    data = source.read_bytes()
    if looks_binary(data):
        parser.error("Academic review requires a text source")
    try:
        output, report = review_academic(data.decode("utf-8"), args.model)
    except (UnicodeError, ValueError, RuntimeError, OSError) as exc:
        eprint(f"Academic review failed: {exc}")
        return 1
    # Exclusive creation prevents a concurrent run from overwriting another result.
    with destination.open("x", encoding="utf-8", newline="") as stream:
        stream.write(output)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(destination), "report": str(report_path), "ok": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
