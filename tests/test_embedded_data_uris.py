"""Tests for embedded data URI inspection and cleaning in SVG, HTML, and Markdown."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from container_meta import (
    clean_html,
    clean_markdown,
    clean_svg,
    inspect_html,
    inspect_markdown,
    inspect_svg,
)

from tests.test_clean_image import _minimal_jpeg_with_app11, _minimal_png_with_text


def test_svg_embedded_png_c2pa_cleaned():
    png_c2pa = _minimal_png_with_text()
    png_b64 = base64.b64encode(png_c2pa).decode("ascii")

    svg_data = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <image width="100" height="100" xlink:href="data:image/png;base64,{png_b64}" />
</svg>""".encode()

    # Inspect
    has_c2pa, has_ai, findings, _ = inspect_svg(svg_data)
    assert has_c2pa is True
    assert has_ai is True
    assert any("embedded data:image/png" in f for f in findings)

    # Clean
    cleaned_bytes, actions = clean_svg(svg_data)
    assert any("cleaned embedded data:image/png" in a for a in actions)

    # Re-inspect
    has_c2pa_after, has_ai_after, _findings_after, _ = inspect_svg(cleaned_bytes)
    assert has_c2pa_after is False
    assert has_ai_after is False
    assert b"c2pa" not in cleaned_bytes.lower()


def test_svg_embedded_base64_with_newlines():
    png_c2pa = _minimal_png_with_text()
    png_b64 = base64.b64encode(png_c2pa).decode("ascii")
    # Split base64 across newlines as common in formatted SVGs
    multiline_b64 = png_b64[:20] + "\n  \r\n  " + png_b64[20:]

    svg_data = f"""<svg xmlns="http://www.w3.org/2000/svg">
  <image href="data:image/png;base64,{multiline_b64}" />
</svg>""".encode()

    cleaned_bytes, actions = clean_svg(svg_data)
    assert any("cleaned embedded data:image/png" in a for a in actions)

    has_c2pa, _, _, _ = inspect_svg(cleaned_bytes)
    assert has_c2pa is False


def test_html_embedded_jpeg_c2pa_cleaned():
    jpeg_c2pa = _minimal_jpeg_with_app11()
    jpeg_b64 = base64.b64encode(jpeg_c2pa).decode("ascii")

    html_text = f"""<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
  <img src="data:image/jpeg;base64,{jpeg_b64}" alt="test" />
</body>
</html>"""

    # Inspect
    has_c2pa, has_ai, findings, _ = inspect_html(html_text)
    assert has_c2pa is True
    assert has_ai is True
    assert any("embedded data:image/jpeg" in f for f in findings)

    # Clean
    cleaned_text, actions = clean_html(html_text)
    assert any("cleaned embedded data:image/jpeg" in a for a in actions)
    assert "c2pa-manifest-fake" not in cleaned_text

    # Re-inspect
    has_c2pa_after, has_ai_after, _, _ = inspect_html(cleaned_text)
    assert has_c2pa_after is False
    assert has_ai_after is False


def test_markdown_embedded_data_uri_cleaned():
    png_c2pa = _minimal_png_with_text()
    png_b64 = base64.b64encode(png_c2pa).decode("ascii")

    md_text = f"""---
title: Sample Article
author: Dev
---

Here is an image:
![diagram](data:image/png;base64,{png_b64})
"""

    has_c2pa, _has_ai, findings, _ = inspect_markdown(md_text)
    assert has_c2pa is True
    assert any("embedded data:image/png" in f for f in findings)

    cleaned_text, actions = clean_markdown(md_text)
    assert any("cleaned embedded data:image/png" in a for a in actions)
    assert "c2pa" not in cleaned_text.lower()


def test_already_clean_embedded_image_no_op():
    # PNG with no metadata
    clean_png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        + b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7V\xfe"
        + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_b64 = base64.b64encode(clean_png).decode("ascii")
    html_text = f'<img src="data:image/png;base64,{png_b64}">'

    cleaned_text, actions = clean_html(html_text)
    # The string must remain exactly identical
    assert cleaned_text == html_text
    assert "no HTML AI meta removed" in actions


def test_data_uri_with_many_params_is_matched():
    # No fixed count/length limit on the parameter list: a URI with far more
    # than 16 short params must still be found and cleaned.
    png_c2pa = _minimal_png_with_text()
    png_b64 = base64.b64encode(png_c2pa).decode("ascii")
    params = "".join(f";p{i}=v{i}" for i in range(40))
    assert params.count(";") > 16
    html_text = f'<img src="data:image/png;base64{params},{png_b64}">'

    has_c2pa, has_ai, findings, _ = inspect_html(html_text)
    assert has_c2pa is True
    assert has_ai is True
    assert any("embedded data:image/png" in f for f in findings)

    cleaned_text, actions = clean_html(html_text)
    assert any("cleaned embedded data:image/png" in a for a in actions)
    assert "c2pa" not in cleaned_text.lower()


def test_corrupted_data_uri_graceful_fallback():
    # Corrupted base64 that shouldn't crash the file cleaner
    bad_html = '<img src="data:image/png;base64,!!!NOT_VALID_BASE64###">'
    cleaned_text, _actions = clean_html(bad_html)
    assert cleaned_text == bad_html


def test_nested_svg_data_uri_cleaned():
    nested_svg = '<svg><metadata><ai:GeneratedBy>DALL-E</ai:GeneratedBy></metadata><rect width="10" height="10"/></svg>'
    nested_b64 = base64.b64encode(nested_svg.encode("utf-8")).decode("ascii")

    parent_html = f'<img src="data:image/svg+xml;base64,{nested_b64}">'

    _has_c2pa, has_ai, _findings, _ = inspect_html(parent_html)
    assert has_ai is True

    cleaned_html, actions = clean_html(parent_html)
    assert any("cleaned embedded data:image/svg+xml" in a for a in actions)
    assert "DALL-E" not in base64.b64decode(cleaned_html.split("base64,")[1].split('"')[0]).decode(
        "utf-8"
    )
