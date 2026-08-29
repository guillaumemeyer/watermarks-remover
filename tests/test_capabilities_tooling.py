"""/capabilities must not report a tool as missing when the image ships it.

Two separate defects, both making the report disagree with the container:

* ``_tool_usable()`` probes with ``--version`` unless ``_VERSION_FLAG`` says
  otherwise. ffmpeg has no ``--version``; it exits 8. So an image that installs
  ffmpeg advertised ``"ffmpeg": false``.
* the core Dockerfile never installed ghostscript, although ``/capabilities``
  advertises it and ``_ghostscript_usable()`` probes for it.

A capability report is what a caller reads before deciding a job is impossible,
so a false negative there reads as a clean verdict about a tool nobody ran.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server


def test_ffmpeg_version_flag_is_declared():
    assert server._VERSION_FLAG.get("ffmpeg") == "-version"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_on_path_is_reported_usable():
    server._tool_usable.cache_clear()
    assert server._tool_usable("ffmpeg") is True


def test_core_image_installs_every_advertised_tool():
    dockerfile = (ROOT / "service" / "Dockerfile").read_text(encoding="utf-8")
    for pkg in ("ffmpeg", "ghostscript", "qpdf", "libimage-exiftool-perl"):
        assert pkg in dockerfile, f"{pkg} is advertised by /capabilities but not installed"
    # c2patool is not an apt package: it is fetched and installed separately, so
    # the loop above would keep passing if that step disappeared.
    assert "/usr/local/bin/c2patool" in dockerfile
