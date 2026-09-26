#!/usr/bin/env python3
"""Install the optional YuNet face model with pinned source and SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from common import subprocess_creationflags

COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_NAME = "face_detection_yunet_2026may.onnx"
MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
LICENSE_SHA256 = "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"
MODEL_URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/face_detection_yunet/{MODEL_NAME}"
LICENSE_URL = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{COMMIT}/models/face_detection_yunet/LICENSE"
MAX_DOWNLOAD = 1024 * 1024


def download_verified(url: str, destination: Path, digest: str) -> None:
    """Never replace an existing file with unverified bytes."""
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
            return
        raise ValueError(f"Existing file has an unexpected checksum: {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "watermarks-remover-setup"})  # noqa: S310 -- pinned HTTPS URLs
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 -- fixed HTTPS URLs
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Download failed SHA-256/size verification: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".yunet-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    """Install optional CPU dependencies and checksum-verified model assets from CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path.home() / ".cache" / "watermarks-remover" / "yunet"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install pinned CPU dependencies into the active virtual environment",
    )
    args = parser.parse_args()
    if args.install_deps:
        if sys.prefix == sys.base_prefix:
            parser.error(
                "--install-deps requires a virtual environment; use the CtrlRegen environment"
            )
        if sys.version_info < (3, 11):  # noqa: UP036 -- standalone installer also runs on Python 3.10
            parser.error("face protection dependencies require Python 3.11 or newer")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(Path(__file__).with_name("requirements-face-protection.txt")),
            ],
            check=True,
            creationflags=subprocess_creationflags,
        )
    destination = args.dir.expanduser().resolve()
    try:
        download_verified(MODEL_URL, destination / MODEL_NAME, MODEL_SHA256)
        download_verified(LICENSE_URL, destination / "LICENSE", LICENSE_SHA256)
    except (OSError, ValueError) as exc:
        print(f"Face model installation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified model: {destination / MODEL_NAME}")
    print(
        "Set WATERMARKS_FACE_MODEL to that path in the service environment, then restart the service."
    )
    print(
        "Use --install-deps with the CtrlRegen virtual-environment interpreter if CPU dependencies are missing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
