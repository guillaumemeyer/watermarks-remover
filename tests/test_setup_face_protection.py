"""Installer integrity checks use in-memory downloads."""

import hashlib
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "scripts"))
import setup_face_protection as setup


def test_verified_download_is_idempotent(monkeypatch, tmp_path):
    """Verify verified download is idempotent."""
    payload = b"verified fixture"
    digest = hashlib.sha256(payload).hexdigest()
    calls = []

    def fetch(*args, **kwargs):
        """Return fixture bytes and record each simulated network request."""
        calls.append(args)
        return io.BytesIO(payload)

    monkeypatch.setattr(setup.urllib.request, "urlopen", fetch)
    dest = tmp_path / "model.onnx"
    setup.download_verified(setup.MODEL_URL, dest, digest)
    setup.download_verified(setup.MODEL_URL, dest, digest)
    assert dest.read_bytes() == payload
    assert len(calls) == 1


def test_checksum_failure_leaves_no_model(monkeypatch, tmp_path):
    """Verify checksum failure leaves no model."""
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    dest = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match="verification"):
        setup.download_verified(setup.MODEL_URL, dest, "0" * 64)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_unknown_file_is_not_replaced(monkeypatch, tmp_path):
    """Verify existing unknown file is not replaced."""
    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"user file")
    with pytest.raises(ValueError, match="Existing file"):
        setup.download_verified(setup.MODEL_URL, dest, "0" * 64)
    assert dest.read_bytes() == b"user file"
