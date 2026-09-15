"""Synthetic face-blending tests; no portraits, downloads, or GPU required."""

import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service" / "scripts"))
import face_protection as fp


@pytest.fixture
def detector(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"mock")
    responses = []

    class FakeDetector:
        def detect(self, image):
            return None, responses.pop(0) if responses else None

    monkeypatch.setattr(
        fp.cv2,
        "FaceDetectorYN",
        type("Factory", (), {"create": staticmethod(lambda *args: FakeDetector())}),
    )
    return model, responses


def detection(x=40, y=40, width=40, height=50):
    return np.array([[x, y, width, height, *([0] * 10), 0.95]], dtype=np.float32)


def pair():
    original = np.full((160, 160, 3), 120, dtype=np.uint8)
    original[55:60, 50:70] = 20  # synthetic high-frequency face detail
    regenerated = np.full_like(original, 65)
    return Image.fromarray(original), Image.fromarray(regenerated)


@pytest.mark.parametrize("method", ["gradient", "feather"])
def test_preserves_background_and_source_inputs(detector, method):
    model, responses = detector
    responses.append(detection())
    original, regenerated = pair()
    original_bytes = original.tobytes()
    output, report, mask = fp.protect_faces(original, regenerated, str(model), method)
    a, b, c = map(np.asarray, (original, regenerated, output))
    assert report["count"] == 1
    assert report["blend_method"] == method
    assert np.array_equal(c[mask == 0], b[mask == 0])
    assert original.tobytes() == original_bytes
    if method == "feather":
        assert np.array_equal(c[mask == 1], a[mask == 1])
    else:
        # Keep the original dark detail while adapting the bright patch to its surroundings.
        assert float(c[56:59, 52:68].mean()) < float(c[65:70, 52:68].mean()) - 20
        assert float(c[65:70, 52:68].mean()) < 100


def test_no_faces_is_exact_noop(detector):
    model, _ = detector
    original, regenerated = pair()
    output, report, mask = fp.protect_faces(original, regenerated, str(model))
    assert output.tobytes() == regenerated.tobytes()
    assert report["count"] == 0
    assert not mask.any()
    assert "No faces detected" in report["warning"]


def test_rotation_fallback_maps_mask_back(detector):
    model, responses = detector
    responses.extend([None, detection()])
    original, regenerated = pair()
    output, report, mask = fp.protect_faces(original, regenerated, str(model))
    assert report["count"] == 1
    assert report["detection_rotation_degrees"] == -30
    assert np.isfinite(mask).all()
    assert output.size == original.size


def test_face_touching_border(detector):
    model, responses = detector
    responses.append(detection(0, 0, 35, 40))
    original, regenerated = pair()
    output, report, mask = fp.protect_faces(original, regenerated, str(model))
    assert report["count"] == 1
    assert np.array_equal(np.asarray(output)[mask == 0], np.asarray(regenerated)[mask == 0])


def test_identical_inputs_stay_identical(detector):
    model, responses = detector
    responses.append(detection())
    original, _ = pair()
    output, _, _ = fp.protect_faces(original, original, str(model))
    assert output.tobytes() == original.tobytes()


def test_missing_model_and_size_mismatch(tmp_path):
    original, regenerated = pair()
    with pytest.raises(ValueError, match="matching image"):
        fp.protect_faces(original, Image.new("RGB", (20, 20)))
    with pytest.raises(ValueError, match="model unavailable"):
        fp.protect_faces(original, regenerated, str(tmp_path / "missing"))
