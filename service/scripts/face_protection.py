"""Optional face protection for CtrlRegen; preserved structure may retain watermarks."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def protect_faces(
    original: Image.Image,
    regenerated: Image.Image,
    model_path: str | None = None,
    blend_method: str = "gradient",
):
    start = time.monotonic()
    if blend_method not in ("gradient", "feather"):
        raise ValueError("blend_method must be gradient or feather")
    if original.size != regenerated.size:
        raise ValueError("Face protection requires matching image dimensions")
    model = Path(model_path or os.environ.get("WATERMARKS_FACE_MODEL", ""))
    if not model.is_file():
        raise ValueError("Face protection model unavailable: set WATERMARKS_FACE_MODEL")
    a = np.asarray(original.convert("RGB"))
    b = np.asarray(regenerated.convert("RGB"))
    h, w = a.shape[:2]
    scale = min(1.0, 1280 / max(w, h))
    resized = cv2.resize(
        cv2.cvtColor(a, cv2.COLOR_RGB2BGR), (max(1, round(w * scale)), max(1, round(h * scale)))
    )
    detector = cv2.FaceDetectorYN.create(
        str(model), "", (resized.shape[1], resized.shape[0]), 0.85, 0.3, 5000
    )
    _, detections = detector.detect(resized)
    inverse_rotation = None
    detection_angle = 0
    if detections is None or len(detections) == 0:
        # Tilted portraits can be missed. Retry only when the first pass is empty.
        for angle in (-30, 30, -60, 60, -90, 90):
            matrix = cv2.getRotationMatrix2D(
                (resized.shape[1] / 2, resized.shape[0] / 2), angle, 1.0
            )
            rotated = cv2.warpAffine(
                resized,
                matrix,
                (resized.shape[1], resized.shape[0]),
                borderMode=cv2.BORDER_REFLECT_101,
            )
            _, detections = detector.detect(rotated)
            if detections is not None and len(detections) > 0:
                inverse_rotation = cv2.invertAffineTransform(matrix)
                detection_angle = angle
                break
    alpha = np.zeros((h, w), dtype=np.float32)
    faces = []
    for detection in [] if detections is None else detections:
        x, y, fw, fh = (float(v) for v in detection[:4])
        if inverse_rotation is not None:
            corners = np.array(
                [[[x, y], [x + fw, y], [x + fw, y + fh], [x, y + fh]]], dtype=np.float32
            )
            corners = cv2.transform(corners, inverse_rotation)[0]
            x, y = corners.min(axis=0)
            right, bottom = corners.max(axis=0)
            fw, fh = right - x, bottom - y
        x, y, fw, fh = (float(v) / scale for v in (x, y, fw, fh))
        if not all(np.isfinite(v) for v in (x, y, fw, fh)) or min(fw, fh) < 16:
            continue
        # Fully preserve the face core. Feather only outside the enlarged ellipse.
        core = np.zeros((h, w), dtype=np.uint8)
        center = (round(x + fw / 2), round(y + fh * 0.48))
        axes = (max(1, round(fw * 0.72)), max(1, round(fh * 0.72)))
        cv2.ellipse(core, center, axes, 0, 0, 360, 1, -1)
        if not core.any():
            continue
        feather = max(8.0, min(48.0, min(fw, fh) * 0.25))
        distance = cv2.distanceTransform(1 - core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        t = np.clip(distance / feather, 0, 1)
        weight = 1 - t * t * (3 - 2 * t)
        alpha = np.maximum(alpha, weight)
        faces.append(
            {
                "bbox_xywh": [x, y, fw, fh],
                "confidence": float(detection[-1]),
                "feather_pixels": feather,
            }
        )
    blended = (
        np.rint(
            a.astype(np.float32) * alpha[..., None] + b.astype(np.float32) * (1 - alpha[..., None])
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    # Explicit exact copies also prevent any rounding drift in protected cores.
    blended[alpha == 1] = a[alpha == 1]
    blended[alpha == 0] = b[alpha == 0]
    if blend_method == "gradient" and faces and not np.array_equal(a, b):
        # Match boundary illumination using Poisson blending instead of mixing
        # differently exposed patches across a visible elliptical feather ring.
        binary = np.uint8(alpha > 0.001) * 255
        # Padding supports detected faces touching an image boundary.
        padding = 4
        source = cv2.copyMakeBorder(
            cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_REFLECT_101,
        )
        target = cv2.copyMakeBorder(
            cv2.cvtColor(b, cv2.COLOR_RGB2BGR),
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_REFLECT_101,
        )
        mask = cv2.copyMakeBorder(
            binary, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
        )
        x, y, mw, mh = cv2.boundingRect(mask)
        center = (x + mw // 2, y + mh // 2)
        composed = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
        blended = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)[
            padding : padding + h, padding : padding + w
        ].copy()
        blended[binary == 0] = b[binary == 0]
    report = {
        "enabled": True,
        "blend_method": blend_method,
        "preserves_exact_face_pixels": blend_method == "feather",
        "faces": faces,
        "count": len(faces),
        "detection_rotation_degrees": detection_angle,
        "seconds": round(time.monotonic() - start, 3),
        "protected_fraction": float((alpha == 1).mean()),
        "warning": "Original face structure is retained; gradient blending adjusts illumination. Invisible watermarks may remain in the protected region."
        if blend_method == "gradient"
        else "Original face pixels are preserved and may retain invisible watermarks.",
    }
    if not faces:
        report["warning"] = (
            "No faces detected; no protection applied. Face detection can miss faces."
        )
    return Image.fromarray(blended), report, alpha
