"""Shared rendering for images and review videos."""

from __future__ import annotations

import cv2

from .models import Detection, Image


def render(
    image: Image,
    detections: tuple[Detection, ...],
    mode: str,
    detection_ms: float,
    ground_truth_count: int,
    object_label: str,
) -> Image:
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
        label = f"{object_label} {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )

    title = (
        f"{mode.upper()} | predicted: {len(detections)} | "
        f"truth: {ground_truth_count} | {detection_ms:.1f} ms"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(
        output,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
