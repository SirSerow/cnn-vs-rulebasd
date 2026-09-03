"""Shared rendering for images and review videos."""

from __future__ import annotations

import cv2
import numpy as np

from .models import Detection, GroundTruthBox, Image, Polygon


def render(
    image: Image,
    detections: tuple[Detection, ...],
    mode: str,
    detection_ms: float,
    ground_truth: tuple[GroundTruthBox, ...],
    object_label: str,
    roi_polygons: tuple[Polygon, ...] = (),
) -> Image:
    output = image.copy()
    roi_contours = [
        np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        for polygon in roi_polygons
        if len(polygon) >= 3
    ]
    if roi_contours:
        roi_mask = np.zeros(output.shape[:2], dtype=np.uint8)
        cv2.drawContours(roi_mask, roi_contours, -1, 255, cv2.FILLED)
        outside_roi = roi_mask == 0
        dark_gray = np.full_like(output, (70, 70, 70))
        dimmed = cv2.addWeighted(output, 0.45, dark_gray, 0.55, 0)
        output[outside_roi] = dimmed[outside_roi]

    region_contours = [
        np.asarray(detection.region_polygon, dtype=np.int32).reshape(-1, 1, 2)
        for detection in detections
        if detection.region_polygon is not None and len(detection.region_polygon) >= 3
    ]
    if region_contours:
        region_mask = np.zeros(output.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, region_contours, -1, 255, cv2.FILLED)

        region_color = np.full_like(output, (0, 165, 255))
        blended = cv2.addWeighted(output, 0.62, region_color, 0.38, 0)
        output[region_mask > 0] = blended[region_mask > 0]
        cv2.drawContours(output, region_contours, -1, (0, 220, 255), 2)

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

    if roi_contours:
        cv2.drawContours(output, roi_contours, -1, (0, 220, 255), 3)

    title = (
        f"{mode.upper()} | predicted: {len(detections)} | "
        f"truth: {len(ground_truth)} | {detection_ms:.1f} ms"
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
    legend = "prediction: green"
    if roi_contours:
        legend += " | ROI: yellow"
    if region_contours:
        legend += " | CV region: orange"
    text_width = cv2.getTextSize(
        legend,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1,
    )[
        0
    ][0]
    top = output.shape[0] - 23
    cv2.rectangle(
        output,
        (0, top),
        (text_width + 14, output.shape[0]),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        legend,
        (7, top + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output
