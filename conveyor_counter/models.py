"""Shared data structures used by every detector backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BBox = tuple[int, int, int, int]
Image = NDArray[np.uint8]
Point = tuple[int, int]
Polygon = tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    bbox_xyxy: BBox
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: BBox
    confidence: float
    class_id: int = 0
    # Classical detectors may expose the contour that produced the box. Neural
    # detectors leave this unset because they predict boxes directly.
    region_polygon: Polygon | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ImageSample:
    image: Image
    image_id: str
    source_path: Path
    ground_truth: tuple[GroundTruthBox, ...]


@dataclass(frozen=True, slots=True)
class ImageResult:
    image_id: str
    detections: tuple[Detection, ...]
    ground_truth: tuple[GroundTruthBox, ...]
    detection_ms: float
