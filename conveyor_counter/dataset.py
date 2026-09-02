"""Load Edge Impulse image datasets into the shared coordinate system."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2

from .models import BBox, GroundTruthBox, ImageSample


class EdgeImpulseImageDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        normalized_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.split_dir = root / split
        self.normalized_size = normalized_size
        labels_path = self.split_dir / "bounding_boxes.labels"
        try:
            manifest = json.loads(labels_path.read_text(encoding="utf-8"))
            self._annotations = manifest["boundingBoxes"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot read Edge Impulse labels from {labels_path}"
            ) from exc
        if not isinstance(self._annotations, dict):
            raise ValueError(f"Expected an image-to-box mapping in {labels_path}")
        self._filenames = sorted(self._annotations)

    def __len__(self) -> int:
        return len(self._filenames)

    def __iter__(self) -> Iterator[ImageSample]:
        for filename in self._filenames:
            yield self[filename]

    def __getitem__(self, filename: str) -> ImageSample:
        source_path = self.split_dir / filename
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Cannot decode image: {source_path}")

        source_height, source_width = image.shape[:2]
        normalized = cv2.resize(
            image,
            self.normalized_size,
            interpolation=cv2.INTER_AREA,
        )
        boxes = tuple(
            GroundTruthBox(
                _scale_box(
                    box,
                    source_size=(source_width, source_height),
                    target_size=self.normalized_size,
                )
            )
            for box in self._annotations[filename]
        )
        return ImageSample(normalized, filename, source_path, boxes)


def _scale_box(
    box: dict[str, object],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> BBox:
    try:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["width"])
        height = float(box["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Edge Impulse box: {box}") from exc

    source_width, source_height = source_size
    target_width, target_height = target_size
    x_scale = target_width / source_width
    y_scale = target_height / source_height
    x1 = round(x * x_scale)
    y1 = round(y * y_scale)
    x2 = round((x + width) * x_scale)
    y2 = round((y + height) * y_scale)
    return (
        max(0, min(x1, target_width)),
        max(0, min(y1, target_height)),
        max(0, min(x2, target_width)),
        max(0, min(y2, target_height)),
    )
