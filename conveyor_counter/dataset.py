"""Load annotated image datasets into the shared coordinate system."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2

from .models import BBox, GroundTruthBox, ImageSample


class EdgeImpulseImageDataset:
    """Edge Impulse image dataset used by the conveyor-cube experiment."""

    def __init__(
        self,
        root: Path,
        split: str,
        normalized_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.split = split
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
                _scale_edge_impulse_box(
                    box,
                    source_size=(source_width, source_height),
                    target_size=self.normalized_size,
                )
            )
            for box in self._annotations[filename]
        )
        return ImageSample(normalized, filename, source_path, boxes)


class CocoImageDataset:
    """Small COCO-format loader used by the road-vehicle experiment."""

    def __init__(
        self,
        root: Path,
        split: str,
        normalized_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.root = root
        self.split = split
        self.normalized_size = normalized_size
        annotation_path = _find_coco_annotation(root, split)
        try:
            manifest = json.loads(annotation_path.read_text(encoding="utf-8"))
            images = manifest["images"]
            annotations = manifest["annotations"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot read COCO labels from {annotation_path}") from exc

        self._images = sorted(images, key=lambda item: item["file_name"])
        self._annotations: dict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue
            self._annotations[int(annotation["image_id"])].append(annotation)

        self._paths_by_name = {
            path.name: path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    def __len__(self) -> int:
        return len(self._images)

    def __iter__(self) -> Iterator[ImageSample]:
        for record in self._images:
            yield self._load(record)

    def _load(self, record: dict[str, object]) -> ImageSample:
        filename = str(record["file_name"])
        source_path = self.root / filename
        if not source_path.is_file():
            source_path = self._paths_by_name.get(Path(filename).name, source_path)

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
                _scale_coco_box(
                    annotation["bbox"],
                    source_size=(source_width, source_height),
                    target_size=self.normalized_size,
                ),
                int(annotation["category_id"]),
            )
            for annotation in self._annotations[int(record["id"])]
        )
        return ImageSample(normalized, filename, source_path, boxes)


def load_dataset(
    root: Path,
    split: str,
    normalized_size: tuple[int, int],
    dataset_format: str,
) -> EdgeImpulseImageDataset | CocoImageDataset:
    if dataset_format == "edge_impulse":
        return EdgeImpulseImageDataset(root, split, normalized_size)
    if dataset_format == "coco":
        return CocoImageDataset(root, split, normalized_size)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def _find_coco_annotation(root: Path, split: str) -> Path:
    preferred = (
        root / "annotations" / f"instance_{split}.json",
        root / "annotations" / f"instances_{split}.json",
    )
    for path in preferred:
        if path.is_file():
            return path

    matches = sorted(
        path
        for path in root.rglob("*.json")
        if split.lower() in path.name.lower() and "source" not in path.name.lower()
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one COCO annotation JSON for split {split!r}, found {matches}"
        )
    return matches[0]


def _scale_edge_impulse_box(
    box: dict[str, object],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> BBox:
    try:
        xywh = (box["x"], box["y"], box["width"], box["height"])
    except KeyError as exc:
        raise ValueError(f"Invalid Edge Impulse box: {box}") from exc
    return _scale_coco_box(xywh, source_size, target_size)


def _scale_coco_box(
    box: object,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> BBox:
    try:
        x, y, width, height = (float(value) for value in box)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid xywh box: {box}") from exc

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
