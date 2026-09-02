#!/usr/bin/env python3
"""Prepare training data, fine-tune YOLO26n, and export its ONNX model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "datasets" / "cubes-on-conveyor-belt"
DEFAULT_PREPARED = REPOSITORY_ROOT / "datasets" / "yolo-cubes"
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "yolo26n-cube.onnx"


def prepare_dataset(source: Path, destination: Path) -> Path:
    """Convert only the source training split to YOLO text annotations."""
    source_split = source / "training"
    manifest = json.loads(
        (source_split / "bounding_boxes.labels").read_text(encoding="utf-8")
    )["boundingBoxes"]
    filenames = sorted(manifest)

    if destination.exists():
        shutil.rmtree(destination)
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True)
        (destination / "labels" / split).mkdir(parents=True)

    # Every fifth sorted image is validation. The frozen public testing split
    # is never copied or read by this training script.
    for index, filename in enumerate(filenames):
        split = "val" if index % 5 == 0 else "train"
        source_image = source_split / filename
        target_image = destination / "images" / split / filename
        shutil.copy2(source_image, target_image)

        from PIL import Image

        with Image.open(source_image) as image:
            image_width, image_height = image.size
        labels = [
            _to_yolo_line(box, image_width, image_height) for box in manifest[filename]
        ]
        (destination / "labels" / split / f"{Path(filename).stem}.txt").write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8",
        )

    dataset_yaml = destination / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(destination.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "cube"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def _to_yolo_line(
    box: dict[str, int],
    image_width: int,
    image_height: int,
) -> str:
    center_x = (box["x"] + box["width"] / 2) / image_width
    center_y = (box["y"] + box["height"] / 2) / image_height
    width = box["width"] / image_width
    height = box["height"] / image_height
    return f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"


def train_and_export(
    dataset_yaml: Path,
    output_model: Path,
    epochs: int,
) -> None:
    from ultralytics import YOLO

    run_name = "yolo26n-cube"
    run_root = REPOSITORY_ROOT / "training_runs"
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=4,
        patience=10,
        project=str(run_root),
        name=run_name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        mosaic=0.0,
        translate=0.05,
        scale=0.2,
    )

    best_model = YOLO(str(run_root / run_name / "weights" / "best.pt"))
    exported = Path(
        best_model.export(
            format="onnx",
            imgsz=(480, 640),
            dynamic=False,
            simplify=True,
        )
    )
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output_model)
    print(f"Exported model: {output_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_yaml = prepare_dataset(args.source, args.prepared)
    train_and_export(dataset_yaml, args.output_model, args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
