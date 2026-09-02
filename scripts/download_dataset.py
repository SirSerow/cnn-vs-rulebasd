#!/usr/bin/env python3
"""Download and validate the pinned Edge Impulse conveyor-cubes dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "datasets" / "cubes-on-conveyor-belt"
DATASET_REPOSITORY = (
    "https://huggingface.co/datasets/edgeimpulse/cubes-on-conveyor-belt"
)
REVISION = "e3d1c8b0c4872b70fcd77d86dec3bde7875e6054"
SPLITS = ("training", "testing")
EXPECTED_IMAGES = {"training": 55, "testing": 15}
EXPECTED_CLASSES = {"blue", "green", "red", "yellow"}
EXPECTED_TOTAL_BOXES = 156
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def run(command: list[str], *, cwd: Path | None = None) -> None:
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def require_download_tools() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download the dataset")
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "git-lfs is required. On Raspberry Pi OS/Debian, run: "
            "sudo apt-get install git-lfs"
        ) from exc


def is_materialized_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as image_file:
        signature = image_file.read(8)
    return signature.startswith(b"\xff\xd8\xff") or signature == b"\x89PNG\r\n\x1a\n"


def validate_dataset(root: Path) -> dict[str, object]:
    total_boxes = 0
    class_counts: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = {}

    if not (root / "info.labels").is_file():
        raise RuntimeError(f"Missing root label manifest: {root / 'info.labels'}")

    for split in SPLITS:
        split_dir = root / split
        labels_path = split_dir / "bounding_boxes.labels"
        if not labels_path.is_file():
            raise RuntimeError(f"Missing annotation file: {labels_path}")

        try:
            manifest = json.loads(labels_path.read_text(encoding="utf-8"))
            annotations = manifest["boundingBoxes"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid annotation manifest: {labels_path}") from exc

        if not isinstance(annotations, dict):
            raise RuntimeError(f"Expected an image-to-box mapping in {labels_path}")

        image_files = {
            path.name
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        annotated_files = set(annotations)
        if image_files != annotated_files:
            missing_images = sorted(annotated_files - image_files)
            missing_labels = sorted(image_files - annotated_files)
            raise RuntimeError(
                f"Image/annotation mismatch in {split}: "
                f"missing images={missing_images[:3]}, "
                f"missing annotations={missing_labels[:3]}"
            )

        for filename, boxes in annotations.items():
            image_path = split_dir / filename
            if not is_materialized_image(image_path):
                raise RuntimeError(
                    f"Image is missing or still an LFS pointer: {image_path}"
                )
            if not isinstance(boxes, list):
                raise RuntimeError(f"Invalid boxes for {image_path}")
            for box in boxes:
                try:
                    label = str(box["label"])
                    values = [box[key] for key in ("x", "y", "width", "height")]
                except (KeyError, TypeError) as exc:
                    raise RuntimeError(f"Invalid box in {labels_path}: {box}") from exc
                if any(not isinstance(value, int) or value < 0 for value in values[:2]):
                    raise RuntimeError(f"Invalid box origin in {labels_path}: {box}")
                if any(
                    not isinstance(value, int) or value <= 0 for value in values[2:]
                ):
                    raise RuntimeError(f"Invalid box size in {labels_path}: {box}")
                class_counts[label] += 1
                total_boxes += 1

        split_counts[split] = {
            "images": len(image_files),
            "boxes": sum(len(boxes) for boxes in annotations.values()),
        }

    actual_images = {split: values["images"] for split, values in split_counts.items()}
    if actual_images != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Pinned dataset image counts changed: expected {EXPECTED_IMAGES}, "
            f"found {actual_images}"
        )
    if set(class_counts) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Pinned dataset classes changed: expected {sorted(EXPECTED_CLASSES)}, "
            f"found {sorted(class_counts)}"
        )
    if total_boxes != EXPECTED_TOTAL_BOXES:
        raise RuntimeError(
            f"Pinned dataset box count changed: expected {EXPECTED_TOTAL_BOXES}, "
            f"found {total_boxes}"
        )

    return {
        "splits": split_counts,
        "total_images": sum(values["images"] for values in split_counts.values()),
        "total_boxes": total_boxes,
        "class_counts": dict(sorted(class_counts.items())),
    }


def source_metadata(counts: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider": "Edge Impulse",
        "mirror": DATASET_REPOSITORY,
        "revision": REVISION,
        "format": "Edge Impulse object detection",
        "source_page": (
            "https://docs.edgeimpulse.com/datasets/image/cubes-on-conveyor-belt-colors"
        ),
        "license": "BSD-3-Clause-Clear",
        "class_names": sorted(EXPECTED_CLASSES),
        "evaluation_class": "cube",
        "downloaded_counts": counts,
    }


def write_metadata(root: Path, counts: dict[str, object]) -> None:
    (root / ".source.json").write_text(
        json.dumps(source_metadata(counts), indent=2) + "\n", encoding="utf-8"
    )


def download_dataset(output: Path) -> None:
    require_download_tools()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary_directory:
        checkout = Path(temporary_directory) / "checkout"
        checkout.mkdir()
        run(["git", "init", "--quiet"], cwd=checkout)
        run(["git", "remote", "add", "origin", DATASET_REPOSITORY], cwd=checkout)
        run(["git", "fetch", "--depth", "1", "origin", REVISION], cwd=checkout)
        run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout)
        run(["git", "lfs", "pull", "--include=training/**,testing/**"], cwd=checkout)

        counts = validate_dataset(checkout)
        write_metadata(checkout, counts)
        shutil.rmtree(checkout / ".git")
        checkout.rename(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned, annotated conveyor-cubes dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dataset destination (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output.exists():
        try:
            counts = validate_dataset(args.output)
        except RuntimeError as exc:
            print(
                f"Existing destination is incomplete: {exc}. "
                "Remove that specific directory or choose another --output.",
                file=sys.stderr,
            )
            return 1
        if not (args.output / ".source.json").is_file():
            write_metadata(args.output, counts)
        print(f"Dataset is already present and valid: {args.output}")
        return 0

    try:
        print(f"Downloading pinned dataset revision {REVISION}...")
        download_dataset(args.output)
        counts = validate_dataset(args.output)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1

    split_summary = ", ".join(
        f"{split}={values['images']} images/{values['boxes']} boxes"
        for split, values in counts["splits"].items()
    )
    print(f"Validated dataset: {split_summary}")
    print(f"Saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
