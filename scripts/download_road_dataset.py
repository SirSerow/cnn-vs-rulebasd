#!/usr/bin/env python3
"""Download and validate PaddleX's small UA-DETRAC vehicle example."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://paddle-model-ecology.bj.bcebos.com/"
    "paddlex/data/vehicle_coco_examples.tar"
)
EXPECTED_SPLIT_SIZES = {"train": 500, "val": 100}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/vehicle-coco-examples"),
    )
    return parser.parse_args()


def main() -> int:
    output = parse_args().output
    if output.is_dir():
        validate(output)
        print(f"Dataset already valid: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp_name:
        temp = Path(temp_name)
        archive = temp / "vehicle_coco_examples.tar"
        digest = download(archive)
        extracted = temp / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)

        candidates = [
            path
            for path in extracted.rglob("*")
            if path.is_dir()
            and any(path.rglob("*.json"))
            and "vehicle_coco_examples" in path.name
        ]
        source = min(candidates, key=lambda path: len(path.parts)) if candidates else extracted
        shutil.move(str(source), output)

    validate(output)
    metadata = {
        "name": "PaddleX vehicle COCO example",
        "source_dataset": "UA-DETRAC",
        "url": URL,
        "archive_sha256": digest,
        "expected_split_sizes": EXPECTED_SPLIT_SIZES,
    }
    (output / ".source.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded and validated: {output}")
    return 0


def download(destination: Path) -> str:
    digest = hashlib.sha256()
    with urlopen(URL, timeout=60) as response, destination.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
        tar.extractall(destination)


def validate(root: Path) -> None:
    image_names = {
        path.name
        for path in root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        matches = [
            path
            for path in root.rglob("*.json")
            if split in path.name.lower() and "source" not in path.name.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {split} annotation JSON, found {matches}")
        manifest = json.loads(matches[0].read_text(encoding="utf-8"))
        images = manifest.get("images", [])
        annotations = manifest.get("annotations", [])
        if len(images) != expected_size:
            raise ValueError(
                f"Expected {expected_size} {split} images, found {len(images)}"
            )
        if not annotations:
            raise ValueError(f"No annotations found for {split}")
        missing = [
            record["file_name"]
            for record in images
            if Path(record["file_name"]).name not in image_names
        ]
        if missing:
            raise ValueError(f"Missing source images, first entries: {missing[:3]}")


if __name__ == "__main__":
    raise SystemExit(main())
