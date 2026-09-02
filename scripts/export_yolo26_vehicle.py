#!/usr/bin/env python3
"""Export pretrained COCO YOLO26n to the ONNX contract used by the app."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/yolo26n-coco.onnx"),
    )
    return parser.parse_args()


def main() -> int:
    output = parse_args().output
    if output.is_file():
        print(f"Model already exists: {output}")
        return 0

    model = YOLO("yolo26n.pt")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=(480, 640),
            batch=1,
            dynamic=False,
            nms=True,
            opset=17,
            simplify=True,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), output)
    print(f"Exported pretrained COCO model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
