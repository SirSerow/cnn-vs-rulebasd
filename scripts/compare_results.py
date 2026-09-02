#!/usr/bin/env python3
"""Create a compact side-by-side report from two benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencv", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/road-comparison.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    opencv = json.loads(args.opencv.read_text(encoding="utf-8"))
    yolo = json.loads(args.yolo.read_text(encoding="utf-8"))
    rows = [
        ("Precision", "precision", "%"),
        ("Recall", "recall", "%"),
        ("F1", "f1", "%"),
        ("Exact-count frames", "exact_count_accuracy", "%"),
        ("Mean absolute count error", "mean_absolute_count_error", "number"),
        ("Total absolute count error", "total_absolute_count_error", "integer"),
        ("Median detector latency", "median_detection_ms", "ms"),
        ("Detector throughput", "detector_images_per_second", "fps"),
    ]

    lines = [
        "# Road-vehicle benchmark",
        "",
        "| Metric | OpenCV MOG2 | YOLO26n ONNX |",
        "|---|---:|---:|",
    ]
    for label, key, unit in rows:
        left = format_value(opencv[key], unit)
        right = format_value(yolo[key], unit)
        lines.append(f"| {label} | {left} | {right} |")

    lines.extend(
        [
            "",
            "Both methods used the same 100 annotated validation frames, "
            "640×480 input, one-class vehicle evaluation, and IoU 0.50 matching.",
            "",
            "YOLO26n used pretrained COCO weights only; it was not fine-tuned "
            "on these road images.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


def format_value(value: float, unit: str) -> str:
    if unit == "%":
        return f"{100 * value:.1f}%"
    if unit == "integer":
        return str(int(value))
    if unit == "ms":
        return f"{value:.2f} ms"
    if unit == "fps":
        return f"{value:.1f} FPS"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
