"""Command-line interface for both detector modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .dataset import EdgeImpulseImageDataset
from .detectors import OpenCVDetector, YoloOnnxDetector
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("opencv", "yolo"), required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/cubes-on-conveyor-belt"),
    )
    parser.add_argument("--split", default="testing")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def build_detector(mode: str, config: dict[str, Any]):
    if mode == "opencv":
        return OpenCVDetector(config["opencv"])
    yolo = config["yolo"]
    return YoloOnnxDetector(
        Path(yolo["model_path"]),
        float(yolo["confidence_threshold"]),
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    size = tuple(config["dataset"]["normalized_size"])
    dataset = EdgeImpulseImageDataset(args.dataset, args.split, size)
    output_dir = args.output or Path("results") / f"{args.mode}-{args.split}"
    summary = run_pipeline(
        dataset=dataset,
        detector=build_detector(args.mode, config),
        mode=args.mode,
        output_dir=output_dir,
        match_iou=float(config["evaluation"]["match_iou"]),
        video_fps=float(config["output"]["video_fps"]),
        seconds_per_image=int(config["output"]["seconds_per_image"]),
        warmup_runs=int(config["benchmark"]["warmup_runs"]),
        write_images=not args.no_images,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    print(f"Results: {output_dir}")
    return 0
