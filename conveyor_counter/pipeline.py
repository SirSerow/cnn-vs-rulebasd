"""Run a detector, evaluate it, and write inspectable results."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2

from .dataset import EdgeImpulseImageDataset
from .detectors.base import Detector
from .evaluation import Summary, summarize
from .models import ImageResult
from .rendering import render


def run_pipeline(
    dataset: EdgeImpulseImageDataset,
    detector: Detector,
    mode: str,
    output_dir: Path,
    match_iou: float,
    video_fps: float,
    seconds_per_image: int,
    warmup_runs: int,
    write_images: bool = True,
) -> Summary:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    if write_images:
        image_dir.mkdir(exist_ok=True)

    video_path = output_dir / f"{mode}-testing-review.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        dataset.normalized_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {video_path}")

    samples = list(dataset)
    if not samples:
        writer.release()
        raise ValueError("The selected dataset split is empty")
    for _ in range(warmup_runs):
        detector.detect(samples[0])

    results: list[ImageResult] = []
    try:
        for index, sample in enumerate(samples, start=1):
            start = time.perf_counter()
            detections = tuple(detector.detect(sample))
            detection_ms = (time.perf_counter() - start) * 1000
            result = ImageResult(
                sample.image_id,
                detections,
                sample.ground_truth,
                detection_ms,
            )
            results.append(result)
            frame = render(sample.image, detections, mode, detection_ms)
            for _ in range(max(1, round(video_fps * seconds_per_image))):
                writer.write(frame)
            if write_images:
                cv2.imwrite(str(image_dir / f"{index:03d}.jpg"), frame)
    finally:
        writer.release()

    summary = summarize(results, match_iou)
    _write_results(output_dir, results, summary)
    return summary


def _write_results(
    output_dir: Path,
    results: list[ImageResult],
    summary: Summary,
) -> None:
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "ground_truth_count",
                "predicted_count",
                "absolute_count_error",
                "detection_ms",
            ],
        )
        writer.writeheader()
        for result in results:
            truth_count = len(result.ground_truth)
            predicted_count = len(result.detections)
            writer.writerow(
                {
                    "image_id": result.image_id,
                    "ground_truth_count": truth_count,
                    "predicted_count": predicted_count,
                    "absolute_count_error": abs(predicted_count - truth_count),
                    "detection_ms": f"{result.detection_ms:.3f}",
                }
            )

    (output_dir / "summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
