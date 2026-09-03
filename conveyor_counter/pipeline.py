"""Run a detector, evaluate it, and write inspectable results."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .dataset import CocoImageDataset, EdgeImpulseImageDataset
from .detectors.base import Detector
from .evaluation import Summary, summarize
from .models import BBox, ImageResult, Polygon
from .rendering import render


def run_pipeline(
    dataset: EdgeImpulseImageDataset | CocoImageDataset,
    detector: Detector,
    mode: str,
    output_dir: Path,
    match_iou: float,
    video_fps: float,
    seconds_per_image: float,
    warmup_runs: int,
    object_label: str = "object",
    write_images: bool = True,
    evaluation_roi: tuple[Polygon, ...] = (),
) -> Summary:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    if write_images:
        image_dir.mkdir(exist_ok=True)

    video_path = output_dir / f"{mode}-{dataset.split}-review.mp4"
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
            detections = tuple(
                detection
                for detection in detections
                if _box_center_is_in_roi(detection.bbox_xyxy, evaluation_roi)
            )
            ground_truth = sample.ground_truth
            result = ImageResult(
                sample.image_id,
                detections,
                ground_truth,
                detection_ms,
            )
            results.append(result)
            frame = render(
                sample.image,
                detections,
                mode,
                detection_ms,
                ground_truth,
                object_label,
                evaluation_roi,
            )
            for _ in range(max(1, round(video_fps * seconds_per_image))):
                writer.write(frame)
            if write_images:
                cv2.imwrite(str(image_dir / f"{index:03d}.jpg"), frame)
    finally:
        writer.release()

    summary = summarize(results, match_iou)
    _write_results(output_dir, results, summary)
    return summary


def _box_center_is_in_roi(bbox: BBox, roi_polygons: tuple[Polygon, ...]) -> bool:
    if not roi_polygons:
        return True
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return any(
        cv2.pointPolygonTest(
            np.asarray(polygon, dtype=np.int32),
            center,
            False,
        )
        >= 0
        for polygon in roi_polygons
    )


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
