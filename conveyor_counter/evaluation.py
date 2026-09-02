"""Small, dependency-free evaluator for one-class object detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median

import numpy as np

from .models import BBox, ImageResult


@dataclass(frozen=True, slots=True)
class Summary:
    images: int
    ground_truth_boxes: int
    predicted_boxes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    mean_absolute_count_error: float
    exact_count_accuracy: float
    total_absolute_count_error: int
    median_detection_ms: float
    p95_detection_ms: float
    detector_images_per_second: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def intersection_over_union(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def summarize(results: list[ImageResult], match_iou: float) -> Summary:
    true_positives = 0
    matched_ious: list[float] = []
    absolute_count_errors: list[int] = []
    exact_counts = 0

    for result in results:
        candidates = sorted(
            (
                (intersection_over_union(pred.bbox_xyxy, truth.bbox_xyxy), pi, ti)
                for pi, pred in enumerate(result.detections)
                for ti, truth in enumerate(result.ground_truth)
            ),
            reverse=True,
        )
        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        for iou, prediction_index, truth_index in candidates:
            if iou < match_iou:
                break
            if prediction_index in used_predictions or truth_index in used_truth:
                continue
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
            matched_ious.append(iou)
        true_positives += len(used_predictions)

        count_error = abs(len(result.detections) - len(result.ground_truth))
        absolute_count_errors.append(count_error)
        exact_counts += count_error == 0

    ground_truth_boxes = sum(len(result.ground_truth) for result in results)
    predicted_boxes = sum(len(result.detections) for result in results)
    false_positives = predicted_boxes - true_positives
    false_negatives = ground_truth_boxes - true_positives
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    detection_times = [result.detection_ms for result in results]

    return Summary(
        images=len(results),
        ground_truth_boxes=ground_truth_boxes,
        predicted_boxes=predicted_boxes,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_safe_ratio(2 * precision * recall, precision + recall),
        mean_matched_iou=mean(matched_ious) if matched_ious else 0.0,
        mean_absolute_count_error=mean(absolute_count_errors),
        exact_count_accuracy=_safe_ratio(exact_counts, len(results)),
        total_absolute_count_error=abs(predicted_boxes - ground_truth_boxes),
        median_detection_ms=median(detection_times),
        p95_detection_ms=float(np.percentile(detection_times, 95)),
        detector_images_per_second=_safe_ratio(1000.0, mean(detection_times)),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
