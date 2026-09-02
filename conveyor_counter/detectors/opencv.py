"""Color and geometry based detector for the controlled conveyor scene."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..models import Detection, ImageSample


class OpenCVDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def detect(self, sample: ImageSample) -> list[Detection]:
        image = sample.image
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros((height, width), dtype=np.uint8)

        for lower, upper in self.config["color_ranges"]:
            mask |= cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )

        roi_start, roi_end = self.config["roi_y"]
        mask[:roi_start] = 0
        mask[roi_end:] = 0

        kernel_size = int(self.config["morphology_kernel"])
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(self.config["closing_iterations"]),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        detections = [
            detection
            for contour in contours
            if (detection := self._contour_to_detection(contour, width, height))
            is not None
        ]
        return sorted(detections, key=lambda item: item.bbox_xyxy[0])

    def _contour_to_detection(
        self,
        contour: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> Detection | None:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if self.config["reject_frame_border"] and (x == 0 or x + width >= image_width):
            return None

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0.0
        rectangularity = area / (width * height)
        aspect_ratio = width / height

        if not (
            self.config["min_area_px"] <= area <= self.config["max_area_px"]
            and self.config["min_size_px"] <= width <= self.config["max_size_px"]
            and self.config["min_size_px"] <= height <= self.config["max_size_px"]
            and self.config["min_aspect_ratio"]
            <= aspect_ratio
            <= self.config["max_aspect_ratio"]
            and solidity >= self.config["min_solidity"]
            and rectangularity >= self.config["min_rectangularity"]
        ):
            return None

        padding = int(self.config["bbox_padding_px"])
        bbox = (
            max(0, x - padding),
            max(0, y - padding),
            min(image_width, x + width + padding),
            min(image_height, y + height + padding),
        )
        return Detection(bbox, confidence=1.0)
