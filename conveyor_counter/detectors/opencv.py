"""Rule-based detectors for controlled color and fixed-camera scenes."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..models import Detection, ImageSample


class OpenCVDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.method = str(config.get("method", "color"))
        self._subtractor = None
        if self.method == "background_subtraction":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=int(config["history"]),
                varThreshold=float(config["var_threshold"]),
                detectShadows=bool(config["detect_shadows"]),
            )

    def detect(self, sample: ImageSample) -> list[Detection]:
        if self.method == "color":
            mask = self._color_mask(sample.image)
        elif self.method == "background_subtraction":
            mask = self._foreground_mask(sample.image)
        else:
            raise ValueError(f"Unsupported OpenCV method: {self.method}")

        height, width = mask.shape
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

    def _color_mask(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in self.config["color_ranges"]:
            mask |= cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
        return self._clean_mask(mask)

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        assert self._subtractor is not None
        mask = self._subtractor.apply(
            image,
            learningRate=float(self.config.get("learning_rate", -1)),
        )
        # MOG2 uses 127 for shadows and 255 for foreground. Cars alone count.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return self._clean_mask(mask)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        roi_start, roi_end = self.config["roi_y"]
        mask[: int(roi_start)] = 0
        mask[int(roi_end) :] = 0

        kernel_size = int(self.config["morphology_kernel"])
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=int(self.config.get("opening_iterations", 1)),
        )
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(self.config["closing_iterations"]),
        )

    def _contour_to_detection(
        self,
        contour: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> Detection | None:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if self.config["reject_frame_border"] and (
            x == 0 or x + width >= image_width
        ):
            return None

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0.0
        rectangularity = area / (width * height)
        aspect_ratio = width / height
        min_width = self.config.get("min_width_px", self.config.get("min_size_px"))
        max_width = self.config.get("max_width_px", self.config.get("max_size_px"))
        min_height = self.config.get("min_height_px", self.config.get("min_size_px"))
        max_height = self.config.get("max_height_px", self.config.get("max_size_px"))

        if not (
            self.config["min_area_px"] <= area <= self.config["max_area_px"]
            and min_width <= width <= max_width
            and min_height <= height <= max_height
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
