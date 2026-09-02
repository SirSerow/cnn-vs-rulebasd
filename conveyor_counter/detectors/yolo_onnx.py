"""Minimal ONNX Runtime adapter for end-to-end YOLO26 detection exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ..models import Detection, ImageSample


class YoloOnnxDetector:
    """Read YOLO26's ``[x1, y1, x2, y2, score, class]`` output.

    Export the model in its default end-to-end mode. No extra NMS is necessary.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
        class_ids: set[int] | None = None,
        session: Any | None = None,
    ) -> None:
        if session is None and not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO ONNX model not found: {model_path}. "
                "Train/export it before running --mode yolo."
            )
        self.session = session or ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold
        self.class_ids = class_ids

    def detect(self, sample: ImageSample) -> list[Detection]:
        rgb = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
        output = np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        return self._decode(output, sample.image.shape[1], sample.image.shape[0])

    def _decode(
        self,
        output: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
            raise ValueError(
                "Expected an end-to-end YOLO26 output shaped (1, N, 6), "
                f"received {output.shape}"
            )

        detections: list[Detection] = []
        for x1, y1, x2, y2, confidence, class_id in output[0]:
            decoded_class_id = int(class_id)
            if confidence < self.confidence_threshold:
                continue
            if self.class_ids is not None and decoded_class_id not in self.class_ids:
                continue
            box = (
                int(np.clip(round(float(x1)), 0, image_width)),
                int(np.clip(round(float(y1)), 0, image_height)),
                int(np.clip(round(float(x2)), 0, image_width)),
                int(np.clip(round(float(y2)), 0, image_height)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(Detection(box, float(confidence), decoded_class_id))
        return detections
