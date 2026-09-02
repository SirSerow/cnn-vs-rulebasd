from pathlib import Path

import numpy as np

from conveyor_counter.detectors.yolo_onnx import YoloOnnxDetector
from conveyor_counter.models import ImageSample


class FakeInput:
    name = "images"


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, inputs):
        assert inputs["images"].shape == (1, 3, 480, 640)
        return [
            np.asarray(
                [[[10, 20, 30, 40, 0.9, 0], [1, 2, 3, 4, 0.1, 0]]],
                dtype=np.float32,
            )
        ]


def test_yolo_detector_decodes_end_to_end_output():
    sample = ImageSample(
        np.zeros((480, 640, 3), dtype=np.uint8),
        "sample",
        Path("sample.jpg"),
        (),
    )
    detector = YoloOnnxDetector(Path("unused.onnx"), 0.35, session=FakeSession())

    detections = detector.detect(sample)

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (10, 20, 30, 40)
    assert abs(detections[0].confidence - 0.9) < 1e-6
