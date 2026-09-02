from pathlib import Path

import cv2
import numpy as np

from conveyor_counter.detectors.opencv import OpenCVDetector
from conveyor_counter.models import ImageSample


def test_opencv_detector_finds_a_colored_cube():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (250, 200), (320, 270), (0, 0, 255), -1)
    config = {
        "roi_y": [115, 380],
        "color_ranges": [[[0, 60, 40], [12, 255, 255]]],
        "morphology_kernel": 5,
        "closing_iterations": 2,
        "min_area_px": 300,
        "max_area_px": 15000,
        "min_size_px": 25,
        "max_size_px": 140,
        "min_aspect_ratio": 0.45,
        "max_aspect_ratio": 2.2,
        "min_solidity": 0.65,
        "min_rectangularity": 0.35,
        "bbox_padding_px": 5,
        "reject_frame_border": True,
    }
    sample = ImageSample(image, "synthetic", Path("synthetic.jpg"), ())

    detections = OpenCVDetector(config).detect(sample)

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (245, 195, 326, 276)
