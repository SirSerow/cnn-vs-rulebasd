import json

import cv2
import numpy as np

from conveyor_counter.dataset import EdgeImpulseImageDataset


def test_dataset_resizes_image_and_box(tmp_path):
    split = tmp_path / "testing"
    split.mkdir()
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(split / "sample.jpg"), image)
    (split / "bounding_boxes.labels").write_text(
        json.dumps(
            {
                "boundingBoxes": {
                    "sample.jpg": [
                        {"label": "red", "x": 10, "y": 5, "width": 20, "height": 10}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    sample = next(iter(EdgeImpulseImageDataset(tmp_path, "testing", (200, 100))))

    assert sample.image.shape == (100, 200, 3)
    assert sample.ground_truth[0].bbox_xyxy == (20, 10, 60, 30)
