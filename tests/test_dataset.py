import json

import cv2
import numpy as np

from conveyor_counter.dataset import CocoImageDataset, EdgeImpulseImageDataset


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


def test_coco_dataset_loads_ordered_frames_and_boxes(tmp_path):
    images = tmp_path / "val"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    cv2.imwrite(str(images / "frame.jpg"), np.zeros((50, 100, 3), dtype=np.uint8))
    (annotations / "instance_val.json").write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "val/frame.jpg"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 2,
                        "bbox": [10, 5, 20, 10],
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 2, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )

    sample = next(iter(CocoImageDataset(tmp_path, "val", (200, 100))))

    assert sample.image.shape == (100, 200, 3)
    assert sample.ground_truth[0].bbox_xyxy == (20, 10, 60, 30)
    assert sample.ground_truth[0].class_id == 2
