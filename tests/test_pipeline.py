from pathlib import Path

import numpy as np

from conveyor_counter.models import Detection, ImageSample
from conveyor_counter.pipeline import _box_center_is_in_roi, run_pipeline


def test_box_center_roi_filter_accepts_only_boxes_inside_a_polygon():
    roi = (((20, 20), (100, 20), (100, 100), (20, 100)),)

    assert _box_center_is_in_roi((30, 30, 50, 50), roi)
    assert not _box_center_is_in_roi((110, 30, 130, 50), roi)
    assert _box_center_is_in_roi((110, 30, 130, 50), ())


class OneImageDataset:
    split = "testing"
    normalized_size = (16, 16)

    def __iter__(self):
        yield ImageSample(
            np.zeros((16, 16, 3), dtype=np.uint8),
            "sample.jpg",
            Path("sample.jpg"),
            (),
        )


class OneBoxDetector:
    def detect(self, _sample):
        return [Detection((2, 2, 8, 8), 1.0)]


def test_metrics_only_pipeline_skips_all_rendering(tmp_path, monkeypatch):
    def unexpected_render(*_args, **_kwargs):
        raise AssertionError("metrics-only execution must not render frames")

    monkeypatch.setattr("conveyor_counter.pipeline.render", unexpected_render)
    output = tmp_path / "metrics"

    run_pipeline(
        dataset=OneImageDataset(),
        detector=OneBoxDetector(),
        mode="opencv",
        output_dir=output,
        match_iou=0.5,
        video_fps=1.0,
        seconds_per_image=1.0,
        warmup_runs=1,
        write_images=False,
        write_video=False,
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "results.csv",
        "summary.json",
    ]
