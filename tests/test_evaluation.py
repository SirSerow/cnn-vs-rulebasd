from conveyor_counter.evaluation import intersection_over_union, summarize
from conveyor_counter.models import Detection, GroundTruthBox, ImageResult


def test_iou_and_summary():
    truth = GroundTruthBox((10, 10, 30, 30))
    detection = Detection((10, 10, 30, 30), 0.9)
    result = ImageResult("sample", (detection,), (truth,), 10.0)

    assert intersection_over_union(detection.bbox_xyxy, truth.bbox_xyxy) == 1.0
    summary = summarize([result], match_iou=0.5)
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.exact_count_accuracy == 1.0
    assert summary.detector_images_per_second == 100.0


def test_total_absolute_count_error_sums_frame_errors():
    results = [
        ImageResult("under", (), (GroundTruthBox((0, 0, 10, 10)),), 1.0),
        ImageResult(
            "over",
            (Detection((0, 0, 10, 10), 1.0), Detection((20, 20, 30, 30), 1.0)),
            (GroundTruthBox((0, 0, 10, 10)),),
            1.0,
        ),
    ]

    assert summarize(results, match_iou=0.5).total_absolute_count_error == 2
