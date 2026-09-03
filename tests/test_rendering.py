import numpy as np

from conveyor_counter.models import Detection, GroundTruthBox
from conveyor_counter.rendering import render


def test_render_fills_opencv_detection_region_but_not_its_whole_box():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = Detection(
        (35, 35, 85, 85),
        confidence=1.0,
        region_polygon=((40, 40), (80, 40), (60, 70)),
    )

    output = render(
        image,
        (detection,),
        mode="opencv",
        detection_ms=1.0,
        ground_truth=(GroundTruthBox((90, 40, 110, 70)),),
        object_label="object",
    )

    # The triangular contour is orange, while another pixel inside the
    # rectangular box remains unchanged.
    assert output[50, 60, 2] > output[50, 60, 1] > output[50, 60, 0]
    assert np.array_equal(output[75, 75], image[75, 75])


def test_render_dims_pixels_outside_roi_and_draws_a_thick_border():
    image = np.full((120, 160, 3), 200, dtype=np.uint8)

    output = render(
        image,
        (),
        mode="yolo",
        detection_ms=1.0,
        ground_truth=(),
        object_label="object",
        roi_polygons=(((40, 40), (120, 40), (120, 100), (40, 100)),),
    )

    assert np.array_equal(output[70, 80], image[70, 80])
    assert np.all(output[70, 20] < image[70, 20])
    assert tuple(output[40, 80]) == (0, 220, 255)


def test_render_does_not_draw_ground_truth_boxes():
    image = np.full((120, 160, 3), 200, dtype=np.uint8)

    output = render(
        image,
        (),
        mode="yolo",
        detection_ms=1.0,
        ground_truth=(GroundTruthBox((90, 40, 110, 70)),),
        object_label="object",
    )

    assert np.array_equal(output[40, 90], image[40, 90])
