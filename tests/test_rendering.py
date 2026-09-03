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
