from conveyor_counter.pipeline import _box_center_is_in_roi


def test_box_center_roi_filter_accepts_only_boxes_inside_a_polygon():
    roi = (((20, 20), (100, 20), (100, 100), (20, 100)),)

    assert _box_center_is_in_roi((30, 30, 50, 50), roi)
    assert not _box_center_is_in_roi((110, 30, 130, 50), roi)
    assert _box_center_is_in_roi((110, 30, 130, 50), ())
