import numpy as np

from goal_mapping import classify_goal_zone, compute_goal_homography, map_point_to_goal_uv


def test_goal_homography_maps_corners_to_unit_goal():
    corners = {
        "bottom_left": [10, 100],
        "bottom_right": [110, 100],
        "top_right": [110, 0],
        "top_left": [10, 0],
    }
    homography = compute_goal_homography(corners)
    u, v = map_point_to_goal_uv((60, 50), homography)
    assert np.isclose(u, 0.5)
    assert np.isclose(v, 0.5)


def test_classify_goal_zone():
    assert classify_goal_zone(0.1, 0.2) == {
        "goal_zone_horizontal": "left",
        "goal_zone_vertical": "low",
    }
    assert classify_goal_zone(0.5, 0.5) == {
        "goal_zone_horizontal": "center",
        "goal_zone_vertical": "middle",
    }
    assert classify_goal_zone(None, None) == {
        "goal_zone_horizontal": "unknown",
        "goal_zone_vertical": "unknown",
    }

