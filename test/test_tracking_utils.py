import numpy as np
import pytest

from rdk_pan_tilt_tracker.tracking_utils import get_face_target
from rdk_pan_tilt_tracker.tracking_utils import is_facing_camera
from rdk_pan_tilt_tracker.tracking_utils import StableTrueTrigger


def keypoints(nose=(50.0, 60.0), eyes=((40.0, 50.0), (60.0, 50.0))):
    points = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros(17, dtype=np.float32)

    points[0] = nose
    points[1] = eyes[0]
    points[2] = eyes[1]
    points[3] = (30.0, 52.0)
    points[4] = (70.0, 52.0)
    scores[:5] = 0.9

    return points, scores


def test_front_face_is_accepted():
    points, scores = keypoints()
    assert is_facing_camera(points, scores)


def test_sideways_face_is_rejected_by_nose_offset():
    points, scores = keypoints(nose=(58.0, 60.0))
    assert not is_facing_camera(points, scores)


def test_tilted_face_is_rejected_by_eye_height():
    points, scores = keypoints(eyes=((40.0, 45.0), (60.0, 55.0)))
    assert not is_facing_camera(points, scores)


def test_low_confidence_eye_is_rejected():
    points, scores = keypoints()
    scores[2] = 0.1
    assert not is_facing_camera(points, scores)


def test_rdk_face_target_contains_facing_status():
    points, scores = keypoints()
    target = get_face_target(points, scores)
    assert target is not None
    assert target['facing'] is True


def test_invalid_keypoints_raise_clear_error():
    with pytest.raises(ValueError):
        is_facing_camera(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.float32)
        )


def test_stable_true_trigger_emits_once_and_rearms():
    trigger = StableTrueTrigger(
        hold_seconds=0.8,
        reset_seconds=1.0,
        cooldown_seconds=2.0
    )

    assert not trigger.update(True, 0.0)
    assert not trigger.update(True, 0.7)
    assert trigger.update(True, 0.8)
    assert not trigger.update(True, 1.0)
    assert not trigger.update(False, 1.1)
    assert not trigger.update(False, 2.1)
    assert not trigger.update(True, 2.2)
    assert trigger.update(True, 3.0)
