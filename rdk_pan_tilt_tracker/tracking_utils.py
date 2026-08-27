from typing import Tuple

import numpy as np


TIME_EPSILON = 1e-6


class StableTrueTrigger:
    """Emit once after a boolean condition remains true for a duration."""

    def __init__(
        self,
        hold_seconds=0.8,
        reset_seconds=1.0,
        cooldown_seconds=5.0
    ):
        if min(hold_seconds, reset_seconds, cooldown_seconds) < 0.0:
            raise ValueError('trigger durations must be non-negative')

        self.hold_seconds = float(hold_seconds)
        self.reset_seconds = float(reset_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.true_since = None
        self.false_since = None
        self.last_triggered = None
        self.latched = False

    def update(self, condition, now_seconds):
        """Return True once when a stable condition becomes actionable."""
        now_seconds = float(now_seconds)

        if condition:
            self.false_since = None
            if self.true_since is None:
                self.true_since = now_seconds

            held_long_enough = (
                now_seconds - self.true_since + TIME_EPSILON
                >= self.hold_seconds
            )
            cooldown_finished = (
                self.last_triggered is None
                or now_seconds - self.last_triggered + TIME_EPSILON
                >= self.cooldown_seconds
            )

            if not self.latched and held_long_enough and cooldown_finished:
                self.latched = True
                self.last_triggered = now_seconds
                return True

            return False

        self.true_since = None
        if self.false_since is None:
            self.false_since = now_seconds

        if (
            now_seconds - self.false_since + TIME_EPSILON
            >= self.reset_seconds
        ):
            self.latched = False

        return False


def split_keypoints(
    keypoints,
    scores=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize COCO keypoints from Ultralytics or RDKX5 adapters.

    ``keypoints`` may be an ``(N, 3)`` array containing x, y, confidence,
    or an ``(N, 2)`` array accompanied by a one-dimensional ``scores`` array.
    """
    points = np.asarray(keypoints, dtype=np.float32)

    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
        raise ValueError('keypoints must have shape (N, 2+) with N >= 5')

    if scores is None:
        if points.shape[1] < 3:
            raise ValueError(
                'scores are required when keypoints do not contain confidence'
            )
        confidence = points[:, 2]
    else:
        confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
        if confidence.shape[0] != points.shape[0]:
            raise ValueError('scores and keypoints must have the same length')

    return points[:, :2], confidence


# ================================================================
# Face direction
# ================================================================

def is_facing_camera(
    person,
    score=None,
    keypoint_confidence=0.30,
    nose_offset_threshold=0.25,
    eye_height_threshold=0.20,
    min_eye_distance=5.0
):
    """
    Estimate whether the face is approximately facing the camera.

    COCO keypoints:
        0: nose
        1: left_eye
        2: right_eye
        3: left_ear
        4: right_ear
    """

    person, score = split_keypoints(person, score)

    # Nose + both eyes are required
    for index in [0, 1, 2]:

        if score[index] < keypoint_confidence:
            return False

    nose = person[0]

    left_eye = person[1]
    right_eye = person[2]

    nose_x = float(nose[0])

    left_eye_x = float(left_eye[0])
    left_eye_y = float(left_eye[1])

    right_eye_x = float(right_eye[0])
    right_eye_y = float(right_eye[1])

    # ------------------------------------------------------------
    # Eye distance
    # ------------------------------------------------------------

    eye_distance = abs(
        right_eye_x - left_eye_x
    )

    if eye_distance < min_eye_distance:
        return False

    # ------------------------------------------------------------
    # Nose position relative to eye center
    # ------------------------------------------------------------

    eye_center_x = (
        left_eye_x + right_eye_x
    ) / 2.0

    nose_offset_ratio = abs(
        nose_x - eye_center_x
    ) / eye_distance

    # ------------------------------------------------------------
    # Eye height difference
    # ------------------------------------------------------------

    eye_height_ratio = abs(
        left_eye_y - right_eye_y
    ) / eye_distance

    # ------------------------------------------------------------
    # Judgment
    # ------------------------------------------------------------

    if nose_offset_ratio > nose_offset_threshold:
        return False

    if eye_height_ratio > eye_height_threshold:
        return False

    return True


# ================================================================
# Face target
# ================================================================

def get_face_target(
    person,
    score=None,
    keypoint_confidence=0.30,
    nose_offset_threshold=0.25,
    eye_height_threshold=0.20
):
    """
    Calculate face center and face orientation information.
    """

    person, score = split_keypoints(person, score)
    face_points = []

    # Nose / eyes / ears
    for index in [0, 1, 2, 3, 4]:

        if score[index] >= keypoint_confidence:

            face_points.append(
                person[index]
            )

    if len(face_points) < 2:
        return None

    face_points = np.asarray(
        face_points,
        dtype=np.float32
    )

    face_x = float(
        np.mean(face_points[:, 0])
    )

    face_y = float(
        np.mean(face_points[:, 1])
    )

    facing = is_facing_camera(
        person=person,
        score=score,
        keypoint_confidence=keypoint_confidence,
        nose_offset_threshold=nose_offset_threshold,
        eye_height_threshold=eye_height_threshold
    )

    return {
        'x': face_x,
        'y': face_y,
        'type': 'face',
        'facing': facing
    }
