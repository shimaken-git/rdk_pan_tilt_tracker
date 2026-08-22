
import os
import sys
from argparse import Namespace

import numpy as np


# ================================================================
# D-Robotics YOLO11 Pose sample
# ================================================================

RDK_PYDEV_ROOT = "/app/pydev_demo"

RDK_POSE_ROOT = os.path.join(
    RDK_PYDEV_ROOT,
    "04_pose_sample"
)

RDK_POSE_SAMPLE = os.path.join(
    RDK_POSE_ROOT,
    "01_ultralytics_yolo11_pose"
)

# ------------------------------------------------
# Important:
#
# /app/pydev_demo/utils
# を import できるようにする
# ------------------------------------------------

if RDK_PYDEV_ROOT not in sys.path:
    sys.path.insert(0, RDK_PYDEV_ROOT)

# YOLO11 Pose sample module
if RDK_POSE_SAMPLE not in sys.path:
    sys.path.insert(0, RDK_POSE_SAMPLE)


try:
    from ultralytics_yolo11_pose import YoloV11_Pose

except ImportError as e:
    raise ImportError(
        "Failed to import RDK X5 YOLO11 Pose sample.\n"
        f"Expected directory: {RDK_POSE_SAMPLE}\n"
        f"Original error: {e}"
    )


class RdkYolo11Pose:

    def __init__(
        self,
        model_path,
        score_threshold=0.25,
        bpu_cores=None,
        priority=0
    ):

        if bpu_cores is None:
            bpu_cores = [0]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"BPU model not found: {model_path}"
            )

        opt = Namespace()

        opt.model_path = model_path
        opt.score_thres = float(score_threshold)

        self.model = YoloV11_Pose(opt)

        self.model.set_scheduling_params(
            priority=int(priority),
            bpu_cores=bpu_cores
        )

    def infer(self, frame):

        height, width = frame.shape[:2]

        input_tensor = self.model.pre_process(frame)

        outputs = self.model.forward(input_tensor)

        (
            ids,
            scores,
            boxes,
            keypoints_xy,
            keypoints_score
        ) = self.model.post_process(
            outputs,
            height,
            width
        )

        return (
            ids,
            scores,
            boxes,
            keypoints_xy,
            keypoints_score
        )
