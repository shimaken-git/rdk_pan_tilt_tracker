#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from dynamixel_handler_msgs.msg import DynamixelGoal
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from rdk_pan_tilt_tracker.rdk_yolo11_pose import RdkYolo11Pose
from rdk_pan_tilt_tracker.tracking_utils import get_face_target
from rdk_pan_tilt_tracker.tracking_utils import StableTrueTrigger


class PanTiltTracker(Node):

    def __init__(self):

        super().__init__('pan_tilt_tracker')

        # ========================================================
        # Parameters
        # ========================================================

        self.declare_parameter(
            'camera_device',
            '/dev/video0'
        )

        self.declare_parameter(
            'camera_width',
            1280
        )

        self.declare_parameter(
            'camera_height',
            480
        )

        self.declare_parameter(
            'camera_fps',
            30
        )

        self.declare_parameter(
            'headless',
            False
        )

        self.declare_parameter(
            'model_path',
            '/app/pydev_demo/04_pose_sample/'
            '01_ultralytics_yolo11_pose/'
            'yolo11n_pose_bayese_640x640_nv12.bin'
        )

        self.declare_parameter(
            'confidence',
            0.25
        )

        self.declare_parameter(
            'keypoint_confidence',
            0.30
        )

        self.declare_parameter(
            'face_front_nose_offset_threshold',
            0.25
        )

        self.declare_parameter(
            'face_front_eye_height_threshold',
            0.20
        )

        self.declare_parameter('reach_out_hold_seconds', 0.8)
        self.declare_parameter('reach_out_reset_seconds', 1.0)
        self.declare_parameter('reach_out_cooldown_seconds', 5.0)

        # --------------------------------------------------------
        # Tracking
        # auto = raised hand -> hand, otherwise face
        # face = always face
        # hand = always wrist
        # --------------------------------------------------------

        self.declare_parameter(
            'tracking_mode',
            'auto'
        )

        # --------------------------------------------------------
        # Dynamixel
        # --------------------------------------------------------

        self.declare_parameter('pan_id', 4)
        self.declare_parameter('tilt_id', 6)
        self.declare_parameter('pan_goal_current_ma', 500.0)
        self.declare_parameter('tilt_goal_current_ma', 500.0)
        self.declare_parameter(
            'dynamixel_goal_topic',
            '/robot/dynamixel/command/goal'
        )

        self.declare_parameter('pan_initial', 0.0)
        self.declare_parameter('tilt_initial', 0.0)

        self.declare_parameter('pan_direction', 1.0)
        self.declare_parameter('tilt_direction', 1.0)
        self.declare_parameter('tilt_ros_direction', -1.0)

        self.declare_parameter('pan_min', -90.0)
        self.declare_parameter('pan_max', 90.0)

        self.declare_parameter('tilt_min', -45.0)
        self.declare_parameter('tilt_max', 45.0)

        self.declare_parameter('pan_kp', 0.035)
        self.declare_parameter('tilt_kp', 0.035)

        self.declare_parameter('max_pan_step', 1.0)
        self.declare_parameter('max_tilt_step', 1.0)

        self.declare_parameter('deadzone_x', 30.0)
        self.declare_parameter('deadzone_y', 30.0)

        self.declare_parameter('filter_alpha', 0.25)

        # ========================================================
        # Read parameters
        # ========================================================

        self.camera_device = self.get_parameter(
            'camera_device'
        ).value

        self.camera_width = self.get_parameter(
            'camera_width'
        ).value

        self.camera_height = self.get_parameter(
            'camera_height'
        ).value

        self.camera_fps = self.get_parameter(
            'camera_fps'
        ).value

        self.headless = self.get_parameter(
            'headless'
        ).value

        model_path = self.get_parameter(
            'model_path'
        ).value

        confidence = self.get_parameter(
            'confidence'
        ).value

        self.kpt_confidence = self.get_parameter(
            'keypoint_confidence'
        ).value

        self.face_front_nose_offset_threshold = self.get_parameter(
            'face_front_nose_offset_threshold'
        ).value

        self.face_front_eye_height_threshold = self.get_parameter(
            'face_front_eye_height_threshold'
        ).value

        reach_out_hold_seconds = self.get_parameter(
            'reach_out_hold_seconds'
        ).value

        reach_out_reset_seconds = self.get_parameter(
            'reach_out_reset_seconds'
        ).value

        reach_out_cooldown_seconds = self.get_parameter(
            'reach_out_cooldown_seconds'
        ).value

        self.tracking_mode = self.get_parameter(
            'tracking_mode'
        ).value

        self.pan_id = self.get_parameter('pan_id').value
        self.tilt_id = self.get_parameter('tilt_id').value
        self.pan_goal_current_ma = self.get_parameter(
            'pan_goal_current_ma'
        ).value
        self.tilt_goal_current_ma = self.get_parameter(
            'tilt_goal_current_ma'
        ).value
        self.dynamixel_goal_topic = self.get_parameter(
            'dynamixel_goal_topic'
        ).value

        self.pan_angle = self.get_parameter(
            'pan_initial'
        ).value

        self.tilt_angle = self.get_parameter(
            'tilt_initial'
        ).value

        self.pan_direction = self.get_parameter(
            'pan_direction'
        ).value

        self.tilt_direction = self.get_parameter(
            'tilt_direction'
        ).value

        self.tilt_ros_direction = self.get_parameter(
            'tilt_ros_direction'
        ).value

        self.pan_min = self.get_parameter('pan_min').value
        self.pan_max = self.get_parameter('pan_max').value

        self.tilt_min = self.get_parameter('tilt_min').value
        self.tilt_max = self.get_parameter('tilt_max').value

        self.pan_kp = self.get_parameter('pan_kp').value
        self.tilt_kp = self.get_parameter('tilt_kp').value

        self.max_pan_step = self.get_parameter(
            'max_pan_step'
        ).value

        self.max_tilt_step = self.get_parameter(
            'max_tilt_step'
        ).value

        self.deadzone_x = self.get_parameter(
            'deadzone_x'
        ).value

        self.deadzone_y = self.get_parameter(
            'deadzone_y'
        ).value

        self.alpha = self.get_parameter(
            'filter_alpha'
        ).value

        # ========================================================
        # BPU YOLO11 Pose
        # ========================================================

        self.get_logger().info(
            f'Loading BPU model: {model_path}'
        )

        self.pose_model = RdkYolo11Pose(
            model_path=model_path,
            score_threshold=confidence,
            bpu_cores=[0],
            priority=0
        )

        self.get_logger().info(
            'YOLO11 Pose BPU model loaded'
        )

        # ========================================================
        # Stereo USB Camera
        # ========================================================

        self.cap = cv2.VideoCapture(
            self.camera_device,
            cv2.CAP_V4L2
        )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG')
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.camera_width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.camera_height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.camera_fps
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f'Cannot open camera {self.camera_device}'
            )

        # ========================================================
        # ROS publisher
        # ========================================================

        self.position_pub = self.create_publisher(
            DynamixelGoal,
            self.dynamixel_goal_topic,
            10
        )

        self.face_detected_pub = self.create_publisher(
            Bool,
            '~/face_detected',
            10
        )

        self.face_facing_pub = self.create_publisher(
            Bool,
            '~/face_facing',
            10
        )

        self.reach_out_request_pub = self.create_publisher(
            Bool,
            '~/reach_out_request',
            10
        )

        self.camera_joint_state_pub = self.create_publisher(
            JointState,
            '~/camera_joint_state',
            10
        )

        # ========================================================
        # Filter state
        # ========================================================

        self.filtered_x = None
        self.filtered_y = None

        self.target_type = 'none'
        self.face_detected = False
        self.face_facing = False
        self.reach_out_trigger = StableTrueTrigger(
            hold_seconds=reach_out_hold_seconds,
            reset_seconds=reach_out_reset_seconds,
            cooldown_seconds=reach_out_cooldown_seconds
        )

        # ========================================================
        # Timer
        #
        # Run at the requested camera rate. Actual FPS is bounded by
        # camera capture and synchronous BPU inference time.
        # ========================================================

        self.timer = self.create_timer(
            1.0 / max(float(self.camera_fps), 1.0),
            self.update
        )

        self.get_logger().info(
            'Pan Tilt Tracker started'
        )

    # ============================================================
    # Find face / hand
    # ============================================================

    def find_target(
        self,
        keypoints_xy,
        keypoints_score
    ):

        if len(keypoints_xy) == 0:
            return None

        # --------------------------------------------------------
        # For now choose the first detected person.
        # Later we can add person tracking / stereo association.
        # --------------------------------------------------------

        person = keypoints_xy[0]
        score = keypoints_score[0]

        # COCO
        #
        # 0 nose
        # 1 left eye
        # 2 right eye
        # 3 left ear
        # 4 right ear
        #
        # 5 left shoulder
        # 6 right shoulder
        #
        # 9 left wrist
        # 10 right wrist

        # ========================================================
        # Face
        # ========================================================

        face_result = get_face_target(
            person=person,
            score=score,
            keypoint_confidence=self.kpt_confidence,
            nose_offset_threshold=(
                self.face_front_nose_offset_threshold
            ),
            eye_height_threshold=(
                self.face_front_eye_height_threshold
            )
        )

        face_target = None

        if face_result is not None:
            face_target = (
                face_result['x'],
                face_result['y'],
                'face',
                face_result['facing']
            )

        # ========================================================
        # Hand / wrist
        # ========================================================

        hand_candidates = []

        # Left wrist
        if score[9] >= self.kpt_confidence:

            hand_candidates.append(
                (
                    float(person[9][0]),
                    float(person[9][1]),
                    9
                )
            )

        # Right wrist
        if score[10] >= self.kpt_confidence:

            hand_candidates.append(
                (
                    float(person[10][0]),
                    float(person[10][1]),
                    10
                )
            )

        hand_target = None

        # ========================================================
        # AUTO mode:
        # wrist must be above corresponding shoulder
        # ========================================================

        if self.tracking_mode == 'auto':

            raised_hands = []

            for x, y, index in hand_candidates:

                if index == 9:
                    shoulder_index = 5
                else:
                    shoulder_index = 6

                if (
                    score[shoulder_index]
                    >= self.kpt_confidence
                ):

                    shoulder_y = person[
                        shoulder_index
                    ][1]

                    if y < shoulder_y:

                        raised_hands.append(
                            (x, y)
                        )

            if raised_hands:

                # Highest hand in image
                hand = min(
                    raised_hands,
                    key=lambda p: p[1]
                )

                hand_target = (
                    hand[0],
                    hand[1],
                    'hand',
                    False
                )

        elif self.tracking_mode == 'hand':

            if hand_candidates:

                hand = min(
                    hand_candidates,
                    key=lambda p: p[1]
                )

                hand_target = (
                    hand[0],
                    hand[1],
                    'hand',
                    False
                )

        # ========================================================
        # Select target
        # ========================================================

        if self.tracking_mode == 'face':

            return face_target

        if hand_target is not None:

            return hand_target

        return face_target

    # ============================================================
    # Main
    # ============================================================

    def update(self):

        ret, frame = self.cap.read()

        if not ret:

            self.get_logger().warning(
                'Failed to read camera frame'
            )

            return

        height, width = frame.shape[:2]

        # ========================================================
        # Split stereo frame
        # ========================================================

        half_width = width // 2

        left_frame = frame[:, :half_width].copy()
        right_frame = frame[:, half_width:]

        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_frame.shape[:2]

        left_center_x = left_w / 2.0
        left_center_y = left_h / 2.0

        right_center_x = right_w / 2.0
        right_center_y = right_h / 2.0

        # ========================================================
        # Left BPU inference
        # ========================================================

        (
            left_ids,
            left_scores,
            left_boxes,
            left_kpts,
            left_kpt_scores
        ) = self.pose_model.infer(
            left_frame
        )

        left_target = self.find_target(
            left_kpts,
            left_kpt_scores
        )

        # ========================================================
        # Use the left camera only to keep one BPU inference per update.
        # ========================================================

        right_target = None
        tracking_target = left_target

        # ========================================================
        # Pan / Tilt tracking
        # ========================================================

        if tracking_target is not None:

            (
                target_x,
                target_y,
                target_type,
                target_facing
            ) = tracking_target

            self.target_type = target_type
            self.face_detected = target_type == 'face'
            self.face_facing = (
                self.face_detected and target_facing
            )

            # ----------------------------------------------------
            # EMA
            # ----------------------------------------------------

            if self.filtered_x is None:

                self.filtered_x = target_x
                self.filtered_y = target_y

            else:

                self.filtered_x = (
                    self.alpha * target_x
                    +
                    (1.0 - self.alpha)
                    * self.filtered_x
                )

                self.filtered_y = (
                    self.alpha * target_y
                    +
                    (1.0 - self.alpha)
                    * self.filtered_y
                )

            # ----------------------------------------------------
            # Error
            # ----------------------------------------------------

            error_x = (
                self.filtered_x
                - left_center_x
            )

            error_y = (
                self.filtered_y
                - left_center_y
            )

            # ----------------------------------------------------
            # Dead zone
            # ----------------------------------------------------

            if abs(error_x) < self.deadzone_x:
                error_x = 0.0

            if abs(error_y) < self.deadzone_y:
                error_y = 0.0

            # ----------------------------------------------------
            # Pan
            # ----------------------------------------------------

            pan_delta = (
                self.pan_direction
                * self.pan_kp
                * error_x
            )

            pan_delta = np.clip(
                pan_delta,
                -self.max_pan_step,
                self.max_pan_step
            )

            self.pan_angle += pan_delta

            self.pan_angle = float(
                np.clip(
                    self.pan_angle,
                    self.pan_min,
                    self.pan_max
                )
            )

            # ----------------------------------------------------
            # Tilt
            # ----------------------------------------------------

            tilt_delta = (
                self.tilt_direction
                * self.tilt_kp
                * error_y
            )

            tilt_delta = np.clip(
                tilt_delta,
                -self.max_tilt_step,
                self.max_tilt_step
            )

            self.tilt_angle += tilt_delta

            self.tilt_angle = float(
                np.clip(
                    self.tilt_angle,
                    self.tilt_min,
                    self.tilt_max
                )
            )

            self.publish_position()

        else:

            self.target_type = 'none'
            self.face_detected = False
            self.face_facing = False

            # Do not keep old EMA target
            self.filtered_x = None
            self.filtered_y = None

        self.publish_face_status()
        self.publish_camera_joint_state()
        self.publish_reach_out_request()

        if not self.headless:

            # ====================================================
            # Draw
            # ====================================================

            self.draw_debug(
                left_frame,
                left_center_x,
                left_center_y,
                left_target
            )

            self.draw_debug(
                right_frame,
                right_center_x,
                right_center_y,
                right_target
            )

            stereo_display = np.hstack(
                (
                    left_frame,
                    right_frame
                )
            )

            cv2.imshow(
                'Pan Tilt Tracker',
                stereo_display
            )

            key = cv2.waitKey(1)

            if key == 27:
                rclpy.shutdown()

    # ============================================================
    # Debug display
    # ============================================================

    def draw_debug(
        self,
        frame,
        center_x,
        center_y,
        target
    ):

        cv2.drawMarker(
            frame,
            (
                int(center_x),
                int(center_y)
            ),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            30,
            2
        )

        if target is None:
            return

        target_x, target_y, target_type, target_facing = target

        if target_type == 'face':

            # Red
            color = (
                0,
                0,
                255
            )

        else:

            # Green
            color = (
                0,
                255,
                0
            )

        x = int(target_x)
        y = int(target_y)

        cv2.circle(
            frame,
            (x, y),
            8,
            color,
            -1
        )

        cv2.drawMarker(
            frame,
            (x, y),
            color,
            cv2.MARKER_CROSS,
            35,
            2
        )

        cv2.line(
            frame,
            (
                int(center_x),
                int(center_y)
            ),
            (x, y),
            color,
            2
        )

        cv2.putText(
            frame,
            target_type,
            (
                x + 10,
                y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        if target_type == 'face':
            facing_text = (
                'FRONT' if target_facing else 'NOT FRONT'
            )
            cv2.putText(
                frame,
                facing_text,
                (
                    x + 10,
                    y + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ============================================================
    # Dynamixel
    # ============================================================

    def publish_position(self):

        msg = DynamixelGoal()

        msg.id_list = [
            self.pan_id,
            self.tilt_id
        ]

        msg.position_deg = [
            float(self.pan_angle),
            float(self.tilt_angle)
        ]

        msg.current_ma = [
            float(self.pan_goal_current_ma),
            float(self.tilt_goal_current_ma)
        ]

        self.position_pub.publish(msg)

    def publish_face_status(self):
        detected_msg = Bool()
        detected_msg.data = self.face_detected
        self.face_detected_pub.publish(detected_msg)

        facing_msg = Bool()
        facing_msg.data = self.face_facing
        self.face_facing_pub.publish(facing_msg)

    def publish_reach_out_request(self):
        now_seconds = self.get_clock().now().nanoseconds / 1e9
        if not self.reach_out_trigger.update(
            self.face_facing,
            now_seconds
        ):
            return

        request_msg = Bool()
        request_msg.data = True
        self.reach_out_request_pub.publish(request_msg)
        self.get_logger().info(
            'Reach-out requested: stable frontal face detected'
        )

    def publish_camera_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['neck_joint', 'neck_pan_joint']
        msg.position = [
            float(np.deg2rad(
                self.tilt_ros_direction * self.tilt_angle
            )),
            float(np.deg2rad(self.pan_angle))
        ]
        self.camera_joint_state_pub.publish(msg)

    # ============================================================
    # Shutdown
    # ============================================================

    def destroy_node(self):

        if self.cap is not None:
            self.cap.release()

        if not self.headless:
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = PanTiltTracker()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
