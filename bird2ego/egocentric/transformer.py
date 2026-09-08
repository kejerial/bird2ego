"""Egocentric coordinate transformation.

Transforms 3D coordinates from world/camera frame to egocentric (first-person) frame.
The egocentric frame is centered at the estimated eye position, looking forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import JOINT_IDX


@dataclass
class EgocentricFrame:
    """Egocentric representation for a single frame."""

    frame_idx: int
    timestamp: float

    # Head pose in world frame
    head_position: np.ndarray  # (3,) estimated eye/head position
    head_forward: np.ndarray  # (3,) forward direction (gaze)
    head_up: np.ndarray  # (3,) up direction
    head_right: np.ndarray  # (3,) right direction

    # Hands in egocentric frame (relative to head, looking forward)
    # Coordinate system: x=right, y=down, z=forward
    left_hand_ego: Optional[np.ndarray] = None  # (21, 3) or None if not visible
    right_hand_ego: Optional[np.ndarray] = None  # (21, 3) or None if not visible

    # Hand states
    left_hand_pinching: bool = False
    left_pinch_distance: float = 0.0
    right_hand_pinching: bool = False
    right_pinch_distance: float = 0.0

    # Arm keypoints in egocentric frame (shoulder, elbow, wrist)
    left_arm_ego: Optional[np.ndarray] = None  # (3, 3) shoulder, elbow, wrist
    right_arm_ego: Optional[np.ndarray] = None  # (3, 3) shoulder, elbow, wrist

    # Objects in egocentric frame
    objects_ego: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # obj_id -> (3,) center position

    # Quality metrics
    head_confidence: float = 0.0
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0


@dataclass
class EgocentricTimeSeries:
    """Time-series of egocentric data."""

    frames: List[EgocentricFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get_left_hand_trajectory(self) -> np.ndarray:
        """Get (T, 21, 3) array of left hand positions in ego frame."""
        trajectories = []
        for f in self.frames:
            if f.left_hand_ego is not None:
                trajectories.append(f.left_hand_ego)
            else:
                trajectories.append(np.zeros((21, 3)))
        return np.array(trajectories)

    def get_right_hand_trajectory(self) -> np.ndarray:
        """Get (T, 21, 3) array of right hand positions in ego frame."""
        trajectories = []
        for f in self.frames:
            if f.right_hand_ego is not None:
                trajectories.append(f.right_hand_ego)
            else:
                trajectories.append(np.zeros((21, 3)))
        return np.array(trajectories)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "num_frames": self.num_frames,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "timestamp": f.timestamp,
                    "head_position": f.head_position.tolist(),
                    "head_forward": f.head_forward.tolist(),
                    "left_hand_ego": f.left_hand_ego.tolist()
                    if f.left_hand_ego is not None
                    else None,
                    "right_hand_ego": f.right_hand_ego.tolist()
                    if f.right_hand_ego is not None
                    else None,
                    "left_hand_pinching": f.left_hand_pinching,
                    "left_pinch_distance": f.left_pinch_distance,
                    "right_hand_pinching": f.right_hand_pinching,
                    "right_pinch_distance": f.right_pinch_distance,
                    "left_arm_ego": f.left_arm_ego.tolist() if f.left_arm_ego is not None else None,
                    "right_arm_ego": f.right_arm_ego.tolist()
                    if f.right_arm_ego is not None
                    else None,
                    "objects_ego": {k: v.tolist() for k, v in f.objects_ego.items()},
                    "head_confidence": f.head_confidence,
                    "left_hand_confidence": f.left_hand_confidence,
                    "right_hand_confidence": f.right_hand_confidence,
                }
                for f in self.frames
            ],
        }


class EgocentricTransformer:
    """Transform third-person coordinates to egocentric frame.

    The egocentric frame is:
    - Origin: Estimated eye position (between eyes, slightly in front of head)
    - Z-axis: Forward (gaze direction, estimated from head orientation)
    - Y-axis: Down
    - X-axis: Right

    This matches typical egocentric camera conventions for robotics/VLA.
    """

    def __init__(
        self,
        eye_offset_forward: float = 0.1,  # How far in front of head center the "eye" is
        default_gaze_down_angle: float = 30.0,  # Default angle looking down at hands (degrees)
        pinch_threshold: float = 0.05,  # Normalized distance for pinch detection
    ):
        """Initialize transformer.

        Args:
            eye_offset_forward: Offset from head center to eye position (normalized).
            default_gaze_down_angle: Default angle (degrees) the head looks down.
            pinch_threshold: Threshold for pinch detection (normalized 3D distance).
        """
        self.eye_offset_forward = eye_offset_forward
        self.default_gaze_down_angle = np.radians(default_gaze_down_angle)
        self.pinch_threshold = pinch_threshold

    def estimate_head_pose(
        self,
        pose_3d: np.ndarray,  # (17, 3) COCO17 3D keypoints
        pose_conf: np.ndarray,  # (17,) confidences
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Estimate head position and orientation from body pose.

        Returns:
            (head_pos, forward, up, right, confidence) tuple
        """
        # Get relevant keypoints
        nose_idx = JOINT_IDX["nose"]
        left_eye_idx = JOINT_IDX["left_eye"]
        right_eye_idx = JOINT_IDX["right_eye"]
        left_ear_idx = JOINT_IDX["left_ear"]
        right_ear_idx = JOINT_IDX["right_ear"]
        left_shoulder_idx = JOINT_IDX["left_shoulder"]
        right_shoulder_idx = JOINT_IDX["right_shoulder"]

        # Check which keypoints are valid
        valid_nose = pose_conf[nose_idx] > 0.3
        valid_left_eye = pose_conf[left_eye_idx] > 0.3
        valid_right_eye = pose_conf[right_eye_idx] > 0.3
        valid_left_ear = pose_conf[left_ear_idx] > 0.3
        valid_right_ear = pose_conf[right_ear_idx] > 0.3
        valid_left_shoulder = pose_conf[left_shoulder_idx] > 0.3
        valid_right_shoulder = pose_conf[right_shoulder_idx] > 0.3

        confidence = 0.0

        # Estimate head center (between ears or eyes, or at nose)
        if valid_left_eye and valid_right_eye:
            head_center = (pose_3d[left_eye_idx] + pose_3d[right_eye_idx]) / 2
            confidence = (pose_conf[left_eye_idx] + pose_conf[right_eye_idx]) / 2
        elif valid_left_ear and valid_right_ear:
            head_center = (pose_3d[left_ear_idx] + pose_3d[right_ear_idx]) / 2
            confidence = (pose_conf[left_ear_idx] + pose_conf[right_ear_idx]) / 2
        elif valid_nose:
            head_center = pose_3d[nose_idx].copy()
            confidence = pose_conf[nose_idx]
        else:
            # Fallback: estimate from shoulders
            if valid_left_shoulder and valid_right_shoulder:
                shoulder_center = (pose_3d[left_shoulder_idx] + pose_3d[right_shoulder_idx]) / 2
                # Head is above shoulders (in Y, which is typically up in 3D)
                head_center = shoulder_center + np.array([0, -0.25, 0])  # Assuming Y is up
                confidence = 0.3
            else:
                return (
                    np.zeros(3),
                    np.array([0, 0, 1]),
                    np.array([0, -1, 0]),
                    np.array([1, 0, 0]),
                    0.0,
                )

        # Estimate forward direction (from ears or shoulders)
        if valid_left_ear and valid_right_ear:
            # Right vector from ears
            right = pose_3d[right_ear_idx] - pose_3d[left_ear_idx]
            right = right / (np.linalg.norm(right) + 1e-8)
        elif valid_left_shoulder and valid_right_shoulder:
            right = pose_3d[right_shoulder_idx] - pose_3d[left_shoulder_idx]
            right = right / (np.linalg.norm(right) + 1e-8)
        else:
            right = np.array([1, 0, 0])

        # Estimate up vector (from shoulders to head)
        if valid_left_shoulder and valid_right_shoulder:
            shoulder_center = (pose_3d[left_shoulder_idx] + pose_3d[right_shoulder_idx]) / 2
            up = head_center - shoulder_center
            up = up / (np.linalg.norm(up) + 1e-8)
        else:
            up = np.array([0, -1, 0])  # Default: Y is up (negative Y in image coords)

        # Forward is cross product of up and right (looking forward)
        forward = np.cross(up, right)
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Re-orthogonalize
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Apply default gaze angle (looking slightly down)
        cos_angle = np.cos(self.default_gaze_down_angle)
        sin_angle = np.sin(self.default_gaze_down_angle)
        # Rotate forward around right axis to look down
        forward_rotated = forward * cos_angle - up * sin_angle
        up_rotated = forward * sin_angle + up * cos_angle

        # Move eye position slightly forward
        eye_position = head_center + forward_rotated * self.eye_offset_forward

        return (eye_position, forward_rotated, up_rotated, right, confidence)

    def transform_to_egocentric(
        self,
        points_3d: np.ndarray,  # (N, 3) points in world frame
        head_pos: np.ndarray,  # (3,) eye position
        forward: np.ndarray,  # (3,) forward direction
        up: np.ndarray,  # (3,) up direction
        right: np.ndarray,  # (3,) right direction
    ) -> np.ndarray:
        """Transform points from world frame to egocentric frame.

        Egocentric frame: x=right, y=down, z=forward

        Args:
            points_3d: Points in world/camera frame
            head_pos: Eye position
            forward, up, right: Head orientation axes

        Returns:
            Points in egocentric frame
        """
        # Translate to head-centered
        centered = points_3d - head_pos

        # Build rotation matrix (world to ego)
        # Ego axes: x=right, y=down (negative up), z=forward
        R = np.stack([right, -up, forward], axis=0)  # (3, 3)

        # Rotate to egocentric frame
        ego_points = centered @ R.T

        return ego_points

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        pose_3d: Optional[np.ndarray],  # (17, 3) body pose
        pose_conf: Optional[np.ndarray],  # (17,) confidences
        left_hand_3d: Optional[np.ndarray],  # (21, 3) left hand landmarks
        left_hand_conf: float,
        right_hand_3d: Optional[np.ndarray],  # (21, 3) right hand landmarks
        right_hand_conf: float,
        objects: Optional[Dict[int, np.ndarray]] = None,  # obj_id -> (3,) center
    ) -> EgocentricFrame:
        """Process a single frame and return egocentric representation.

        Args:
            frame_idx: Frame index
            timestamp: Timestamp in seconds
            pose_3d: Body pose 3D keypoints (COCO17)
            pose_conf: Body pose confidences
            left_hand_3d: Left hand 21 landmarks in 3D
            left_hand_conf: Overall left hand confidence
            right_hand_3d: Right hand 21 landmarks in 3D
            right_hand_conf: Overall right hand confidence
            objects: Optional dict of object centers in 3D

        Returns:
            EgocentricFrame with transformed coordinates
        """
        # Default values
        head_pos = np.zeros(3)
        forward = np.array([0, 0, 1])
        up = np.array([0, -1, 0])
        right = np.array([1, 0, 0])
        head_conf = 0.0

        # Estimate head pose from body
        if pose_3d is not None and pose_conf is not None:
            head_pos, forward, up, right, head_conf = self.estimate_head_pose(pose_3d, pose_conf)

        # Transform hands to egocentric frame
        left_hand_ego = None
        right_hand_ego = None
        left_arm_ego = None
        right_arm_ego = None
        left_pinching = False
        left_pinch_dist = 0.0
        right_pinching = False
        right_pinch_dist = 0.0

        if left_hand_3d is not None and left_hand_conf > 0.3:
            left_hand_ego = self.transform_to_egocentric(left_hand_3d, head_pos, forward, up, right)
            # Detect pinch (thumb tip to index tip distance)
            left_pinch_dist = float(np.linalg.norm(left_hand_3d[4] - left_hand_3d[8]))
            left_pinching = bool(left_pinch_dist < self.pinch_threshold)

        if right_hand_3d is not None and right_hand_conf > 0.3:
            right_hand_ego = self.transform_to_egocentric(
                right_hand_3d, head_pos, forward, up, right
            )
            right_pinch_dist = float(np.linalg.norm(right_hand_3d[4] - right_hand_3d[8]))
            right_pinching = bool(right_pinch_dist < self.pinch_threshold)

        # Transform arm keypoints (shoulder, elbow, wrist)
        if pose_3d is not None and pose_conf is not None:
            # Left arm
            l_shoulder_idx = JOINT_IDX["left_shoulder"]
            l_elbow_idx = JOINT_IDX["left_elbow"]
            l_wrist_idx = JOINT_IDX["left_wrist"]
            if (
                pose_conf[l_shoulder_idx] > 0.3
                and pose_conf[l_elbow_idx] > 0.3
                and pose_conf[l_wrist_idx] > 0.3
            ):
                left_arm = np.stack(
                    [pose_3d[l_shoulder_idx], pose_3d[l_elbow_idx], pose_3d[l_wrist_idx]]
                )
                left_arm_ego = self.transform_to_egocentric(left_arm, head_pos, forward, up, right)

            # Right arm
            r_shoulder_idx = JOINT_IDX["right_shoulder"]
            r_elbow_idx = JOINT_IDX["right_elbow"]
            r_wrist_idx = JOINT_IDX["right_wrist"]
            if (
                pose_conf[r_shoulder_idx] > 0.3
                and pose_conf[r_elbow_idx] > 0.3
                and pose_conf[r_wrist_idx] > 0.3
            ):
                right_arm = np.stack(
                    [pose_3d[r_shoulder_idx], pose_3d[r_elbow_idx], pose_3d[r_wrist_idx]]
                )
                right_arm_ego = self.transform_to_egocentric(
                    right_arm, head_pos, forward, up, right
                )

        # Transform objects
        objects_ego = {}
        if objects:
            for obj_id, obj_center in objects.items():
                objects_ego[obj_id] = self.transform_to_egocentric(
                    obj_center.reshape(1, 3), head_pos, forward, up, right
                ).flatten()

        return EgocentricFrame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            head_position=head_pos,
            head_forward=forward,
            head_up=up,
            head_right=right,
            left_hand_ego=left_hand_ego,
            right_hand_ego=right_hand_ego,
            left_hand_pinching=left_pinching,
            left_pinch_distance=left_pinch_dist,
            right_hand_pinching=right_pinching,
            right_pinch_distance=right_pinch_dist,
            left_arm_ego=left_arm_ego,
            right_arm_ego=right_arm_ego,
            objects_ego=objects_ego,
            head_confidence=float(head_conf),
            left_hand_confidence=left_hand_conf,
            right_hand_confidence=right_hand_conf,
        )
