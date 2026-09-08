"""KinematicsProcessor: computes joint angles and velocities.

Optional in v1, includes placeholders for future implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import JOINT_IDX, NUM_JOINTS, SENTINEL_3D, PersonPose

logger = logging.getLogger(__name__)


# Define kinematic chains (parent -> child relationships)
KINEMATIC_CHAINS: Dict[str, List[Tuple[str, str]]] = {
    "spine": [
        ("left_hip", "left_shoulder"),
        ("right_hip", "right_shoulder"),
    ],
    "left_arm": [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
    ],
    "right_arm": [
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ],
    "left_leg": [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
    ],
    "right_leg": [
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ],
}

# Joint angle definitions (joint, parent_joint, child_joint)
JOINT_ANGLES: Dict[str, Tuple[str, str, str]] = {
    "left_elbow": ("left_elbow", "left_shoulder", "left_wrist"),
    "right_elbow": ("right_elbow", "right_shoulder", "right_wrist"),
    "left_knee": ("left_knee", "left_hip", "left_ankle"),
    "right_knee": ("right_knee", "right_hip", "right_ankle"),
    "left_shoulder": ("left_shoulder", "right_shoulder", "left_elbow"),
    "right_shoulder": ("right_shoulder", "left_shoulder", "right_elbow"),
    "left_hip": ("left_hip", "right_hip", "left_knee"),
    "right_hip": ("right_hip", "left_hip", "right_knee"),
}


@dataclass
class KinematicsFrame:
    """Kinematics data for a single frame."""

    frame_idx: int
    joint_angles: Dict[str, float] = field(default_factory=dict)
    joint_velocities_3d: Optional[np.ndarray] = None  # (17, 3)
    joint_speeds: Optional[np.ndarray] = None  # (17,)
    bone_lengths: Dict[str, float] = field(default_factory=dict)


@dataclass
class KinematicsData:
    """Time-series kinematics data."""

    frames: List[KinematicsFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)


class KinematicsProcessor:
    """Computes joint angles, velocities, and other kinematic features.

    This is optional in v1 and includes placeholders for future implementation.
    """

    def __init__(
        self,
        fps: float = 30.0,
        velocity_smoothing: int = 3,
        min_confidence: float = 0.2,
    ):
        """Initialize KinematicsProcessor.

        Args:
            fps: Frame rate for velocity computation.
            velocity_smoothing: Window size for velocity smoothing.
            min_confidence: Minimum confidence for valid joints.
        """
        self.fps = fps
        self.velocity_smoothing = velocity_smoothing
        self.min_confidence = min_confidence

    def compute(self, person_pose: PersonPose) -> KinematicsData:
        """Compute kinematics for all frames.

        Args:
            person_pose: PersonPose object with pose data.

        Returns:
            KinematicsData with computed features.
        """
        T = person_pose.num_frames
        if T == 0:
            return KinematicsData()

        # Get arrays
        coords_3d = person_pose.get_coords_3d_array()  # (T, 17, 3)
        conf_3d = person_pose.get_conf_3d_array()  # (T, 17)

        # Compute velocities
        velocities = self._compute_velocities(coords_3d, conf_3d)

        # Compute for each frame
        frames = []
        for t in range(T):
            kf = KinematicsFrame(frame_idx=t)

            # Joint angles
            kf.joint_angles = self._compute_joint_angles(coords_3d[t], conf_3d[t])

            # Velocities
            if velocities is not None:
                kf.joint_velocities_3d = velocities[t]
                kf.joint_speeds = np.linalg.norm(velocities[t], axis=1)

            # Bone lengths
            kf.bone_lengths = self._compute_bone_lengths(coords_3d[t], conf_3d[t])

            frames.append(kf)

        return KinematicsData(frames=frames)

    def _compute_velocities(
        self,
        coords_3d: np.ndarray,
        conf_3d: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute joint velocities using finite differences.

        Args:
            coords_3d: 3D coordinates, shape (T, 17, 3).
            conf_3d: Confidence scores, shape (T, 17).

        Returns:
            Velocities array, shape (T, 17, 3), or None if not enough data.
        """
        T = coords_3d.shape[0]
        if T < 2:
            return None

        # Simple finite difference
        dt = 1.0 / self.fps
        velocities = np.zeros_like(coords_3d)

        for t in range(T):
            if t == 0:
                # Forward difference
                velocities[t] = (coords_3d[t + 1] - coords_3d[t]) / dt
            elif t == T - 1:
                # Backward difference
                velocities[t] = (coords_3d[t] - coords_3d[t - 1]) / dt
            else:
                # Central difference
                velocities[t] = (coords_3d[t + 1] - coords_3d[t - 1]) / (2 * dt)

        # Zero out velocities for invalid joints
        sentinel = np.array(SENTINEL_3D)
        for t in range(T):
            for j in range(NUM_JOINTS):
                if np.allclose(coords_3d[t, j], sentinel) or conf_3d[t, j] < self.min_confidence:
                    velocities[t, j] = 0.0

        # Apply smoothing
        if self.velocity_smoothing > 1:
            kernel = np.ones(self.velocity_smoothing) / self.velocity_smoothing
            for j in range(NUM_JOINTS):
                for d in range(3):
                    velocities[:, j, d] = np.convolve(velocities[:, j, d], kernel, mode="same")

        return velocities

    def _compute_joint_angles(
        self,
        coords_3d: np.ndarray,
        conf_3d: np.ndarray,
    ) -> Dict[str, float]:
        """Compute joint angles for a single frame.

        Args:
            coords_3d: 3D coordinates, shape (17, 3).
            conf_3d: Confidence scores, shape (17,).

        Returns:
            Dictionary of joint name to angle in degrees.
        """
        angles = {}
        sentinel = np.array(SENTINEL_3D)

        for angle_name, (joint, parent, child) in JOINT_ANGLES.items():
            j_idx = JOINT_IDX[joint]
            p_idx = JOINT_IDX[parent]
            c_idx = JOINT_IDX[child]

            # Check validity
            if (
                np.allclose(coords_3d[j_idx], sentinel)
                or np.allclose(coords_3d[p_idx], sentinel)
                or np.allclose(coords_3d[c_idx], sentinel)
            ):
                continue

            if (
                conf_3d[j_idx] < self.min_confidence
                or conf_3d[p_idx] < self.min_confidence
                or conf_3d[c_idx] < self.min_confidence
            ):
                continue

            # Compute vectors
            v1 = coords_3d[p_idx] - coords_3d[j_idx]
            v2 = coords_3d[c_idx] - coords_3d[j_idx]

            # Compute angle
            angle = self._angle_between(v1, v2)
            angles[angle_name] = float(np.degrees(angle))

        return angles

    def _compute_bone_lengths(
        self,
        coords_3d: np.ndarray,
        conf_3d: np.ndarray,
    ) -> Dict[str, float]:
        """Compute bone lengths for a single frame.

        Args:
            coords_3d: 3D coordinates, shape (17, 3).
            conf_3d: Confidence scores, shape (17,).

        Returns:
            Dictionary of bone name to length.
        """
        lengths = {}
        sentinel = np.array(SENTINEL_3D)

        for chain_name, bones in KINEMATIC_CHAINS.items():
            for parent_name, child_name in bones:
                p_idx = JOINT_IDX[parent_name]
                c_idx = JOINT_IDX[child_name]

                # Check validity
                if np.allclose(coords_3d[p_idx], sentinel) or np.allclose(
                    coords_3d[c_idx], sentinel
                ):
                    continue

                if conf_3d[p_idx] < self.min_confidence or conf_3d[c_idx] < self.min_confidence:
                    continue

                bone_name = f"{parent_name}_{child_name}"
                length = np.linalg.norm(coords_3d[c_idx] - coords_3d[p_idx])
                lengths[bone_name] = float(length)

        return lengths

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors in radians."""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def get_hand_positions(
        self,
        person_pose: PersonPose,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get left and right hand (wrist) positions over time.

        Args:
            person_pose: PersonPose object.

        Returns:
            Tuple of (left_wrist_positions, right_wrist_positions),
            each shape (T, 2) for 2D or (T, 3) for 3D.
        """
        kp_2d = person_pose.get_keypoints_2d_array()  # (T, 17, 2)
        left_wrist = kp_2d[:, JOINT_IDX["left_wrist"]]
        right_wrist = kp_2d[:, JOINT_IDX["right_wrist"]]
        return left_wrist, right_wrist

    def get_hand_velocities(
        self,
        person_pose: PersonPose,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get left and right hand velocities over time.

        Args:
            person_pose: PersonPose object.

        Returns:
            Tuple of (left_velocity, right_velocity), each shape (T,).
        """
        kinematics = self.compute(person_pose)
        if kinematics.num_frames == 0:
            return np.array([]), np.array([])

        left_speeds = []
        right_speeds = []

        for kf in kinematics.frames:
            if kf.joint_speeds is not None:
                left_speeds.append(kf.joint_speeds[JOINT_IDX["left_wrist"]])
                right_speeds.append(kf.joint_speeds[JOINT_IDX["right_wrist"]])
            else:
                left_speeds.append(0.0)
                right_speeds.append(0.0)

        return np.array(left_speeds), np.array(right_speeds)
