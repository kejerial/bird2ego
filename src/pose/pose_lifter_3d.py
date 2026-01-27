"""PoseLifter3D: lifts COCO17 2D to 3D root-relative coordinates."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..utils.timeline import (
    JOINT_IDX,
    NUM_JOINTS,
    SENTINEL_2D,
    SENTINEL_3D,
    SENTINEL_CONF,
    PoseFrame,
)

logger = logging.getLogger(__name__)


class PoseLifter3DBase(ABC):
    """Abstract base class for 3D pose lifting."""

    @abstractmethod
    def lift(
        self,
        keypoints_2d: np.ndarray,
        conf_2d: np.ndarray,
        image_size: tuple,
    ) -> tuple:
        """Lift 2D keypoints to 3D root-relative coordinates.

        Args:
            keypoints_2d: 2D keypoints, shape (17, 2).
            conf_2d: Confidence scores, shape (17,).
            image_size: (width, height) of the image.

        Returns:
            Tuple of (coords_3d, conf_3d) where:
                coords_3d: shape (17, 3), root-relative, units="arb"
                conf_3d: shape (17,)
        """
        pass


class StubPoseLifter3D(PoseLifter3DBase):
    """Stub 3D pose lifter that generates plausible 3D poses from 2D.

    Uses simple heuristics to create root-relative 3D coordinates:
    - X: normalized x coordinate relative to image center
    - Y: normalized y coordinate relative to image center
    - Z: estimated depth based on joint type and position
    """

    def __init__(
        self,
        depth_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize stub lifter.

        Args:
            depth_scale: Scale factor for depth estimation.
            seed: Random seed for reproducibility.
        """
        self.depth_scale = depth_scale
        self.rng = np.random.default_rng(seed)

        # Typical depth offsets for each joint (relative to pelvis)
        # Positive Z = towards camera
        self._depth_priors = np.array([
            0.3,   # nose (forward)
            0.35,  # left_eye
            0.35,  # right_eye
            0.25,  # left_ear
            0.25,  # right_ear
            0.0,   # left_shoulder
            0.0,   # right_shoulder
            0.1,   # left_elbow
            -0.1,  # right_elbow
            0.15,  # left_wrist
            -0.15, # right_wrist
            0.0,   # left_hip
            0.0,   # right_hip
            0.05,  # left_knee
            -0.05, # right_knee
            0.0,   # left_ankle
            0.0,   # right_ankle
        ], dtype=np.float32)

    def lift(
        self,
        keypoints_2d: np.ndarray,
        conf_2d: np.ndarray,
        image_size: tuple,
    ) -> tuple:
        """Lift 2D to 3D using simple heuristics."""
        w, h = image_size
        coords_3d = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        conf_3d = np.zeros(NUM_JOINTS, dtype=np.float32)

        # Compute virtual root (pelvis) as midpoint of hips
        left_hip = keypoints_2d[JOINT_IDX["left_hip"]]
        right_hip = keypoints_2d[JOINT_IDX["right_hip"]]
        left_hip_valid = (
            left_hip[0] != SENTINEL_2D[0] and conf_2d[JOINT_IDX["left_hip"]] > 0
        )
        right_hip_valid = (
            right_hip[0] != SENTINEL_2D[0] and conf_2d[JOINT_IDX["right_hip"]] > 0
        )

        if left_hip_valid and right_hip_valid:
            root_2d = (left_hip + right_hip) / 2
            root_valid = True
        elif left_hip_valid:
            root_2d = left_hip
            root_valid = True
        elif right_hip_valid:
            root_2d = right_hip
            root_valid = True
        else:
            # No valid hip, use image center
            root_2d = np.array([w / 2, h / 2], dtype=np.float32)
            root_valid = False

        # Lift each joint
        for j in range(NUM_JOINTS):
            kp = keypoints_2d[j]
            conf = conf_2d[j]

            if kp[0] == SENTINEL_2D[0] or conf <= 0:
                coords_3d[j] = SENTINEL_3D
                conf_3d[j] = SENTINEL_CONF
                continue

            # Normalize 2D coords relative to root
            # X: right is positive
            # Y: down is positive
            x_rel = (kp[0] - root_2d[0]) / w * 2.0
            y_rel = (kp[1] - root_2d[1]) / h * 2.0

            # Estimate depth using prior + small noise
            z_rel = self._depth_priors[j] * self.depth_scale
            z_rel += self.rng.normal(0, 0.05) * self.depth_scale

            coords_3d[j] = [x_rel, y_rel, z_rel]

            # 3D confidence is slightly lower than 2D due to uncertainty
            conf_3d[j] = conf * 0.8

        # If root was valid, ensure it's at origin
        if root_valid:
            # Root is virtual, but hips should be near origin
            hip_mean = (coords_3d[JOINT_IDX["left_hip"]] +
                       coords_3d[JOINT_IDX["right_hip"]]) / 2
            # Shift all coords so hip mean is at origin (xy only)
            for j in range(NUM_JOINTS):
                if coords_3d[j][0] != SENTINEL_3D[0]:
                    coords_3d[j][0] -= hip_mean[0]
                    coords_3d[j][1] -= hip_mean[1]

        return coords_3d, conf_3d


class PoseLifter3D:
    """3D pose lifter with configurable backend.

    Coord frame: root-relative-camera
    Axis convention: x=right, y=down, z=forward (camera)
    Units: "arb" (scale-ambiguous)
    """

    def __init__(
        self,
        backend: str = "stub",
        min_confidence: float = 0.2,
        **kwargs,
    ):
        """Initialize PoseLifter3D.

        Args:
            backend: Backend to use ("stub" for testing).
            min_confidence: Minimum confidence threshold.
            **kwargs: Additional arguments for backend.
        """
        self.min_confidence = min_confidence

        if backend == "stub":
            self._lifter = StubPoseLifter3D(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def lift_frame(
        self,
        pose_frame: PoseFrame,
        image_size: tuple,
    ) -> PoseFrame:
        """Lift 2D pose to 3D for a single frame.

        Args:
            pose_frame: Input PoseFrame with 2D keypoints.
            image_size: (width, height) of the image.

        Returns:
            PoseFrame with 3D coordinates filled in.
        """
        keypoints_2d = np.array(pose_frame.keypoints_2d_px, dtype=np.float32)
        conf_2d = np.array(pose_frame.conf_2d, dtype=np.float32)

        coords_3d, conf_3d = self._lifter.lift(keypoints_2d, conf_2d, image_size)

        # Update pose frame with 3D data
        pose_frame.coords_3d = coords_3d.tolist()
        pose_frame.conf_3d = conf_3d.tolist()

        return pose_frame

    def lift_video(
        self,
        pose_frames: List[PoseFrame],
        image_size: tuple,
    ) -> List[PoseFrame]:
        """Lift 2D poses to 3D for all frames.

        Args:
            pose_frames: List of PoseFrame objects with 2D keypoints.
            image_size: (width, height) of the images.

        Returns:
            List of PoseFrame objects with 3D coordinates.
        """
        return [self.lift_frame(pf, image_size) for pf in pose_frames]
