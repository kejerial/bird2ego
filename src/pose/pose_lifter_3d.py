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
        world_coords_3d: Optional[np.ndarray] = None,
    ) -> tuple:
        """Lift 2D keypoints to 3D root-relative coordinates.

        Args:
            keypoints_2d: 2D keypoints, shape (17, 2).
            conf_2d: Confidence scores, shape (17,).
            image_size: (width, height) of the image.
            world_coords_3d: Optional metric 3D joints, shape (17, 3), supplied
                by the 2D estimator. Backends that derive 3D from 2D ignore it.

        Returns:
            Tuple of (coords_3d, conf_3d) where:
                coords_3d: shape (17, 3), root-relative
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
        world_coords_3d: Optional[np.ndarray] = None,
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


class MediaPipeWorldLifter(PoseLifter3DBase):
    """Use the metric 3D joints MediaPipe already produces.

    MediaPipe Pose Landmarker returns `pose_world_landmarks` in metres, with
    the origin at the hip midpoint and the image axis convention: x right,
    y down, z towards the camera. That matches the pipeline 3D contract, so
    this backend re-roots the joints on the hip midpoint and passes them
    through. It never invents data: a frame without world landmarks gets
    sentinels.
    """

    def __init__(self, min_confidence: float = 0.2):
        """Initialize the lifter.

        Args:
            min_confidence: Minimum 2D confidence for a joint to be kept.
        """
        self.min_confidence = min_confidence

    def lift(
        self,
        keypoints_2d: np.ndarray,
        conf_2d: np.ndarray,
        image_size: tuple,
        world_coords_3d: Optional[np.ndarray] = None,
    ) -> tuple:
        """Pass MediaPipe world landmarks through, re-rooted on the hips."""
        coords_3d = np.full((NUM_JOINTS, 3), SENTINEL_3D, dtype=np.float32)
        conf_3d = np.full(NUM_JOINTS, SENTINEL_CONF, dtype=np.float32)

        if world_coords_3d is None:
            return coords_3d, conf_3d

        world = np.asarray(world_coords_3d, dtype=np.float32)
        valid = (conf_2d > self.min_confidence) & (world[:, 0] != SENTINEL_3D[0])
        if not np.any(valid):
            return coords_3d, conf_3d

        # Re-root on the hip midpoint. MediaPipe already does this, but a
        # missing hip landmark would otherwise shift the whole skeleton.
        left_hip = JOINT_IDX["left_hip"]
        right_hip = JOINT_IDX["right_hip"]
        if valid[left_hip] and valid[right_hip]:
            root = (world[left_hip] + world[right_hip]) / 2.0
        else:
            root = world[valid].mean(axis=0)

        coords_3d[valid] = world[valid] - root
        conf_3d[valid] = conf_2d[valid]

        return coords_3d, conf_3d


class PoseLifter3D:
    """3D pose lifter with configurable backend.

    Coord frame: root-relative-camera
    Axis convention: x=right, y=down, z=forward (camera)
    Units: metres for the "mediapipe_world" backend, "arb" (scale-ambiguous)
    for the "stub" backend.
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

        self.backend = backend

        if backend == "stub":
            self._lifter = StubPoseLifter3D(**kwargs)
        elif backend == "mediapipe_world":
            self._lifter = MediaPipeWorldLifter(
                min_confidence=min_confidence, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'stub' or 'mediapipe_world'"
            )

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
        world_coords_3d = (
            np.array(pose_frame.world_coords_3d, dtype=np.float32)
            if pose_frame.world_coords_3d is not None
            else None
        )

        coords_3d, conf_3d = self._lifter.lift(
            keypoints_2d, conf_2d, image_size, world_coords_3d
        )

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
