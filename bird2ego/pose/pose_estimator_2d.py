"""PoseEstimator2D: returns COCO17 2D joints per frame."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..utils.timeline import (
    NUM_JOINTS,
    SENTINEL_2D,
    SENTINEL_BBOX,
    SENTINEL_CONF,
    PoseFrame,
    create_empty_pose_frame,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection2D:
    """A 2D pose detection for a single person."""

    bbox_xyxy: List[float]  # [x1, y1, x2, y2]
    keypoints_2d: np.ndarray  # (17, 2)
    conf_2d: np.ndarray  # (17,)
    person_conf: float  # overall confidence
    # Metric 3D joints in metres, root at the hip midpoint, when the backend
    # provides them. MediaPipe fills this from pose_world_landmarks.
    world_keypoints_3d: Optional[np.ndarray] = None  # (17, 3) or None


class PoseEstimator2DBase(ABC):
    """Abstract base class for 2D pose estimation."""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> List[Detection2D]:
        """Estimate 2D poses in a single frame.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            List of Detection2D objects, one per detected person.
        """
        pass

    @abstractmethod
    def estimate_batch(self, frames: List[np.ndarray]) -> List[List[Detection2D]]:
        """Estimate 2D poses in a batch of frames.

        Args:
            frames: List of input frames.

        Returns:
            List of detection lists, one per frame.
        """
        pass


class StubPoseEstimator2D(PoseEstimator2DBase):
    """Stub 2D pose estimator that returns plausible random poses.

    Used for testing the pipeline without heavy ML dependencies.
    """

    def __init__(
        self,
        detection_prob: float = 0.95,
        joint_visibility_prob: float = 0.85,
        seed: Optional[int] = None,
    ):
        """Initialize stub estimator.

        Args:
            detection_prob: Probability of detecting a person in each frame.
            joint_visibility_prob: Probability that each joint is visible.
            seed: Random seed for reproducibility.
        """
        self.detection_prob = detection_prob
        self.joint_visibility_prob = joint_visibility_prob
        self.rng = np.random.default_rng(seed)
        self._prev_keypoints: Optional[np.ndarray] = None

    def estimate(self, frame: np.ndarray) -> List[Detection2D]:
        """Generate stub 2D pose detection."""
        h, w = frame.shape[:2]

        # Probabilistically detect a person
        if self.rng.random() > self.detection_prob:
            return []

        # Generate plausible bbox (person in center-ish area)
        cx = w * (0.3 + 0.4 * self.rng.random())
        cy = h * (0.3 + 0.4 * self.rng.random())
        bw = w * (0.2 + 0.3 * self.rng.random())
        bh = h * (0.4 + 0.4 * self.rng.random())

        x1 = max(0, cx - bw / 2)
        y1 = max(0, cy - bh / 2)
        x2 = min(w, cx + bw / 2)
        y2 = min(h, cy + bh / 2)
        bbox = [x1, y1, x2, y2]

        # Generate keypoints within bbox with anatomical structure
        keypoints = self._generate_anatomical_keypoints(x1, y1, x2, y2)

        # Add temporal smoothing if we have previous keypoints
        if self._prev_keypoints is not None:
            alpha = 0.7
            keypoints = alpha * keypoints + (1 - alpha) * self._prev_keypoints
        self._prev_keypoints = keypoints.copy()

        # Generate visibility and confidence
        visible = self.rng.random(NUM_JOINTS) < self.joint_visibility_prob
        conf = np.where(visible, 0.5 + 0.5 * self.rng.random(NUM_JOINTS), 0.0)

        # Mark invisible joints with sentinel
        keypoints_out = keypoints.copy()
        for j in range(NUM_JOINTS):
            if not visible[j]:
                keypoints_out[j] = SENTINEL_2D

        person_conf = float(np.mean(conf[conf > 0])) if np.any(conf > 0) else 0.0

        return [
            Detection2D(
                bbox_xyxy=bbox,
                keypoints_2d=keypoints_out,
                conf_2d=conf,
                person_conf=person_conf,
            )
        ]

    def estimate_batch(self, frames: List[np.ndarray]) -> List[List[Detection2D]]:
        """Estimate poses for a batch of frames."""
        return [self.estimate(f) for f in frames]

    def _generate_anatomical_keypoints(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> np.ndarray:
        """Generate anatomically plausible keypoints within bbox."""
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2

        # Approximate body proportions (head at top, feet at bottom)
        keypoints = np.zeros((NUM_JOINTS, 2), dtype=np.float32)

        # Head region (top 15%)
        head_y = y1 + bh * 0.07
        keypoints[0] = [cx, head_y]  # nose
        keypoints[1] = [cx - bw * 0.05, head_y - bh * 0.02]  # left_eye
        keypoints[2] = [cx + bw * 0.05, head_y - bh * 0.02]  # right_eye
        keypoints[3] = [cx - bw * 0.08, head_y]  # left_ear
        keypoints[4] = [cx + bw * 0.08, head_y]  # right_ear

        # Shoulders (20% from top)
        shoulder_y = y1 + bh * 0.20
        keypoints[5] = [cx - bw * 0.20, shoulder_y]  # left_shoulder
        keypoints[6] = [cx + bw * 0.20, shoulder_y]  # right_shoulder

        # Elbows (35% from top)
        elbow_y = y1 + bh * 0.35
        keypoints[7] = [cx - bw * 0.25, elbow_y]  # left_elbow
        keypoints[8] = [cx + bw * 0.25, elbow_y]  # right_elbow

        # Wrists (50% from top)
        wrist_y = y1 + bh * 0.50
        keypoints[9] = [cx - bw * 0.30, wrist_y]  # left_wrist
        keypoints[10] = [cx + bw * 0.30, wrist_y]  # right_wrist

        # Hips (55% from top)
        hip_y = y1 + bh * 0.55
        keypoints[11] = [cx - bw * 0.12, hip_y]  # left_hip
        keypoints[12] = [cx + bw * 0.12, hip_y]  # right_hip

        # Knees (75% from top)
        knee_y = y1 + bh * 0.75
        keypoints[13] = [cx - bw * 0.12, knee_y]  # left_knee
        keypoints[14] = [cx + bw * 0.12, knee_y]  # right_knee

        # Ankles (95% from top)
        ankle_y = y1 + bh * 0.95
        keypoints[15] = [cx - bw * 0.12, ankle_y]  # left_ankle
        keypoints[16] = [cx + bw * 0.12, ankle_y]  # right_ankle

        # Add small random noise
        noise = self.rng.normal(0, bw * 0.02, keypoints.shape)
        keypoints += noise.astype(np.float32)

        return keypoints


class PoseEstimator2D:
    """2D pose estimator with configurable backend.

    Handles single-person selection from multiple detections.
    """

    def __init__(
        self,
        backend: str = "stub",
        min_confidence: float = 0.2,
        **kwargs,
    ):
        """Initialize PoseEstimator2D.

        Args:
            backend: Backend to use ("stub" for testing).
            min_confidence: Minimum confidence threshold.
            **kwargs: Additional arguments for backend.
        """
        self.min_confidence = min_confidence

        if backend == "stub":
            self._estimator = StubPoseEstimator2D(**kwargs)
        elif backend == "mediapipe":
            from .mediapipe_estimator import MediaPipePoseEstimator

            self._estimator = MediaPipePoseEstimator(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'stub' or 'mediapipe'")

        self._prev_bbox: Optional[List[float]] = None

    def estimate_frame(self, frame: np.ndarray, frame_idx: int) -> PoseFrame:
        """Estimate 2D pose for a single frame.

        Selects single person using highest confidence or closest to previous.

        Args:
            frame: Input frame (BGR, uint8).
            frame_idx: Frame index for output.

        Returns:
            PoseFrame with 2D pose data.
        """
        detections = self._estimator.estimate(frame)

        if not detections:
            self._prev_bbox = None
            return create_empty_pose_frame(frame_idx)

        # Select best detection
        if len(detections) == 1:
            det = detections[0]
        else:
            det = self._select_person(detections)

        self._prev_bbox = det.bbox_xyxy

        # Build visibility arrays
        joint_visible = []
        joint_occluded = []
        joint_in_frame = []

        h, w = frame.shape[:2]

        for j in range(NUM_JOINTS):
            kp = det.keypoints_2d[j]
            conf = det.conf_2d[j]

            # Check if in frame
            in_frame = (
                kp[0] != SENTINEL_2D[0]
                and 0 <= kp[0] < w
                and 0 <= kp[1] < h
                and conf >= self.min_confidence
            )

            # Visible if high confidence, occluded if low but detected
            visible = in_frame and conf >= 0.5
            occluded = in_frame and not visible

            joint_in_frame.append(in_frame)
            joint_visible.append(visible)
            joint_occluded.append(occluded)

        return PoseFrame(
            frame_idx=frame_idx,
            bbox_xyxy=det.bbox_xyxy,
            keypoints_2d_px=det.keypoints_2d.tolist(),
            conf_2d=det.conf_2d.tolist(),
            coords_3d=[SENTINEL_2D + [-1.0] for _ in range(NUM_JOINTS)],  # placeholder
            conf_3d=[SENTINEL_CONF] * NUM_JOINTS,
            joint_visible=joint_visible,
            joint_occluded=joint_occluded,
            joint_in_frame=joint_in_frame,
            world_coords_3d=(
                det.world_keypoints_3d.tolist() if det.world_keypoints_3d is not None else None
            ),
        )

    def estimate_video(self, frames: List[np.ndarray]) -> List[PoseFrame]:
        """Estimate 2D poses for all frames in a video.

        Args:
            frames: List of video frames.

        Returns:
            List of PoseFrame objects.
        """
        return [self.estimate_frame(f, i) for i, f in enumerate(frames)]

    def _select_person(self, detections: List[Detection2D]) -> Detection2D:
        """Select best person from multiple detections.

        Uses IoU with previous bbox if available, otherwise highest confidence.
        """
        if self._prev_bbox is None or self._prev_bbox == SENTINEL_BBOX:
            # Select highest confidence
            return max(detections, key=lambda d: d.person_conf)

        # Select closest to previous bbox by IoU
        best_det = None
        best_iou = -1

        for det in detections:
            iou = self._compute_iou(self._prev_bbox, det.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        return best_det if best_det else detections[0]

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
