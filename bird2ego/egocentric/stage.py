"""Egocentric pipeline stage.

Turns the third-person timeline into a first-person time series. The stage
reads the 3D body pose the lifter produced, estimates the head frame, and
re-expresses hands, arms, and tracked objects in that frame.

Object tracks carry 2D boxes only. The stage maps a box centre into the 3D
pose frame with a similarity fit between the 2D and 3D joints of the same
frame. That places the object on the torso depth plane. It is an
approximation, and `objects_ego` carries it as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import (
    JOINT_IDX,
    SENTINEL_3D,
    Timeline,
    is_bbox_valid,
)
from .transformer import EgocentricTimeSeries, EgocentricTransformer

logger = logging.getLogger(__name__)

# Joints that define the torso plane. The stage puts objects at their depth.
_TORSO_JOINTS = [
    JOINT_IDX["left_shoulder"],
    JOINT_IDX["right_shoulder"],
    JOINT_IDX["left_hip"],
    JOINT_IDX["right_hip"],
]

_WRIST_IDX = {"left": JOINT_IDX["left_wrist"], "right": JOINT_IDX["right_wrist"]}


@dataclass
class ImageToPoseFit:
    """Similarity map from image pixels to the 3D pose frame of one frame."""

    scale: float  # metres per pixel
    tx: float
    ty: float
    z_plane: float

    def apply(self, x_px: float, y_px: float) -> np.ndarray:
        """Map one pixel position onto the torso depth plane."""
        return np.array(
            [self.scale * x_px + self.tx, self.scale * y_px + self.ty, self.z_plane],
            dtype=np.float32,
        )


def fit_image_to_pose_3d(
    keypoints_2d: np.ndarray,
    coords_3d: np.ndarray,
    conf_3d: np.ndarray,
    min_conf: float = 0.2,
) -> Optional[ImageToPoseFit]:
    """Fit a uniform scale and offset from image pixels to 3D pose x and y.

    Args:
        keypoints_2d: 2D joints in pixels, shape (17, 2).
        coords_3d: 3D joints, shape (17, 3).
        conf_3d: 3D joint confidences, shape (17,).
        min_conf: Minimum confidence for a joint to join the fit.

    Returns:
        The fit, or None when fewer than three joints qualify.
    """
    valid = (conf_3d > min_conf) & (coords_3d[:, 0] != SENTINEL_3D[0]) & (keypoints_2d[:, 0] >= 0)
    if int(np.count_nonzero(valid)) < 3:
        return None

    p2 = keypoints_2d[valid].astype(np.float64)
    p3 = coords_3d[valid][:, :2].astype(np.float64)

    c2 = p2.mean(axis=0)
    c3 = p3.mean(axis=0)
    d2 = p2 - c2
    d3 = p3 - c3

    denom = float(np.sum(d2 * d2))
    if denom < 1e-9:
        return None

    scale = float(np.sum(d3 * d2) / denom)
    if not np.isfinite(scale) or abs(scale) < 1e-9:
        return None

    tx = float(c3[0] - scale * c2[0])
    ty = float(c3[1] - scale * c2[1])

    torso_valid = [j for j in _TORSO_JOINTS if valid[j]]
    z_plane = float(np.mean(coords_3d[torso_valid, 2])) if torso_valid else 0.0

    return ImageToPoseFit(scale=scale, tx=tx, ty=ty, z_plane=z_plane)


class EgocentricStage:
    """Build the egocentric time series for a processed timeline."""

    def __init__(
        self,
        width: int = 960,
        height: int = 720,
        fov_horizontal: float = 90.0,
        gaze_down_angle: float = 30.0,
        eye_offset_forward: float = 0.1,
        pinch_threshold: float = 0.05,
        min_joint_conf: float = 0.2,
        detect_hands: bool = False,
        hand_min_confidence: float = 0.5,
    ):
        """Initialize the stage.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
            fov_horizontal: Render horizontal field of view in degrees.
            gaze_down_angle: Angle the head looks down, in degrees.
            eye_offset_forward: Eye offset ahead of the head centre, in metres.
            pinch_threshold: Pinch distance threshold, in metres.
            min_joint_conf: Minimum 3D joint confidence the stage trusts.
            detect_hands: Run the MediaPipe hand landmarker on each frame.
            hand_min_confidence: Hand landmarker detection confidence.
        """
        self.width = width
        self.height = height
        self.fov_horizontal = fov_horizontal
        self.min_joint_conf = min_joint_conf
        self.detect_hands = detect_hands
        self.hand_min_confidence = hand_min_confidence

        self.transformer = EgocentricTransformer(
            eye_offset_forward=eye_offset_forward,
            default_gaze_down_angle=gaze_down_angle,
            pinch_threshold=pinch_threshold,
        )
        self._hand_detector = None

    # -- hands ---------------------------------------------------------------

    def _get_hand_detector(self):
        """Create the MediaPipe hand landmarker once, on first use."""
        if self._hand_detector is None:
            from ..contact.hand_detector import HandDetector

            self._hand_detector = HandDetector(
                num_hands=2,
                min_detection_confidence=self.hand_min_confidence,
                min_tracking_confidence=self.hand_min_confidence,
            )
        return self._hand_detector

    def _hands_for_frame(
        self,
        frame: np.ndarray,
        keypoints_2d: np.ndarray,
        coords_3d: np.ndarray,
        conf_3d: np.ndarray,
        fit: Optional[ImageToPoseFit],
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Detect hands and place their 21 landmarks in the 3D pose frame.

        The landmarker gives landmarks in image space. The stage converts the
        wrist-relative offsets to metres with the frame similarity scale, then
        anchors the hand on the body wrist joint. Each hand goes to the body
        wrist nearest to it in pixels, which is more reliable than the
        landmarker handedness label.

        Args:
            frame: Source BGR frame.
            keypoints_2d: 2D body joints in pixels, shape (17, 2).
            coords_3d: 3D body joints, shape (17, 3).
            conf_3d: 3D body joint confidences, shape (17,).
            fit: Image-to-pose similarity fit for this frame.

        Returns:
            Mapping of "left"/"right" to (landmarks (21, 3), confidence).
        """
        if fit is None:
            return {}

        detections = self._get_hand_detector().detect(frame)
        if not detections:
            return {}

        img_w = frame.shape[1]
        out: Dict[str, Tuple[np.ndarray, float]] = {}

        for hand in detections:
            wrist_px = np.asarray(hand.landmarks[0], dtype=np.float64)

            # Attach to the nearest usable body wrist.
            best_side, best_dist = None, np.inf
            for side, joint in _WRIST_IDX.items():
                if conf_3d[joint] <= self.min_joint_conf:
                    continue
                if keypoints_2d[joint][0] < 0:
                    continue
                dist = float(np.linalg.norm(keypoints_2d[joint] - wrist_px))
                if dist < best_dist:
                    best_side, best_dist = side, dist
            if best_side is None or best_side in out:
                continue

            rel_px = np.asarray(hand.landmarks, dtype=np.float64) - wrist_px
            rel_z_px = (
                np.asarray(hand.landmarks_3d, dtype=np.float64)[:, 2]
                - float(hand.landmarks_3d[0][2])
            ) * img_w

            rel_m = np.zeros((rel_px.shape[0], 3), dtype=np.float32)
            rel_m[:, 0] = rel_px[:, 0] * fit.scale
            rel_m[:, 1] = rel_px[:, 1] * fit.scale
            rel_m[:, 2] = rel_z_px * fit.scale

            anchor = coords_3d[_WRIST_IDX[best_side]]
            out[best_side] = (rel_m + anchor, float(hand.confidence))

        return out

    # -- objects -------------------------------------------------------------

    @staticmethod
    def _index_object_boxes(
        timeline: Timeline,
    ) -> Dict[int, Dict[int, List[float]]]:
        """Index every valid object box by frame index, then by object id."""
        by_frame: Dict[int, Dict[int, List[float]]] = {}
        for object_id, track in timeline.objects.items():
            for obj_frame in track.frames:
                if not is_bbox_valid(obj_frame.bbox_xyxy):
                    continue
                by_frame.setdefault(obj_frame.frame_idx, {})[object_id] = obj_frame.bbox_xyxy
        return by_frame

    @staticmethod
    def _object_centers(
        boxes: Dict[int, List[float]], fit: Optional[ImageToPoseFit]
    ) -> Dict[int, np.ndarray]:
        """Map every visible object box centre into the 3D pose frame."""
        if fit is None:
            return {}
        return {
            object_id: fit.apply((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            for object_id, bbox in boxes.items()
        }

    # -- main ----------------------------------------------------------------

    def run(
        self,
        timeline: Timeline,
        frames: Optional[List[np.ndarray]] = None,
    ) -> EgocentricTimeSeries:
        """Build the egocentric time series.

        Args:
            timeline: Processed timeline with 3D pose and object tracks.
            frames: Source frames. Only the hand landmarker needs them.

        Returns:
            One EgocentricFrame per timeline frame.
        """
        series = EgocentricTimeSeries()
        person_pose = timeline.person_pose
        if person_pose is None or person_pose.num_frames == 0:
            logger.warning("Egocentric stage: no pose data, nothing to transform")
            return series

        use_hands = self.detect_hands and frames is not None
        boxes_by_frame = self._index_object_boxes(timeline)

        for i, pose_frame in enumerate(person_pose.frames):
            timestamp = timeline.frames[i].t if i < len(timeline.frames) else float(i)
            keypoints_2d = np.asarray(pose_frame.keypoints_2d_px, dtype=np.float32)
            coords_3d = np.asarray(pose_frame.coords_3d, dtype=np.float32)
            conf_3d = np.asarray(pose_frame.conf_3d, dtype=np.float32)

            has_pose = bool(np.any(conf_3d > self.min_joint_conf))
            fit = (
                fit_image_to_pose_3d(keypoints_2d, coords_3d, conf_3d, self.min_joint_conf)
                if has_pose
                else None
            )

            hands: Dict[str, Tuple[np.ndarray, float]] = {}
            if use_hands and has_pose and i < len(frames):
                hands = self._hands_for_frame(frames[i], keypoints_2d, coords_3d, conf_3d, fit)

            left = hands.get("left")
            right = hands.get("right")

            ego_frame = self.transformer.process_frame(
                frame_idx=pose_frame.frame_idx,
                timestamp=timestamp,
                pose_3d=coords_3d if has_pose else None,
                pose_conf=conf_3d if has_pose else None,
                left_hand_3d=left[0] if left else None,
                left_hand_conf=left[1] if left else 0.0,
                right_hand_3d=right[0] if right else None,
                right_hand_conf=right[1] if right else 0.0,
                objects=self._object_centers(boxes_by_frame.get(pose_frame.frame_idx, {}), fit),
            )
            series.frames.append(ego_frame)

        return series

    # -- render --------------------------------------------------------------

    def render_video(
        self,
        series: EgocentricTimeSeries,
        output_path: str,
        fps: float,
        object_classes: Optional[Dict[int, str]] = None,
    ) -> Optional[str]:
        """Render the egocentric time series to an MP4 file.

        Args:
            series: Egocentric time series.
            output_path: Destination MP4 path.
            fps: Frame rate of the render.
            object_classes: Object id to class name, for the labels.

        Returns:
            The written path, or None when there is nothing to render.
        """
        if series.num_frames == 0:
            logger.warning("Egocentric render skipped: no frames")
            return None

        import cv2

        from .renderer import EgocentricRenderer

        renderer = EgocentricRenderer(
            width=self.width,
            height=self.height,
            fov_horizontal=self.fov_horizontal,
        )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(float(fps), 1.0),
            (self.width, self.height),
        )
        if not writer.isOpened():
            logger.warning("Egocentric render skipped: cannot open %s", path)
            return None

        try:
            for ego_frame in series.frames:
                writer.write(renderer.render(ego_frame, object_classes=object_classes))
        finally:
            writer.release()

        return str(path)


def summarize_series(series: EgocentricTimeSeries) -> Dict[str, int]:
    """Count what the egocentric stage actually produced.

    Args:
        series: Egocentric time series.

    Returns:
        Counts of frames, frames with a head pose, hands, and objects.
    """
    return {
        "num_frames": series.num_frames,
        "frames_with_head": sum(1 for f in series.frames if f.head_confidence > 0.0),
        "frames_with_left_hand": sum(1 for f in series.frames if f.left_hand_ego is not None),
        "frames_with_right_hand": sum(1 for f in series.frames if f.right_hand_ego is not None),
        "frames_with_arms": sum(
            1 for f in series.frames if f.left_arm_ego is not None or f.right_arm_ego is not None
        ),
        "frames_with_objects": sum(1 for f in series.frames if f.objects_ego),
    }
