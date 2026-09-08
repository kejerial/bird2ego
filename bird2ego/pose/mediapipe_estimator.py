"""MediaPipe-based 2D pose estimator.

Works from any camera angle (front, side, back).
Uses MediaPipe Tasks API (0.10+).
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .pose_estimator_2d import PoseEstimator2DBase, Detection2D
from ..utils.timeline import NUM_JOINTS, SENTINEL_2D, SENTINEL_3D

logger = logging.getLogger(__name__)

# MediaPipe Pose Landmarker to COCO17 mapping
# MediaPipe has 33 landmarks, COCO17 has 17
MP_TO_COCO17 = {
    0: 0,    # nose -> nose
    2: 1,    # left_eye_inner -> left_eye (approx)
    5: 2,    # right_eye_inner -> right_eye (approx)
    7: 3,    # left_ear -> left_ear
    8: 4,    # right_ear -> right_ear
    11: 5,   # left_shoulder -> left_shoulder
    12: 6,   # right_shoulder -> right_shoulder
    13: 7,   # left_elbow -> left_elbow
    14: 8,   # right_elbow -> right_elbow
    15: 9,   # left_wrist -> left_wrist
    16: 10,  # right_wrist -> right_wrist
    23: 11,  # left_hip -> left_hip
    24: 12,  # right_hip -> right_hip
    25: 13,  # left_knee -> left_knee
    26: 14,  # right_knee -> right_knee
    27: 15,  # left_ankle -> left_ankle
    28: 16,  # right_ankle -> right_ankle
}

# Model URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def get_model_path() -> str:
    """Download model if needed and return path."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "pose_landmarker_lite.task"
    
    if not model_path.exists():
        print(f"Downloading MediaPipe pose model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete!")
    
    return str(model_path)


class MediaPipePoseEstimator(PoseEstimator2DBase):
    """MediaPipe Pose Landmarker estimator (Tasks API).
    
    Works well from:
    - Front view
    - Side view (left/right)
    - Back view
    - Diagonal views
    - Various distances
    
    Note: Best performance when full body is visible.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        """Initialize MediaPipe Pose Landmarker.

        Args:
            model_complexity: 0, 1, or 2 (ignored in new API, uses lite model).
            min_detection_confidence: Minimum detection confidence [0.0, 1.0].
            min_tracking_confidence: Minimum tracking confidence [0.0, 1.0].
            static_image_mode: If True, treats each image independently.
        """
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        self.mp = mp
        
        # Get model path (downloads if needed)
        model_path = get_model_path()
        
        # Create options
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
        )
        
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.static_mode = static_image_mode
        self.frame_timestamp = 0
        
        print(f"  MediaPipe Pose Landmarker loaded")

    def estimate(self, frame: np.ndarray) -> List[Detection2D]:
        """Estimate 2D pose using MediaPipe.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            List with single Detection2D if person found, empty otherwise.
        """
        h, w = frame.shape[:2]
        
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process frame
        if self.static_mode:
            results = self.detector.detect(mp_image)
        else:
            self.frame_timestamp += 33  # ~30fps
            results = self.detector.detect_for_video(mp_image, self.frame_timestamp)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return []
        
        # Get first person's landmarks
        landmarks = results.pose_landmarks[0]

        # Metric 3D landmarks, in metres, origin at the hip midpoint.
        # MediaPipe uses the same axis convention as the image: x right,
        # y down, z towards the camera.
        world_landmarks = None
        if getattr(results, "pose_world_landmarks", None):
            world_landmarks = results.pose_world_landmarks[0]
        
        # Convert MediaPipe landmarks to COCO17 format
        keypoints = np.full((NUM_JOINTS, 2), SENTINEL_2D, dtype=np.float32)
        conf = np.zeros(NUM_JOINTS, dtype=np.float32)
        
        world_keypoints = None
        if world_landmarks is not None:
            world_keypoints = np.full((NUM_JOINTS, 3), SENTINEL_3D, dtype=np.float32)

        for mp_idx, coco_idx in MP_TO_COCO17.items():
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                # MediaPipe gives normalized coords [0, 1]
                x = lm.x * w
                y = lm.y * h
                visibility = lm.visibility if hasattr(lm, 'visibility') else 0.9
                
                # Only use if visible enough
                if visibility > 0.1:
                    keypoints[coco_idx] = [x, y]
                    conf[coco_idx] = visibility
                    if world_keypoints is not None and mp_idx < len(world_landmarks):
                        wlm = world_landmarks[mp_idx]
                        world_keypoints[coco_idx] = [wlm.x, wlm.y, wlm.z]
                else:
                    keypoints[coco_idx] = SENTINEL_2D
                    conf[coco_idx] = 0.0
        
        # Compute bounding box from visible keypoints
        valid_mask = conf > 0.1
        if not np.any(valid_mask):
            return []
        
        valid_kps = keypoints[valid_mask]
        x1 = float(max(0, np.min(valid_kps[:, 0]) - 20))
        y1 = float(max(0, np.min(valid_kps[:, 1]) - 20))
        x2 = float(min(w, np.max(valid_kps[:, 0]) + 20))
        y2 = float(min(h, np.max(valid_kps[:, 1]) + 20))
        
        # Overall confidence
        person_conf = float(np.mean(conf[valid_mask]))
        
        return [
            Detection2D(
                bbox_xyxy=[x1, y1, x2, y2],
                keypoints_2d=keypoints,
                conf_2d=conf,
                person_conf=person_conf,
                world_keypoints_3d=world_keypoints,
            )
        ]

    def estimate_batch(self, frames: List[np.ndarray]) -> List[List[Detection2D]]:
        """Estimate poses for multiple frames."""
        return [self.estimate(f) for f in frames]

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()
