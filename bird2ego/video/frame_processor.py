"""FrameProcessor: resize, letterbox, and normalize frames."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ResizePolicy(str, Enum):
    """How to resize frames."""
    NONE = "none"  # Keep original size
    RESIZE = "resize"  # Simple resize (may distort aspect ratio)
    LETTERBOX = "letterbox"  # Pad to maintain aspect ratio
    CROP_CENTER = "crop_center"  # Center crop


@dataclass
class ProcessedFrame:
    """A processed frame with metadata."""
    frame: np.ndarray
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int


class FrameProcessor:
    """Processes frames with resize/letterbox/normalization.

    Preserves metadata for coordinate mapping back to original space.
    """

    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        resize_policy: ResizePolicy = ResizePolicy.NONE,
        normalize: bool = False,
        pad_value: int = 114,  # gray padding for letterbox
    ):
        """Initialize FrameProcessor.

        Args:
            target_width: Target width after processing.
            target_height: Target height after processing.
            resize_policy: How to handle resizing.
            normalize: Whether to normalize pixel values to [0, 1].
            pad_value: Padding value for letterbox (0-255).
        """
        self.target_width = target_width
        self.target_height = target_height
        self.resize_policy = resize_policy
        self.normalize = normalize
        self.pad_value = pad_value

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Process a single frame.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            ProcessedFrame with processed data and metadata.
        """
        original_height, original_width = frame.shape[:2]

        if self.resize_policy == ResizePolicy.NONE:
            return ProcessedFrame(
                frame=frame.copy() if self.normalize else frame,
                original_width=original_width,
                original_height=original_height,
                processed_width=original_width,
                processed_height=original_height,
                scale_x=1.0,
                scale_y=1.0,
                pad_left=0,
                pad_top=0,
            )

        target_w = self.target_width or original_width
        target_h = self.target_height or original_height

        if self.resize_policy == ResizePolicy.RESIZE:
            processed, scale_x, scale_y, pad_left, pad_top = self._resize(
                frame, target_w, target_h
            )
        elif self.resize_policy == ResizePolicy.LETTERBOX:
            processed, scale_x, scale_y, pad_left, pad_top = self._letterbox(
                frame, target_w, target_h
            )
        elif self.resize_policy == ResizePolicy.CROP_CENTER:
            processed, scale_x, scale_y, pad_left, pad_top = self._crop_center(
                frame, target_w, target_h
            )
        else:
            raise ValueError(f"Unknown resize policy: {self.resize_policy}")

        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        return ProcessedFrame(
            frame=processed,
            original_width=original_width,
            original_height=original_height,
            processed_width=target_w,
            processed_height=target_h,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    def process_batch(
        self, frames: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[ProcessedFrame]]:
        """Process a batch of frames.

        Args:
            frames: List of input frames.

        Returns:
            Tuple of (stacked_array, list_of_processed_frames).
        """
        processed_list = [self.process(f) for f in frames]
        stacked = np.stack([p.frame for p in processed_list], axis=0)
        return stacked, processed_list

    def _resize(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """Simple resize (may distort aspect ratio)."""
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        scale_x = target_w / w
        scale_y = target_h / h
        return resized, scale_x, scale_y, 0, 0

    def _letterbox(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """Letterbox resize maintaining aspect ratio."""
        h, w = frame.shape[:2]

        # Compute scale to fit within target
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        # Add padding
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value),
        )

        return padded, scale, scale, pad_left, pad_top

    def _crop_center(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """Center crop to target size."""
        h, w = frame.shape[:2]

        # Scale to cover target area
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        cropped = resized[start_y : start_y + target_h, start_x : start_x + target_w]

        # Note: pad_left/pad_top are negative offsets for crop
        return cropped, scale, scale, -start_x, -start_y

    def map_coords_to_original(
        self,
        coords: np.ndarray,
        processed_frame: ProcessedFrame,
    ) -> np.ndarray:
        """Map coordinates from processed space back to original image space.

        Args:
            coords: Coordinates in processed space, shape (..., 2) as (x, y).
            processed_frame: ProcessedFrame metadata.

        Returns:
            Coordinates in original image space.
        """
        coords = np.asarray(coords, dtype=np.float32)

        # Remove padding offset
        coords[..., 0] = coords[..., 0] - processed_frame.pad_left
        coords[..., 1] = coords[..., 1] - processed_frame.pad_top

        # Reverse scale
        coords[..., 0] = coords[..., 0] / processed_frame.scale_x
        coords[..., 1] = coords[..., 1] / processed_frame.scale_y

        return coords

    def map_coords_to_processed(
        self,
        coords: np.ndarray,
        processed_frame: ProcessedFrame,
    ) -> np.ndarray:
        """Map coordinates from original image space to processed space.

        Args:
            coords: Coordinates in original space, shape (..., 2) as (x, y).
            processed_frame: ProcessedFrame metadata.

        Returns:
            Coordinates in processed image space.
        """
        coords = np.asarray(coords, dtype=np.float32)

        # Apply scale
        coords[..., 0] = coords[..., 0] * processed_frame.scale_x
        coords[..., 1] = coords[..., 1] * processed_frame.scale_y

        # Add padding offset
        coords[..., 0] = coords[..., 0] + processed_frame.pad_left
        coords[..., 1] = coords[..., 1] + processed_frame.pad_top

        return coords
