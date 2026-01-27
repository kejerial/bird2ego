"""VideoLoader: reads video files and extracts frames with timestamps."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata about a video file."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # seconds
    fourcc: str


class VideoLoader:
    """Loads video files and extracts frames at configured fps.

    Supports MP4, AVI, and other formats readable by OpenCV.
    """

    def __init__(
        self,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ):
        """Initialize VideoLoader.

        Args:
            target_fps: Target frame rate for extraction. If None, uses original fps.
            max_frames: Maximum number of frames to extract. If None, extracts all.
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    def open(self, video_path: str) -> VideoMetadata:
        """Open a video file and return its metadata.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with video properties.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video cannot be opened.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        duration = frame_count / fps if fps > 0 else 0.0

        self._metadata = VideoMetadata(
            path=str(path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            fourcc=fourcc,
        )

        logger.info(
            f"Opened video: {path.name}, {width}x{height}, "
            f"{fps:.2f} fps, {frame_count} frames, {duration:.2f}s"
        )

        return self._metadata

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._metadata = None

    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Return current video metadata."""
        return self._metadata

    def read_frames(
        self,
        video_path: Optional[str] = None,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Read frames from video, yielding (frame_idx, timestamp, frame).

        Args:
            video_path: Path to video. If None, uses already opened video.

        Yields:
            Tuple of (frame_idx, timestamp_seconds, frame_bgr).
        """
        if video_path is not None:
            self.open(video_path)

        if self._cap is None or self._metadata is None:
            raise RuntimeError("No video opened. Call open() first.")

        original_fps = self._metadata.fps
        target_fps = self.target_fps if self.target_fps else original_fps

        # Compute frame step for target fps
        if target_fps >= original_fps:
            frame_step = 1
            effective_fps = original_fps
        else:
            frame_step = int(round(original_fps / target_fps))
            effective_fps = original_fps / frame_step

        logger.info(
            f"Extracting frames: original_fps={original_fps:.2f}, "
            f"target_fps={target_fps:.2f}, effective_fps={effective_fps:.2f}, "
            f"frame_step={frame_step}"
        )

        frame_count = 0
        output_idx = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # Check if this frame should be extracted
            if frame_count % frame_step == 0:
                timestamp = frame_count / original_fps
                yield (output_idx, timestamp, frame)
                output_idx += 1

                if self.max_frames and output_idx >= self.max_frames:
                    logger.info(f"Reached max_frames limit: {self.max_frames}")
                    break

            frame_count += 1

        logger.info(f"Extracted {output_idx} frames from {frame_count} total")

    def read_all_frames(
        self,
        video_path: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], List[float], VideoMetadata]:
        """Read all frames into memory.

        Args:
            video_path: Path to video. If None, uses already opened video.

        Returns:
            Tuple of (frames_list, timestamps_list, metadata).
        """
        if video_path is not None:
            self.open(video_path)

        frames = []
        timestamps = []

        for frame_idx, timestamp, frame in self.read_frames():
            frames.append(frame)
            timestamps.append(timestamp)

        return frames, timestamps, self._metadata

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
