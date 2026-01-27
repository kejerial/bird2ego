"""TemporalAlignment: ensures consistent timestamps and frame indexing."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..utils.timeline import FrameInfo, Timeline, create_empty_timeline

logger = logging.getLogger(__name__)


class TemporalAlignment:
    """Guarantees monotonic timestamps and consistent indexing across modules.

    Handles:
    - Creating timeline from video metadata
    - Validating timestamp monotonicity
    - Frame-to-time and time-to-frame mapping
    - Reindexing data to match canonical timeline
    """

    def __init__(self, tolerance: float = 1e-6):
        """Initialize TemporalAlignment.

        Args:
            tolerance: Tolerance for timestamp comparisons.
        """
        self.tolerance = tolerance

    def create_timeline_from_video(
        self,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
        t0: float = 0.0,
        timestamps: Optional[List[float]] = None,
    ) -> Timeline:
        """Create a Timeline from video metadata.

        Args:
            fps: Frame rate of the video.
            frame_count: Number of frames.
            width: Frame width.
            height: Frame height.
            t0: Start time offset.
            timestamps: Optional list of actual timestamps from video.
                       If None, synthesizes timestamps from fps.

        Returns:
            Initialized Timeline object.
        """
        if timestamps is not None:
            # Use provided timestamps
            if len(timestamps) != frame_count:
                raise ValueError(
                    f"Timestamps length ({len(timestamps)}) != frame_count ({frame_count})"
                )
            # Validate monotonicity
            if not self.is_monotonic(timestamps):
                logger.warning("Provided timestamps are not monotonic, fixing...")
                timestamps = self.fix_monotonicity(timestamps)

            frames = [
                FrameInfo(frame_idx=i, t=t, width=width, height=height)
                for i, t in enumerate(timestamps)
            ]
            timeline = Timeline(
                fps_extracted=fps,
                t0=timestamps[0] if timestamps else t0,
                timestamp_source="video",
                frame_to_time_rule="from_video_timestamps",
                frames=frames,
            )
        else:
            # Synthesize timestamps
            timeline = create_empty_timeline(
                num_frames=frame_count,
                fps=fps,
                width=width,
                height=height,
                t0=t0,
            )

        logger.info(
            f"Created timeline: {timeline.num_frames} frames, "
            f"{timeline.fps_extracted:.2f} fps, {timeline.duration:.2f}s"
        )

        return timeline

    def is_monotonic(self, timestamps: List[float]) -> bool:
        """Check if timestamps are strictly monotonically increasing.

        Args:
            timestamps: List of timestamps.

        Returns:
            True if strictly monotonic.
        """
        if len(timestamps) < 2:
            return True
        ts = np.array(timestamps)
        return bool(np.all(np.diff(ts) > -self.tolerance))

    def fix_monotonicity(self, timestamps: List[float]) -> List[float]:
        """Fix non-monotonic timestamps by interpolating.

        Args:
            timestamps: List of timestamps (possibly non-monotonic).

        Returns:
            Fixed timestamps that are monotonic.
        """
        if len(timestamps) < 2:
            return timestamps

        ts = np.array(timestamps, dtype=np.float64)
        fixed = np.zeros_like(ts)
        fixed[0] = ts[0]

        for i in range(1, len(ts)):
            if ts[i] <= fixed[i - 1]:
                # Interpolate: assume constant frame rate
                if i > 1:
                    dt = fixed[i - 1] - fixed[i - 2]
                else:
                    dt = 1.0 / 30.0  # Default to 30fps if first pair
                fixed[i] = fixed[i - 1] + dt
                logger.warning(
                    f"Fixed non-monotonic timestamp at frame {i}: "
                    f"{ts[i]:.6f} -> {fixed[i]:.6f}"
                )
            else:
                fixed[i] = ts[i]

        return fixed.tolist()

    def validate_timeline(self, timeline: Timeline) -> Tuple[bool, List[str]]:
        """Validate a timeline for consistency.

        Args:
            timeline: Timeline to validate.

        Returns:
            Tuple of (is_valid, list_of_warnings).
        """
        warnings = []

        # Check frame indices are contiguous 0..T-1
        indices = timeline.get_frame_indices()
        expected = list(range(timeline.num_frames))
        if indices != expected:
            warnings.append(
                f"Frame indices not contiguous: got {indices[:5]}..., "
                f"expected {expected[:5]}..."
            )

        # Check timestamps are monotonic
        if timeline.num_frames > 1:
            ts = timeline.get_timestamps()
            if not self.is_monotonic(ts.tolist()):
                warnings.append("Timestamps are not monotonically increasing")

        # Check fps consistency
        if timeline.num_frames > 1:
            ts = timeline.get_timestamps()
            actual_fps = (timeline.num_frames - 1) / (ts[-1] - ts[0])
            if abs(actual_fps - timeline.fps_extracted) > 1.0:
                warnings.append(
                    f"FPS mismatch: extracted={timeline.fps_extracted:.2f}, "
                    f"computed={actual_fps:.2f}"
                )

        # Run timeline's own validation
        warnings.extend(timeline.validate_alignment())

        is_valid = len(warnings) == 0
        if not is_valid:
            for w in warnings:
                logger.warning(f"Timeline validation: {w}")

        return is_valid, warnings

    def align_data_to_timeline(
        self,
        data: np.ndarray,
        data_timestamps: List[float],
        timeline: Timeline,
        interpolate: bool = False,
    ) -> np.ndarray:
        """Align data with different timestamps to the canonical timeline.

        Args:
            data: Data array with shape (N, ...) where N is number of data points.
            data_timestamps: Timestamps for each data point.
            timeline: Target timeline to align to.
            interpolate: If True, interpolate missing frames. If False, use nearest.

        Returns:
            Aligned data array with shape (T, ...) matching timeline.
        """
        if len(data_timestamps) != data.shape[0]:
            raise ValueError(
                f"Data length ({data.shape[0]}) != timestamps length ({len(data_timestamps)})"
            )

        target_ts = timeline.get_timestamps()
        T = len(target_ts)

        # Initialize output with same dtype
        out_shape = (T,) + data.shape[1:]
        aligned = np.zeros(out_shape, dtype=data.dtype)

        data_ts = np.array(data_timestamps)

        for i, t in enumerate(target_ts):
            if interpolate and data.ndim <= 2:
                # Linear interpolation for 1D/2D data
                aligned[i] = np.interp(t, data_ts, data)
            else:
                # Nearest neighbor
                idx = int(np.argmin(np.abs(data_ts - t)))
                aligned[i] = data[idx]

        return aligned

    def resample_timeline(
        self,
        timeline: Timeline,
        target_fps: float,
    ) -> Timeline:
        """Resample a timeline to a different frame rate.

        Args:
            timeline: Original timeline.
            target_fps: Target frame rate.

        Returns:
            New Timeline at target frame rate.
        """
        if timeline.num_frames == 0:
            return timeline

        original_ts = timeline.get_timestamps()
        t_start = original_ts[0]
        t_end = original_ts[-1]
        duration = t_end - t_start

        new_frame_count = int(duration * target_fps) + 1
        new_timestamps = [t_start + i / target_fps for i in range(new_frame_count)]

        # Get width/height from first frame
        width = timeline.frames[0].width if timeline.frames else 640
        height = timeline.frames[0].height if timeline.frames else 480

        new_frames = [
            FrameInfo(frame_idx=i, t=t, width=width, height=height)
            for i, t in enumerate(new_timestamps)
        ]

        return Timeline(
            fps_extracted=target_fps,
            t0=t_start,
            timestamp_source="resampled",
            frame_to_time_rule="frame_idx / fps_extracted",
            frames=new_frames,
        )
