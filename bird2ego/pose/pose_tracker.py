"""PoseTracker: temporal smoothing for 2D and 3D keypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from ..utils.timeline import NUM_JOINTS, SENTINEL_2D, SENTINEL_3D, PoseFrame

logger = logging.getLogger(__name__)


class SmoothingMethod(str, Enum):
    """Smoothing method for temporal filtering."""
    NONE = "none"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    ONE_EURO = "one_euro"


@dataclass
class OneEuroParams:
    """Parameters for One Euro filter."""
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


class OneEuroFilter:
    """One Euro Filter for smooth signal filtering.

    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """Initialize One Euro Filter.

        Args:
            freq: Sampling frequency (fps).
            min_cutoff: Minimum cutoff frequency.
            beta: Cutoff slope.
            d_cutoff: Derivative cutoff frequency.
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset filter state."""
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter a new sample.

        Args:
            x: Input sample.

        Returns:
            Filtered sample.
        """
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()

        # Compute derivative
        dx = (x - self._x_prev) * self.freq

        # Filter derivative
        alpha_d = self._smoothing_factor(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev

        # Compute cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filter signal
        alpha = self._smoothing_factor(cutoff)
        x_hat = alpha * x + (1 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat

        return x_hat

    def _smoothing_factor(self, cutoff: np.ndarray) -> np.ndarray:
        """Compute smoothing factor from cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)


class PoseTracker:
    """Temporal smoothing and tracking for pose data.

    Applies smoothing to 2D and/or 3D keypoints to reduce jitter.
    Handles tracking failures by interpolating over short gaps.
    """

    def __init__(
        self,
        method: SmoothingMethod = SmoothingMethod.ONE_EURO,
        fps: float = 30.0,
        window_size: int = 5,
        alpha: float = 0.5,
        one_euro_params: Optional[OneEuroParams] = None,
        max_gap_frames: int = 5,
    ):
        """Initialize PoseTracker.

        Args:
            method: Smoothing method to use.
            fps: Frame rate for One Euro filter.
            window_size: Window size for moving average.
            alpha: Smoothing factor for exponential smoothing.
            one_euro_params: Parameters for One Euro filter.
            max_gap_frames: Maximum gap to interpolate over.
        """
        self.method = method
        self.fps = fps
        self.window_size = window_size
        self.alpha = alpha
        self.max_gap_frames = max_gap_frames

        if one_euro_params is None:
            one_euro_params = OneEuroParams()
        self.one_euro_params = one_euro_params

        # Filters for each joint dimension
        self._filters_2d: Optional[List[List[OneEuroFilter]]] = None
        self._filters_3d: Optional[List[List[OneEuroFilter]]] = None

    def smooth_poses(
        self,
        pose_frames: List[PoseFrame],
        smooth_2d: bool = True,
        smooth_3d: bool = True,
    ) -> List[PoseFrame]:
        """Apply temporal smoothing to pose data.

        Args:
            pose_frames: List of PoseFrame objects.
            smooth_2d: Whether to smooth 2D keypoints.
            smooth_3d: Whether to smooth 3D coordinates.

        Returns:
            List of smoothed PoseFrame objects.
        """
        if not pose_frames:
            return pose_frames

        if self.method == SmoothingMethod.NONE:
            return pose_frames

        # Extract arrays
        T = len(pose_frames)
        kp_2d = np.array([pf.keypoints_2d_px for pf in pose_frames])  # (T, 17, 2)
        kp_3d = np.array([pf.coords_3d for pf in pose_frames])  # (T, 17, 3)
        conf_2d = np.array([pf.conf_2d for pf in pose_frames])  # (T, 17)
        conf_3d = np.array([pf.conf_3d for pf in pose_frames])  # (T, 17)

        # Smooth each joint
        if smooth_2d:
            kp_2d = self._smooth_array(kp_2d, conf_2d, SENTINEL_2D)
        if smooth_3d:
            kp_3d = self._smooth_array(kp_3d, conf_3d, SENTINEL_3D)

        # Update pose frames
        for t in range(T):
            pose_frames[t].keypoints_2d_px = kp_2d[t].tolist()
            pose_frames[t].coords_3d = kp_3d[t].tolist()

        return pose_frames

    def _smooth_array(
        self,
        data: np.ndarray,
        conf: np.ndarray,
        sentinel: List[float],
    ) -> np.ndarray:
        """Smooth a (T, J, D) array of keypoints.

        Args:
            data: Keypoint data, shape (T, J, D).
            conf: Confidence scores, shape (T, J).
            sentinel: Sentinel value for missing data.

        Returns:
            Smoothed data array.
        """
        T, J, D = data.shape
        smoothed = data.copy()
        sentinel_arr = np.array(sentinel)

        for j in range(J):
            # Check which frames have valid data
            valid = np.array([
                not np.allclose(data[t, j], sentinel_arr) and conf[t, j] > 0
                for t in range(T)
            ])

            if not np.any(valid):
                continue

            # Interpolate over gaps
            data_j = self._interpolate_gaps(data[:, j], valid)

            # Apply smoothing
            if self.method == SmoothingMethod.MOVING_AVERAGE:
                smoothed[:, j] = self._moving_average(data_j, valid)
            elif self.method == SmoothingMethod.EXPONENTIAL:
                smoothed[:, j] = self._exponential_smooth(data_j, valid)
            elif self.method == SmoothingMethod.ONE_EURO:
                smoothed[:, j] = self._one_euro_smooth(data_j, valid, D)

            # Restore sentinels for invalid frames (beyond interpolation)
            for t in range(T):
                if not valid[t] and not self._in_gap(valid, t):
                    smoothed[t, j] = sentinel_arr

        return smoothed

    def _interpolate_gaps(
        self, data: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Interpolate over short gaps in data.

        Args:
            data: Data for one joint, shape (T, D).
            valid: Boolean mask of valid frames.

        Returns:
            Data with gaps interpolated.
        """
        T, D = data.shape
        result = data.copy()

        # Find gap start/end indices
        i = 0
        while i < T:
            if not valid[i]:
                # Find gap end
                gap_start = i
                while i < T and not valid[i]:
                    i += 1
                gap_end = i

                gap_len = gap_end - gap_start

                # Only interpolate short gaps with valid neighbors
                if gap_len <= self.max_gap_frames:
                    if gap_start > 0 and gap_end < T:
                        # Linear interpolation
                        for d in range(D):
                            result[gap_start:gap_end, d] = np.interp(
                                range(gap_start, gap_end),
                                [gap_start - 1, gap_end],
                                [data[gap_start - 1, d], data[gap_end, d]],
                            )
            else:
                i += 1

        return result

    def _in_gap(self, valid: np.ndarray, t: int) -> bool:
        """Check if frame t is in an interpolatable gap."""
        if valid[t]:
            return False

        # Find gap bounds
        start = t
        while start > 0 and not valid[start - 1]:
            start -= 1
        end = t
        while end < len(valid) - 1 and not valid[end + 1]:
            end += 1

        gap_len = end - start + 1
        has_left = start > 0
        has_right = end < len(valid) - 1

        return gap_len <= self.max_gap_frames and has_left and has_right

    def _moving_average(
        self, data: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Apply moving average smoothing."""
        T = len(data)
        result = data.copy()
        half_w = self.window_size // 2

        for t in range(T):
            if not valid[t]:
                continue
            start = max(0, t - half_w)
            end = min(T, t + half_w + 1)
            window_valid = valid[start:end]
            if np.sum(window_valid) > 0:
                result[t] = np.mean(data[start:end][window_valid], axis=0)

        return result

    def _exponential_smooth(
        self, data: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Apply exponential smoothing."""
        T = len(data)
        result = data.copy()

        prev = None
        for t in range(T):
            if not valid[t]:
                continue
            if prev is None:
                prev = data[t]
            else:
                result[t] = self.alpha * data[t] + (1 - self.alpha) * prev
                prev = result[t]

        return result

    def _one_euro_smooth(
        self, data: np.ndarray, valid: np.ndarray, dim: int
    ) -> np.ndarray:
        """Apply One Euro filter smoothing."""
        T = len(data)
        result = data.copy()

        # Create filter for each dimension
        filters = [
            OneEuroFilter(
                freq=self.fps,
                min_cutoff=self.one_euro_params.min_cutoff,
                beta=self.one_euro_params.beta,
                d_cutoff=self.one_euro_params.d_cutoff,
            )
            for _ in range(dim)
        ]

        for t in range(T):
            if not valid[t]:
                # Reset filters on gaps
                for f in filters:
                    f.reset()
                continue

            for d in range(dim):
                result[t, d] = filters[d](np.array([data[t, d]]))[0]

        return result
