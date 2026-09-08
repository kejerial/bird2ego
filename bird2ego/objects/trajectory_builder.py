"""TrajectoryBuilder: builds per-object trajectory time series."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import (
    ObjectFrame,
    ObjectTrack,
    ObjectState,
    Timeline,
    SENTINEL_BBOX,
    SENTINEL_CONF,
    create_empty_object_frame,
    create_empty_object_track,
)
from .object_detector import Detection
from .object_tracker import ObjectTracker

logger = logging.getLogger(__name__)


class TrajectoryBuilder:
    """Builds per-object trajectory time series aligned to timeline.

    Ensures all tracks have the same length T with sentinels for missing frames.
    """

    def __init__(
        self,
        interpolate_gaps: bool = True,
        max_gap_frames: int = 10,
    ):
        """Initialize TrajectoryBuilder.

        Args:
            interpolate_gaps: Whether to interpolate over short gaps.
            max_gap_frames: Maximum gap to interpolate over.
        """
        self.interpolate_gaps = interpolate_gaps
        self.max_gap_frames = max_gap_frames

    def build_from_tracker(
        self,
        tracker: ObjectTracker,
        detections_per_frame: List[List[Detection]],
        num_frames: int,
    ) -> Dict[int, ObjectTrack]:
        """Build trajectories from tracker output.

        Args:
            tracker: Object tracker instance.
            detections_per_frame: Detections for each frame.
            num_frames: Total number of frames.

        Returns:
            Dictionary mapping object_id to ObjectTrack.
        """
        tracker.reset()

        # Track through video and collect results
        frame_results: Dict[int, Dict[int, Detection]] = {
            i: {} for i in range(num_frames)
        }

        for frame_idx, dets in enumerate(detections_per_frame):
            results = tracker.update(dets, frame_idx)
            for track_id, det in results:
                frame_results[frame_idx][track_id] = det

        # Get all track IDs
        all_track_ids = set()
        for frame_dets in frame_results.values():
            all_track_ids.update(frame_dets.keys())

        # Build tracks
        tracks = {}
        for track_id in all_track_ids:
            # Find first detection to get class info
            first_det = None
            for frame_idx in range(num_frames):
                if track_id in frame_results[frame_idx]:
                    first_det = frame_results[frame_idx][track_id]
                    break

            if first_det is None:
                continue

            # Create track with all frames
            track = create_empty_object_track(
                object_id=track_id,
                class_name=first_det.class_name,
                class_id=first_det.class_id,
                num_frames=num_frames,
            )

            # Fill in detections
            for frame_idx in range(num_frames):
                if track_id in frame_results[frame_idx]:
                    det = frame_results[frame_idx][track_id]
                    track.frames[frame_idx].bbox_xyxy = det.bbox_xyxy
                    track.frames[frame_idx].conf = det.confidence

            tracks[track_id] = track

        # Interpolate gaps if enabled
        if self.interpolate_gaps:
            tracks = self._interpolate_all_tracks(tracks)

        return tracks

    def build_from_detections(
        self,
        detections_per_frame: List[List[Detection]],
        tracker: Optional[ObjectTracker] = None,
    ) -> Dict[int, ObjectTrack]:
        """Build trajectories from detections with optional tracking.

        Args:
            detections_per_frame: Detections for each frame.
            tracker: Optional tracker. If None, creates default IoU tracker.

        Returns:
            Dictionary mapping object_id to ObjectTrack.
        """
        if tracker is None:
            from .object_tracker import ObjectTracker
            tracker = ObjectTracker(backend="iou")

        num_frames = len(detections_per_frame)
        return self.build_from_tracker(tracker, detections_per_frame, num_frames)

    def add_tracks_to_timeline(
        self,
        timeline: Timeline,
        tracks: Dict[int, ObjectTrack],
    ) -> Timeline:
        """Add object tracks to a timeline.

        Args:
            timeline: Timeline to add tracks to.
            tracks: Dictionary of object tracks.

        Returns:
            Updated timeline.
        """
        # Ensure tracks are aligned to timeline length
        T = timeline.num_frames
        aligned_tracks = {}

        for obj_id, track in tracks.items():
            if track.num_frames == T:
                aligned_tracks[obj_id] = track
            elif track.num_frames < T:
                # Extend track with empty frames
                new_track = create_empty_object_track(
                    object_id=obj_id,
                    class_name=track.class_name,
                    class_id=track.class_id,
                    num_frames=T,
                )
                for i, frame in enumerate(track.frames):
                    if i < T:
                        new_track.frames[i] = frame
                aligned_tracks[obj_id] = new_track
            else:
                # Truncate track
                track.frames = track.frames[:T]
                aligned_tracks[obj_id] = track

        timeline.objects = aligned_tracks
        return timeline

    def _interpolate_all_tracks(
        self,
        tracks: Dict[int, ObjectTrack],
    ) -> Dict[int, ObjectTrack]:
        """Interpolate gaps in all tracks.

        Args:
            tracks: Dictionary of object tracks.

        Returns:
            Tracks with gaps interpolated.
        """
        for track in tracks.values():
            self._interpolate_track(track)
        return tracks

    def _interpolate_track(self, track: ObjectTrack) -> None:
        """Interpolate gaps in a single track.

        Args:
            track: Object track to interpolate.
        """
        T = track.num_frames
        if T < 3:
            return

        # Find valid frames
        valid = np.array([
            f.bbox_xyxy != SENTINEL_BBOX for f in track.frames
        ])

        # Find gaps
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
                        self._interpolate_gap(track, gap_start, gap_end)
            else:
                i += 1

    def _interpolate_gap(
        self,
        track: ObjectTrack,
        gap_start: int,
        gap_end: int,
    ) -> None:
        """Interpolate a gap in a track.

        Args:
            track: Object track.
            gap_start: First frame of gap.
            gap_end: First frame after gap.
        """
        start_frame = track.frames[gap_start - 1]
        end_frame = track.frames[gap_end]

        start_bbox = np.array(start_frame.bbox_xyxy)
        end_bbox = np.array(end_frame.bbox_xyxy)
        start_conf = start_frame.conf
        end_conf = end_frame.conf

        gap_len = gap_end - gap_start

        for i, t in enumerate(range(gap_start, gap_end)):
            alpha = (i + 1) / (gap_len + 1)

            # Linear interpolation
            bbox = ((1 - alpha) * start_bbox + alpha * end_bbox).tolist()
            conf = (1 - alpha) * start_conf + alpha * end_conf

            track.frames[t].bbox_xyxy = bbox
            track.frames[t].conf = conf * 0.5  # Reduce confidence for interpolated

    def get_trajectory_array(
        self,
        track: ObjectTrack,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get trajectory as numpy arrays.

        Args:
            track: Object track.

        Returns:
            Tuple of (bbox_centers, bbox_sizes, confidences).
            bbox_centers: (T, 2) array of (cx, cy).
            bbox_sizes: (T, 2) array of (width, height).
            confidences: (T,) array.
        """
        T = track.num_frames
        centers = np.zeros((T, 2), dtype=np.float32)
        sizes = np.zeros((T, 2), dtype=np.float32)
        confs = np.zeros(T, dtype=np.float32)

        for t, frame in enumerate(track.frames):
            bbox = frame.bbox_xyxy
            if bbox != SENTINEL_BBOX:
                centers[t, 0] = (bbox[0] + bbox[2]) / 2
                centers[t, 1] = (bbox[1] + bbox[3]) / 2
                sizes[t, 0] = bbox[2] - bbox[0]
                sizes[t, 1] = bbox[3] - bbox[1]
                confs[t] = frame.conf
            else:
                centers[t] = [-1, -1]
                sizes[t] = [-1, -1]
                confs[t] = 0

        return centers, sizes, confs

    def compute_velocity(
        self,
        track: ObjectTrack,
        fps: float = 30.0,
    ) -> np.ndarray:
        """Compute velocity for a track.

        Args:
            track: Object track.
            fps: Frame rate.

        Returns:
            (T,) array of speeds (pixels per second).
        """
        centers, _, confs = self.get_trajectory_array(track)
        T = len(centers)

        if T < 2:
            return np.zeros(T)

        # Compute displacements
        velocities = np.zeros(T)
        dt = 1.0 / fps

        for t in range(1, T):
            if confs[t] > 0 and confs[t - 1] > 0:
                dx = centers[t, 0] - centers[t - 1, 0]
                dy = centers[t, 1] - centers[t - 1, 1]
                velocities[t] = np.sqrt(dx**2 + dy**2) / dt

        return velocities
