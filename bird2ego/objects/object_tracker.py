"""ObjectTracker: tracks objects across frames."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .object_detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with history."""
    track_id: int
    class_id: int
    class_name: str
    last_bbox: List[float]
    last_conf: float
    last_seen_frame: int
    age: int = 0  # frames since creation
    hits: int = 0  # successful matches


class IoUTracker:
    """Simple IoU-based multi-object tracker.

    Matches detections to tracks using IoU overlap.
    Creates new tracks for unmatched detections.
    Removes tracks that haven't been seen recently.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        """Initialize IoU tracker.

        Args:
            iou_threshold: Minimum IoU for matching.
            max_age: Maximum frames without detection before removing track.
            min_hits: Minimum hits before track is confirmed.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id: int = 0
        self._frame_idx: int = 0

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracks = {}
        self._next_id = 0
        self._frame_idx = 0

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
    ) -> List[Tuple[int, Detection]]:
        """Update tracks with new detections.

        Args:
            detections: List of detections for current frame.
            frame_idx: Current frame index.

        Returns:
            List of (track_id, detection) pairs for matched detections.
        """
        self._frame_idx = frame_idx

        if not detections:
            # Age all tracks
            self._age_tracks()
            return []

        # Compute IoU matrix
        track_ids = list(self._tracks.keys())
        if track_ids:
            iou_matrix = self._compute_iou_matrix(track_ids, detections)
            matches, unmatched_dets, unmatched_tracks = self._hungarian_match(
                iou_matrix, track_ids, detections
            )
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = []

        # Update matched tracks
        results = []
        for track_id, det_idx in matches:
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.last_bbox = det.bbox_xyxy
            track.last_conf = det.confidence
            track.last_seen_frame = frame_idx
            track.age += 1
            track.hits += 1
            results.append((track_id, det))

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._create_track(det, frame_idx)
            # Only return new tracks that meet min_hits
            if self._tracks[track_id].hits >= self.min_hits:
                results.append((track_id, det))

        # Remove old tracks
        self._remove_old_tracks()

        return results

    def get_active_tracks(self) -> Dict[int, TrackedObject]:
        """Get all active confirmed tracks."""
        return {
            tid: t for tid, t in self._tracks.items()
            if t.hits >= self.min_hits
        }

    def _create_track(self, det: Detection, frame_idx: int) -> int:
        """Create a new track."""
        track_id = self._next_id
        self._next_id += 1

        self._tracks[track_id] = TrackedObject(
            track_id=track_id,
            class_id=det.class_id,
            class_name=det.class_name,
            last_bbox=det.bbox_xyxy,
            last_conf=det.confidence,
            last_seen_frame=frame_idx,
            age=1,
            hits=1,
        )

        return track_id

    def _age_tracks(self) -> None:
        """Age all tracks by one frame."""
        for track in self._tracks.values():
            track.age += 1

    def _remove_old_tracks(self) -> None:
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        for track_id, track in self._tracks.items():
            frames_since_seen = self._frame_idx - track.last_seen_frame
            if frames_since_seen > self.max_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]

    def _compute_iou_matrix(
        self,
        track_ids: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        """Compute IoU matrix between tracks and detections."""
        n_tracks = len(track_ids)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for i, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for j, det in enumerate(detections):
                # Only match same class
                if track.class_id == det.class_id:
                    iou_matrix[i, j] = self._compute_iou(
                        track.last_bbox, det.bbox_xyxy
                    )

        return iou_matrix

    def _hungarian_match(
        self,
        iou_matrix: np.ndarray,
        track_ids: List[int],
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match tracks to detections using greedy algorithm.

        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks).
            matches is list of (track_id, detection_index) pairs.
        """
        n_tracks, n_dets = iou_matrix.shape
        matches = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))

        # Greedy matching (simple but effective for small numbers)
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Find best match
            best_iou = -1
            best_track_idx = -1
            best_det_idx = -1

            for track_idx in unmatched_tracks:
                for det_idx in unmatched_dets:
                    if iou_matrix[track_idx, det_idx] > best_iou:
                        best_iou = iou_matrix[track_idx, det_idx]
                        best_track_idx = track_idx
                        best_det_idx = det_idx

            if best_iou < self.iou_threshold:
                break

            # Make match
            track_id = track_ids[best_track_idx]
            matches.append((track_id, best_det_idx))
            unmatched_tracks.remove(best_track_idx)
            unmatched_dets.remove(best_det_idx)

        return matches, unmatched_dets, unmatched_tracks

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


class ObjectTracker:
    """Object tracker with configurable backend.

    Outputs stable object_ids across frames.
    """

    def __init__(
        self,
        backend: str = "iou",
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        **kwargs,
    ):
        """Initialize ObjectTracker.

        Args:
            backend: Tracker backend ("iou" for simple IoU tracker).
            iou_threshold: Minimum IoU for matching.
            max_age: Maximum frames without detection before removing track.
            min_hits: Minimum hits before track is confirmed.
            **kwargs: Additional arguments for backend.
        """
        if backend == "iou":
            self._tracker = IoUTracker(
                iou_threshold=iou_threshold,
                max_age=max_age,
                min_hits=min_hits,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.reset()

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
    ) -> List[Tuple[int, Detection]]:
        """Update tracks with new detections.

        Args:
            detections: List of detections for current frame.
            frame_idx: Current frame index.

        Returns:
            List of (track_id, detection) pairs.
        """
        return self._tracker.update(detections, frame_idx)

    def track_video(
        self,
        detections_per_frame: List[List[Detection]],
    ) -> Dict[int, List[Tuple[int, Optional[Detection]]]]:
        """Track objects through entire video.

        Args:
            detections_per_frame: List of detection lists, one per frame.

        Returns:
            Dictionary mapping track_id to list of (frame_idx, detection) pairs.
            Detection is None for frames where track was not detected.
        """
        self.reset()

        # Track through video
        all_results: Dict[int, List[Tuple[int, Optional[Detection]]]] = {}

        for frame_idx, dets in enumerate(detections_per_frame):
            results = self.update(dets, frame_idx)

            for track_id, det in results:
                if track_id not in all_results:
                    all_results[track_id] = []
                all_results[track_id].append((frame_idx, det))

        return all_results

    def get_active_tracks(self) -> Dict[int, TrackedObject]:
        """Get all active confirmed tracks."""
        return self._tracker.get_active_tracks()
