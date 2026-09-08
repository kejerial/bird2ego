"""StateClassifier: derives object states from tracks + motion + contacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import (
    SENTINEL_BBOX,
    Containment,
    InteractionState,
    MotionState,
    ObjectState,
    ObjectTrack,
    SupportRelation,
)

logger = logging.getLogger(__name__)


@dataclass
class StateClassifierConfig:
    """Configuration for state classification."""

    motion_threshold: float = 5.0  # pixels per frame
    motion_window: int = 3  # frames for motion averaging
    hand_contact_distance: float = 50.0  # pixels
    fixture_classes: List[str] = None  # classes considered fixtures
    container_classes: List[str] = None  # classes considered containers

    def __post_init__(self):
        if self.fixture_classes is None:
            self.fixture_classes = ["fixture", "machine", "station"]
        if self.container_classes is None:
            self.container_classes = ["bin", "box", "container"]


class StateClassifier:
    """Classifies object states from tracks, motion, and contacts.

    Computes:
    - support_relation: where the object is supported
    - motion_state: whether the object is moving
    - interaction_state: contact/grasp state
    - containment: whether object is in a container
    """

    def __init__(self, config: Optional[StateClassifierConfig] = None):
        """Initialize StateClassifier.

        Args:
            config: Configuration for classification thresholds.
        """
        self.config = config or StateClassifierConfig()

    def classify_motion_only(
        self,
        object_tracks: Dict[int, ObjectTrack],
    ) -> Dict[int, ObjectTrack]:
        """Classify motion state for all objects (no contact info yet).

        This is the first pass before contacts are computed.

        Args:
            object_tracks: Dictionary of object tracks.

        Returns:
            Updated object tracks with motion_state filled in.
        """
        for track in object_tracks.values():
            self._classify_track_motion(track)
        return object_tracks

    def classify_with_contacts(
        self,
        object_tracks: Dict[int, ObjectTrack],
        hand_positions: Optional[Dict[str, np.ndarray]] = None,
        contact_frames: Optional[Dict[int, List[int]]] = None,
        grasp_frames: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, ObjectTrack]:
        """Refine object states using contact information.

        Args:
            object_tracks: Dictionary of object tracks.
            hand_positions: Dict with "left" and "right" keys, each (T, 2) array.
            contact_frames: Dict mapping object_id to list of contact frame indices.
            grasp_frames: Dict mapping object_id to list of grasp frame indices.

        Returns:
            Updated object tracks with full state information.
        """
        for obj_id, track in object_tracks.items():
            T = track.num_frames

            for t in range(T):
                frame = track.frames[t]

                # Check if in contact or grasped
                is_contact = (
                    contact_frames is not None
                    and obj_id in contact_frames
                    and t in contact_frames[obj_id]
                )
                is_grasped = (
                    grasp_frames is not None
                    and obj_id in grasp_frames
                    and t in grasp_frames[obj_id]
                )

                # Update interaction state
                if is_grasped:
                    frame.state.interaction_state = InteractionState.GRASPED
                    frame.state.support_relation = SupportRelation.HAND
                elif is_contact:
                    frame.state.interaction_state = InteractionState.IN_CONTACT
                else:
                    frame.state.interaction_state = InteractionState.NONE

                # Classify support relation if not grasped
                if not is_grasped:
                    frame.state.support_relation = self._classify_support(track, t, object_tracks)

                # Classify containment
                frame.state.containment = self._classify_containment(track, t, object_tracks)

        return object_tracks

    def _classify_track_motion(self, track: ObjectTrack) -> None:
        """Classify motion state for a single track."""
        T = track.num_frames
        if T < 2:
            return

        # Compute bbox centers
        centers = []
        valid = []
        for frame in track.frames:
            if frame.bbox_xyxy != SENTINEL_BBOX:
                cx = (frame.bbox_xyxy[0] + frame.bbox_xyxy[2]) / 2
                cy = (frame.bbox_xyxy[1] + frame.bbox_xyxy[3]) / 2
                centers.append([cx, cy])
                valid.append(True)
            else:
                centers.append([0, 0])
                valid.append(False)

        centers = np.array(centers)
        valid = np.array(valid)

        # Compute motion (displacement per frame)
        for t in range(T):
            if not valid[t]:
                continue

            # Get window of frames
            start = max(0, t - self.config.motion_window)
            end = min(T, t + self.config.motion_window + 1)

            # Compute average displacement in window
            displacements = []
            for i in range(start, end - 1):
                if valid[i] and valid[i + 1]:
                    dx = centers[i + 1, 0] - centers[i, 0]
                    dy = centers[i + 1, 1] - centers[i, 1]
                    displacements.append(np.sqrt(dx**2 + dy**2))

            if displacements:
                avg_motion = np.mean(displacements)
                if avg_motion > self.config.motion_threshold:
                    track.frames[t].state.motion_state = MotionState.MOVING
                else:
                    track.frames[t].state.motion_state = MotionState.STATIC
            else:
                track.frames[t].state.motion_state = MotionState.STATIC

    def _classify_support(
        self,
        track: ObjectTrack,
        frame_idx: int,
        all_tracks: Dict[int, ObjectTrack],
    ) -> SupportRelation:
        """Classify support relation for a single frame.

        Args:
            track: The object track.
            frame_idx: Frame index.
            all_tracks: All object tracks for spatial reasoning.

        Returns:
            SupportRelation for this frame.
        """
        frame = track.frames[frame_idx]
        if frame.bbox_xyxy == SENTINEL_BBOX:
            return SupportRelation.UNKNOWN

        bbox = frame.bbox_xyxy
        obj_bottom = bbox[3]  # y2
        obj_cx = (bbox[0] + bbox[2]) / 2

        # Check if on a fixture
        for other_id, other_track in all_tracks.items():
            if other_id == track.object_id:
                continue
            if other_track.class_name not in self.config.fixture_classes:
                continue
            if frame_idx >= other_track.num_frames:
                continue

            other_frame = other_track.frames[frame_idx]
            if other_frame.bbox_xyxy == SENTINEL_BBOX:
                continue

            other_bbox = other_frame.bbox_xyxy
            # Check if object is above fixture
            if other_bbox[0] < obj_cx < other_bbox[2] and abs(obj_bottom - other_bbox[1]) < 20:
                return SupportRelation.FIXTURE

        # Default to table if object is in lower part of frame
        return SupportRelation.TABLE

    def _classify_containment(
        self,
        track: ObjectTrack,
        frame_idx: int,
        all_tracks: Dict[int, ObjectTrack],
    ) -> Containment:
        """Classify containment for a single frame.

        Args:
            track: The object track.
            frame_idx: Frame index.
            all_tracks: All object tracks for spatial reasoning.

        Returns:
            Containment state for this frame.
        """
        frame = track.frames[frame_idx]
        if frame.bbox_xyxy == SENTINEL_BBOX:
            return Containment.UNKNOWN

        bbox = frame.bbox_xyxy
        obj_cx = (bbox[0] + bbox[2]) / 2
        obj_cy = (bbox[1] + bbox[3]) / 2

        # Check if inside a container
        for other_id, other_track in all_tracks.items():
            if other_id == track.object_id:
                continue
            if other_track.class_name not in self.config.container_classes:
                continue
            if frame_idx >= other_track.num_frames:
                continue

            other_frame = other_track.frames[frame_idx]
            if other_frame.bbox_xyxy == SENTINEL_BBOX:
                continue

            other_bbox = other_frame.bbox_xyxy
            # Check if object center is inside container
            if other_bbox[0] < obj_cx < other_bbox[2] and other_bbox[1] < obj_cy < other_bbox[3]:
                if "bin" in other_track.class_name.lower():
                    return Containment.IN_BIN
                else:
                    return Containment.IN_BOX

        return Containment.NONE

    def get_state_changes(
        self,
        track: ObjectTrack,
    ) -> List[Tuple[int, str, ObjectState, ObjectState]]:
        """Detect state changes in a track.

        Args:
            track: Object track to analyze.

        Returns:
            List of (frame_idx, change_type, old_state, new_state) tuples.
        """
        changes = []
        T = track.num_frames

        for t in range(1, T):
            old_state = track.frames[t - 1].state
            new_state = track.frames[t].state

            # Check each state dimension
            if old_state.motion_state != new_state.motion_state:
                changes.append((t, "motion_state", old_state, new_state))

            if old_state.support_relation != new_state.support_relation:
                changes.append((t, "support_relation", old_state, new_state))

            if old_state.interaction_state != new_state.interaction_state:
                changes.append((t, "interaction_state", old_state, new_state))

            if old_state.containment != new_state.containment:
                changes.append((t, "containment", old_state, new_state))

        return changes
