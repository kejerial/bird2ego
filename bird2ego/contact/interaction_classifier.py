"""InteractionClassifier: classifies contact interactions using heuristics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import (
    Containment,
    HandSide,
    ObjectTrack,
    PersonPose,
    SupportRelation,
)
from .contact_detector import ContactFrame

logger = logging.getLogger(__name__)


class InteractionLabel(str, Enum):
    """Labels for hand-object interactions."""

    GRASP = "grasp"
    PUSH = "push"
    PULL = "pull"
    INSERT = "insert"
    REMOVE = "remove"
    TOUCH = "touch"
    UNKNOWN = "unknown"


@dataclass
class InteractionClassifierConfig:
    """Configuration for interaction classification."""

    # Motion thresholds
    motion_threshold: float = 5.0  # pixels per frame for "moving"
    motion_correlation_threshold: float = 0.7  # hand-object motion correlation

    # Duration thresholds (frames)
    min_grasp_duration: int = 5
    min_push_duration: int = 3

    # Spatial thresholds
    insert_proximity_threshold: float = 30.0  # pixels to fixture/container


@dataclass
class ClassifiedInteraction:
    """A classified interaction instance."""

    hand: HandSide
    object_id: int
    label: InteractionLabel
    confidence: float
    frame_start: int
    frame_end: int
    evidence: Dict


class InteractionClassifier:
    """Classifies hand-object interactions based on motion patterns.

    Heuristic rules:
    - grasp: sustained contact + object moves with hand
    - push: contact + object moves after hand motion (away from hand)
    - pull: contact + object moves after hand motion (towards hand)
    - insert: contact near fixture/bin + state change containment/support_relation
    - remove: opposite of insert
    - touch: brief contact without significant motion
    """

    def __init__(self, config: Optional[InteractionClassifierConfig] = None):
        """Initialize InteractionClassifier.

        Args:
            config: Configuration for classification.
        """
        self.config = config or InteractionClassifierConfig()

    def classify_interactions(
        self,
        contacts_per_frame: List[List[ContactFrame]],
        person_pose: PersonPose,
        object_tracks: Dict[int, ObjectTrack],
        fps: float = 30.0,
    ) -> List[ClassifiedInteraction]:
        """Classify all interactions in a video.

        Args:
            contacts_per_frame: Contact detections for all frames.
            person_pose: PersonPose with pose data.
            object_tracks: Dictionary of object tracks.
            fps: Frame rate.

        Returns:
            List of classified interactions.
        """
        # Find contact segments (continuous contact periods)
        segments = self._find_contact_segments(contacts_per_frame)

        # Classify each segment
        interactions = []
        for hand, object_id, frame_start, frame_end in segments:
            interaction = self._classify_segment(
                hand,
                object_id,
                frame_start,
                frame_end,
                contacts_per_frame,
                person_pose,
                object_tracks,
                fps,
            )
            if interaction is not None:
                interactions.append(interaction)

        return interactions

    def _find_contact_segments(
        self,
        contacts_per_frame: List[List[ContactFrame]],
    ) -> List[Tuple[HandSide, int, int, int]]:
        """Find continuous contact segments.

        Args:
            contacts_per_frame: Contact detections for all frames.

        Returns:
            List of (hand, object_id, frame_start, frame_end) tuples.
        """
        # Track active contacts
        active: Dict[Tuple[HandSide, int], int] = {}  # key -> start frame
        segments = []

        for frame_idx, frame_contacts in enumerate(contacts_per_frame):
            # Get current contacts
            current_contacts = set()
            for cf in frame_contacts:
                if cf.is_contact:
                    current_contacts.add((cf.hand, cf.object_id))

            # Check for ended segments
            ended_keys = []
            for key in active:
                if key not in current_contacts:
                    start = active[key]
                    segments.append((key[0], key[1], start, frame_idx - 1))
                    ended_keys.append(key)

            for key in ended_keys:
                del active[key]

            # Check for new segments
            for key in current_contacts:
                if key not in active:
                    active[key] = frame_idx

        # Close remaining segments
        T = len(contacts_per_frame)
        for key, start in active.items():
            segments.append((key[0], key[1], start, T - 1))

        return segments

    def _classify_segment(
        self,
        hand: HandSide,
        object_id: int,
        frame_start: int,
        frame_end: int,
        contacts_per_frame: List[List[ContactFrame]],
        person_pose: PersonPose,
        object_tracks: Dict[int, ObjectTrack],
        fps: float,
    ) -> Optional[ClassifiedInteraction]:
        """Classify a single contact segment.

        Args:
            hand: Which hand.
            object_id: Object ID.
            frame_start: Start frame.
            frame_end: End frame (inclusive).
            contacts_per_frame: All contacts.
            person_pose: Pose data.
            object_tracks: Object tracks.
            fps: Frame rate.

        Returns:
            ClassifiedInteraction or None if cannot classify.
        """
        duration = frame_end - frame_start + 1

        if object_id not in object_tracks:
            return None

        track = object_tracks[object_id]

        # Compute hand and object motion during segment
        hand_motion = self._compute_hand_motion(person_pose, hand, frame_start, frame_end)
        object_motion = self._compute_object_motion(track, frame_start, frame_end)

        # Compute motion correlation
        motion_corr = self._compute_motion_correlation(
            person_pose, hand, track, frame_start, frame_end
        )

        # Get state changes
        state_before = track.frames[frame_start].state if frame_start < track.num_frames else None
        state_after = track.frames[min(frame_end, track.num_frames - 1)].state

        # Build evidence
        evidence = {
            "duration_frames": duration,
            "hand_motion": float(hand_motion),
            "object_motion": float(object_motion),
            "motion_correlation": float(motion_corr),
        }

        # Classification rules
        label, confidence = self._apply_classification_rules(
            duration,
            hand_motion,
            object_motion,
            motion_corr,
            state_before,
            state_after,
            track,
            frame_start,
            frame_end,
        )

        if label == InteractionLabel.UNKNOWN and duration < 3:
            return None  # Skip very short unknown interactions

        return ClassifiedInteraction(
            hand=hand,
            object_id=object_id,
            label=label,
            confidence=confidence,
            frame_start=frame_start,
            frame_end=frame_end,
            evidence=evidence,
        )

    def _compute_hand_motion(
        self,
        person_pose: PersonPose,
        hand: HandSide,
        frame_start: int,
        frame_end: int,
    ) -> float:
        """Compute total hand motion during segment.

        Args:
            person_pose: Pose data.
            hand: Which hand.
            frame_start: Start frame.
            frame_end: End frame.

        Returns:
            Total motion in pixels.
        """
        from ..utils.timeline import JOINT_IDX, SENTINEL_2D

        wrist_idx = JOINT_IDX["left_wrist" if hand == HandSide.LEFT else "right_wrist"]
        total_motion = 0.0

        for t in range(frame_start, min(frame_end, person_pose.num_frames - 1)):
            kp1 = person_pose.frames[t].keypoints_2d_px[wrist_idx]
            kp2 = person_pose.frames[t + 1].keypoints_2d_px[wrist_idx]

            if kp1 != SENTINEL_2D and kp2 != SENTINEL_2D:
                dx = kp2[0] - kp1[0]
                dy = kp2[1] - kp1[1]
                total_motion += np.sqrt(dx**2 + dy**2)

        return total_motion

    def _compute_object_motion(
        self,
        track: ObjectTrack,
        frame_start: int,
        frame_end: int,
    ) -> float:
        """Compute total object motion during segment.

        Args:
            track: Object track.
            frame_start: Start frame.
            frame_end: End frame.

        Returns:
            Total motion in pixels.
        """
        from ..utils.timeline import SENTINEL_BBOX

        total_motion = 0.0

        for t in range(frame_start, min(frame_end, track.num_frames - 1)):
            bbox1 = track.frames[t].bbox_xyxy
            bbox2 = track.frames[t + 1].bbox_xyxy

            if bbox1 != SENTINEL_BBOX and bbox2 != SENTINEL_BBOX:
                cx1 = (bbox1[0] + bbox1[2]) / 2
                cy1 = (bbox1[1] + bbox1[3]) / 2
                cx2 = (bbox2[0] + bbox2[2]) / 2
                cy2 = (bbox2[1] + bbox2[3]) / 2

                dx = cx2 - cx1
                dy = cy2 - cy1
                total_motion += np.sqrt(dx**2 + dy**2)

        return total_motion

    def _compute_motion_correlation(
        self,
        person_pose: PersonPose,
        hand: HandSide,
        track: ObjectTrack,
        frame_start: int,
        frame_end: int,
    ) -> float:
        """Compute correlation between hand and object motion.

        Args:
            person_pose: Pose data.
            hand: Which hand.
            track: Object track.
            frame_start: Start frame.
            frame_end: End frame.

        Returns:
            Correlation coefficient (-1 to 1).
        """
        from ..utils.timeline import JOINT_IDX, SENTINEL_2D, SENTINEL_BBOX

        wrist_idx = JOINT_IDX["left_wrist" if hand == HandSide.LEFT else "right_wrist"]

        hand_vels = []
        obj_vels = []

        for t in range(frame_start, min(frame_end, person_pose.num_frames - 1)):
            if t >= track.num_frames - 1:
                break

            kp1 = person_pose.frames[t].keypoints_2d_px[wrist_idx]
            kp2 = person_pose.frames[t + 1].keypoints_2d_px[wrist_idx]
            bbox1 = track.frames[t].bbox_xyxy
            bbox2 = track.frames[t + 1].bbox_xyxy

            if (
                kp1 != SENTINEL_2D
                and kp2 != SENTINEL_2D
                and bbox1 != SENTINEL_BBOX
                and bbox2 != SENTINEL_BBOX
            ):
                # Hand velocity
                hv = np.sqrt((kp2[0] - kp1[0]) ** 2 + (kp2[1] - kp1[1]) ** 2)
                hand_vels.append(hv)

                # Object velocity
                cx1 = (bbox1[0] + bbox1[2]) / 2
                cy1 = (bbox1[1] + bbox1[3]) / 2
                cx2 = (bbox2[0] + bbox2[2]) / 2
                cy2 = (bbox2[1] + bbox2[3]) / 2
                ov = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
                obj_vels.append(ov)

        if len(hand_vels) < 3:
            return 0.0

        # Compute correlation
        hand_arr = np.array(hand_vels)
        obj_arr = np.array(obj_vels)

        if np.std(hand_arr) < 1e-6 or np.std(obj_arr) < 1e-6:
            return 0.0

        corr = np.corrcoef(hand_arr, obj_arr)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _apply_classification_rules(
        self,
        duration: int,
        hand_motion: float,
        object_motion: float,
        motion_corr: float,
        state_before,
        state_after,
        track: ObjectTrack,
        frame_start: int,
        frame_end: int,
    ) -> Tuple[InteractionLabel, float]:
        """Apply heuristic rules to classify interaction.

        Returns:
            Tuple of (label, confidence).
        """
        # Check for grasp: sustained contact with high motion correlation
        if (
            duration >= self.config.min_grasp_duration
            and motion_corr > self.config.motion_correlation_threshold
            and object_motion > self.config.motion_threshold * duration * 0.5
        ):
            return InteractionLabel.GRASP, min(0.9, 0.5 + motion_corr * 0.4)

        # Check for insert/remove based on state changes
        if state_before is not None and state_after is not None:
            # Insert: containment changed to in_bin/in_box
            if state_before.containment == Containment.NONE and state_after.containment in [
                Containment.IN_BIN,
                Containment.IN_BOX,
            ]:
                return InteractionLabel.INSERT, 0.8

            # Remove: containment changed from in_bin/in_box to none
            if (
                state_before.containment in [Containment.IN_BIN, Containment.IN_BOX]
                and state_after.containment == Containment.NONE
            ):
                return InteractionLabel.REMOVE, 0.8

            # Check for place: support_relation changed from hand to table/fixture
            if (
                state_before.support_relation == SupportRelation.HAND
                and state_after.support_relation in [SupportRelation.TABLE, SupportRelation.FIXTURE]
            ):
                return InteractionLabel.GRASP, 0.7  # End of grasp

        # Check for push/pull based on object motion without high correlation
        if (
            object_motion > self.config.motion_threshold * duration
            and motion_corr < self.config.motion_correlation_threshold
        ):
            # Determine direction relative to hand motion
            # Simplified: assume push if object moves
            if duration >= self.config.min_push_duration:
                return InteractionLabel.PUSH, 0.6

        # Brief contact without significant motion
        if duration < self.config.min_grasp_duration:
            return InteractionLabel.TOUCH, 0.5

        return InteractionLabel.UNKNOWN, 0.3
