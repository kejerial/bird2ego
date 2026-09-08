"""ContactDetector: detects hand-object contacts using geometry and motion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.timeline import (
    JOINT_IDX,
    SENTINEL_2D,
    SENTINEL_BBOX,
    HandSide,
    ObjectTrack,
    PersonPose,
)

logger = logging.getLogger(__name__)


@dataclass
class ContactDetectorConfig:
    """Configuration for contact detection."""

    # Distance threshold for contact (pixels)
    contact_distance_threshold: float = 50.0

    # Minimum confidence for joints to be usable
    min_joint_conf_2d: float = 0.2

    # Temporal hysteresis (frames)
    contact_onset_frames: int = 2  # frames of contact before onset
    contact_offset_frames: int = 3  # frames without contact before offset

    # Use hand center (wrist + elbow midpoint) instead of just wrist
    use_hand_center: bool = True


@dataclass
class ContactFrame:
    """Contact information for a single frame."""

    frame_idx: int
    hand: HandSide
    object_id: int
    distance: float
    is_contact: bool
    hand_position: Tuple[float, float]
    object_center: Tuple[float, float]


class ContactDetector:
    """Detects contacts between hands and objects.

    Uses hand keypoints (wrist as proxy, or wrist+elbow midpoint as hand center)
    and object bounding boxes to detect spatial proximity.

    Applies temporal hysteresis to avoid flickering.
    """

    def __init__(self, config: Optional[ContactDetectorConfig] = None):
        """Initialize ContactDetector.

        Args:
            config: Configuration for contact detection.
        """
        self.config = config or ContactDetectorConfig()

        # State for temporal hysteresis
        self._contact_counts: Dict[Tuple[HandSide, int], int] = {}
        self._no_contact_counts: Dict[Tuple[HandSide, int], int] = {}
        self._active_contacts: Dict[Tuple[HandSide, int], bool] = {}

    def reset(self) -> None:
        """Reset detector state."""
        self._contact_counts = {}
        self._no_contact_counts = {}
        self._active_contacts = {}

    def detect_frame(
        self,
        person_pose: PersonPose,
        object_tracks: Dict[int, ObjectTrack],
        frame_idx: int,
    ) -> List[ContactFrame]:
        """Detect contacts in a single frame.

        Args:
            person_pose: PersonPose with pose data.
            object_tracks: Dictionary of object tracks.
            frame_idx: Current frame index.

        Returns:
            List of ContactFrame objects for detected contacts.
        """
        contacts = []

        # Get pose frame
        if frame_idx >= person_pose.num_frames:
            return contacts
        pose_frame = person_pose.frames[frame_idx]

        # Get hand positions
        left_pos = self._get_hand_position(pose_frame, HandSide.LEFT)
        right_pos = self._get_hand_position(pose_frame, HandSide.RIGHT)

        # Check each object
        for obj_id, track in object_tracks.items():
            if frame_idx >= track.num_frames:
                continue

            obj_frame = track.frames[frame_idx]
            if obj_frame.bbox_xyxy == SENTINEL_BBOX:
                continue

            # Compute object center
            bbox = obj_frame.bbox_xyxy
            obj_cx = (bbox[0] + bbox[2]) / 2
            obj_cy = (bbox[1] + bbox[3]) / 2
            obj_center = (obj_cx, obj_cy)

            # Check left hand
            if left_pos is not None:
                distance = self._compute_hand_object_distance(left_pos, bbox)
                is_contact = self._update_contact_state(HandSide.LEFT, obj_id, distance)
                contacts.append(
                    ContactFrame(
                        frame_idx=frame_idx,
                        hand=HandSide.LEFT,
                        object_id=obj_id,
                        distance=distance,
                        is_contact=is_contact,
                        hand_position=left_pos,
                        object_center=obj_center,
                    )
                )

            # Check right hand
            if right_pos is not None:
                distance = self._compute_hand_object_distance(right_pos, bbox)
                is_contact = self._update_contact_state(HandSide.RIGHT, obj_id, distance)
                contacts.append(
                    ContactFrame(
                        frame_idx=frame_idx,
                        hand=HandSide.RIGHT,
                        object_id=obj_id,
                        distance=distance,
                        is_contact=is_contact,
                        hand_position=right_pos,
                        object_center=obj_center,
                    )
                )

        return contacts

    def detect_video(
        self,
        person_pose: PersonPose,
        object_tracks: Dict[int, ObjectTrack],
    ) -> List[List[ContactFrame]]:
        """Detect contacts for all frames.

        Args:
            person_pose: PersonPose with pose data.
            object_tracks: Dictionary of object tracks.

        Returns:
            List of contact lists, one per frame.
        """
        self.reset()
        T = person_pose.num_frames
        return [self.detect_frame(person_pose, object_tracks, t) for t in range(T)]

    def _get_hand_position(
        self,
        pose_frame,
        hand: HandSide,
    ) -> Optional[Tuple[float, float]]:
        """Get hand position from pose.

        Uses wrist as proxy, optionally with elbow for hand center.

        Args:
            pose_frame: PoseFrame object.
            hand: Which hand.

        Returns:
            (x, y) position or None if not available.
        """
        if hand == HandSide.LEFT:
            wrist_idx = JOINT_IDX["left_wrist"]
            elbow_idx = JOINT_IDX["left_elbow"]
        else:
            wrist_idx = JOINT_IDX["right_wrist"]
            elbow_idx = JOINT_IDX["right_elbow"]

        # Check wrist usability
        wrist_kp = pose_frame.keypoints_2d_px[wrist_idx]
        wrist_conf = pose_frame.conf_2d[wrist_idx]
        wrist_in_frame = pose_frame.joint_in_frame[wrist_idx]

        wrist_usable = (
            wrist_in_frame
            and wrist_conf >= self.config.min_joint_conf_2d
            and wrist_kp != SENTINEL_2D
        )

        if not wrist_usable:
            return None

        wrist_pos = (wrist_kp[0], wrist_kp[1])

        # Optionally compute hand center
        if not self.config.use_hand_center:
            return wrist_pos

        # Check elbow usability
        elbow_kp = pose_frame.keypoints_2d_px[elbow_idx]
        elbow_conf = pose_frame.conf_2d[elbow_idx]
        elbow_in_frame = pose_frame.joint_in_frame[elbow_idx]

        elbow_usable = (
            elbow_in_frame
            and elbow_conf >= self.config.min_joint_conf_2d
            and elbow_kp != SENTINEL_2D
        )

        if elbow_usable:
            # Hand center = midpoint of wrist and elbow (towards hand)
            # Actually use wrist + 0.3*(wrist - elbow) to extend towards fingers
            dx = wrist_kp[0] - elbow_kp[0]
            dy = wrist_kp[1] - elbow_kp[1]
            hand_x = wrist_kp[0] + 0.3 * dx
            hand_y = wrist_kp[1] + 0.3 * dy
            return (hand_x, hand_y)
        else:
            return wrist_pos

    def _compute_hand_object_distance(
        self,
        hand_pos: Tuple[float, float],
        bbox: List[float],
    ) -> float:
        """Compute distance from hand to object bbox.

        Uses distance to nearest point on bbox, not center.

        Args:
            hand_pos: Hand (x, y) position.
            bbox: Object bbox [x1, y1, x2, y2].

        Returns:
            Distance in pixels.
        """
        hx, hy = hand_pos
        x1, y1, x2, y2 = bbox

        # Find nearest point on bbox
        nearest_x = max(x1, min(hx, x2))
        nearest_y = max(y1, min(hy, y2))

        # Check if hand is inside bbox
        if x1 <= hx <= x2 and y1 <= hy <= y2:
            return 0.0

        # Compute distance
        dx = hx - nearest_x
        dy = hy - nearest_y
        return float(np.sqrt(dx**2 + dy**2))

    def _update_contact_state(
        self,
        hand: HandSide,
        object_id: int,
        distance: float,
    ) -> bool:
        """Update contact state with temporal hysteresis.

        Args:
            hand: Which hand.
            object_id: Object ID.
            distance: Distance between hand and object.

        Returns:
            True if contact is active.
        """
        key = (hand, object_id)
        is_close = distance < self.config.contact_distance_threshold

        # Initialize if needed
        if key not in self._active_contacts:
            self._active_contacts[key] = False
            self._contact_counts[key] = 0
            self._no_contact_counts[key] = 0

        currently_active = self._active_contacts[key]

        if is_close:
            self._contact_counts[key] += 1
            self._no_contact_counts[key] = 0

            # Onset: need consistent contact
            if not currently_active:
                if self._contact_counts[key] >= self.config.contact_onset_frames:
                    self._active_contacts[key] = True
        else:
            self._no_contact_counts[key] += 1
            self._contact_counts[key] = 0

            # Offset: need consistent no-contact
            if currently_active:
                if self._no_contact_counts[key] >= self.config.contact_offset_frames:
                    self._active_contacts[key] = False

        return self._active_contacts[key]

    def get_contact_frames_per_object(
        self,
        contacts_per_frame: List[List[ContactFrame]],
    ) -> Dict[int, List[int]]:
        """Get frames where each object is in contact.

        Args:
            contacts_per_frame: Contacts for all frames.

        Returns:
            Dict mapping object_id to list of contact frame indices.
        """
        result: Dict[int, List[int]] = {}

        for frame_contacts in contacts_per_frame:
            for cf in frame_contacts:
                if cf.is_contact:
                    if cf.object_id not in result:
                        result[cf.object_id] = []
                    if cf.frame_idx not in result[cf.object_id]:
                        result[cf.object_id].append(cf.frame_idx)

        return result
