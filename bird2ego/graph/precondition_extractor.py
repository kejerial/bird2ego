"""PreconditionExtractor: extracts required states at segment start."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.timeline import (
    ActionSegment,
    Containment,
    InteractionState,
    MotionState,
    ObjectState,
    SupportRelation,
    Timeline,
)

logger = logging.getLogger(__name__)


@dataclass
class ObjectPrecondition:
    """Required state for an object before an action."""

    object_id: int
    support_relation: Optional[SupportRelation] = None
    motion_state: Optional[MotionState] = None
    interaction_state: Optional[InteractionState] = None
    containment: Optional[Containment] = None

    def matches(self, state: ObjectState) -> bool:
        """Check if a state matches this precondition.

        Args:
            state: ObjectState to check.

        Returns:
            True if state satisfies all specified conditions.
        """
        if self.support_relation is not None:
            if state.support_relation != self.support_relation:
                return False
        if self.motion_state is not None:
            if state.motion_state != self.motion_state:
                return False
        if self.interaction_state is not None:
            if state.interaction_state != self.interaction_state:
                return False
        if self.containment is not None:
            if state.containment != self.containment:
                return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {"object_id": self.object_id}
        if self.support_relation is not None:
            result["support_relation"] = self.support_relation.value
        if self.motion_state is not None:
            result["motion_state"] = self.motion_state.value
        if self.interaction_state is not None:
            result["interaction_state"] = self.interaction_state.value
        if self.containment is not None:
            result["containment"] = self.containment.value
        return result


@dataclass
class SegmentPreconditions:
    """Preconditions for an action segment."""

    seg_id: int
    object_preconditions: List[ObjectPrecondition] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "seg_id": self.seg_id,
            "objects": [p.to_dict() for p in self.object_preconditions],
        }


class PreconditionExtractor:
    """Extracts required states at segment start for involved objects.

    For each segment, captures the state of all involved objects
    at the frame just before the segment starts.
    """

    def __init__(self, lookback_frames: int = 1):
        """Initialize PreconditionExtractor.

        Args:
            lookback_frames: Number of frames to look back for preconditions.
        """
        self.lookback_frames = lookback_frames

    def extract(
        self,
        segments: List[ActionSegment],
        timeline: Timeline,
    ) -> Dict[int, SegmentPreconditions]:
        """Extract preconditions for all segments.

        Args:
            segments: List of action segments.
            timeline: Timeline with object tracks.

        Returns:
            Dictionary mapping seg_id to SegmentPreconditions.
        """
        result = {}

        for segment in segments:
            preconditions = self._extract_segment_preconditions(segment, timeline)
            result[segment.seg_id] = preconditions

        return result

    def _extract_segment_preconditions(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> SegmentPreconditions:
        """Extract preconditions for a single segment.

        Args:
            segment: Action segment.
            timeline: Timeline with object tracks.

        Returns:
            SegmentPreconditions for this segment.
        """
        preconditions = SegmentPreconditions(seg_id=segment.seg_id)

        # Get frame just before segment (or segment start if at beginning)
        precond_frame = max(0, segment.frame_start - self.lookback_frames)

        # Extract preconditions for each involved object
        for obj_id in segment.objects_involved:
            if obj_id not in timeline.objects:
                continue

            track = timeline.objects[obj_id]
            if precond_frame >= track.num_frames:
                continue

            state = track.frames[precond_frame].state

            # Create precondition capturing relevant state
            precond = ObjectPrecondition(
                object_id=obj_id,
                support_relation=state.support_relation,
                motion_state=state.motion_state,
                interaction_state=state.interaction_state,
                containment=state.containment,
            )
            preconditions.object_preconditions.append(precond)

        return preconditions

    def get_required_states(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> Dict[int, ObjectState]:
        """Get required object states for a segment.

        Args:
            segment: Action segment.
            timeline: Timeline with object tracks.

        Returns:
            Dictionary mapping object_id to required ObjectState.
        """
        precond_frame = max(0, segment.frame_start - self.lookback_frames)
        states = {}

        for obj_id in segment.objects_involved:
            if obj_id not in timeline.objects:
                continue

            track = timeline.objects[obj_id]
            if precond_frame >= track.num_frames:
                continue

            states[obj_id] = track.frames[precond_frame].state

        return states
