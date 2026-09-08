"""PostconditionExtractor: extracts state changes at segment end."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.timeline import (
    ActionSegment,
    ObjectState,
    Timeline,
)

logger = logging.getLogger(__name__)


@dataclass
class StateChange:
    """A change in a single state dimension."""

    dimension: str  # "support_relation", "motion_state", etc.
    from_value: str
    to_value: str


@dataclass
class ObjectPostcondition:
    """State changes for an object after an action."""

    object_id: int
    final_state: Optional[ObjectState] = None
    changes: List[StateChange] = field(default_factory=list)

    def has_changes(self) -> bool:
        """Check if there are any state changes."""
        return len(self.changes) > 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "object_id": self.object_id,
            "changes": [
                {"dimension": c.dimension, "from": c.from_value, "to": c.to_value}
                for c in self.changes
            ],
        }
        if self.final_state:
            result["final_state"] = {
                "support_relation": self.final_state.support_relation.value,
                "motion_state": self.final_state.motion_state.value,
                "interaction_state": self.final_state.interaction_state.value,
                "containment": self.final_state.containment.value,
            }
        return result


@dataclass
class SegmentPostconditions:
    """Postconditions (state changes) for an action segment."""

    seg_id: int
    object_postconditions: List[ObjectPostcondition] = field(default_factory=list)

    def has_effects(self) -> bool:
        """Check if this segment has any effects (state changes)."""
        return any(p.has_changes() for p in self.object_postconditions)

    def get_final_states(self) -> Dict[int, ObjectState]:
        """Get final states for all objects."""
        return {
            p.object_id: p.final_state
            for p in self.object_postconditions
            if p.final_state is not None
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "seg_id": self.seg_id,
            "objects": [p.to_dict() for p in self.object_postconditions],
        }


class PostconditionExtractor:
    """Extracts state deltas at segment end.

    For each segment, captures what changed in the state of involved objects
    between segment start and end.
    """

    def __init__(self, lookahead_frames: int = 1):
        """Initialize PostconditionExtractor.

        Args:
            lookahead_frames: Number of frames to look ahead for postconditions.
        """
        self.lookahead_frames = lookahead_frames

    def extract(
        self,
        segments: List[ActionSegment],
        timeline: Timeline,
    ) -> Dict[int, SegmentPostconditions]:
        """Extract postconditions for all segments.

        Args:
            segments: List of action segments.
            timeline: Timeline with object tracks.

        Returns:
            Dictionary mapping seg_id to SegmentPostconditions.
        """
        result = {}

        for segment in segments:
            postconditions = self._extract_segment_postconditions(segment, timeline)
            result[segment.seg_id] = postconditions

        return result

    def _extract_segment_postconditions(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> SegmentPostconditions:
        """Extract postconditions for a single segment.

        Args:
            segment: Action segment.
            timeline: Timeline with object tracks.

        Returns:
            SegmentPostconditions for this segment.
        """
        postconditions = SegmentPostconditions(seg_id=segment.seg_id)

        # Get frames for comparison
        start_frame = max(0, segment.frame_start)
        end_frame = min(segment.frame_end + self.lookahead_frames, timeline.num_frames - 1)

        # Extract postconditions for each involved object
        for obj_id in segment.objects_involved:
            if obj_id not in timeline.objects:
                continue

            track = timeline.objects[obj_id]
            if start_frame >= track.num_frames or end_frame >= track.num_frames:
                continue

            start_state = track.frames[start_frame].state
            end_state = track.frames[end_frame].state

            # Find changes
            changes = self._compare_states(start_state, end_state)

            postcond = ObjectPostcondition(
                object_id=obj_id,
                final_state=end_state,
                changes=changes,
            )
            postconditions.object_postconditions.append(postcond)

        return postconditions

    def _compare_states(
        self,
        start_state: ObjectState,
        end_state: ObjectState,
    ) -> List[StateChange]:
        """Compare two states and return list of changes.

        Args:
            start_state: State at segment start.
            end_state: State at segment end.

        Returns:
            List of StateChange objects.
        """
        changes = []

        if start_state.support_relation != end_state.support_relation:
            changes.append(
                StateChange(
                    dimension="support_relation",
                    from_value=start_state.support_relation.value,
                    to_value=end_state.support_relation.value,
                )
            )

        if start_state.motion_state != end_state.motion_state:
            changes.append(
                StateChange(
                    dimension="motion_state",
                    from_value=start_state.motion_state.value,
                    to_value=end_state.motion_state.value,
                )
            )

        if start_state.interaction_state != end_state.interaction_state:
            changes.append(
                StateChange(
                    dimension="interaction_state",
                    from_value=start_state.interaction_state.value,
                    to_value=end_state.interaction_state.value,
                )
            )

        if start_state.containment != end_state.containment:
            changes.append(
                StateChange(
                    dimension="containment",
                    from_value=start_state.containment.value,
                    to_value=end_state.containment.value,
                )
            )

        return changes

    def get_state_deltas(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> Dict[int, List[StateChange]]:
        """Get state deltas for all objects in a segment.

        Args:
            segment: Action segment.
            timeline: Timeline with object tracks.

        Returns:
            Dictionary mapping object_id to list of StateChange.
        """
        postconds = self._extract_segment_postconditions(segment, timeline)
        return {p.object_id: p.changes for p in postconds.object_postconditions}

    def get_final_states(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> Dict[int, ObjectState]:
        """Get final object states after a segment.

        Args:
            segment: Action segment.
            timeline: Timeline with object tracks.

        Returns:
            Dictionary mapping object_id to final ObjectState.
        """
        end_frame = min(segment.frame_end + self.lookahead_frames, timeline.num_frames - 1)
        states = {}

        for obj_id in segment.objects_involved:
            if obj_id not in timeline.objects:
                continue

            track = timeline.objects[obj_id]
            if end_frame >= track.num_frames:
                continue

            states[obj_id] = track.frames[end_frame].state

        return states
