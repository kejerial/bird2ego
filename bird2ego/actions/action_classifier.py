"""ActionClassifier: labels action segments based on event/state patterns."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..utils.timeline import (
    ActionSegment,
    ContactEvent,
    Timeline,
)

logger = logging.getLogger(__name__)


class ActionLabel(str, Enum):
    """Standard action labels."""

    PICK = "pick"
    PLACE = "place"
    INSERT = "insert"
    REMOVE = "remove"
    PUSH = "push"
    PULL = "pull"
    INSPECT = "inspect"
    IDLE = "idle"
    REACH = "reach"
    RETRACT = "retract"
    MANIPULATE = "manipulate"
    UNKNOWN = "unknown"


@dataclass
class ActionClassifierConfig:
    """Configuration for action classification."""

    # Confidence thresholds
    high_conf_threshold: float = 0.8
    medium_conf_threshold: float = 0.5

    # Pattern matching weights
    use_contact_patterns: bool = True
    use_motion_patterns: bool = True
    use_state_patterns: bool = True


class ActionClassifier:
    """Classifies action segments based on event/state patterns.

    Uses deterministic rules mapping from:
    - Contact events during segment
    - Object motion states
    - Support relation changes
    - Containment changes

    to action labels like pick, place, insert, remove, etc.
    """

    def __init__(self, config: Optional[ActionClassifierConfig] = None):
        """Initialize ActionClassifier.

        Args:
            config: Configuration for classification.
        """
        self.config = config or ActionClassifierConfig()

    def classify_segments(
        self,
        segments: List[ActionSegment],
        timeline: Timeline,
    ) -> List[ActionSegment]:
        """Classify all segments.

        Args:
            segments: List of unlabeled segments.
            timeline: Timeline with events and objects.

        Returns:
            Segments with labels and confidence updated.
        """
        for segment in segments:
            label, confidence, evidence = self._classify_segment(segment, timeline)
            segment.label = label.value
            segment.conf = confidence
            segment.evidence.update(evidence)

        return segments

    def _classify_segment(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> tuple:
        """Classify a single segment.

        Args:
            segment: Segment to classify.
            timeline: Timeline with context.

        Returns:
            Tuple of (label, confidence, evidence).
        """
        evidence = {}

        # Get events during this segment
        events = self._get_events_in_segment(segment, timeline)
        evidence["num_events"] = len(events)

        # Get object state changes
        state_changes = self._get_state_changes_in_segment(segment, timeline)
        evidence["state_changes"] = [str(sc) for sc in state_changes]

        # Apply classification rules
        label, conf = self._apply_rules(segment, events, state_changes, timeline)
        evidence["rule_matched"] = label.value

        return label, conf, evidence

    def _get_events_in_segment(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> List[ContactEvent]:
        """Get contact events overlapping with segment.

        Args:
            segment: Action segment.
            timeline: Timeline with events.

        Returns:
            List of overlapping events.
        """
        events = []
        for event in timeline.events:
            # Check overlap
            if event.frame_end >= segment.frame_start and event.frame_start <= segment.frame_end:
                events.append(event)
        return events

    def _get_state_changes_in_segment(
        self,
        segment: ActionSegment,
        timeline: Timeline,
    ) -> List[Dict]:
        """Get object state changes during segment.

        Args:
            segment: Action segment.
            timeline: Timeline with objects.

        Returns:
            List of state change dictionaries.
        """
        changes = []

        for obj_id in segment.objects_involved:
            if obj_id not in timeline.objects:
                continue

            track = timeline.objects[obj_id]

            # Get states at start and end
            start_frame = max(0, segment.frame_start)
            end_frame = min(segment.frame_end, track.num_frames - 1)

            if start_frame >= track.num_frames or end_frame < 0:
                continue

            start_state = track.frames[start_frame].state
            end_state = track.frames[end_frame].state

            # Check each state dimension
            if start_state.support_relation != end_state.support_relation:
                changes.append(
                    {
                        "object_id": obj_id,
                        "type": "support_relation",
                        "from": start_state.support_relation.value,
                        "to": end_state.support_relation.value,
                    }
                )

            if start_state.motion_state != end_state.motion_state:
                changes.append(
                    {
                        "object_id": obj_id,
                        "type": "motion_state",
                        "from": start_state.motion_state.value,
                        "to": end_state.motion_state.value,
                    }
                )

            if start_state.containment != end_state.containment:
                changes.append(
                    {
                        "object_id": obj_id,
                        "type": "containment",
                        "from": start_state.containment.value,
                        "to": end_state.containment.value,
                    }
                )

            if start_state.interaction_state != end_state.interaction_state:
                changes.append(
                    {
                        "object_id": obj_id,
                        "type": "interaction_state",
                        "from": start_state.interaction_state.value,
                        "to": end_state.interaction_state.value,
                    }
                )

        return changes

    def _apply_rules(
        self,
        segment: ActionSegment,
        events: List[ContactEvent],
        state_changes: List[Dict],
        timeline: Timeline,
    ) -> tuple:
        """Apply classification rules.

        Args:
            segment: Segment to classify.
            events: Contact events in segment.
            state_changes: State changes in segment.
            timeline: Timeline context.

        Returns:
            Tuple of (label, confidence).
        """
        # Extract patterns
        event_labels = set(e.label for e in events)
        support_changes = [sc for sc in state_changes if sc["type"] == "support_relation"]
        containment_changes = [sc for sc in state_changes if sc["type"] == "containment"]
        motion_changes = [sc for sc in state_changes if sc["type"] == "motion_state"]

        # Rule 1: PICK - grasp event + support changes from table/fixture to hand
        if "grasp" in event_labels:
            for sc in support_changes:
                if sc["from"] in ["table", "fixture", "unknown"] and sc["to"] == "hand":
                    return ActionLabel.PICK, 0.9

        # Rule 2: PLACE - grasp ends + support changes from hand to table/fixture
        if "grasp" in event_labels:
            for sc in support_changes:
                if sc["from"] == "hand" and sc["to"] in ["table", "fixture"]:
                    return ActionLabel.PLACE, 0.9

        # Rule 3: INSERT - containment changes to in_bin/in_box
        for sc in containment_changes:
            if sc["to"] in ["in_bin", "in_box"]:
                return ActionLabel.INSERT, 0.85

        # Rule 4: REMOVE - containment changes from in_bin/in_box to none
        for sc in containment_changes:
            if sc["from"] in ["in_bin", "in_box"] and sc["to"] == "none":
                return ActionLabel.REMOVE, 0.85

        # Rule 5: PUSH - push event detected
        if "push" in event_labels:
            return ActionLabel.PUSH, 0.7

        # Rule 6: PULL - pull event detected
        if "pull" in event_labels:
            return ActionLabel.PULL, 0.7

        # Rule 7: INSPECT - touch event without significant state changes
        if "touch" in event_labels and not state_changes:
            return ActionLabel.INSPECT, 0.6

        # Rule 8: REACH - motion towards objects before contact
        if motion_changes and not events:
            for sc in motion_changes:
                if sc["to"] == "moving":
                    return ActionLabel.REACH, 0.5

        # Rule 9: RETRACT - motion away from objects after contact
        if motion_changes and events:
            for sc in motion_changes:
                if sc["to"] == "static":
                    return ActionLabel.RETRACT, 0.5

        # Rule 10: MANIPULATE - grasp with motion but no support change
        if "grasp" in event_labels and motion_changes and not support_changes:
            return ActionLabel.MANIPULATE, 0.6

        # Rule 11: IDLE - no events and no state changes
        if not events and not state_changes:
            return ActionLabel.IDLE, 0.8

        # Default: UNKNOWN
        if events:
            return ActionLabel.MANIPULATE, 0.4
        return ActionLabel.UNKNOWN, 0.3

    def add_segments_to_timeline(
        self,
        timeline: Timeline,
        segments: List[ActionSegment],
    ) -> Timeline:
        """Add classified segments to timeline.

        Args:
            timeline: Timeline to update.
            segments: Classified segments.

        Returns:
            Updated timeline.
        """
        timeline.segments = segments
        return timeline
