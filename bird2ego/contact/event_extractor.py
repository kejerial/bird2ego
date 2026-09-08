"""EventExtractor: extracts contact events with onset/offset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..utils.timeline import ContactEvent, HandSide, Timeline
from .contact_detector import ContactFrame
from .interaction_classifier import ClassifiedInteraction, InteractionLabel

logger = logging.getLogger(__name__)


@dataclass
class EventExtractorConfig:
    """Configuration for event extraction."""

    # Minimum event duration (frames)
    min_event_duration: int = 3

    # Merge events with small gaps (frames)
    merge_gap_threshold: int = 5

    # Minimum confidence to emit event
    min_confidence: float = 0.3


class EventExtractor:
    """Extracts contact events from detections and classifications.

    Produces ContactEvent objects with:
    - onset/offset times
    - hand and object participants
    - interaction labels
    - confidence scores
    """

    def __init__(self, config: Optional[EventExtractorConfig] = None):
        """Initialize EventExtractor.

        Args:
            config: Configuration for event extraction.
        """
        self.config = config or EventExtractorConfig()
        self._next_event_id = 0

    def reset(self) -> None:
        """Reset extractor state."""
        self._next_event_id = 0

    def extract_events(
        self,
        classified_interactions: List[ClassifiedInteraction],
        timeline: Timeline,
    ) -> List[ContactEvent]:
        """Extract contact events from classified interactions.

        Args:
            classified_interactions: List of classified interactions.
            timeline: Timeline for timestamp conversion.

        Returns:
            List of ContactEvent objects.
        """
        self.reset()

        # Filter by confidence and duration
        filtered = [
            ci
            for ci in classified_interactions
            if (
                ci.confidence >= self.config.min_confidence
                and ci.frame_end - ci.frame_start + 1 >= self.config.min_event_duration
            )
        ]

        # Merge nearby interactions of same type
        merged = self._merge_nearby_interactions(filtered)

        # Convert to ContactEvent objects
        events = []
        for ci in merged:
            event = self._interaction_to_event(ci, timeline)
            events.append(event)

        return events

    def extract_from_contacts(
        self,
        contacts_per_frame: List[List[ContactFrame]],
        timeline: Timeline,
        default_label: str = "contact",
    ) -> List[ContactEvent]:
        """Extract basic contact events without interaction classification.

        Args:
            contacts_per_frame: Contact detections for all frames.
            timeline: Timeline for timestamp conversion.
            default_label: Default label for events.

        Returns:
            List of ContactEvent objects.
        """
        self.reset()

        # Find contact segments
        segments = self._find_contact_segments(contacts_per_frame)

        # Convert to events
        events = []
        for hand, object_id, frame_start, frame_end in segments:
            duration = frame_end - frame_start + 1
            if duration < self.config.min_event_duration:
                continue

            t_start = timeline.frame_to_time(frame_start)
            t_end = timeline.frame_to_time(frame_end)

            # Compute confidence from contact distances
            avg_distance = self._compute_avg_distance(
                contacts_per_frame, hand, object_id, frame_start, frame_end
            )
            conf = max(0.3, 1.0 - avg_distance / 100.0)

            event = ContactEvent(
                event_id=self._next_event_id,
                event_type="contact",
                t_start=t_start,
                t_end=t_end,
                frame_start=frame_start,
                frame_end=frame_end,
                hand=hand,
                object_id=object_id,
                label=default_label,
                conf=conf,
                evidence={"avg_distance": avg_distance},
            )
            self._next_event_id += 1
            events.append(event)

        return events

    def add_events_to_timeline(
        self,
        timeline: Timeline,
        events: List[ContactEvent],
    ) -> Timeline:
        """Add contact events to a timeline.

        Args:
            timeline: Timeline to add events to.
            events: List of ContactEvent objects.

        Returns:
            Updated timeline.
        """
        timeline.events = events
        return timeline

    def _interaction_to_event(
        self,
        ci: ClassifiedInteraction,
        timeline: Timeline,
    ) -> ContactEvent:
        """Convert ClassifiedInteraction to ContactEvent.

        Args:
            ci: Classified interaction.
            timeline: Timeline for timestamp conversion.

        Returns:
            ContactEvent object.
        """
        t_start = timeline.frame_to_time(ci.frame_start)
        t_end = timeline.frame_to_time(ci.frame_end)

        event = ContactEvent(
            event_id=self._next_event_id,
            event_type="contact",
            t_start=t_start,
            t_end=t_end,
            frame_start=ci.frame_start,
            frame_end=ci.frame_end,
            hand=ci.hand,
            object_id=ci.object_id,
            label=ci.label.value,
            conf=ci.confidence,
            evidence=ci.evidence,
        )
        self._next_event_id += 1
        return event

    def _merge_nearby_interactions(
        self,
        interactions: List[ClassifiedInteraction],
    ) -> List[ClassifiedInteraction]:
        """Merge nearby interactions of the same type.

        Args:
            interactions: List of classified interactions.

        Returns:
            Merged list of interactions.
        """
        if not interactions:
            return []

        # Group by (hand, object_id, label)
        groups: Dict[tuple, List[ClassifiedInteraction]] = {}
        for ci in interactions:
            key = (ci.hand, ci.object_id, ci.label)
            if key not in groups:
                groups[key] = []
            groups[key].append(ci)

        merged = []
        for key, group in groups.items():
            # Sort by start frame
            group.sort(key=lambda x: x.frame_start)

            # Merge consecutive with small gaps
            current = group[0]
            for ci in group[1:]:
                gap = ci.frame_start - current.frame_end - 1
                if gap <= self.config.merge_gap_threshold:
                    # Merge
                    current = ClassifiedInteraction(
                        hand=current.hand,
                        object_id=current.object_id,
                        label=current.label,
                        confidence=(current.confidence + ci.confidence) / 2,
                        frame_start=current.frame_start,
                        frame_end=ci.frame_end,
                        evidence={**current.evidence, "merged": True},
                    )
                else:
                    merged.append(current)
                    current = ci

            merged.append(current)

        # Sort by start frame
        merged.sort(key=lambda x: x.frame_start)
        return merged

    def _find_contact_segments(
        self,
        contacts_per_frame: List[List[ContactFrame]],
    ) -> List[tuple]:
        """Find continuous contact segments.

        Args:
            contacts_per_frame: Contact detections for all frames.

        Returns:
            List of (hand, object_id, frame_start, frame_end) tuples.
        """
        # Track active contacts
        active: Dict[tuple, int] = {}  # key -> start frame
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

    def _compute_avg_distance(
        self,
        contacts_per_frame: List[List[ContactFrame]],
        hand: HandSide,
        object_id: int,
        frame_start: int,
        frame_end: int,
    ) -> float:
        """Compute average contact distance during segment.

        Args:
            contacts_per_frame: Contact detections.
            hand: Which hand.
            object_id: Object ID.
            frame_start: Start frame.
            frame_end: End frame.

        Returns:
            Average distance in pixels.
        """
        distances = []
        for t in range(frame_start, frame_end + 1):
            if t < len(contacts_per_frame):
                for cf in contacts_per_frame[t]:
                    if cf.hand == hand and cf.object_id == object_id:
                        distances.append(cf.distance)
                        break

        return sum(distances) / len(distances) if distances else 0.0

    def get_grasp_frames_per_object(
        self,
        events: List[ContactEvent],
    ) -> Dict[int, List[int]]:
        """Get frames where each object is being grasped.

        Args:
            events: List of contact events.

        Returns:
            Dict mapping object_id to list of grasp frame indices.
        """
        result: Dict[int, List[int]] = {}

        for event in events:
            if event.label == InteractionLabel.GRASP.value:
                if event.object_id not in result:
                    result[event.object_id] = []
                for t in range(event.frame_start, event.frame_end + 1):
                    if t not in result[event.object_id]:
                        result[event.object_id].append(t)

        return result
