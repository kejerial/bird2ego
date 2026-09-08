"""ActionSegmenter: segments actions based on event/state changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..utils.timeline import (
    ActionSegment,
    Timeline,
)

logger = logging.getLogger(__name__)


class SegmentTrigger(str, Enum):
    """Types of triggers that can start/end segments."""

    CONTACT_ONSET = "contact_onset"
    CONTACT_OFFSET = "contact_offset"
    SUPPORT_CHANGE = "support_change"
    MOTION_START = "motion_start"
    MOTION_STOP = "motion_stop"
    CONTAINMENT_CHANGE = "containment_change"
    IDLE_TIMEOUT = "idle_timeout"


@dataclass
class SegmentBoundary:
    """A segment boundary detected by the segmenter."""

    frame_idx: int
    trigger: SegmentTrigger
    objects_involved: List[int]
    confidence: float
    evidence: Dict


@dataclass
class ActionSegmenterConfig:
    """Configuration for action segmentation."""

    # Minimum segment duration (frames)
    min_segment_duration: int = 5

    # Idle detection
    idle_threshold_frames: int = 30  # frames without events to trigger idle

    # State change detection
    detect_contact_changes: bool = True
    detect_motion_changes: bool = True
    detect_support_changes: bool = True
    detect_containment_changes: bool = True


class ActionSegmenter:
    """Segments video into action segments based on event/state changes.

    Derives segment boundaries from:
    - Contact on/off transitions
    - Support_relation changes
    - Object motion state changes
    - Containment changes

    Deterministic and debuggable baseline implementation.
    """

    def __init__(self, config: Optional[ActionSegmenterConfig] = None):
        """Initialize ActionSegmenter.

        Args:
            config: Configuration for segmentation.
        """
        self.config = config or ActionSegmenterConfig()

    def segment(
        self,
        timeline: Timeline,
    ) -> List[SegmentBoundary]:
        """Detect segment boundaries from timeline data.

        Args:
            timeline: Timeline with events and object tracks.

        Returns:
            List of segment boundaries.
        """
        boundaries = []

        # Detect contact-based boundaries
        if self.config.detect_contact_changes:
            boundaries.extend(self._detect_contact_boundaries(timeline))

        # Detect state-based boundaries
        if self.config.detect_motion_changes:
            boundaries.extend(self._detect_motion_boundaries(timeline))

        if self.config.detect_support_changes:
            boundaries.extend(self._detect_support_boundaries(timeline))

        if self.config.detect_containment_changes:
            boundaries.extend(self._detect_containment_boundaries(timeline))

        # Sort by frame index
        boundaries.sort(key=lambda b: b.frame_idx)

        # Remove duplicates (same frame, merge triggers)
        boundaries = self._merge_duplicate_boundaries(boundaries)

        return boundaries

    def boundaries_to_segments(
        self,
        boundaries: List[SegmentBoundary],
        timeline: Timeline,
    ) -> List[ActionSegment]:
        """Convert boundaries to action segments.

        Args:
            boundaries: List of segment boundaries.
            timeline: Timeline for timestamp conversion.

        Returns:
            List of ActionSegment objects (without labels).
        """
        if not boundaries:
            # Single segment covering entire video
            if timeline.num_frames > 0:
                return [
                    ActionSegment(
                        seg_id=0,
                        t_start=timeline.frame_to_time(0),
                        t_end=timeline.frame_to_time(timeline.num_frames - 1),
                        frame_start=0,
                        frame_end=timeline.num_frames - 1,
                        label="unknown",
                        conf=0.5,
                        objects_involved=[],
                        evidence={"no_boundaries": True},
                    )
                ]
            return []

        segments = []
        seg_id = 0

        # Add segment before first boundary
        first_frame = boundaries[0].frame_idx
        if first_frame > 0:
            segments.append(
                ActionSegment(
                    seg_id=seg_id,
                    t_start=timeline.frame_to_time(0),
                    t_end=timeline.frame_to_time(first_frame - 1),
                    frame_start=0,
                    frame_end=first_frame - 1,
                    label="unknown",
                    conf=0.5,
                    objects_involved=[],
                    evidence={"pre_first_boundary": True},
                )
            )
            seg_id += 1

        # Add segments between boundaries
        for i in range(len(boundaries)):
            start_frame = boundaries[i].frame_idx
            if i + 1 < len(boundaries):
                end_frame = boundaries[i + 1].frame_idx - 1
            else:
                end_frame = timeline.num_frames - 1

            if end_frame - start_frame + 1 < self.config.min_segment_duration:
                continue

            # Collect objects from boundary
            objects = boundaries[i].objects_involved.copy()

            segments.append(
                ActionSegment(
                    seg_id=seg_id,
                    t_start=timeline.frame_to_time(start_frame),
                    t_end=timeline.frame_to_time(end_frame),
                    frame_start=start_frame,
                    frame_end=end_frame,
                    label="unknown",
                    conf=boundaries[i].confidence,
                    objects_involved=objects,
                    evidence={
                        "trigger": boundaries[i].trigger.value,
                        **boundaries[i].evidence,
                    },
                )
            )
            seg_id += 1

        return segments

    def _detect_contact_boundaries(
        self,
        timeline: Timeline,
    ) -> List[SegmentBoundary]:
        """Detect boundaries from contact events.

        Args:
            timeline: Timeline with events.

        Returns:
            List of segment boundaries.
        """
        boundaries = []

        for event in timeline.events:
            # Contact onset
            boundaries.append(
                SegmentBoundary(
                    frame_idx=event.frame_start,
                    trigger=SegmentTrigger.CONTACT_ONSET,
                    objects_involved=[event.object_id],
                    confidence=event.conf,
                    evidence={
                        "event_id": event.event_id,
                        "label": event.label,
                        "hand": event.hand.value,
                    },
                )
            )

            # Contact offset
            boundaries.append(
                SegmentBoundary(
                    frame_idx=event.frame_end + 1,
                    trigger=SegmentTrigger.CONTACT_OFFSET,
                    objects_involved=[event.object_id],
                    confidence=event.conf,
                    evidence={
                        "event_id": event.event_id,
                        "label": event.label,
                        "hand": event.hand.value,
                    },
                )
            )

        return boundaries

    def _detect_motion_boundaries(
        self,
        timeline: Timeline,
    ) -> List[SegmentBoundary]:
        """Detect boundaries from object motion state changes.

        Args:
            timeline: Timeline with object tracks.

        Returns:
            List of segment boundaries.
        """
        from ..utils.timeline import MotionState

        boundaries = []

        for obj_id, track in timeline.objects.items():
            for t in range(1, track.num_frames):
                prev_state = track.frames[t - 1].state.motion_state
                curr_state = track.frames[t].state.motion_state

                if prev_state != curr_state:
                    if curr_state == MotionState.MOVING:
                        trigger = SegmentTrigger.MOTION_START
                    else:
                        trigger = SegmentTrigger.MOTION_STOP

                    boundaries.append(
                        SegmentBoundary(
                            frame_idx=t,
                            trigger=trigger,
                            objects_involved=[obj_id],
                            confidence=0.7,
                            evidence={
                                "object_id": obj_id,
                                "prev_state": prev_state.value,
                                "curr_state": curr_state.value,
                            },
                        )
                    )

        return boundaries

    def _detect_support_boundaries(
        self,
        timeline: Timeline,
    ) -> List[SegmentBoundary]:
        """Detect boundaries from support relation changes.

        Args:
            timeline: Timeline with object tracks.

        Returns:
            List of segment boundaries.
        """
        boundaries = []

        for obj_id, track in timeline.objects.items():
            for t in range(1, track.num_frames):
                prev_support = track.frames[t - 1].state.support_relation
                curr_support = track.frames[t].state.support_relation

                if prev_support != curr_support:
                    boundaries.append(
                        SegmentBoundary(
                            frame_idx=t,
                            trigger=SegmentTrigger.SUPPORT_CHANGE,
                            objects_involved=[obj_id],
                            confidence=0.8,
                            evidence={
                                "object_id": obj_id,
                                "prev_support": prev_support.value,
                                "curr_support": curr_support.value,
                            },
                        )
                    )

        return boundaries

    def _detect_containment_boundaries(
        self,
        timeline: Timeline,
    ) -> List[SegmentBoundary]:
        """Detect boundaries from containment state changes.

        Args:
            timeline: Timeline with object tracks.

        Returns:
            List of segment boundaries.
        """
        boundaries = []

        for obj_id, track in timeline.objects.items():
            for t in range(1, track.num_frames):
                prev_cont = track.frames[t - 1].state.containment
                curr_cont = track.frames[t].state.containment

                if prev_cont != curr_cont:
                    boundaries.append(
                        SegmentBoundary(
                            frame_idx=t,
                            trigger=SegmentTrigger.CONTAINMENT_CHANGE,
                            objects_involved=[obj_id],
                            confidence=0.85,
                            evidence={
                                "object_id": obj_id,
                                "prev_containment": prev_cont.value,
                                "curr_containment": curr_cont.value,
                            },
                        )
                    )

        return boundaries

    def _merge_duplicate_boundaries(
        self,
        boundaries: List[SegmentBoundary],
    ) -> List[SegmentBoundary]:
        """Merge boundaries at the same frame.

        Args:
            boundaries: List of boundaries sorted by frame.

        Returns:
            Merged list of boundaries.
        """
        if not boundaries:
            return []

        merged = []
        current = boundaries[0]

        for b in boundaries[1:]:
            if b.frame_idx == current.frame_idx:
                # Merge: combine objects, keep highest confidence
                objects = list(set(current.objects_involved + b.objects_involved))
                evidence = {**current.evidence, **b.evidence}
                current = SegmentBoundary(
                    frame_idx=current.frame_idx,
                    trigger=current.trigger,  # Keep first trigger
                    objects_involved=objects,
                    confidence=max(current.confidence, b.confidence),
                    evidence=evidence,
                )
            else:
                merged.append(current)
                current = b

        merged.append(current)
        return merged
