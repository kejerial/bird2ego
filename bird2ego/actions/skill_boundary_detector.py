"""SkillBoundaryDetector: merges and refines segment boundaries."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from ..utils.timeline import ActionSegment, Timeline
from .action_segmenter import SegmentBoundary

logger = logging.getLogger(__name__)


@dataclass
class SkillBoundaryConfig:
    """Configuration for skill boundary detection."""
    # Minimum segment duration (frames)
    min_duration_frames: int = 10

    # Merge segments shorter than this (frames)
    merge_threshold_frames: int = 5

    # Hysteresis for boundary stability (frames)
    hysteresis_frames: int = 3

    # Maximum segments to produce
    max_segments: Optional[int] = None


class SkillBoundaryDetector:
    """Refines segment boundaries for stability.

    Applies:
    - Minimum duration enforcement
    - Merging of short segments
    - Hysteresis for boundary stability
    """

    def __init__(self, config: Optional[SkillBoundaryConfig] = None):
        """Initialize SkillBoundaryDetector.

        Args:
            config: Configuration for boundary detection.
        """
        self.config = config or SkillBoundaryConfig()

    def refine_segments(
        self,
        segments: List[ActionSegment],
        timeline: Timeline,
    ) -> List[ActionSegment]:
        """Refine action segments.

        Args:
            segments: List of raw segments.
            timeline: Timeline for context.

        Returns:
            Refined list of segments.
        """
        if not segments:
            return []

        # Apply minimum duration
        segments = self._enforce_min_duration(segments, timeline)

        # Merge short segments
        segments = self._merge_short_segments(segments, timeline)

        # Apply hysteresis
        segments = self._apply_hysteresis(segments)

        # Limit max segments if configured
        if self.config.max_segments and len(segments) > self.config.max_segments:
            segments = self._limit_segments(segments)

        # Reassign segment IDs
        for i, seg in enumerate(segments):
            seg.seg_id = i

        return segments

    def refine_boundaries(
        self,
        boundaries: List[SegmentBoundary],
        timeline: Timeline,
    ) -> List[SegmentBoundary]:
        """Refine segment boundaries.

        Args:
            boundaries: List of raw boundaries.
            timeline: Timeline for context.

        Returns:
            Refined list of boundaries.
        """
        if not boundaries:
            return []

        # Remove boundaries too close together
        refined = [boundaries[0]]

        for b in boundaries[1:]:
            last = refined[-1]
            gap = b.frame_idx - last.frame_idx

            if gap >= self.config.hysteresis_frames:
                refined.append(b)
            else:
                # Keep the higher confidence one
                if b.confidence > last.confidence:
                    refined[-1] = b

        return refined

    def _enforce_min_duration(
        self,
        segments: List[ActionSegment],
        timeline: Timeline,
    ) -> List[ActionSegment]:
        """Remove segments shorter than minimum duration.

        Args:
            segments: List of segments.
            timeline: Timeline for context.

        Returns:
            Filtered list of segments.
        """
        return [
            seg for seg in segments
            if seg.frame_end - seg.frame_start + 1 >= self.config.min_duration_frames
        ]

    def _merge_short_segments(
        self,
        segments: List[ActionSegment],
        timeline: Timeline,
    ) -> List[ActionSegment]:
        """Merge segments shorter than merge threshold with neighbors.

        Args:
            segments: List of segments.
            timeline: Timeline for context.

        Returns:
            Merged list of segments.
        """
        if len(segments) <= 1:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            seg = segments[i]
            duration = seg.frame_end - seg.frame_start + 1

            if duration < self.config.merge_threshold_frames and i > 0:
                # Merge with previous segment
                prev = merged[-1]
                merged[-1] = ActionSegment(
                    seg_id=prev.seg_id,
                    t_start=prev.t_start,
                    t_end=seg.t_end,
                    frame_start=prev.frame_start,
                    frame_end=seg.frame_end,
                    label=prev.label,  # Keep previous label
                    conf=(prev.conf + seg.conf) / 2,
                    objects_involved=list(set(prev.objects_involved + seg.objects_involved)),
                    evidence={**prev.evidence, "merged_with": seg.seg_id},
                )
            elif duration < self.config.merge_threshold_frames and i < len(segments) - 1:
                # Merge with next segment
                next_seg = segments[i + 1]
                merged.append(
                    ActionSegment(
                        seg_id=seg.seg_id,
                        t_start=seg.t_start,
                        t_end=next_seg.t_end,
                        frame_start=seg.frame_start,
                        frame_end=next_seg.frame_end,
                        label=next_seg.label,  # Use next label
                        conf=(seg.conf + next_seg.conf) / 2,
                        objects_involved=list(set(seg.objects_involved + next_seg.objects_involved)),
                        evidence={**next_seg.evidence, "merged_with": seg.seg_id},
                    )
                )
                i += 1  # Skip next segment
            else:
                merged.append(seg)

            i += 1

        return merged

    def _apply_hysteresis(
        self,
        segments: List[ActionSegment],
    ) -> List[ActionSegment]:
        """Apply hysteresis to segment boundaries.

        Shifts boundaries slightly for stability.

        Args:
            segments: List of segments.

        Returns:
            Segments with adjusted boundaries.
        """
        if len(segments) <= 1:
            return segments

        # Ensure no gaps or overlaps between adjacent segments
        for i in range(len(segments) - 1):
            curr = segments[i]
            next_seg = segments[i + 1]

            gap = next_seg.frame_start - curr.frame_end - 1

            if gap != 0:
                # Adjust to remove gap/overlap
                mid = (curr.frame_end + next_seg.frame_start) // 2
                segments[i].frame_end = mid
                segments[i].t_end = segments[i].t_start + (mid - segments[i].frame_start) / 30.0  # Approximate
                segments[i + 1].frame_start = mid + 1
                segments[i + 1].t_start = segments[i].t_end

        return segments

    def _limit_segments(
        self,
        segments: List[ActionSegment],
    ) -> List[ActionSegment]:
        """Limit number of segments by merging lowest confidence ones.

        Args:
            segments: List of segments.

        Returns:
            Limited list of segments.
        """
        while len(segments) > self.config.max_segments:
            # Find lowest confidence segment
            min_conf = float('inf')
            min_idx = 0

            for i, seg in enumerate(segments):
                if seg.conf < min_conf:
                    min_conf = seg.conf
                    min_idx = i

            # Merge with neighbor
            if min_idx == 0:
                merge_with = 1
            elif min_idx == len(segments) - 1:
                merge_with = len(segments) - 2
            else:
                # Merge with lower confidence neighbor
                if segments[min_idx - 1].conf < segments[min_idx + 1].conf:
                    merge_with = min_idx - 1
                else:
                    merge_with = min_idx + 1

            # Perform merge
            idx1 = min(min_idx, merge_with)
            idx2 = max(min_idx, merge_with)

            seg1 = segments[idx1]
            seg2 = segments[idx2]

            merged = ActionSegment(
                seg_id=seg1.seg_id,
                t_start=seg1.t_start,
                t_end=seg2.t_end,
                frame_start=seg1.frame_start,
                frame_end=seg2.frame_end,
                label=seg1.label if seg1.conf > seg2.conf else seg2.label,
                conf=(seg1.conf + seg2.conf) / 2,
                objects_involved=list(set(seg1.objects_involved + seg2.objects_involved)),
                evidence={**seg1.evidence, "merged_for_limit": True},
            )

            segments = segments[:idx1] + [merged] + segments[idx2 + 1:]

        return segments
