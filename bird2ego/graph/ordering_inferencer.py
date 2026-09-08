"""OrderingInferencer: infers causal and temporal ordering between segments."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..utils.timeline import ActionSegment, ObjectState, Timeline
from .precondition_extractor import ObjectPrecondition, SegmentPreconditions
from .postcondition_extractor import SegmentPostconditions, StateChange

logger = logging.getLogger(__name__)


class EdgeType(str, Enum):
    """Types of edges in the task graph."""
    CAUSAL = "causal"  # A's postcondition satisfies B's precondition
    TEMPORAL = "temporal"  # A comes before B but no causal link found


@dataclass
class InferredEdge:
    """An inferred edge between two segments."""
    from_seg_id: int
    to_seg_id: int
    edge_type: EdgeType
    confidence: float
    evidence: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "from": self.from_seg_id,
            "to": self.to_seg_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class OrderingInferencer:
    """Infers ordering relationships between action segments.

    Creates edges:
    - CAUSAL: if A's postcondition satisfies B's precondition (state match)
    - TEMPORAL: if A comes before B but no causal explanation exists

    Causal edges take precedence over temporal edges.
    """

    def __init__(
        self,
        require_shared_objects: bool = True,
        min_causal_confidence: float = 0.5,
        add_temporal_fallback: bool = True,
    ):
        """Initialize OrderingInferencer.

        Args:
            require_shared_objects: If True, only consider pairs with shared objects.
            min_causal_confidence: Minimum confidence for causal edges.
            add_temporal_fallback: If True, add temporal edges when no causal found.
        """
        self.require_shared_objects = require_shared_objects
        self.min_causal_confidence = min_causal_confidence
        self.add_temporal_fallback = add_temporal_fallback

    def infer_edges(
        self,
        segments: List[ActionSegment],
        preconditions: Dict[int, SegmentPreconditions],
        postconditions: Dict[int, SegmentPostconditions],
        timeline: Timeline,
    ) -> List[InferredEdge]:
        """Infer all edges between segments.

        Args:
            segments: List of action segments.
            preconditions: Preconditions for each segment.
            postconditions: Postconditions for each segment.
            timeline: Timeline context.

        Returns:
            List of InferredEdge objects.
        """
        edges = []

        # Sort segments by time
        sorted_segments = sorted(segments, key=lambda s: s.frame_start)

        # Track which pairs have causal edges
        causal_pairs: Set[Tuple[int, int]] = set()

        # Check all pairs where A comes before B
        for i, seg_a in enumerate(sorted_segments):
            for seg_b in sorted_segments[i + 1:]:
                # Skip if no overlap in time (A must end before B starts)
                if seg_a.frame_end >= seg_b.frame_start:
                    continue

                # Check if they share objects
                shared_objects = set(seg_a.objects_involved) & set(seg_b.objects_involved)
                if self.require_shared_objects and not shared_objects:
                    continue

                # Try to find causal relationship
                causal_edge = self._find_causal_edge(
                    seg_a, seg_b,
                    preconditions.get(seg_a.seg_id),
                    postconditions.get(seg_a.seg_id),
                    preconditions.get(seg_b.seg_id),
                    shared_objects,
                )

                if causal_edge and causal_edge.confidence >= self.min_causal_confidence:
                    edges.append(causal_edge)
                    causal_pairs.add((seg_a.seg_id, seg_b.seg_id))

        # Add temporal edges for adjacent segments without causal edges
        if self.add_temporal_fallback:
            for i in range(len(sorted_segments) - 1):
                seg_a = sorted_segments[i]
                seg_b = sorted_segments[i + 1]

                if (seg_a.seg_id, seg_b.seg_id) not in causal_pairs:
                    temporal_edge = InferredEdge(
                        from_seg_id=seg_a.seg_id,
                        to_seg_id=seg_b.seg_id,
                        edge_type=EdgeType.TEMPORAL,
                        confidence=0.5,
                        evidence={
                            "reason": "temporal_adjacency",
                            "gap_frames": seg_b.frame_start - seg_a.frame_end,
                        },
                    )
                    edges.append(temporal_edge)

        return edges

    def _find_causal_edge(
        self,
        seg_a: ActionSegment,
        seg_b: ActionSegment,
        precond_a: Optional[SegmentPreconditions],
        postcond_a: Optional[SegmentPostconditions],
        precond_b: Optional[SegmentPreconditions],
        shared_objects: Set[int],
    ) -> Optional[InferredEdge]:
        """Try to find a causal edge from A to B.

        Args:
            seg_a: First segment.
            seg_b: Second segment.
            precond_a: A's preconditions.
            postcond_a: A's postconditions.
            precond_b: B's preconditions.
            shared_objects: Objects involved in both segments.

        Returns:
            InferredEdge if causal link found, None otherwise.
        """
        if postcond_a is None or precond_b is None:
            return None

        # Get A's final states
        a_final_states = postcond_a.get_final_states()

        # Get B's required states
        b_precond_map = {
            p.object_id: p for p in precond_b.object_preconditions
        }

        # Check if A's effects enable B's preconditions
        matched_objects = []
        total_confidence = 0.0

        for obj_id in shared_objects:
            if obj_id not in a_final_states or obj_id not in b_precond_map:
                continue

            final_state = a_final_states[obj_id]
            required = b_precond_map[obj_id]

            if required.matches(final_state):
                matched_objects.append(obj_id)
                total_confidence += 0.8  # Base confidence for match

        if not matched_objects:
            return None

        # Also check if A produces changes that B requires
        a_changes = {
            p.object_id: p.changes
            for p in postcond_a.object_postconditions
        }

        for obj_id in matched_objects:
            if obj_id in a_changes and a_changes[obj_id]:
                total_confidence += 0.1  # Bonus for explicit state change

        confidence = min(1.0, total_confidence / max(1, len(shared_objects)))

        return InferredEdge(
            from_seg_id=seg_a.seg_id,
            to_seg_id=seg_b.seg_id,
            edge_type=EdgeType.CAUSAL,
            confidence=confidence,
            evidence={
                "matched_objects": matched_objects,
                "shared_objects": list(shared_objects),
                "a_label": seg_a.label,
                "b_label": seg_b.label,
            },
        )

    def get_predecessors(
        self,
        seg_id: int,
        edges: List[InferredEdge],
    ) -> List[int]:
        """Get predecessor segment IDs.

        Args:
            seg_id: Target segment ID.
            edges: List of edges.

        Returns:
            List of predecessor segment IDs.
        """
        return [e.from_seg_id for e in edges if e.to_seg_id == seg_id]

    def get_successors(
        self,
        seg_id: int,
        edges: List[InferredEdge],
    ) -> List[int]:
        """Get successor segment IDs.

        Args:
            seg_id: Source segment ID.
            edges: List of edges.

        Returns:
            List of successor segment IDs.
        """
        return [e.to_seg_id for e in edges if e.from_seg_id == seg_id]

    def filter_edges_by_type(
        self,
        edges: List[InferredEdge],
        edge_type: EdgeType,
    ) -> List[InferredEdge]:
        """Filter edges by type.

        Args:
            edges: List of edges.
            edge_type: Type to filter for.

        Returns:
            Filtered list of edges.
        """
        return [e for e in edges if e.edge_type == edge_type]
