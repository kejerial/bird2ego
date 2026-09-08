"""Tests for graph export validity."""
import json
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bird2ego.graph import (
    EdgeType,
    InferredEdge,
    OrderingInferencer,
    PostconditionExtractor,
    PreconditionExtractor,
    TaskGraphBuilder,
)
from bird2ego.output import GraphExporter
from bird2ego.utils.timeline import ActionSegment


class TestTaskGraphBuilder:
    """Tests for TaskGraphBuilder."""

    def test_build_empty_graph(self):
        """Test building graph with no segments."""
        builder = TaskGraphBuilder()
        graph = builder.build([], [])

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_build_graph_with_segments(self):
        """Test building graph with segments."""
        segments = [
            ActionSegment(
                seg_id=0, t_start=0.0, t_end=1.0,
                frame_start=0, frame_end=30,
                label="pick", conf=0.9, objects_involved=[1],
            ),
            ActionSegment(
                seg_id=1, t_start=1.0, t_end=2.0,
                frame_start=30, frame_end=60,
                label="place", conf=0.8, objects_involved=[1],
            ),
        ]

        edges = [
            InferredEdge(
                from_seg_id=0, to_seg_id=1,
                edge_type=EdgeType.CAUSAL,
                confidence=0.85,
                evidence={"test": True},
            ),
        ]

        builder = TaskGraphBuilder()
        graph = builder.build(segments, edges)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1

    def test_node_attributes(self):
        """Test that nodes have correct attributes."""
        segments = [
            ActionSegment(
                seg_id=0, t_start=0.0, t_end=1.0,
                frame_start=0, frame_end=30,
                label="pick", conf=0.9, objects_involved=[1, 2],
            ),
        ]

        builder = TaskGraphBuilder()
        graph = builder.build(segments, [])

        node_data = graph.nodes[0]
        assert node_data["node_id"] == 0
        assert node_data["label"] == "pick"
        assert node_data["t_start"] == 0.0
        assert node_data["t_end"] == 1.0
        assert node_data["objects"] == [1, 2]
        assert node_data["conf"] == 0.9

    def test_edge_attributes(self):
        """Test that edges have correct attributes."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
            ActionSegment(seg_id=1, t_start=1.0, t_end=2.0,
                         frame_start=30, frame_end=60, label="place", conf=0.8),
        ]

        edges = [
            InferredEdge(
                from_seg_id=0, to_seg_id=1,
                edge_type=EdgeType.CAUSAL,
                confidence=0.85,
                evidence={"reason": "test"},
            ),
        ]

        builder = TaskGraphBuilder()
        graph = builder.build(segments, edges)

        edge_data = graph.edges[0, 1]
        assert edge_data["edge_type"] == "causal"
        assert edge_data["conf"] == 0.85
        assert "reason" in edge_data["evidence"]


class TestGraphExport:
    """Tests for graph export functionality."""

    def test_export_json(self):
        """Test JSON export."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
            ActionSegment(seg_id=1, t_start=1.0, t_end=2.0,
                         frame_start=30, frame_end=60, label="place", conf=0.8),
        ]

        edges = [
            InferredEdge(
                from_seg_id=0, to_seg_id=1,
                edge_type=EdgeType.TEMPORAL,
                confidence=0.5,
                evidence={},
            ),
        ]

        builder = TaskGraphBuilder()
        builder.build(segments, edges)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = GraphExporter(output_dir=tmpdir)
            json_path = exporter.export_json(builder)

            # Verify file exists
            assert Path(json_path).exists()

            # Load and verify content
            with open(json_path) as f:
                data = json.load(f)

            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) == 2
            assert len(data["edges"]) == 1

    def test_export_graphml(self):
        """Test GraphML export."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
        ]

        builder = TaskGraphBuilder()
        builder.build(segments, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = GraphExporter(output_dir=tmpdir)
            graphml_path = exporter.export_graphml(builder)

            # Verify file exists
            assert Path(graphml_path).exists()

            # Verify it's valid GraphML
            loaded = nx.read_graphml(graphml_path)
            assert loaded.number_of_nodes() == 1

    def test_load_json(self):
        """Test loading graph from JSON."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
            ActionSegment(seg_id=1, t_start=1.0, t_end=2.0,
                         frame_start=30, frame_end=60, label="place", conf=0.8),
        ]

        builder = TaskGraphBuilder()
        builder.build(segments, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = GraphExporter(output_dir=tmpdir)
            json_path = exporter.export_json(builder)

            # Load and verify
            loaded = exporter.load_json(json_path)
            assert loaded.number_of_nodes() == 2


class TestNodeCountEqualsSegments:
    """Test that node count equals segment count."""

    def test_node_count_matches_segments(self):
        """Verify graph node count == segment count."""
        # Create various numbers of segments
        for n in [1, 5, 10, 20]:
            segments = [
                ActionSegment(
                    seg_id=i,
                    t_start=i * 1.0,
                    t_end=(i + 1) * 1.0,
                    frame_start=i * 30,
                    frame_end=(i + 1) * 30,
                    label="action",
                    conf=0.8,
                )
                for i in range(n)
            ]

            builder = TaskGraphBuilder()
            graph = builder.build(segments, [])

            assert graph.number_of_nodes() == len(segments), \
                f"Expected {len(segments)} nodes, got {graph.number_of_nodes()}"

    def test_validation_catches_mismatch(self):
        """Test that validation detects node count mismatch."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
            ActionSegment(seg_id=1, t_start=1.0, t_end=2.0,
                         frame_start=30, frame_end=60, label="place", conf=0.8),
        ]

        builder = TaskGraphBuilder()
        # Build with only one segment
        builder.build(segments[:1], [])

        # Validate against full segments list
        warnings = builder.validate(segments)
        assert any("count" in w.lower() for w in warnings)


class TestEdgeTypes:
    """Tests for edge type handling."""

    def test_causal_edges(self):
        """Test causal edge creation."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
            ActionSegment(seg_id=1, t_start=1.0, t_end=2.0,
                         frame_start=30, frame_end=60, label="place", conf=0.8),
        ]

        edges = [
            InferredEdge(
                from_seg_id=0, to_seg_id=1,
                edge_type=EdgeType.CAUSAL,
                confidence=0.9,
                evidence={},
            ),
        ]

        builder = TaskGraphBuilder()
        graph = builder.build(segments, edges)

        edge_data = graph.edges[0, 1]
        assert edge_data["edge_type"] == "causal"

    def test_temporal_edges(self):
        """Test temporal edge creation."""
        segments = [
            ActionSegment(seg_id=0, t_start=0.0, t_end=1.0,
                         frame_start=0, frame_end=30, label="pick", conf=0.9),
            ActionSegment(seg_id=1, t_start=1.0, t_end=2.0,
                         frame_start=30, frame_end=60, label="place", conf=0.8),
        ]

        edges = [
            InferredEdge(
                from_seg_id=0, to_seg_id=1,
                edge_type=EdgeType.TEMPORAL,
                confidence=0.5,
                evidence={},
            ),
        ]

        builder = TaskGraphBuilder()
        graph = builder.build(segments, edges)

        edge_data = graph.edges[0, 1]
        assert edge_data["edge_type"] == "temporal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
