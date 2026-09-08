"""Tests for timeline timestamp validity."""

import numpy as np
import pytest

# Add src to path
from bird2ego.utils.timeline import (
    create_empty_object_track,
    create_empty_person_pose,
    create_empty_timeline,
)
from bird2ego.video.temporal_alignment import TemporalAlignment


class TestTimelineCreation:
    """Tests for timeline creation."""

    def test_create_empty_timeline(self):
        """Test creating an empty timeline."""
        timeline = create_empty_timeline(
            num_frames=100,
            fps=30.0,
            width=640,
            height=480,
            t0=0.0,
        )

        assert timeline.num_frames == 100
        assert timeline.fps_extracted == 30.0
        assert timeline.t0 == 0.0
        assert timeline.timestamp_source == "synthetic"

    def test_timeline_duration(self):
        """Test timeline duration calculation."""
        timeline = create_empty_timeline(
            num_frames=90,
            fps=30.0,
            width=640,
            height=480,
        )

        # 90 frames at 30fps = 3 seconds (from t=0 to t=89/30)
        expected_duration = (90 - 1) / 30.0
        assert abs(timeline.duration - expected_duration) < 0.001


class TestTimestampMonotonicity:
    """Tests for timestamp monotonicity."""

    def test_timestamps_monotonic(self):
        """Test that timestamps are strictly monotonic."""
        timeline = create_empty_timeline(
            num_frames=100,
            fps=30.0,
            width=640,
            height=480,
        )

        timestamps = timeline.get_timestamps()
        diffs = np.diff(timestamps)
        assert np.all(diffs > 0), "Timestamps must be strictly increasing"

    def test_timestamps_consistent_with_fps(self):
        """Test that timestamps are consistent with fps."""
        fps = 30.0
        timeline = create_empty_timeline(
            num_frames=100,
            fps=fps,
            width=640,
            height=480,
        )

        timestamps = timeline.get_timestamps()
        expected_dt = 1.0 / fps

        diffs = np.diff(timestamps)
        assert np.allclose(diffs, expected_dt), "Timestamps should be evenly spaced"

    def test_frame_indices_contiguous(self):
        """Test that frame indices are contiguous 0..T-1."""
        timeline = create_empty_timeline(
            num_frames=50,
            fps=30.0,
            width=640,
            height=480,
        )

        indices = timeline.get_frame_indices()
        expected = list(range(50))
        assert indices == expected


class TestTimelineAlignment:
    """Tests for timeline alignment across modules."""

    def test_pose_alignment(self):
        """Test that pose is aligned to timeline."""
        T = 100
        timeline = create_empty_timeline(
            num_frames=T,
            fps=30.0,
            width=640,
            height=480,
        )

        pose = create_empty_person_pose(T)
        timeline.person_pose = pose

        warnings = timeline.validate_alignment()
        assert not any("Pose" in w for w in warnings)

    def test_object_alignment(self):
        """Test that objects are aligned to timeline."""
        T = 100
        timeline = create_empty_timeline(
            num_frames=T,
            fps=30.0,
            width=640,
            height=480,
        )

        track = create_empty_object_track(
            object_id=0,
            class_name="box",
            class_id=0,
            num_frames=T,
        )
        timeline.objects[0] = track

        warnings = timeline.validate_alignment()
        assert not any("Object" in w for w in warnings)

    def test_misaligned_pose_detected(self):
        """Test that misaligned pose is detected."""
        T = 100
        timeline = create_empty_timeline(
            num_frames=T,
            fps=30.0,
            width=640,
            height=480,
        )

        # Create pose with wrong number of frames
        pose = create_empty_person_pose(T - 10)
        timeline.person_pose = pose

        warnings = timeline.validate_alignment()
        assert any("Pose" in w for w in warnings)


class TestTemporalAlignmentModule:
    """Tests for TemporalAlignment module."""

    def test_is_monotonic(self):
        """Test monotonicity checker."""
        aligner = TemporalAlignment()

        # Monotonic
        assert aligner.is_monotonic([0.0, 0.1, 0.2, 0.3])

        # Not monotonic
        assert not aligner.is_monotonic([0.0, 0.1, 0.05, 0.3])

    def test_fix_monotonicity(self):
        """Test monotonicity fixer."""
        aligner = TemporalAlignment()

        # Non-monotonic timestamps
        bad_ts = [0.0, 0.1, 0.05, 0.3, 0.25, 0.4]
        fixed_ts = aligner.fix_monotonicity(bad_ts)

        # Should be monotonic after fixing
        assert aligner.is_monotonic(fixed_ts)

    def test_validate_timeline(self):
        """Test timeline validation."""
        aligner = TemporalAlignment()

        timeline = create_empty_timeline(
            num_frames=100,
            fps=30.0,
            width=640,
            height=480,
        )

        valid, warnings = aligner.validate_timeline(timeline)
        assert valid
        assert len(warnings) == 0


class TestFrameTimeConversion:
    """Tests for frame-to-time and time-to-frame conversion."""

    def test_frame_to_time(self):
        """Test frame index to timestamp conversion."""
        timeline = create_empty_timeline(
            num_frames=100,
            fps=30.0,
            width=640,
            height=480,
        )

        # Frame 0 should be at t=0
        assert timeline.frame_to_time(0) == 0.0

        # Frame 30 should be at t=1.0
        assert abs(timeline.frame_to_time(30) - 1.0) < 0.001

    def test_time_to_frame(self):
        """Test timestamp to frame index conversion."""
        timeline = create_empty_timeline(
            num_frames=100,
            fps=30.0,
            width=640,
            height=480,
        )

        # t=0 should be frame 0
        assert timeline.time_to_frame(0.0) == 0

        # t=1.0 should be frame 30
        assert timeline.time_to_frame(1.0) == 30

    def test_roundtrip_conversion(self):
        """Test that frame->time->frame roundtrip is consistent."""
        timeline = create_empty_timeline(
            num_frames=100,
            fps=30.0,
            width=640,
            height=480,
        )

        for frame_idx in [0, 15, 30, 45, 60, 75, 90]:
            t = timeline.frame_to_time(frame_idx)
            recovered_idx = timeline.time_to_frame(t)
            assert recovered_idx == frame_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
