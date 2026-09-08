"""Tests for the egocentric transform and the egocentric pipeline stage."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.egocentric import EgocentricStage, fit_image_to_pose_3d, summarize_series
from src.egocentric.transformer import EgocentricTransformer
from src.pipeline import PipelineOrchestrator
from src.utils import JOINT_IDX, NUM_JOINTS, PipelineConfig

# Body measurements of the synthetic subject, in metres.
SHOULDER_HALF_WIDTH = 0.2
HEAD_HEIGHT = 0.3


def make_pose(facing: str) -> tuple:
    """Build a synthetic upright body pose with a known facing direction.

    The frame is the pipeline 3D frame: x right, y down, z forward. The
    subject stands upright, so the shoulder line fixes the facing.

    Args:
        facing: "plus_z" or "plus_x".

    Returns:
        Tuple of (coords_3d (17, 3), conf (17,)).
    """
    coords = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    conf = np.ones(NUM_JOINTS, dtype=np.float32)

    if facing == "plus_z":
        left_shoulder = np.array([-SHOULDER_HALF_WIDTH, 0.0, 0.0])
        right_shoulder = np.array([SHOULDER_HALF_WIDTH, 0.0, 0.0])
    elif facing == "plus_x":
        left_shoulder = np.array([0.0, 0.0, SHOULDER_HALF_WIDTH])
        right_shoulder = np.array([0.0, 0.0, -SHOULDER_HALF_WIDTH])
    else:
        raise ValueError(facing)

    coords[JOINT_IDX["left_shoulder"]] = left_shoulder
    coords[JOINT_IDX["right_shoulder"]] = right_shoulder

    # Eyes sit HEAD_HEIGHT above the shoulder midpoint. Negative y is up.
    head_center = np.array([0.0, -HEAD_HEIGHT, 0.0])
    eye_offset = (right_shoulder - left_shoulder) * 0.1
    coords[JOINT_IDX["left_eye"]] = head_center - eye_offset
    coords[JOINT_IDX["right_eye"]] = head_center + eye_offset

    # Ears are not used: the shoulders already define the facing.
    conf[JOINT_IDX["left_ear"]] = 0.0
    conf[JOINT_IDX["right_ear"]] = 0.0
    conf[JOINT_IDX["nose"]] = 0.0

    return coords, conf


def make_hand(center: np.ndarray, spread: float = 0.02) -> np.ndarray:
    """Build 21 hand landmarks spread around a centre point."""
    hand = np.tile(center.astype(np.float32), (21, 1))
    hand[:, 0] += np.arange(21, dtype=np.float32) * spread
    return hand


class TestEgocentricTransform:
    """The ego frame is x=right, y=down, z=forward, origin at the eye."""

    @staticmethod
    def _transformer() -> EgocentricTransformer:
        """A transformer with no gaze tilt and no eye offset."""
        return EgocentricTransformer(
            eye_offset_forward=0.0, default_gaze_down_angle=0.0
        )

    def test_head_frame_for_subject_facing_plus_z(self):
        """A subject facing +z gets the identity rotation."""
        coords, conf = make_pose("plus_z")
        eye, forward, up, right, head_conf = self._transformer().estimate_head_pose(
            coords, conf
        )

        assert np.allclose(eye, [0.0, -HEAD_HEIGHT, 0.0], atol=1e-6)
        assert np.allclose(forward, [0.0, 0.0, 1.0], atol=1e-6)
        assert np.allclose(up, [0.0, -1.0, 0.0], atol=1e-6)
        assert np.allclose(right, [1.0, 0.0, 0.0], atol=1e-6)
        assert head_conf > 0.0

    def test_points_translate_only_when_facing_plus_z(self):
        """With an identity rotation the transform is a translation."""
        coords, conf = make_pose("plus_z")
        transformer = self._transformer()
        eye, forward, up, right, _ = transformer.estimate_head_pose(coords, conf)

        world = np.array([[0.3, 0.1, 0.5], [-0.2, -0.3, 0.0]], dtype=np.float32)
        ego = transformer.transform_to_egocentric(world, eye, forward, up, right)

        assert np.allclose(ego, world - eye, atol=1e-6)
        # A point 0.4 m below the eye reads as y = +0.4, because y is down.
        assert ego[0][1] == pytest.approx(0.4, abs=1e-6)

    def test_facing_plus_x_rotates_the_axes(self):
        """A subject facing +x maps world +x onto ego forward."""
        coords, conf = make_pose("plus_x")
        transformer = self._transformer()
        eye, forward, up, right, _ = transformer.estimate_head_pose(coords, conf)

        assert np.allclose(forward, [1.0, 0.0, 0.0], atol=1e-6)

        # One metre straight ahead of the subject.
        world = (eye + np.array([1.0, 0.0, 0.0])).reshape(1, 3)
        ego = transformer.transform_to_egocentric(world, eye, forward, up, right)
        assert np.allclose(ego[0], [0.0, 0.0, 1.0], atol=1e-6)

        # One metre to the subject's right is world -z.
        world = (eye + np.array([0.0, 0.0, -1.0])).reshape(1, 3)
        ego = transformer.transform_to_egocentric(world, eye, forward, up, right)
        assert np.allclose(ego[0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_process_frame_places_hands_and_objects(self):
        """Hands and objects land at their world offset from the eye."""
        coords, conf = make_pose("plus_z")
        transformer = self._transformer()

        hand_center = np.array([0.25, 0.1, 0.45], dtype=np.float32)
        object_center = np.array([0.0, 0.0, 0.8], dtype=np.float32)

        ego = transformer.process_frame(
            frame_idx=7,
            timestamp=0.25,
            pose_3d=coords,
            pose_conf=conf,
            left_hand_3d=None,
            left_hand_conf=0.0,
            right_hand_3d=make_hand(hand_center),
            right_hand_conf=0.9,
            objects={3: object_center},
        )

        assert ego.frame_idx == 7
        assert ego.timestamp == pytest.approx(0.25)
        assert ego.left_hand_ego is None
        assert ego.right_hand_ego is not None
        assert ego.right_hand_ego.shape == (21, 3)

        eye = np.array([0.0, -HEAD_HEIGHT, 0.0], dtype=np.float32)
        assert np.allclose(ego.right_hand_ego[0], hand_center - eye, atol=1e-6)
        assert np.allclose(ego.objects_ego[3], object_center - eye, atol=1e-6)

    def test_pinch_uses_thumb_and_index_tips(self):
        """A closed thumb-index pair reads as a pinch."""
        coords, conf = make_pose("plus_z")
        transformer = EgocentricTransformer(
            eye_offset_forward=0.0, default_gaze_down_angle=0.0, pinch_threshold=0.05
        )

        hand = np.zeros((21, 3), dtype=np.float32)
        hand[4] = [0.30, 0.00, 0.40]  # thumb tip
        hand[8] = [0.31, 0.00, 0.40]  # index tip, 1 cm away

        ego = transformer.process_frame(
            frame_idx=0,
            timestamp=0.0,
            pose_3d=coords,
            pose_conf=conf,
            left_hand_3d=None,
            left_hand_conf=0.0,
            right_hand_3d=hand,
            right_hand_conf=0.9,
        )
        assert ego.right_hand_pinching is True
        assert ego.right_pinch_distance == pytest.approx(0.01, abs=1e-6)

        hand[8] = [0.50, 0.00, 0.40]  # 20 cm away
        ego = transformer.process_frame(
            frame_idx=1,
            timestamp=0.0,
            pose_3d=coords,
            pose_conf=conf,
            left_hand_3d=None,
            left_hand_conf=0.0,
            right_hand_3d=hand,
            right_hand_conf=0.9,
        )
        assert ego.right_hand_pinching is False


class TestImageToPoseFit:
    """The stage maps object boxes into the 3D pose frame."""

    def test_recovers_a_known_similarity(self):
        """The fit recovers the scale and offset used to build the data."""
        rng = np.random.default_rng(0)
        keypoints_2d = rng.uniform(50, 600, size=(NUM_JOINTS, 2)).astype(np.float32)

        scale, tx, ty = 0.004, -1.2, -0.8
        coords_3d = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        coords_3d[:, 0] = scale * keypoints_2d[:, 0] + tx
        coords_3d[:, 1] = scale * keypoints_2d[:, 1] + ty
        conf = np.ones(NUM_JOINTS, dtype=np.float32)

        fit = fit_image_to_pose_3d(keypoints_2d, coords_3d, conf)
        assert fit is not None
        assert fit.scale == pytest.approx(scale, rel=1e-4)
        assert fit.tx == pytest.approx(tx, abs=1e-4)
        assert fit.ty == pytest.approx(ty, abs=1e-4)

        mapped = fit.apply(320.0, 240.0)
        assert mapped[0] == pytest.approx(scale * 320.0 + tx, abs=1e-4)
        assert mapped[1] == pytest.approx(scale * 240.0 + ty, abs=1e-4)

    def test_returns_none_without_enough_joints(self):
        """Fewer than three usable joints give no fit."""
        keypoints_2d = np.full((NUM_JOINTS, 2), -1.0, dtype=np.float32)
        coords_3d = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        conf = np.zeros(NUM_JOINTS, dtype=np.float32)
        assert fit_image_to_pose_3d(keypoints_2d, coords_3d, conf) is None


def write_test_video(path: Path, num_frames: int = 12) -> None:
    """Write a small clip so the pipeline has a real file to read."""
    import cv2

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from make_synthetic_video import draw_frame

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    assert writer.isOpened()
    try:
        for i in range(num_frames):
            writer.write(draw_frame(320, 240, i / num_frames))
    finally:
        writer.release()


class TestPipelineEgocentricStage:
    """The stage runs inside PipelineOrchestrator and reaches the exports."""

    @staticmethod
    def _config() -> PipelineConfig:
        """Stub backends plus the egocentric stage."""
        config = PipelineConfig()
        config.egocentric.enabled = True
        config.egocentric.export_frames = True
        config.egocentric.render = True
        config.egocentric.detect_hands = False
        config.egocentric.render_width = 160
        config.egocentric.render_height = 120
        config.segmentation.min_duration = 2
        return config

    def test_export_contains_ego_frames(self, tmp_path):
        """The run writes egocentric.json with one entry per frame."""
        video = tmp_path / "clip.mp4"
        write_test_video(video)

        paths = PipelineOrchestrator(self._config()).process(
            str(video), str(tmp_path / "out"), run_id="test_ego"
        )

        assert "egocentric" in paths
        data = json.loads(Path(paths["egocentric"]).read_text())
        assert data["schema_version"] == "1.0"
        assert data["type"] == "egocentric_trajectory"
        assert data["num_frames"] == 12
        assert len(data["frames"]) == 12
        assert data["video_info"]["units"] == "m"

        first = data["frames"][0]
        assert first["frame_idx"] == 0
        assert len(first["head"]["position"]) == 3
        assert len(first["head"]["forward"]) == 3
        # Stub detections give three tracked objects to place in the ego frame.
        assert any(f["objects"] for f in data["frames"])

    def test_render_writes_a_video(self, tmp_path):
        """render: true writes egocentric.mp4 next to the JSON."""
        video = tmp_path / "clip.mp4"
        write_test_video(video)

        paths = PipelineOrchestrator(self._config()).process(
            str(video), str(tmp_path / "out"), run_id="test_render"
        )

        assert "egocentric_video" in paths
        rendered = Path(paths["egocentric_video"])
        assert rendered.exists()
        assert rendered.stat().st_size > 0

    def test_stage_is_off_by_default(self, tmp_path):
        """Without the config block the pipeline writes no ego output."""
        video = tmp_path / "clip.mp4"
        write_test_video(video)

        config = PipelineConfig()
        config.segmentation.min_duration = 2
        paths = PipelineOrchestrator(config).process(
            str(video), str(tmp_path / "out"), run_id="test_off"
        )

        assert "egocentric" not in paths
        assert "egocentric_video" not in paths


class TestSummarizeSeries:
    """The summary counts what the stage actually produced."""

    def test_counts_are_zero_without_pose(self):
        """No pose data gives an empty series and zero counts."""
        from src.utils import create_empty_timeline

        timeline = create_empty_timeline(fps=30.0, num_frames=5, width=64, height=48)
        series = EgocentricStage().run(timeline)

        assert series.num_frames == 0
        assert summarize_series(series)["num_frames"] == 0
