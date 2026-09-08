"""Tests for pose schema validity."""
import json

import pytest

# Add src to path

from bird2ego.utils.timeline import (
    COCO17_JOINT_NAMES,
    NUM_JOINTS,
    SENTINEL_2D,
    SENTINEL_3D,
    SENTINEL_BBOX,
    PersonPose,
    PoseFrame,
    create_empty_person_pose,
    create_empty_pose_frame,
)


class TestCOCO17Schema:
    """Tests for COCO17 joint schema."""

    def test_joint_count(self):
        """Verify exactly 17 joints."""
        assert NUM_JOINTS == 17
        assert len(COCO17_JOINT_NAMES) == 17

    def test_joint_names(self):
        """Verify expected joint names are present."""
        expected = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        assert COCO17_JOINT_NAMES == expected

    def test_sentinel_values(self):
        """Verify sentinel values are correct."""
        assert SENTINEL_2D == [-1.0, -1.0]
        assert SENTINEL_3D == [-1.0, -1.0, -1.0]
        assert SENTINEL_BBOX == [-1.0, -1.0, -1.0, -1.0]


class TestPoseFrame:
    """Tests for PoseFrame data structure."""

    def test_empty_pose_frame(self):
        """Test creating an empty pose frame."""
        frame = create_empty_pose_frame(0)

        assert frame.frame_idx == 0
        assert frame.bbox_xyxy == SENTINEL_BBOX
        assert len(frame.keypoints_2d_px) == NUM_JOINTS
        assert len(frame.conf_2d) == NUM_JOINTS
        assert len(frame.coords_3d) == NUM_JOINTS
        assert len(frame.conf_3d) == NUM_JOINTS
        assert len(frame.joint_visible) == NUM_JOINTS
        assert len(frame.joint_occluded) == NUM_JOINTS
        assert len(frame.joint_in_frame) == NUM_JOINTS

    def test_empty_pose_frame_sentinels(self):
        """Test that empty pose frame uses correct sentinels."""
        frame = create_empty_pose_frame(5)

        for j in range(NUM_JOINTS):
            assert frame.keypoints_2d_px[j] == SENTINEL_2D
            assert frame.conf_2d[j] == 0.0
            assert frame.coords_3d[j] == SENTINEL_3D
            assert frame.conf_3d[j] == 0.0
            assert frame.joint_visible[j] is False
            assert frame.joint_occluded[j] is False
            assert frame.joint_in_frame[j] is False


class TestPersonPose:
    """Tests for PersonPose data structure."""

    def test_empty_person_pose(self):
        """Test creating empty pose for multiple frames."""
        T = 10
        pose = create_empty_person_pose(T)

        assert pose.num_frames == T
        assert pose.skeleton_type == "COCO17"
        assert pose.joint_names == COCO17_JOINT_NAMES
        assert pose.root_joint_index == -1
        assert pose.root_joint_name == "pelvis_virtual"

    def test_frame_indices(self):
        """Test frame indices are sequential."""
        T = 5
        pose = create_empty_person_pose(T)

        indices = pose.get_frame_indices()
        assert indices == [0, 1, 2, 3, 4]

    def test_array_shapes(self):
        """Test that array accessors return correct shapes."""
        T = 10
        pose = create_empty_person_pose(T)

        bbox = pose.get_bbox_array()
        assert bbox.shape == (T, 4)

        kp_2d = pose.get_keypoints_2d_array()
        assert kp_2d.shape == (T, NUM_JOINTS, 2)

        conf_2d = pose.get_conf_2d_array()
        assert conf_2d.shape == (T, NUM_JOINTS)

        coords_3d = pose.get_coords_3d_array()
        assert coords_3d.shape == (T, NUM_JOINTS, 3)

        conf_3d = pose.get_conf_3d_array()
        assert conf_3d.shape == (T, NUM_JOINTS)

        joint_in_frame = pose.get_joint_in_frame_array()
        assert joint_in_frame.shape == (T, NUM_JOINTS)


class TestPoseSchemaValidation:
    """Tests for pose schema validation."""

    def test_pose_json_schema_fields(self):
        """Test that pose can be serialized with required fields."""
        pose = create_empty_person_pose(5)

        # Build the schema structure
        schema = {
            "schema_version": "1.0",
            "run": {"run_id": "test"},
            "image": {"width": 640, "height": 480, "space": "processed_frames"},
            "person": {
                "bbox_xyxy": pose.get_bbox_array().tolist(),
                "skeleton_type": pose.skeleton_type,
                "joint_names": pose.joint_names,
                "root_joint_index": pose.root_joint_index,
                "root_joint_name": pose.root_joint_name,
                "root_definition": pose.root_definition,
                "frame_idx": pose.get_frame_indices(),
                "keypoints_2d_px": pose.get_keypoints_2d_array().tolist(),
                "conf_2d": pose.get_conf_2d_array().tolist(),
                "pose_3d_rootrel": {
                    "coords": pose.get_coords_3d_array().tolist(),
                    "units": "arb",
                    "coord_frame": "root-relative-camera",
                    "axis_convention": "x=right, y=down, z=forward (camera)",
                    "conf_3d": pose.get_conf_3d_array().tolist(),
                },
                "visibility": {
                    "joint_visible": [[False] * NUM_JOINTS for _ in range(5)],
                    "joint_occluded": [[False] * NUM_JOINTS for _ in range(5)],
                    "joint_in_frame": pose.get_joint_in_frame_array().tolist(),
                },
                "missing_joint_convention": {
                    "coord_sentinel_2d": SENTINEL_2D,
                    "coord_sentinel_3d": SENTINEL_3D,
                    "conf_sentinel": 0.0,
                },
            },
        }

        # Verify it can be serialized to JSON
        json_str = json.dumps(schema)
        assert len(json_str) > 0

        # Verify key fields
        parsed = json.loads(json_str)
        assert parsed["person"]["skeleton_type"] == "COCO17"
        assert len(parsed["person"]["joint_names"]) == 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
