"""Shared timeline container and data contracts for the vision pipeline.

All modules reference the same timestamps for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COCO17_JOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_JOINTS: int = 17

# Joint indices for convenience
JOINT_IDX: Dict[str, int] = {name: i for i, name in enumerate(COCO17_JOINT_NAMES)}

# Sentinel values for missing data
SENTINEL_2D: List[float] = [-1.0, -1.0]
SENTINEL_3D: List[float] = [-1.0, -1.0, -1.0]
SENTINEL_BBOX: List[float] = [-1.0, -1.0, -1.0, -1.0]
SENTINEL_CONF: float = 0.0


# ---------------------------------------------------------------------------
# Enums for object states
# ---------------------------------------------------------------------------


class SupportRelation(str, Enum):
    """Where the object is supported."""

    HAND = "hand"
    TABLE = "table"
    FIXTURE = "fixture"
    UNKNOWN = "unknown"


class MotionState(str, Enum):
    """Whether the object is moving."""

    STATIC = "static"
    MOVING = "moving"


class InteractionState(str, Enum):
    """Interaction state of an object."""

    IN_CONTACT = "in_contact"
    GRASPED = "grasped"
    NONE = "none"


class Containment(str, Enum):
    """Containment state of an object."""

    IN_BIN = "in_bin"
    IN_BOX = "in_box"
    NONE = "none"
    UNKNOWN = "unknown"


class HandSide(str, Enum):
    """Which hand."""

    LEFT = "left"
    RIGHT = "right"


# ---------------------------------------------------------------------------
# Frame metadata
# ---------------------------------------------------------------------------


@dataclass
class FrameInfo:
    """Metadata for a single frame."""

    frame_idx: int
    t: float  # timestamp in seconds
    width: int
    height: int


# ---------------------------------------------------------------------------
# Pose data structures
# ---------------------------------------------------------------------------


@dataclass
class PoseFrame:
    """Pose data for a single frame."""

    frame_idx: int
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] or SENTINEL_BBOX
    keypoints_2d_px: List[List[float]]  # (17, 2)
    conf_2d: List[float]  # (17,)
    coords_3d: List[List[float]]  # (17, 3)
    conf_3d: List[float]  # (17,)
    joint_visible: List[bool]  # (17,)
    joint_occluded: List[bool]  # (17,)
    joint_in_frame: List[bool]  # (17,)
    # Metric 3D joints straight from the estimator, when the backend supplies
    # them. MediaPipe pose_world_landmarks fill this. None means the estimator
    # gives no metric 3D and the lifter must derive coords_3d another way.
    world_coords_3d: Optional[List[List[float]]] = None


@dataclass
class PersonPose:
    """Time-series pose data for a single person."""

    skeleton_type: str = "COCO17"
    joint_names: List[str] = field(default_factory=lambda: COCO17_JOINT_NAMES.copy())
    root_joint_index: int = -1  # virtual root
    root_joint_name: str = "pelvis_virtual"
    root_definition: str = "pelvis_virtual = 0.5*(left_hip + right_hip)"
    frames: List[PoseFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get_frame_indices(self) -> List[int]:
        return [f.frame_idx for f in self.frames]

    def get_bbox_array(self) -> np.ndarray:
        """Return (T, 4) array of bboxes."""
        return np.array([f.bbox_xyxy for f in self.frames], dtype=np.float32)

    def get_keypoints_2d_array(self) -> np.ndarray:
        """Return (T, 17, 2) array of 2D keypoints."""
        return np.array([f.keypoints_2d_px for f in self.frames], dtype=np.float32)

    def get_conf_2d_array(self) -> np.ndarray:
        """Return (T, 17) array of 2D confidences."""
        return np.array([f.conf_2d for f in self.frames], dtype=np.float32)

    def get_coords_3d_array(self) -> np.ndarray:
        """Return (T, 17, 3) array of 3D coords."""
        return np.array([f.coords_3d for f in self.frames], dtype=np.float32)

    def get_conf_3d_array(self) -> np.ndarray:
        """Return (T, 17) array of 3D confidences."""
        return np.array([f.conf_3d for f in self.frames], dtype=np.float32)

    def get_joint_in_frame_array(self) -> np.ndarray:
        """Return (T, 17) bool array."""
        return np.array([f.joint_in_frame for f in self.frames], dtype=bool)


# ---------------------------------------------------------------------------
# Object tracking data structures
# ---------------------------------------------------------------------------


@dataclass
class ObjectState:
    """State of an object at a single frame."""

    support_relation: SupportRelation = SupportRelation.UNKNOWN
    motion_state: MotionState = MotionState.STATIC
    interaction_state: InteractionState = InteractionState.NONE
    containment: Containment = Containment.UNKNOWN


@dataclass
class ObjectFrame:
    """Object data for a single frame."""

    frame_idx: int
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] or SENTINEL_BBOX
    conf: float
    state: ObjectState = field(default_factory=ObjectState)


@dataclass
class ObjectTrack:
    """Time-series track for a single object."""

    object_id: int
    class_name: str
    class_id: int
    frames: List[ObjectFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get_bbox_array(self) -> np.ndarray:
        """Return (T, 4) array of bboxes."""
        return np.array([f.bbox_xyxy for f in self.frames], dtype=np.float32)

    def get_conf_array(self) -> np.ndarray:
        """Return (T,) array of confidences."""
        return np.array([f.conf for f in self.frames], dtype=np.float32)


# ---------------------------------------------------------------------------
# Hand pose data structures
# ---------------------------------------------------------------------------


@dataclass
class HandFrame:
    """Hand data for a single frame."""

    frame_idx: int
    handedness: str  # "Left" or "Right"
    landmarks_2d: List[List[float]]  # (21, 2) pixel coordinates
    landmarks_3d: List[List[float]]  # (21, 3) normalized 3D coordinates
    conf: List[float]  # (21,) per-landmark confidence
    overall_confidence: float
    is_detected: bool  # True if hand was detected (not interpolated)


@dataclass
class HandPose:
    """Time-series hand data for a single hand."""

    handedness: str  # "Left" or "Right"
    frames: List[HandFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def get_frame_indices(self) -> List[int]:
        return [f.frame_idx for f in self.frames]

    def get_landmarks_2d_array(self) -> np.ndarray:
        """Return (T, 21, 2) array of 2D landmarks."""
        return np.array([f.landmarks_2d for f in self.frames], dtype=np.float32)

    def get_landmarks_3d_array(self) -> np.ndarray:
        """Return (T, 21, 3) array of 3D landmarks."""
        return np.array([f.landmarks_3d for f in self.frames], dtype=np.float32)

    def get_conf_array(self) -> np.ndarray:
        """Return (T, 21) array of confidences."""
        return np.array([f.conf for f in self.frames], dtype=np.float32)


# ---------------------------------------------------------------------------
# Contact events
# ---------------------------------------------------------------------------


@dataclass
class ContactEvent:
    """A contact event between hand and object."""

    event_id: int
    event_type: str  # "contact"
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    hand: HandSide
    object_id: int
    label: str  # "grasp", "push", "pull", "insert", etc.
    conf: float
    evidence: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action segments
# ---------------------------------------------------------------------------


@dataclass
class ActionSegment:
    """An action segment with start/end times."""

    seg_id: int
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    label: str
    conf: float
    objects_involved: List[int] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main Timeline Container
# ---------------------------------------------------------------------------


@dataclass
class Timeline:
    """Canonical timeline container shared across all modules.

    All data is aligned to the same frame indices and timestamps.
    """

    # Video metadata
    fps_extracted: float = 30.0
    t0: float = 0.0
    timestamp_source: str = "synthetic"  # "video" or "synthetic"
    frame_to_time_rule: str = "frame_idx / fps_extracted"

    # Frame info
    frames: List[FrameInfo] = field(default_factory=list)

    # Pose data (single person)
    person_pose: Optional[PersonPose] = None

    # Hand pose data (left and right hands)
    hands: Dict[str, HandPose] = field(default_factory=dict)  # key: "Left" or "Right"

    # Object tracks keyed by object_id
    objects: Dict[int, ObjectTrack] = field(default_factory=dict)

    # Contact events (v1: contact events only)
    events: List[ContactEvent] = field(default_factory=list)

    # Action segments
    segments: List[ActionSegment] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].t - self.frames[0].t

    def get_timestamps(self) -> np.ndarray:
        """Return (T,) array of timestamps."""
        return np.array([f.t for f in self.frames], dtype=np.float64)

    def get_frame_indices(self) -> List[int]:
        """Return list of frame indices."""
        return [f.frame_idx for f in self.frames]

    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to timestamp."""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return self.t0 + frame_idx / self.fps_extracted
        return self.frames[frame_idx].t

    def time_to_frame(self, t: float) -> int:
        """Convert timestamp to nearest frame index."""
        if not self.frames:
            return int((t - self.t0) * self.fps_extracted)
        timestamps = self.get_timestamps()
        idx = int(np.argmin(np.abs(timestamps - t)))
        return self.frames[idx].frame_idx

    def validate_alignment(self) -> List[str]:
        """Check that all data is aligned to timeline. Returns list of warnings."""
        warnings = []
        T = self.num_frames

        # Check frame indices are 0..T-1
        expected_indices = list(range(T))
        actual_indices = self.get_frame_indices()
        if actual_indices != expected_indices:
            warnings.append(
                f"Frame indices not contiguous: expected {expected_indices[:5]}..., "
                f"got {actual_indices[:5]}..."
            )

        # Check timestamps are monotonic
        if T > 1:
            ts = self.get_timestamps()
            if not np.all(np.diff(ts) > 0):
                warnings.append("Timestamps are not strictly monotonic")

        # Check pose alignment
        if self.person_pose is not None:
            pose_T = self.person_pose.num_frames
            if pose_T != T:
                warnings.append(f"Pose has {pose_T} frames, expected {T}")

        # Check object alignment
        for obj_id, track in self.objects.items():
            obj_T = track.num_frames
            if obj_T != T:
                warnings.append(f"Object {obj_id} has {obj_T} frames, expected {T}")

        return warnings


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_empty_timeline(
    num_frames: int,
    fps: float,
    width: int,
    height: int,
    t0: float = 0.0,
) -> Timeline:
    """Create an empty timeline with frame metadata initialized."""
    frames = []
    for i in range(num_frames):
        t = t0 + i / fps
        frames.append(FrameInfo(frame_idx=i, t=t, width=width, height=height))

    return Timeline(
        fps_extracted=fps,
        t0=t0,
        timestamp_source="synthetic",
        frame_to_time_rule="frame_idx / fps_extracted",
        frames=frames,
    )


def create_empty_pose_frame(frame_idx: int) -> PoseFrame:
    """Create a pose frame with all joints marked as missing."""
    return PoseFrame(
        frame_idx=frame_idx,
        bbox_xyxy=SENTINEL_BBOX.copy(),
        keypoints_2d_px=[SENTINEL_2D.copy() for _ in range(NUM_JOINTS)],
        conf_2d=[SENTINEL_CONF] * NUM_JOINTS,
        coords_3d=[SENTINEL_3D.copy() for _ in range(NUM_JOINTS)],
        conf_3d=[SENTINEL_CONF] * NUM_JOINTS,
        joint_visible=[False] * NUM_JOINTS,
        joint_occluded=[False] * NUM_JOINTS,
        joint_in_frame=[False] * NUM_JOINTS,
    )


def create_empty_person_pose(num_frames: int) -> PersonPose:
    """Create empty pose data for all frames."""
    frames = [create_empty_pose_frame(i) for i in range(num_frames)]
    return PersonPose(frames=frames)


def create_empty_object_frame(frame_idx: int) -> ObjectFrame:
    """Create an object frame with missing data."""
    return ObjectFrame(
        frame_idx=frame_idx,
        bbox_xyxy=SENTINEL_BBOX.copy(),
        conf=SENTINEL_CONF,
        state=ObjectState(),
    )


def create_empty_object_track(
    object_id: int,
    class_name: str,
    class_id: int,
    num_frames: int,
) -> ObjectTrack:
    """Create empty object track for all frames."""
    frames = [create_empty_object_frame(i) for i in range(num_frames)]
    return ObjectTrack(
        object_id=object_id,
        class_name=class_name,
        class_id=class_id,
        frames=frames,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def is_joint_usable(
    joint_in_frame: bool,
    conf: float,
    min_conf: float,
) -> bool:
    """Check if a joint is usable based on visibility and confidence.

    A joint is usable iff joint_in_frame == True AND conf >= min_conf.
    """
    return joint_in_frame and conf >= min_conf


def is_bbox_valid(bbox: List[float]) -> bool:
    """Check if a bbox is valid (not sentinel)."""
    return bbox != SENTINEL_BBOX and bbox[0] >= 0


def compute_virtual_root(
    keypoints_2d: List[List[float]],
    conf_2d: List[float],
    joint_in_frame: List[bool],
) -> tuple:
    """Compute virtual pelvis as midpoint of left_hip and right_hip.

    Returns (x, y, conf, valid) tuple.
    """
    left_hip_idx = JOINT_IDX["left_hip"]
    right_hip_idx = JOINT_IDX["right_hip"]

    left_valid = joint_in_frame[left_hip_idx] and conf_2d[left_hip_idx] > 0
    right_valid = joint_in_frame[right_hip_idx] and conf_2d[right_hip_idx] > 0

    if left_valid and right_valid:
        x = (keypoints_2d[left_hip_idx][0] + keypoints_2d[right_hip_idx][0]) / 2
        y = (keypoints_2d[left_hip_idx][1] + keypoints_2d[right_hip_idx][1]) / 2
        conf = (conf_2d[left_hip_idx] + conf_2d[right_hip_idx]) / 2
        return (x, y, conf, True)
    elif left_valid:
        return (
            keypoints_2d[left_hip_idx][0],
            keypoints_2d[left_hip_idx][1],
            conf_2d[left_hip_idx],
            True,
        )
    elif right_valid:
        return (
            keypoints_2d[right_hip_idx][0],
            keypoints_2d[right_hip_idx][1],
            conf_2d[right_hip_idx],
            True,
        )
    else:
        return (0.0, 0.0, 0.0, False)
