"""Configuration loading and validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Video processing configuration."""
    target_fps: Optional[float] = 30.0
    max_frames: Optional[int] = None


@dataclass
class FrameConfig:
    """Frame processing configuration."""
    resize_policy: str = "none"
    target_width: int = 640
    target_height: int = 480
    normalize: bool = False


@dataclass
class OneEuroConfig:
    """One Euro filter configuration."""
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


@dataclass
class SmoothingConfig:
    """Pose smoothing configuration."""
    method: str = "one_euro"
    window_size: int = 5
    alpha: float = 0.5
    one_euro: OneEuroConfig = field(default_factory=OneEuroConfig)


@dataclass
class PoseConfig:
    """Pose estimation configuration."""
    backend_2d: str = "stub"
    backend_3d: str = "stub"
    min_joint_conf_2d: float = 0.2
    min_joint_conf_3d: float = 0.2
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)


@dataclass
class ObjectsConfig:
    """Object detection configuration."""
    detector_backend: str = "stub"
    classes: Optional[List[str]] = None
    confidence_threshold: float = 0.5
    stub_num_objects: int = 3


@dataclass
class TrackerConfig:
    """Object tracking configuration."""
    backend: str = "iou"
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    interpolate_gaps: bool = True
    max_gap_frames: int = 10


@dataclass
class ContactConfig:
    """Contact detection configuration."""
    distance_threshold: float = 50.0
    onset_frames: int = 2
    offset_frames: int = 3
    use_hand_center: bool = True


@dataclass
class InteractionConfig:
    """Interaction classification configuration."""
    motion_threshold: float = 5.0
    motion_correlation_threshold: float = 0.7
    min_grasp_duration: int = 5
    min_push_duration: int = 3


@dataclass
class SegmentationConfig:
    """Action segmentation configuration."""
    min_duration: int = 10
    merge_threshold: int = 5
    hysteresis: int = 3
    detect_contact_changes: bool = True
    detect_motion_changes: bool = True
    detect_support_changes: bool = True
    detect_containment_changes: bool = True


@dataclass
class StateConfig:
    """State classification configuration."""
    motion_threshold: float = 5.0
    motion_window: int = 3
    fixture_classes: List[str] = field(
        default_factory=lambda: ["fixture", "machine", "station"]
    )
    container_classes: List[str] = field(
        default_factory=lambda: ["bin", "box", "container"]
    )


@dataclass
class GraphConfig:
    """Task graph configuration."""
    require_shared_objects: bool = True
    min_causal_confidence: float = 0.5
    add_temporal_fallback: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    json_indent: int = 2
    export_visualization: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    video: VideoConfig = field(default_factory=VideoConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    state: StateConfig = field(default_factory=StateConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> PipelineConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        PipelineConfig object.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        return PipelineConfig()

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return _dict_to_config(data)


def _dict_to_config(data: Dict[str, Any]) -> PipelineConfig:
    """Convert dictionary to PipelineConfig.

    Args:
        data: Dictionary from YAML.

    Returns:
        PipelineConfig object.
    """
    config = PipelineConfig()

    # Video config
    if "video" in data:
        v = data["video"]
        config.video = VideoConfig(
            target_fps=v.get("target_fps", config.video.target_fps),
            max_frames=v.get("max_frames", config.video.max_frames),
        )

    # Frame config
    if "frame" in data:
        f = data["frame"]
        config.frame = FrameConfig(
            resize_policy=f.get("resize_policy", config.frame.resize_policy),
            target_width=f.get("target_width", config.frame.target_width),
            target_height=f.get("target_height", config.frame.target_height),
            normalize=f.get("normalize", config.frame.normalize),
        )

    # Pose config
    if "pose" in data:
        p = data["pose"]
        smoothing_data = p.get("smoothing", {})
        one_euro_data = smoothing_data.get("one_euro", {})

        one_euro = OneEuroConfig(
            min_cutoff=one_euro_data.get("min_cutoff", 1.0),
            beta=one_euro_data.get("beta", 0.007),
            d_cutoff=one_euro_data.get("d_cutoff", 1.0),
        )

        smoothing = SmoothingConfig(
            method=smoothing_data.get("method", "one_euro"),
            window_size=smoothing_data.get("window_size", 5),
            alpha=smoothing_data.get("alpha", 0.5),
            one_euro=one_euro,
        )

        config.pose = PoseConfig(
            backend_2d=p.get("backend_2d", config.pose.backend_2d),
            backend_3d=p.get("backend_3d", config.pose.backend_3d),
            min_joint_conf_2d=p.get("min_joint_conf_2d", config.pose.min_joint_conf_2d),
            min_joint_conf_3d=p.get("min_joint_conf_3d", config.pose.min_joint_conf_3d),
            smoothing=smoothing,
        )

    # Objects config
    if "objects" in data:
        o = data["objects"]
        config.objects = ObjectsConfig(
            detector_backend=o.get("detector_backend", config.objects.detector_backend),
            classes=o.get("classes", config.objects.classes),
            confidence_threshold=o.get("confidence_threshold", config.objects.confidence_threshold),
            stub_num_objects=o.get("stub_num_objects", config.objects.stub_num_objects),
        )

    # Tracker config
    if "tracker" in data:
        t = data["tracker"]
        config.tracker = TrackerConfig(
            backend=t.get("backend", config.tracker.backend),
            iou_threshold=t.get("iou_threshold", config.tracker.iou_threshold),
            max_age=t.get("max_age", config.tracker.max_age),
            min_hits=t.get("min_hits", config.tracker.min_hits),
            interpolate_gaps=t.get("interpolate_gaps", config.tracker.interpolate_gaps),
            max_gap_frames=t.get("max_gap_frames", config.tracker.max_gap_frames),
        )

    # Contact config
    if "contact" in data:
        c = data["contact"]
        config.contact = ContactConfig(
            distance_threshold=c.get("distance_threshold", config.contact.distance_threshold),
            onset_frames=c.get("onset_frames", config.contact.onset_frames),
            offset_frames=c.get("offset_frames", config.contact.offset_frames),
            use_hand_center=c.get("use_hand_center", config.contact.use_hand_center),
        )

    # Interaction config
    if "interaction" in data:
        i = data["interaction"]
        config.interaction = InteractionConfig(
            motion_threshold=i.get("motion_threshold", config.interaction.motion_threshold),
            motion_correlation_threshold=i.get("motion_correlation_threshold", config.interaction.motion_correlation_threshold),
            min_grasp_duration=i.get("min_grasp_duration", config.interaction.min_grasp_duration),
            min_push_duration=i.get("min_push_duration", config.interaction.min_push_duration),
        )

    # Segmentation config
    if "segmentation" in data:
        s = data["segmentation"]
        config.segmentation = SegmentationConfig(
            min_duration=s.get("min_duration", config.segmentation.min_duration),
            merge_threshold=s.get("merge_threshold", config.segmentation.merge_threshold),
            hysteresis=s.get("hysteresis", config.segmentation.hysteresis),
            detect_contact_changes=s.get("detect_contact_changes", config.segmentation.detect_contact_changes),
            detect_motion_changes=s.get("detect_motion_changes", config.segmentation.detect_motion_changes),
            detect_support_changes=s.get("detect_support_changes", config.segmentation.detect_support_changes),
            detect_containment_changes=s.get("detect_containment_changes", config.segmentation.detect_containment_changes),
        )

    # State config
    if "state" in data:
        st = data["state"]
        config.state = StateConfig(
            motion_threshold=st.get("motion_threshold", config.state.motion_threshold),
            motion_window=st.get("motion_window", config.state.motion_window),
            fixture_classes=st.get("fixture_classes", config.state.fixture_classes),
            container_classes=st.get("container_classes", config.state.container_classes),
        )

    # Graph config
    if "graph" in data:
        g = data["graph"]
        config.graph = GraphConfig(
            require_shared_objects=g.get("require_shared_objects", config.graph.require_shared_objects),
            min_causal_confidence=g.get("min_causal_confidence", config.graph.min_causal_confidence),
            add_temporal_fallback=g.get("add_temporal_fallback", config.graph.add_temporal_fallback),
        )

    # Output config
    if "output" in data:
        out = data["output"]
        config.output = OutputConfig(
            json_indent=out.get("json_indent", config.output.json_indent),
            export_visualization=out.get("export_visualization", config.output.export_visualization),
        )

    return config


def save_config(config: PipelineConfig, path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: PipelineConfig to save.
        path: Output path.
    """
    data = _config_to_dict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {path}")


def _config_to_dict(config: PipelineConfig) -> Dict[str, Any]:
    """Convert PipelineConfig to dictionary.

    Args:
        config: PipelineConfig object.

    Returns:
        Dictionary for YAML serialization.
    """
    return {
        "video": {
            "target_fps": config.video.target_fps,
            "max_frames": config.video.max_frames,
        },
        "frame": {
            "resize_policy": config.frame.resize_policy,
            "target_width": config.frame.target_width,
            "target_height": config.frame.target_height,
            "normalize": config.frame.normalize,
        },
        "pose": {
            "backend_2d": config.pose.backend_2d,
            "backend_3d": config.pose.backend_3d,
            "min_joint_conf_2d": config.pose.min_joint_conf_2d,
            "min_joint_conf_3d": config.pose.min_joint_conf_3d,
            "smoothing": {
                "method": config.pose.smoothing.method,
                "window_size": config.pose.smoothing.window_size,
                "alpha": config.pose.smoothing.alpha,
                "one_euro": {
                    "min_cutoff": config.pose.smoothing.one_euro.min_cutoff,
                    "beta": config.pose.smoothing.one_euro.beta,
                    "d_cutoff": config.pose.smoothing.one_euro.d_cutoff,
                },
            },
        },
        "objects": {
            "detector_backend": config.objects.detector_backend,
            "classes": config.objects.classes,
            "confidence_threshold": config.objects.confidence_threshold,
            "stub_num_objects": config.objects.stub_num_objects,
        },
        "tracker": {
            "backend": config.tracker.backend,
            "iou_threshold": config.tracker.iou_threshold,
            "max_age": config.tracker.max_age,
            "min_hits": config.tracker.min_hits,
            "interpolate_gaps": config.tracker.interpolate_gaps,
            "max_gap_frames": config.tracker.max_gap_frames,
        },
        "contact": {
            "distance_threshold": config.contact.distance_threshold,
            "onset_frames": config.contact.onset_frames,
            "offset_frames": config.contact.offset_frames,
            "use_hand_center": config.contact.use_hand_center,
        },
        "interaction": {
            "motion_threshold": config.interaction.motion_threshold,
            "motion_correlation_threshold": config.interaction.motion_correlation_threshold,
            "min_grasp_duration": config.interaction.min_grasp_duration,
            "min_push_duration": config.interaction.min_push_duration,
        },
        "segmentation": {
            "min_duration": config.segmentation.min_duration,
            "merge_threshold": config.segmentation.merge_threshold,
            "hysteresis": config.segmentation.hysteresis,
            "detect_contact_changes": config.segmentation.detect_contact_changes,
            "detect_motion_changes": config.segmentation.detect_motion_changes,
            "detect_support_changes": config.segmentation.detect_support_changes,
            "detect_containment_changes": config.segmentation.detect_containment_changes,
        },
        "state": {
            "motion_threshold": config.state.motion_threshold,
            "motion_window": config.state.motion_window,
            "fixture_classes": config.state.fixture_classes,
            "container_classes": config.state.container_classes,
        },
        "graph": {
            "require_shared_objects": config.graph.require_shared_objects,
            "min_causal_confidence": config.graph.min_causal_confidence,
            "add_temporal_fallback": config.graph.add_temporal_fallback,
        },
        "output": {
            "json_indent": config.output.json_indent,
            "export_visualization": config.output.export_visualization,
        },
    }
