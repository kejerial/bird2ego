"""Main pipeline orchestrator for video processing."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .actions import (
    ActionClassifier,
    ActionSegmenter,
    ActionSegmenterConfig,
    SkillBoundaryConfig,
    SkillBoundaryDetector,
)
from .egocentric import (
    EgocentricStage,
    EgocentricTimeSeries,
    export_egocentric_json,
    summarize_series,
)
from .contact import (
    ContactDetector,
    ContactDetectorConfig,
    EventExtractor,
    InteractionClassifier,
    InteractionClassifierConfig,
)
from .graph import (
    OrderingInferencer,
    PostconditionExtractor,
    PreconditionExtractor,
    TaskGraphBuilder,
)
from .objects import (
    ObjectDetector,
    ObjectTracker,
    StateClassifier,
    StateClassifierConfig,
    TrajectoryBuilder,
)
from .output import GraphExporter, JSONExporter
from .pose import (
    KinematicsProcessor,
    OneEuroParams,
    PoseEstimator2D,
    PoseLifter3D,
    PoseTracker,
    SmoothingMethod,
)
from .utils import (
    PersonPose,
    PipelineConfig,
    Timeline,
    load_config,
)
from .video import (
    FrameProcessor,
    ResizePolicy,
    TemporalAlignment,
    VideoLoader,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full video processing pipeline.

    Connects all modules end-to-end with explicit dependency order:
    1) detect/track objects
    2) estimate pose
    3) infer contacts
    4) derive object states using motion-only pass, then refine with contacts

    Ensures all outputs share the same timestamps.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self._init_modules()

    def _init_modules(self) -> None:
        """Initialize all pipeline modules from config."""
        cfg = self.config

        # Video loading
        self.video_loader = VideoLoader(
            target_fps=cfg.video.target_fps,
            max_frames=cfg.video.max_frames,
        )

        # Frame processing
        resize_policy = ResizePolicy(cfg.frame.resize_policy)
        self.frame_processor = FrameProcessor(
            target_width=cfg.frame.target_width,
            target_height=cfg.frame.target_height,
            resize_policy=resize_policy,
            normalize=cfg.frame.normalize,
        )

        # Temporal alignment
        self.temporal_alignment = TemporalAlignment()

        # Pose estimation
        self.pose_estimator_2d = PoseEstimator2D(
            backend=cfg.pose.backend_2d,
            min_confidence=cfg.pose.min_joint_conf_2d,
        )

        self.pose_lifter_3d = PoseLifter3D(
            backend=cfg.pose.backend_3d,
            min_confidence=cfg.pose.min_joint_conf_3d,
        )

        smoothing_method = SmoothingMethod(cfg.pose.smoothing.method)
        one_euro_params = OneEuroParams(
            min_cutoff=cfg.pose.smoothing.one_euro.min_cutoff,
            beta=cfg.pose.smoothing.one_euro.beta,
            d_cutoff=cfg.pose.smoothing.one_euro.d_cutoff,
        )
        self.pose_tracker = PoseTracker(
            method=smoothing_method,
            fps=cfg.video.target_fps or 30.0,
            window_size=cfg.pose.smoothing.window_size,
            alpha=cfg.pose.smoothing.alpha,
            one_euro_params=one_euro_params,
        )

        self.kinematics_processor = KinematicsProcessor(
            fps=cfg.video.target_fps or 30.0,
            min_confidence=cfg.pose.min_joint_conf_2d,
        )

        # Object detection and tracking
        self.object_detector = ObjectDetector(
            backend=cfg.objects.detector_backend,
            classes=cfg.objects.classes,
            confidence_threshold=cfg.objects.confidence_threshold,
            num_objects=cfg.objects.stub_num_objects,
        )

        self.object_tracker = ObjectTracker(
            backend=cfg.tracker.backend,
            iou_threshold=cfg.tracker.iou_threshold,
            max_age=cfg.tracker.max_age,
            min_hits=cfg.tracker.min_hits,
        )

        self.trajectory_builder = TrajectoryBuilder(
            interpolate_gaps=cfg.tracker.interpolate_gaps,
            max_gap_frames=cfg.tracker.max_gap_frames,
        )

        state_classifier_config = StateClassifierConfig(
            motion_threshold=cfg.state.motion_threshold,
            motion_window=cfg.state.motion_window,
            fixture_classes=cfg.state.fixture_classes,
            container_classes=cfg.state.container_classes,
        )
        self.state_classifier = StateClassifier(config=state_classifier_config)

        # Contact detection
        contact_config = ContactDetectorConfig(
            contact_distance_threshold=cfg.contact.distance_threshold,
            min_joint_conf_2d=cfg.pose.min_joint_conf_2d,
            contact_onset_frames=cfg.contact.onset_frames,
            contact_offset_frames=cfg.contact.offset_frames,
            use_hand_center=cfg.contact.use_hand_center,
        )
        self.contact_detector = ContactDetector(config=contact_config)

        interaction_config = InteractionClassifierConfig(
            motion_threshold=cfg.interaction.motion_threshold,
            motion_correlation_threshold=cfg.interaction.motion_correlation_threshold,
            min_grasp_duration=cfg.interaction.min_grasp_duration,
            min_push_duration=cfg.interaction.min_push_duration,
        )
        self.interaction_classifier = InteractionClassifier(config=interaction_config)

        self.event_extractor = EventExtractor()

        # Action segmentation
        segmenter_config = ActionSegmenterConfig(
            min_segment_duration=cfg.segmentation.min_duration,
            detect_contact_changes=cfg.segmentation.detect_contact_changes,
            detect_motion_changes=cfg.segmentation.detect_motion_changes,
            detect_support_changes=cfg.segmentation.detect_support_changes,
            detect_containment_changes=cfg.segmentation.detect_containment_changes,
        )
        self.action_segmenter = ActionSegmenter(config=segmenter_config)

        boundary_config = SkillBoundaryConfig(
            min_duration_frames=cfg.segmentation.min_duration,
            merge_threshold_frames=cfg.segmentation.merge_threshold,
            hysteresis_frames=cfg.segmentation.hysteresis,
        )
        self.skill_boundary_detector = SkillBoundaryDetector(config=boundary_config)

        self.action_classifier = ActionClassifier()

        # Graph construction
        self.precondition_extractor = PreconditionExtractor()
        self.postcondition_extractor = PostconditionExtractor()
        self.ordering_inferencer = OrderingInferencer(
            require_shared_objects=cfg.graph.require_shared_objects,
            min_causal_confidence=cfg.graph.min_causal_confidence,
            add_temporal_fallback=cfg.graph.add_temporal_fallback,
        )
        self.task_graph_builder = TaskGraphBuilder()

        # Egocentric transform
        self.egocentric_stage: Optional[EgocentricStage] = None
        if cfg.egocentric.enabled:
            self.egocentric_stage = EgocentricStage(
                width=cfg.egocentric.render_width,
                height=cfg.egocentric.render_height,
                fov_horizontal=cfg.egocentric.fov_horizontal,
                gaze_down_angle=cfg.egocentric.gaze_down_angle,
                eye_offset_forward=cfg.egocentric.eye_offset_forward,
                pinch_threshold=cfg.egocentric.pinch_threshold,
                min_joint_conf=cfg.pose.min_joint_conf_3d,
                detect_hands=cfg.egocentric.detect_hands,
                hand_min_confidence=cfg.egocentric.hand_min_confidence,
            )

    def process(
        self,
        video_path: str,
        output_dir: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Process a video end-to-end.

        Args:
            video_path: Path to input video.
            output_dir: Directory for output files.
            run_id: Optional run identifier.

        Returns:
            Dictionary mapping output type to file path.
        """
        start_time = time.time()
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting pipeline for {video_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Run ID: {run_id}")

        # Step 1: Load video
        logger.info("Step 1: Loading video...")
        frames, timestamps, metadata = self._load_video(video_path)
        logger.info(f"  Loaded {len(frames)} frames")

        # Step 2: Create timeline
        logger.info("Step 2: Creating timeline...")
        timeline = self.temporal_alignment.create_timeline_from_video(
            fps=metadata.fps,
            frame_count=len(frames),
            width=metadata.width,
            height=metadata.height,
            t0=0.0,
            timestamps=timestamps,
        )
        logger.info(f"  Timeline: {timeline.num_frames} frames, {timeline.duration:.2f}s")

        # Step 3: Object detection and tracking
        logger.info("Step 3: Detecting and tracking objects...")
        detections = self.object_detector.detect_video(frames)
        object_tracks = self.trajectory_builder.build_from_detections(
            detections, self.object_tracker
        )
        self.trajectory_builder.add_tracks_to_timeline(timeline, object_tracks)
        logger.info(f"  Tracked {len(object_tracks)} objects")

        # Step 4: Motion-only state classification (before contacts)
        logger.info("Step 4: Classifying object motion states...")
        self.state_classifier.classify_motion_only(timeline.objects)

        # Step 5: Pose estimation
        logger.info("Step 5: Estimating poses...")
        pose_frames = self.pose_estimator_2d.estimate_video(frames)
        pose_frames = self.pose_lifter_3d.lift_video(
            pose_frames, (metadata.width, metadata.height)
        )
        pose_frames = self.pose_tracker.smooth_poses(pose_frames)

        person_pose = PersonPose(frames=pose_frames)
        timeline.person_pose = person_pose
        logger.info(f"  Estimated {person_pose.num_frames} pose frames")

        # Step 6: Contact detection
        logger.info("Step 6: Detecting contacts...")
        contacts_per_frame = self.contact_detector.detect_video(
            person_pose, timeline.objects
        )
        contact_frames = self.contact_detector.get_contact_frames_per_object(
            contacts_per_frame
        )
        logger.info(f"  Detected contacts for {len(contact_frames)} objects")

        # Step 7: Interaction classification
        logger.info("Step 7: Classifying interactions...")
        classified_interactions = self.interaction_classifier.classify_interactions(
            contacts_per_frame, person_pose, timeline.objects,
            fps=self.config.video.target_fps or 30.0,
        )
        logger.info(f"  Classified {len(classified_interactions)} interactions")

        # Step 8: Extract contact events
        logger.info("Step 8: Extracting contact events...")
        events = self.event_extractor.extract_events(classified_interactions, timeline)
        self.event_extractor.add_events_to_timeline(timeline, events)
        grasp_frames = self.event_extractor.get_grasp_frames_per_object(events)
        logger.info(f"  Extracted {len(events)} events")

        # Step 9: Refine object states with contact info
        logger.info("Step 9: Refining object states with contacts...")
        self.state_classifier.classify_with_contacts(
            timeline.objects,
            contact_frames=contact_frames,
            grasp_frames=grasp_frames,
        )

        # Step 10: Action segmentation
        logger.info("Step 10: Segmenting actions...")
        boundaries = self.action_segmenter.segment(timeline)
        segments = self.action_segmenter.boundaries_to_segments(boundaries, timeline)
        segments = self.skill_boundary_detector.refine_segments(segments, timeline)
        segments = self.action_classifier.classify_segments(segments, timeline)
        self.action_classifier.add_segments_to_timeline(timeline, segments)
        logger.info(f"  Created {len(segments)} action segments")

        # Step 11: Extract preconditions and postconditions
        logger.info("Step 11: Extracting pre/postconditions...")
        preconditions = self.precondition_extractor.extract(segments, timeline)
        postconditions = self.postcondition_extractor.extract(segments, timeline)

        # Step 12: Build task graph
        logger.info("Step 12: Building task graph...")
        edges = self.ordering_inferencer.infer_edges(
            segments, preconditions, postconditions, timeline
        )
        graph = self.task_graph_builder.build(
            segments, edges, preconditions, postconditions
        )
        logger.info(
            f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        # Step 13: Egocentric transform
        ego_series: Optional[EgocentricTimeSeries] = None
        if self.egocentric_stage is not None:
            logger.info("Step 13: Building egocentric view...")
            ego_series = self.egocentric_stage.run(timeline, frames)
            counts = summarize_series(ego_series)
            logger.info(
                "  Ego: %d frames, %d with head pose, %d with hands, %d with objects",
                counts["num_frames"],
                counts["frames_with_head"],
                counts["frames_with_left_hand"] + counts["frames_with_right_hand"],
                counts["frames_with_objects"],
            )

        # Step 14: Validate timeline
        logger.info("Step 14: Validating timeline...")
        valid, warnings = self.temporal_alignment.validate_timeline(timeline)
        if not valid:
            for w in warnings:
                logger.warning(f"  {w}")
        else:
            logger.info("  Timeline valid")

        # Step 15: Export outputs
        logger.info("Step 15: Exporting outputs...")
        output_paths = self._export_outputs(
            timeline, output_dir, run_id, ego_series
        )

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f}s")

        return output_paths

    def _load_video(self, video_path: str):
        """Load video frames and timestamps.

        Args:
            video_path: Path to video file.

        Returns:
            Tuple of (frames, timestamps, metadata).
        """
        with self.video_loader as loader:
            frames, timestamps, metadata = loader.read_all_frames(video_path)
        return frames, timestamps, metadata

    def _export_outputs(
        self,
        timeline: Timeline,
        output_dir: str,
        run_id: str,
        ego_series: Optional[EgocentricTimeSeries] = None,
    ) -> Dict[str, str]:
        """Export all outputs.

        Args:
            timeline: Processed timeline.
            output_dir: Output directory.
            run_id: Run identifier.
            ego_series: Egocentric time series, when the stage ran.

        Returns:
            Dictionary mapping output type to file path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Export JSON files
        json_exporter = JSONExporter(
            output_dir=str(output_path),
            indent=self.config.output.json_indent,
            run_id=run_id,
        )
        json_paths = json_exporter.export_all(timeline)
        paths.update(json_paths)

        # Export graph files
        graph_exporter = GraphExporter(
            output_dir=str(output_path),
            indent=self.config.output.json_indent,
        )
        graph_paths = graph_exporter.export_all(self.task_graph_builder)
        paths.update({f"graph_{k}": v for k, v in graph_paths.items()})

        # Export egocentric outputs
        if ego_series is not None:
            paths.update(
                self._export_egocentric(ego_series, timeline, output_path, run_id)
            )

        logger.info(f"Exported {len(paths)} output files to {output_path}")
        return paths

    def _export_egocentric(
        self,
        ego_series: EgocentricTimeSeries,
        timeline: Timeline,
        output_path: Path,
        run_id: str,
    ) -> Dict[str, str]:
        """Write the egocentric JSON and, when configured, the render video.

        Args:
            ego_series: Egocentric time series.
            timeline: Processed timeline, for metadata and object classes.
            output_path: Output directory.
            run_id: Run identifier.

        Returns:
            Dictionary mapping output type to file path.
        """
        cfg = self.config.egocentric
        paths: Dict[str, str] = {}
        fps = timeline.fps_extracted or self.config.video.target_fps or 30.0

        if cfg.export_frames:
            ego_json = output_path / "egocentric.json"
            export_egocentric_json(
                ego_series,
                str(ego_json),
                video_info={
                    "run_id": run_id,
                    "fps": fps,
                    "num_frames": timeline.num_frames,
                    "width": timeline.frames[0].width if timeline.frames else 0,
                    "height": timeline.frames[0].height if timeline.frames else 0,
                    "units": "m",
                    "frame": "egocentric: x=right, y=down, z=forward",
                },
                action_labels=[
                    {
                        "frame_start": seg.frame_start,
                        "frame_end": seg.frame_end,
                        "action": seg.label,
                        "object_id": (
                            seg.objects_involved[0] if seg.objects_involved else None
                        ),
                    }
                    for seg in timeline.segments
                ],
                indent=self.config.output.json_indent,
            )
            paths["egocentric"] = str(ego_json)

        if cfg.render and self.egocentric_stage is not None:
            object_classes = {
                oid: track.class_name for oid, track in timeline.objects.items()
            }
            rendered = self.egocentric_stage.render_video(
                ego_series,
                str(output_path / "egocentric.mp4"),
                fps=fps,
                object_classes=object_classes,
            )
            if rendered:
                paths["egocentric_video"] = rendered

        return paths


def run_pipeline(
    video_path: str,
    config_path: str,
    output_dir: str,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """Run the complete pipeline.

    Args:
        video_path: Path to input video.
        config_path: Path to YAML configuration file.
        output_dir: Directory for output files.
        run_id: Optional run identifier.

    Returns:
        Dictionary mapping output type to file path.
    """
    # Load config
    config = load_config(config_path)

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    return pipeline.process(video_path, output_dir, run_id)
