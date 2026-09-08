#!/usr/bin/env python
"""Video file test bench for step-by-step visualization of pipeline."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bird2ego.pose import PoseEstimator2D, PoseLifter3D, PoseTracker, SmoothingMethod
from bird2ego.objects import ObjectDetector, ObjectTracker, StateClassifier, TrajectoryBuilder
from bird2ego.contact import ContactDetector, ContactDetectorConfig, InteractionClassifier
from bird2ego.actions import ActionSegmenter, ActionClassifier
from bird2ego.utils.timeline import (
    COCO17_JOINT_NAMES,
    JOINT_IDX,
    NUM_JOINTS,
    SENTINEL_2D,
    SENTINEL_BBOX,
    PersonPose,
    Timeline,
    create_empty_timeline,
)
from bird2ego.video import VideoLoader, TemporalAlignment


# Colors for visualization (BGR)
COLORS = {
    "pose": (0, 255, 0),
    "pose_line": (0, 200, 0),
    "bbox_person": (255, 0, 0),
    "bbox_object": (0, 165, 255),
    "contact": (0, 0, 255),
    "text": (255, 255, 255),
    "text_bg": (0, 0, 0),
    "segment": (255, 255, 0),
}

SKELETON_CONNECTIONS = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


class VideoTestBench:
    """Test bench for processing and visualizing video files."""

    def __init__(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        show_pose: bool = True,
        show_objects: bool = True,
        show_contacts: bool = True,
        show_actions: bool = True,
    ):
        """Initialize video test bench.

        Args:
            video_path: Path to video file.
            target_fps: Target FPS for processing (None = original).
            show_pose: Show pose visualization.
            show_objects: Show object visualization.
            show_contacts: Show contact visualization.
            show_actions: Show action segment visualization.
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.show_pose = show_pose
        self.show_objects = show_objects
        self.show_contacts = show_contacts
        self.show_actions = show_actions

        # Load video
        self._load_video()

        # Initialize components
        self._init_components()

        # Process video
        self._process_video()

    def _load_video(self):
        """Load video file."""
        print(f"Loading video: {self.video_path}")
        self.loader = VideoLoader(target_fps=self.target_fps)
        self.frames, self.timestamps, self.metadata = self.loader.read_all_frames(self.video_path)
        print(f"  Loaded {len(self.frames)} frames at {self.metadata.fps:.2f} fps")
        print(f"  Resolution: {self.metadata.width}x{self.metadata.height}")
        print(f"  Duration: {self.metadata.duration:.2f}s")

        # Create timeline
        self.aligner = TemporalAlignment()
        self.timeline = self.aligner.create_timeline_from_video(
            fps=self.metadata.fps,
            frame_count=len(self.frames),
            width=self.metadata.width,
            height=self.metadata.height,
            timestamps=self.timestamps,
        )

    def _init_components(self):
        """Initialize pipeline components."""
        # Pose
        self.pose_estimator = PoseEstimator2D(backend="stub", seed=42)
        self.pose_lifter = PoseLifter3D(backend="stub")
        self.pose_smoother = PoseTracker(method=SmoothingMethod.ONE_EURO)

        # Objects
        self.object_detector = ObjectDetector(backend="stub", num_objects=3, seed=42)
        self.object_tracker = ObjectTracker(backend="iou")
        self.trajectory_builder = TrajectoryBuilder()
        self.state_classifier = StateClassifier()

        # Contacts
        self.contact_detector = ContactDetector()
        self.interaction_classifier = InteractionClassifier()

        # Actions
        self.action_segmenter = ActionSegmenter()
        self.action_classifier = ActionClassifier()

    def _process_video(self):
        """Process the entire video through the pipeline."""
        print("\nProcessing video...")

        # Pose estimation
        print("  Estimating poses...")
        self.pose_frames = self.pose_estimator.estimate_video(self.frames)
        self.pose_frames = self.pose_lifter.lift_video(
            self.pose_frames, (self.metadata.width, self.metadata.height)
        )
        self.pose_frames = self.pose_smoother.smooth_poses(self.pose_frames)
        self.timeline.person_pose = PersonPose(frames=self.pose_frames)

        # Object detection and tracking
        print("  Detecting and tracking objects...")
        detections = self.object_detector.detect_video(self.frames)
        self.object_tracks = self.trajectory_builder.build_from_detections(
            detections, self.object_tracker
        )
        self.trajectory_builder.add_tracks_to_timeline(self.timeline, self.object_tracks)

        # State classification
        print("  Classifying object states...")
        self.state_classifier.classify_motion_only(self.timeline.objects)

        # Contact detection
        print("  Detecting contacts...")
        self.contacts_per_frame = self.contact_detector.detect_video(
            self.timeline.person_pose, self.timeline.objects
        )
        contact_frames = self.contact_detector.get_contact_frames_per_object(
            self.contacts_per_frame
        )

        # Classify interactions
        print("  Classifying interactions...")
        self.interactions = self.interaction_classifier.classify_interactions(
            self.contacts_per_frame,
            self.timeline.person_pose,
            self.timeline.objects,
            fps=self.metadata.fps,
        )

        # Refine states with contacts
        self.state_classifier.classify_with_contacts(
            self.timeline.objects,
            contact_frames=contact_frames,
        )

        # Action segmentation
        print("  Segmenting actions...")
        boundaries = self.action_segmenter.segment(self.timeline)
        self.segments = self.action_segmenter.boundaries_to_segments(boundaries, self.timeline)
        self.segments = self.action_classifier.classify_segments(self.segments, self.timeline)
        self.timeline.segments = self.segments

        print(f"  Found {len(self.segments)} action segments")
        print("Processing complete!")

    def run(self):
        """Run the interactive visualization."""
        print("\n" + "=" * 60)
        print("Video Test Bench - Interactive Viewer")
        print("=" * 60)
        print("Controls:")
        print("  SPACE    - Play/Pause")
        print("  LEFT     - Previous frame")
        print("  RIGHT    - Next frame")
        print("  HOME     - First frame")
        print("  END      - Last frame")
        print("  P        - Toggle pose")
        print("  O        - Toggle objects")
        print("  C        - Toggle contacts")
        print("  A        - Toggle actions")
        print("  S        - Save current frame")
        print("  Q / ESC  - Quit")
        print("=" * 60)

        current_frame = 0
        playing = False
        last_time = time.time()
        playback_fps = self.metadata.fps

        while True:
            # Get visualization
            vis_frame = self._visualize_frame(current_frame)

            # Add playback info
            status = "PLAYING" if playing else "PAUSED"
            cv2.putText(vis_frame, f"[{status}] Frame {current_frame}/{len(self.frames)-1}",
                       (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Video Test Bench", vis_frame)

            # Handle playback
            wait_time = 1 if not playing else int(1000 / playback_fps)
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                playing = not playing
            elif key == 81 or key == 2424832:  # LEFT
                playing = False
                current_frame = max(0, current_frame - 1)
            elif key == 83 or key == 2555904:  # RIGHT
                playing = False
                current_frame = min(len(self.frames) - 1, current_frame + 1)
            elif key == 80 or key == 2359296:  # HOME
                playing = False
                current_frame = 0
            elif key == 87 or key == 2293760:  # END
                playing = False
                current_frame = len(self.frames) - 1
            elif key == ord('p'):
                self.show_pose = not self.show_pose
            elif key == ord('o'):
                self.show_objects = not self.show_objects
            elif key == ord('c'):
                self.show_contacts = not self.show_contacts
            elif key == ord('a'):
                self.show_actions = not self.show_actions
            elif key == ord('s'):
                filename = f"frame_{current_frame:05d}.png"
                cv2.imwrite(filename, vis_frame)
                print(f"Saved: {filename}")

            # Auto-advance when playing
            if playing:
                current_time = time.time()
                if current_time - last_time >= 1.0 / playback_fps:
                    last_time = current_time
                    current_frame = min(len(self.frames) - 1, current_frame + 1)
                    if current_frame >= len(self.frames) - 1:
                        playing = False

        cv2.destroyAllWindows()

    def _visualize_frame(self, frame_idx: int) -> np.ndarray:
        """Create visualization for a single frame."""
        frame = self.frames[frame_idx].copy()

        # Draw objects
        if self.show_objects:
            for obj_id, track in self.timeline.objects.items():
                if frame_idx < track.num_frames:
                    obj_frame = track.frames[frame_idx]
                    bbox = obj_frame.bbox_xyxy
                    if bbox != SENTINEL_BBOX:
                        pt1 = (int(bbox[0]), int(bbox[1]))
                        pt2 = (int(bbox[2]), int(bbox[3]))
                        cv2.rectangle(frame, pt1, pt2, COLORS["bbox_object"], 2)

                        # State info
                        state = obj_frame.state
                        label = f"#{obj_id} {track.class_name}"
                        state_str = f"{state.motion_state.value}"
                        cv2.putText(frame, label, (pt1[0], pt1[1] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["bbox_object"], 1)
                        cv2.putText(frame, state_str, (pt1[0], pt1[1] - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["bbox_object"], 1)

        # Draw pose
        if self.show_pose and frame_idx < len(self.pose_frames):
            pose_frame = self.pose_frames[frame_idx]
            keypoints = pose_frame.keypoints_2d_px
            conf = pose_frame.conf_2d

            # Skeleton
            for j1_name, j2_name in SKELETON_CONNECTIONS:
                idx1, idx2 = JOINT_IDX[j1_name], JOINT_IDX[j2_name]
                if (keypoints[idx1] != SENTINEL_2D and keypoints[idx2] != SENTINEL_2D and
                    conf[idx1] > 0.2 and conf[idx2] > 0.2):
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    cv2.line(frame, pt1, pt2, COLORS["pose_line"], 2)

            # Keypoints
            for j in range(NUM_JOINTS):
                if keypoints[j] != SENTINEL_2D and conf[j] > 0.2:
                    pt = (int(keypoints[j][0]), int(keypoints[j][1]))
                    cv2.circle(frame, pt, 4, COLORS["pose"], -1)

        # Draw contacts
        if self.show_contacts and frame_idx < len(self.contacts_per_frame):
            for cf in self.contacts_per_frame[frame_idx]:
                if cf.is_contact:
                    # Draw line from hand to object
                    hand_pt = (int(cf.hand_position[0]), int(cf.hand_position[1]))
                    obj_pt = (int(cf.object_center[0]), int(cf.object_center[1]))
                    cv2.line(frame, hand_pt, obj_pt, COLORS["contact"], 2)
                    cv2.circle(frame, hand_pt, 10, COLORS["contact"], 2)

        # Draw action segment
        if self.show_actions:
            current_segment = None
            for seg in self.segments:
                if seg.frame_start <= frame_idx <= seg.frame_end:
                    current_segment = seg
                    break

            if current_segment:
                # Draw segment bar at bottom
                h = frame.shape[0]
                progress = (frame_idx - current_segment.frame_start) / max(1, current_segment.frame_end - current_segment.frame_start)
                bar_width = int(frame.shape[1] * 0.3)
                bar_x = frame.shape[1] - bar_width - 10
                cv2.rectangle(frame, (bar_x, h - 40), (bar_x + bar_width, h - 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, h - 40), (bar_x + int(bar_width * progress), h - 20), COLORS["segment"], -1)

                # Label
                label = f"Action: {current_segment.label} ({current_segment.conf:.2f})"
                cv2.putText(frame, label, (bar_x, h - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["segment"], 1)

        # Info overlay
        info_y = 20
        cv2.putText(frame, f"Time: {self.timestamps[frame_idx]:.2f}s", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

        return frame

    def export_video(self, output_path: str, include_annotations: bool = True):
        """Export processed video with annotations.

        Args:
            output_path: Output video path.
            include_annotations: Whether to include visualization annotations.
        """
        print(f"Exporting video to {output_path}...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.metadata.fps,
            (self.metadata.width, self.metadata.height),
        )

        for i in range(len(self.frames)):
            if include_annotations:
                frame = self._visualize_frame(i)
            else:
                frame = self.frames[i]
            out.write(frame)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(self.frames)} frames")

        out.release()
        print(f"Exported to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video file test bench for vision pipeline",
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for processing (default: original)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export annotated video to this path",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose visualization",
    )
    parser.add_argument(
        "--no-objects",
        action="store_true",
        help="Disable object visualization",
    )
    parser.add_argument(
        "--no-contacts",
        action="store_true",
        help="Disable contact visualization",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Disable action visualization",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1

    bench = VideoTestBench(
        video_path=args.video,
        target_fps=args.fps,
        show_pose=not args.no_pose,
        show_objects=not args.no_objects,
        show_contacts=not args.no_contacts,
        show_actions=not args.no_actions,
    )

    if args.export:
        bench.export_video(args.export)
    else:
        bench.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
