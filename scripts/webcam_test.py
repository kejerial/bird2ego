#!/usr/bin/env python
"""Webcam test bench for visualizing pipeline components in real-time.

Uses REAL pose estimation (MediaPipe) and object detection (YOLO) by default.
Works from any camera angle: front, side, back, diagonal.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bird2ego.pose import PoseEstimator2D, PoseLifter3D
from bird2ego.objects import ObjectDetector, ObjectTracker
from bird2ego.contact import ContactDetector, ContactDetectorConfig
from bird2ego.contact.hand_detector import HandDetector, ImprovedContactDetector, FINGERTIPS
from bird2ego.egocentric import EgocentricTransformer, EgocentricRenderer
from bird2ego.utils.timeline import (
    COCO17_JOINT_NAMES,
    JOINT_IDX,
    NUM_JOINTS,
    SENTINEL_2D,
    SENTINEL_BBOX,
    PersonPose,
    ObjectTrack,
    create_empty_object_track,
)


# Colors for visualization (BGR)
COLORS = {
    "pose": (0, 255, 0),       # Green
    "pose_line": (0, 200, 0),  # Dark green
    "bbox_person": (255, 0, 0),  # Blue
    "bbox_object": (0, 165, 255),  # Orange
    "contact": (0, 0, 255),    # Red
    "grab": (0, 0, 255),       # Red
    "hand": (255, 200, 100),   # Light blue
    "fingertip": (0, 255, 255), # Yellow
    "text": (255, 255, 255),   # White
    "text_bg": (0, 0, 0),      # Black
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # Head
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Left arm
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    # Right arm
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    # Left leg
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


class WebcamTestBench:
    """Real-time test bench for visualizing pipeline components."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        show_pose: bool = True,
        show_objects: bool = True,
        show_contacts: bool = True,
        show_3d: bool = False,
        show_ego: bool = False,  # Egocentric view overlay
        pose_backend: str = "mediapipe",
        object_backend: str = "yolo",
        # Performance options
        skip_frames: int = 1,  # Skip 1 frame (process every 2nd) for better performance
        lite_mode: bool = True,  # Use lighter models by default for better speed
    ):
        """Initialize the test bench.

        Args:
            camera_id: Camera device ID.
            width: Frame width.
            height: Frame height.
            fps: Target frame rate.
            show_pose: Whether to show pose estimation.
            show_objects: Whether to show object detection.
            show_contacts: Whether to show contact detection.
            show_3d: Whether to show 3D pose visualization.
            show_ego: Whether to show egocentric view overlay.
            pose_backend: "mediapipe" (real) or "stub" (fake).
            object_backend: "yolo" (real) or "stub" (fake).
            skip_frames: Skip N frames between heavy detection (0=all, 2=every 3rd).
            lite_mode: Use lighter/faster models.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.show_pose = show_pose
        self.show_objects = show_objects
        self.show_contacts = show_contacts
        self.show_3d = show_3d
        self.show_ego = show_ego
        self.pose_backend = pose_backend
        self.object_backend = object_backend
        self.skip_frames = skip_frames
        self.lite_mode = lite_mode

        # Initialize components
        self._init_components()

        # State
        self.frame_idx = 0
        self.last_time = time.time()
        self.fps_display = 0.0
        
        # Cached results for frame skipping
        self.cached_objects = []
        self.cached_tracked_objects = []

        # Object tracking state
        self.object_tracks: Dict[int, ObjectTrack] = {}

    def _init_components(self):
        """Initialize pipeline components with real backends."""
        print(f"\nInitializing components...")
        print(f"  Pose backend: {self.pose_backend}")
        print(f"  Object backend: {self.object_backend}")
        if self.lite_mode:
            print("  Using optimized models for speed")
        if self.skip_frames > 0:
            print(f"  Frame skip: YOLO runs every {self.skip_frames + 1} frames (pose/hands every frame)")
        
        # Pose estimation - MediaPipe works from ANY angle
        if self.pose_backend == "mediapipe":
            # Use lite model (complexity=0) for better performance
            # Full model (complexity=1) is slower but slightly more accurate
            model_complexity = 0  # Always use lite for better speed
            self.pose_estimator = PoseEstimator2D(
                backend="mediapipe",
                min_confidence=0.5,  # Higher threshold to filter false detections
                model_complexity=model_complexity,  # 0=lite (fast), 1=full (slower)
                min_detection_confidence=0.5,  # Higher to reduce false positives
                min_tracking_confidence=0.5,
            )
            print(f"  MediaPipe Pose loaded (complexity={model_complexity}, optimized for speed)")
        else:
            self.pose_estimator = PoseEstimator2D(
                backend="stub",
                min_confidence=0.2,
                seed=42,
            )
            print("  Stub pose estimator (fake data)")
        
        self.pose_lifter = PoseLifter3D(
            backend="stub",  # 3D lifting still uses stub
            min_confidence=0.2,
        )

        # Object detection - YOLO works from ANY angle
        # Exclude person (class 0) since we track that with pose estimation
        if self.object_backend == "yolo":
            # Common object classes to detect (excluding person=0)
            object_classes = list(range(1, 80))  # All classes except person
            
            self.object_detector = ObjectDetector(
                backend="yolo",
                confidence_threshold=0.4,  # Balanced threshold
                model_name="yolov8n.pt",  # Nano model for speed
                conf_threshold=0.4,
                classes=object_classes,  # Exclude person
                verbose=False,
            )
            print("  YOLOv8n loaded (nano model for speed)")
        else:
            self.object_detector = ObjectDetector(
                backend="stub",
                confidence_threshold=0.5,
                num_objects=3,
                seed=42,
            )
            print("  Stub object detector (fake data)")
        
        self.object_tracker = ObjectTracker(
            backend="iou",
            iou_threshold=0.3,
            max_age=30,
            min_hits=2,  # Faster confirmation
        )

        # Hand detection (detailed 21-point hand tracking)
        # Balanced thresholds for speed and detection quality
        det_conf = 0.4  # Good balance for detection
        track_conf = 0.4  # Good balance for tracking
        self.hand_detector = HandDetector(
            num_hands=2,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        
        # Improved contact detection using hand landmarks
        self.contact_detector_improved = ImprovedContactDetector(
            contact_distance=30.0,  # Fingertip to object distance
            grab_distance=50.0,
            temporal_smoothing=5,
        )
        
        # Legacy contact detector (fallback)
        contact_config = ContactDetectorConfig(
            contact_distance_threshold=60.0,
            min_joint_conf_2d=0.3,
            contact_onset_frames=2,
            contact_offset_frames=3,
            use_hand_center=True,
        )
        self.contact_detector = ContactDetector(config=contact_config)

        # Egocentric view transformation
        self.ego_transformer = EgocentricTransformer(
            eye_offset_forward=0.1,
            default_gaze_down_angle=30.0,
            pinch_threshold=0.05,
        )
        # Full-size renderer for separate window (VR-like view)
        self.ego_renderer = EgocentricRenderer(
            width=960,
            height=720,
            fov_horizontal=90.0,
            show_grid=True,
            show_horizon=True,
        )
        self.current_ego_frame = None
        
        # Egocentric recording
        self.ego_recording = False
        self.ego_writer = None
        self.ego_record_path = None
        self.ego_frames_recorded = 0
        
        # State
        self.pose_history: List = []
        self.current_hands = []
        self.current_contacts = []
        print("  Components ready!\n")

    def run(self):
        """Run the webcam test bench."""
        print("=" * 60)
        print("Webcam Test Bench - REAL Detection")
        print("=" * 60)
        print("This uses REAL pose estimation and object detection!")
        print("Works from ANY camera angle: front, side, back, diagonal")
        print("")
        print("Controls:")
        print("  Q / ESC  - Quit")
        print("  P        - Toggle pose visualization")
        print("  O        - Toggle object visualization")
        print("  C        - Toggle contact visualization")
        print("  3        - Toggle 3D pose view")
        print("  E        - Toggle egocentric (first-person) view")
        print("  V        - Start/Stop recording egocentric MP4")
        print("  R        - Reset trackers")
        print("  S        - Save screenshot")
        print("=" * 60)

        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        print("\nWave your arms around! Point at objects!")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Mirror the frame for more intuitive interaction
                frame = cv2.flip(frame, 1)

                # Process frame
                vis_frame = self._process_frame(frame)

                # Show frame
                cv2.imshow("Vision Pipeline - Real Detection", vis_frame)

                # Show 3D view if enabled
                if self.show_3d and len(self.pose_history) > 0:
                    vis_3d = self._draw_3d_view()
                    cv2.imshow("3D Pose View", vis_3d)
                
                # Show egocentric view in separate window
                if self.show_ego and self.current_ego_frame is not None:
                    ego_view = self._render_egocentric_window()
                    cv2.imshow("Egocentric View (First Person)", ego_view)
                    
                    # Write to recording if active
                    if self.ego_recording and self.ego_writer is not None:
                        self.ego_writer.write(ego_view)
                        self.ego_frames_recorded += 1
                # Note: Don't try to destroy window here - it causes errors if window never existed

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('p'):
                    self.show_pose = not self.show_pose
                    print(f"Pose visualization: {'ON' if self.show_pose else 'OFF'}")
                elif key == ord('o'):
                    self.show_objects = not self.show_objects
                    print(f"Object visualization: {'ON' if self.show_objects else 'OFF'}")
                elif key == ord('c'):
                    self.show_contacts = not self.show_contacts
                    print(f"Contact visualization: {'ON' if self.show_contacts else 'OFF'}")
                elif key == ord('3'):
                    self.show_3d = not self.show_3d
                    if not self.show_3d:
                        try:
                            cv2.destroyWindow("3D Pose View")
                        except cv2.error:
                            pass  # Window may not exist yet
                    print(f"3D view: {'ON' if self.show_3d else 'OFF'}")
                elif key == ord('e'):
                    self.show_ego = not self.show_ego
                    if not self.show_ego:
                        try:
                            cv2.destroyWindow("Egocentric View (First Person)")
                        except cv2.error:
                            pass  # Window may not exist yet
                        # Stop recording if active
                        if self.ego_recording:
                            self._stop_ego_recording()
                    print(f"Egocentric view: {'ON' if self.show_ego else 'OFF'}")
                elif key == ord('v'):
                    if self.show_ego:
                        if self.ego_recording:
                            self._stop_ego_recording()
                        else:
                            self._start_ego_recording()
                    else:
                        print("Enable egocentric view (E) first to record")
                elif key == ord('r'):
                    self._reset()
                    print("Trackers reset")
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, vis_frame)
                    print(f"Saved screenshot: {filename}")

                self.frame_idx += 1

        finally:
            # Stop egocentric recording if active
            if self.ego_recording:
                self._stop_ego_recording()
            
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return visualization."""
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.fps_display = 0.9 * self.fps_display + 0.1 * (1.0 / max(dt, 0.001))

        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Determine if we should run heavy detection this frame
        run_detection = (self.skip_frames == 0) or (self.frame_idx % (self.skip_frames + 1) == 0)

        # Run pose estimation (body skeleton) - always run, it's fast
        pose_frame = None
        if self.show_pose:
            pose_frame = self.pose_estimator.estimate_frame(frame, self.frame_idx)
            pose_frame = self.pose_lifter.lift_frame(pose_frame, (w, h))
            self.pose_history.append(pose_frame)
            if len(self.pose_history) > 30:
                self.pose_history.pop(0)

        # Run hand detection - always run, important for interaction
        # Only detect hands that MediaPipe can see (no prediction when occluded)
        self.current_hands = []
        if self.show_contacts or self.show_pose:
            self.current_hands = self.hand_detector.detect(frame)

        # Run object detection and tracking
        # Only run YOLO every N frames when skip_frames > 0
        tracked_objects = []
        if self.show_objects or self.show_contacts:
            if run_detection:
                # Full detection
                detections = self.object_detector.detect_frame(frame)
                results = self.object_tracker.update(detections, self.frame_idx)
                tracked_objects = results
                self.cached_tracked_objects = results  # Cache for skipped frames

                # Update tracks dictionary
                for track_id, det in results:
                    if track_id not in self.object_tracks:
                        self.object_tracks[track_id] = create_empty_object_track(
                            object_id=track_id,
                            class_name=det.class_name,
                            class_id=det.class_id,
                            num_frames=1,
                        )
                    # Update track frame
                    self.object_tracks[track_id].frames[0].bbox_xyxy = det.bbox_xyxy
                    self.object_tracks[track_id].frames[0].conf = det.confidence
            else:
                # Use cached results
                tracked_objects = self.cached_tracked_objects

        # Draw visualizations
        if self.show_objects:
            vis_frame = self._draw_objects(vis_frame, tracked_objects)

        if self.show_pose and pose_frame is not None:
            vis_frame = self._draw_pose(vis_frame, pose_frame)

        if self.show_contacts and pose_frame is not None:
            vis_frame = self._draw_contacts(vis_frame, pose_frame, tracked_objects)

        # Compute egocentric frame (for separate window, not overlay)
        if self.show_ego:
            self.current_ego_frame = self._compute_egocentric(
                pose_frame, tracked_objects
            )

        # Draw info overlay
        vis_frame = self._draw_info(vis_frame)

        return vis_frame
    
    def _compute_egocentric(self, pose_frame, tracked_objects) -> 'EgocentricFrame':
        """Compute egocentric frame from current data.
        
        This creates a TRUE first-person POV by:
        1. Estimating body orientation from shoulders
        2. Computing body-relative arm positions
        3. Rendering from a fixed first-person viewpoint
        
        The egocentric view should ALWAYS look like first-person, regardless
        of how the person is oriented to the camera.
        """
        from bird2ego.egocentric.transformer import EgocentricFrame
        
        timestamp = self.frame_idx / self.fps
        
        # Initialize outputs
        left_hand_ego = None
        left_hand_conf = 0.0
        left_pinching = False
        left_pinch_dist = 0.0
        right_hand_ego = None
        right_hand_conf = 0.0
        right_pinching = False
        right_pinch_dist = 0.0
        left_arm_ego = None
        right_arm_ego = None
        objects_ego = {}
        
        # Get body orientation from pose
        body_center = None
        body_right = None  # Direction from left to right shoulder
        body_forward = None  # Direction person is facing
        shoulder_width = 0.0
        
        if pose_frame is not None:
            keypoints = np.array(pose_frame.keypoints_2d_px)
            conf = np.array(pose_frame.conf_2d)
            
            # Get shoulder positions (indices 5=left_shoulder, 6=right_shoulder)
            left_shoulder_conf = conf[5]
            right_shoulder_conf = conf[6]
            
            if left_shoulder_conf > 0.3 and right_shoulder_conf > 0.3:
                left_shoulder = keypoints[5]  # Person's left shoulder
                right_shoulder = keypoints[6]  # Person's right shoulder
                
                # Body center between shoulders
                body_center = (left_shoulder + right_shoulder) / 2
                
                # Right direction (from person's left to right shoulder)
                shoulder_vec = right_shoulder - left_shoulder
                shoulder_width = np.linalg.norm(shoulder_vec)
                
                if shoulder_width > 10:  # Minimum pixel distance
                    body_right = shoulder_vec / shoulder_width
                    # Forward is perpendicular to right (in 2D, this is approximate)
                    body_forward = np.array([-body_right[1], body_right[0]])
        
        # If we don't have body orientation, use defaults
        if body_center is None:
            body_center = np.array([self.width / 2, self.height / 2])
        if body_right is None:
            body_right = np.array([1.0, 0.0])
            body_forward = np.array([0.0, -1.0])
        if shoulder_width < 10:
            shoulder_width = self.width * 0.3  # Default shoulder width
        
        # Compute arms in body-relative coordinates
        if pose_frame is not None:
            keypoints = np.array(pose_frame.keypoints_2d_px)
            conf = np.array(pose_frame.conf_2d)
            
            # Process LEFT arm (person's left = appears on left in egocentric)
            # Indices: 5=left_shoulder, 7=left_elbow, 9=left_wrist
            if conf[5] > 0.3 and conf[7] > 0.3 and conf[9] > 0.3:
                left_arm_ego = np.zeros((3, 3), dtype=np.float32)
                for i, idx in enumerate([5, 7, 9]):
                    pt = keypoints[idx]
                    rel = pt - body_center  # Relative to body center
                    
                    # Project onto body axes
                    x_body = np.dot(rel, body_right) / shoulder_width  # Left/right
                    y_body = np.dot(rel, body_forward) / shoulder_width  # Forward/back (depth estimate)
                    
                    # In egocentric view:
                    # x = left/right (-1 = left, +1 = right)
                    # y = vertical position (always below eye level for arms)
                    # z = depth (forward)
                    
                    # Map to fixed egocentric positions
                    # Arms should appear in lower corners of view
                    ego_x = -0.6 - 0.3 * (1 - i/2)  # Left side, shoulder further out
                    ego_y = 0.3 + i * 0.15  # Lower in view, wrist lowest
                    ego_z = 0.4 + i * 0.1 + abs(y_body) * 0.2  # Depth based on forward position
                    
                    # Add some variation based on actual arm position
                    ego_x += x_body * 0.3
                    ego_y += (pt[1] - body_center[1]) / self.height * 0.5
                    
                    left_arm_ego[i] = [ego_x, ego_y, ego_z]
            
            # Process RIGHT arm (person's right = appears on right in egocentric)
            # Indices: 6=right_shoulder, 8=right_elbow, 10=right_wrist
            if conf[6] > 0.3 and conf[8] > 0.3 and conf[10] > 0.3:
                right_arm_ego = np.zeros((3, 3), dtype=np.float32)
                for i, idx in enumerate([6, 8, 10]):
                    pt = keypoints[idx]
                    rel = pt - body_center
                    
                    x_body = np.dot(rel, body_right) / shoulder_width
                    y_body = np.dot(rel, body_forward) / shoulder_width
                    
                    # Right arm on right side
                    ego_x = 0.6 + 0.3 * (1 - i/2)  # Right side
                    ego_y = 0.3 + i * 0.15
                    ego_z = 0.4 + i * 0.1 + abs(y_body) * 0.2
                    
                    ego_x += x_body * 0.3
                    ego_y += (pt[1] - body_center[1]) / self.height * 0.5
                    
                    right_arm_ego[i] = [ego_x, ego_y, ego_z]
        
        # Process hands - map to egocentric based on wrist position
        for hand in self.current_hands:
            if hand.landmarks is not None:
                landmarks_2d = hand.landmarks
                wrist_2d = landmarks_2d[0]
                
                # Determine if this hand is on left or right side of body
                wrist_rel = wrist_2d - body_center
                x_body = np.dot(wrist_rel, body_right) / shoulder_width
                
                # Determine hand offset in egocentric view
                if x_body < 0:  # Person's left hand
                    base_x = -0.5
                    is_left = True
                else:  # Person's right hand
                    base_x = 0.5
                    is_left = False
                
                # Convert all hand landmarks to egocentric
                ego_3d = np.zeros((21, 3), dtype=np.float32)
                
                # Get hand bounding box for scaling
                hand_min = landmarks_2d.min(axis=0)
                hand_max = landmarks_2d.max(axis=0)
                hand_size = max(hand_max[0] - hand_min[0], hand_max[1] - hand_min[1])
                hand_center = (hand_min + hand_max) / 2
                
                for i in range(21):
                    # Position relative to hand center, normalized
                    rel_to_hand = (landmarks_2d[i] - hand_center) / max(hand_size, 1)
                    
                    # Scale and position in egocentric view
                    ego_x = base_x + rel_to_hand[0] * 0.4
                    ego_y = 0.4 + rel_to_hand[1] * 0.4  # Hands in lower half of view
                    
                    # Depth from MediaPipe z
                    if hand.landmarks_3d is not None:
                        ego_z = 0.5 + hand.landmarks_3d[i, 2] * 0.3
                    else:
                        ego_z = 0.5
                    
                    ego_3d[i] = [ego_x, ego_y, ego_z]
                
                # Calculate pinch
                thumb_tip = landmarks_2d[4]
                index_tip = landmarks_2d[8]
                pinch_dist = np.linalg.norm(thumb_tip - index_tip) / max(hand_size, 1)
                is_pinching = pinch_dist < 0.3
                
                if is_left:
                    left_hand_ego = ego_3d
                    left_hand_conf = hand.confidence
                    left_pinching = is_pinching
                    left_pinch_dist = pinch_dist
                else:
                    right_hand_ego = ego_3d
                    right_hand_conf = hand.confidence
                    right_pinching = is_pinching
                    right_pinch_dist = pinch_dist
        
        # Convert objects to egocentric
        # Objects should appear in front, positioned relative to body
        for track_id, det in tracked_objects:
            bbox = det.bbox_xyxy
            obj_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            # Position relative to body center
            rel = obj_center - body_center
            x_body = np.dot(rel, body_right) / shoulder_width
            y_offset = (obj_center[1] - body_center[1]) / self.height
            
            # Map to egocentric
            ego_x = x_body * 0.8  # Left/right based on body-relative position
            ego_y = y_offset  # Vertical offset
            
            # Depth from bbox size (smaller = further)
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            ego_z = max(0.4, 1.2 - bbox_size / max(self.width, self.height) * 1.5)
            
            objects_ego[track_id] = np.array([ego_x, ego_y, ego_z])
        
        return EgocentricFrame(
            frame_idx=self.frame_idx,
            timestamp=timestamp,
            head_position=np.array([0, 0, 0]),
            head_forward=np.array([0, 0, 1]),
            head_up=np.array([0, -1, 0]),
            head_right=np.array([1, 0, 0]),
            left_hand_ego=left_hand_ego,
            right_hand_ego=right_hand_ego,
            left_hand_pinching=left_pinching,
            left_pinch_distance=left_pinch_dist,
            right_hand_pinching=right_pinching,
            right_pinch_distance=right_pinch_dist,
            left_arm_ego=left_arm_ego,
            right_arm_ego=right_arm_ego,
            objects_ego=objects_ego,
            head_confidence=0.9,
            left_hand_confidence=left_hand_conf,
            right_hand_confidence=right_hand_conf,
        )
    
    def _render_egocentric_window(self) -> np.ndarray:
        """Render full egocentric view for separate window."""
        if self.current_ego_frame is None:
            # Return blank frame
            return np.zeros((720, 960, 3), dtype=np.uint8)
        
        # Get object class names
        object_classes = {}
        for track_id, track in self.object_tracks.items():
            object_classes[track_id] = track.class_name
        
        # Render with full renderer
        ego_view = self.ego_renderer.render(
            self.current_ego_frame,
            object_classes=object_classes,
            show_labels=True,
        )
        
        # Add recording indicator
        if self.ego_recording:
            cv2.circle(ego_view, (ego_view.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                ego_view, f"REC {self.ego_frames_recorded}",
                (ego_view.shape[1] - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        
        return ego_view
    
    def _start_ego_recording(self):
        """Start recording egocentric view to MP4."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ego_record_path = f"data/processed/egocentric_{timestamp}.mp4"
        
        # Ensure directory exists
        import os
        os.makedirs("data/processed", exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.ego_writer = cv2.VideoWriter(
            self.ego_record_path,
            fourcc,
            self.fps,
            (960, 720)
        )
        
        self.ego_recording = True
        self.ego_frames_recorded = 0
        print(f"Recording started: {self.ego_record_path}")
    
    def _stop_ego_recording(self):
        """Stop recording egocentric view."""
        if self.ego_writer is not None:
            self.ego_writer.release()
            self.ego_writer = None
        
        self.ego_recording = False
        print(f"Recording stopped: {self.ego_frames_recorded} frames saved to {self.ego_record_path}")
        self.ego_record_path = None
        self.ego_frames_recorded = 0

    def _draw_pose(self, frame: np.ndarray, pose_frame) -> np.ndarray:
        """Draw pose skeleton on frame."""
        keypoints = pose_frame.keypoints_2d_px
        conf = pose_frame.conf_2d

        # Head keypoints (nose, eyes, ears) are more prone to false positives
        # Use higher confidence threshold for them
        HEAD_KEYPOINTS = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
        HEAD_CONF_THRESHOLD = 0.6  # Higher threshold for head keypoints
        BODY_CONF_THRESHOLD = 0.5  # Standard threshold for body joints

        def get_conf_threshold(joint_idx: int) -> float:
            """Get confidence threshold based on joint type."""
            return HEAD_CONF_THRESHOLD if joint_idx in HEAD_KEYPOINTS else BODY_CONF_THRESHOLD

        # Draw skeleton lines
        for joint1_name, joint2_name in SKELETON_CONNECTIONS:
            idx1 = JOINT_IDX[joint1_name]
            idx2 = JOINT_IDX[joint2_name]

            conf1_thresh = get_conf_threshold(idx1)
            conf2_thresh = get_conf_threshold(idx2)

            if (keypoints[idx1] != SENTINEL_2D and keypoints[idx2] != SENTINEL_2D and
                conf[idx1] > conf1_thresh and conf[idx2] > conf2_thresh):
                pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                cv2.line(frame, pt1, pt2, COLORS["pose_line"], 3)

        # Draw keypoints
        for j in range(NUM_JOINTS):
            conf_thresh = get_conf_threshold(j)
            if keypoints[j] != SENTINEL_2D and conf[j] > conf_thresh:
                pt = (int(keypoints[j][0]), int(keypoints[j][1]))
                # Size based on confidence
                radius = int(4 + 6 * conf[j])
                cv2.circle(frame, pt, radius, COLORS["pose"], -1)
                cv2.circle(frame, pt, radius, (0, 0, 0), 1)

        return frame

    def _draw_objects(
        self,
        frame: np.ndarray,
        tracked_objects: List[Tuple[int, any]],
    ) -> np.ndarray:
        """Draw object bounding boxes."""
        for track_id, det in tracked_objects:
            bbox = det.bbox_xyxy
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))

            # Draw bbox
            cv2.rectangle(frame, pt1, pt2, COLORS["bbox_object"], 2)

            # Draw label
            label = f"{det.class_name} #{track_id} ({det.confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (pt1[0], pt1[1] - label_h - 5),
                         (pt1[0] + label_w, pt1[1]), COLORS["bbox_object"], -1)
            cv2.putText(frame, label, (pt1[0], pt1[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

        return frame

    def _draw_contacts(
        self,
        frame: np.ndarray,
        pose_frame,
        tracked_objects: List[Tuple[int, any]],
    ) -> np.ndarray:
        """Draw improved contact detection using hand landmarks."""
        
        # Prepare objects for contact detection
        objects_for_contact = [
            (track_id, det.bbox_xyxy, det.class_name)
            for track_id, det in tracked_objects
        ]
        
        # Run improved contact detection with hand landmarks
        if self.current_hands:
            self.current_contacts = self.contact_detector_improved.detect_contacts(
                self.current_hands,
                objects_for_contact,
            )
        else:
            self.current_contacts = []
        
        # Draw hand landmarks
        for hand in self.current_hands:
            # Different colors for detected vs interpolated (predicted) hands
            is_predicted = getattr(hand, 'is_interpolated', False)
            
            # Draw all 21 hand landmarks
            for i, pt in enumerate(hand.landmarks):
                x, y = int(pt[0]), int(pt[1])
                
                # Get per-landmark confidence if available
                lm_conf = 1.0
                if hand.landmark_confidence is not None and hand.landmark_confidence[i] is not None:
                    lm_conf = float(hand.landmark_confidence[i])
                
                # Fingertips are larger and colored differently
                if i in FINGERTIPS:
                    if is_predicted:
                        color = (0, 165, 255)  # Orange for predicted fingertips
                    else:
                        color = (0, 255, 255)  # Yellow for detected fingertips
                    radius = max(3, int(6 * lm_conf))
                else:
                    if is_predicted:
                        color = (180, 150, 100)  # Dimmer for predicted
                    else:
                        color = (255, 200, 100)  # Light blue for detected
                    radius = max(2, int(3 * lm_conf))
                
                cv2.circle(frame, (x, y), radius, color, -1)
            
            # Draw hand connections
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17),  # Palm
            ]
            line_color = (180, 150, 100) if is_predicted else (255, 200, 100)
            for start, end in hand_connections:
                pt1 = (int(hand.landmarks[start][0]), int(hand.landmarks[start][1]))
                pt2 = (int(hand.landmarks[end][0]), int(hand.landmarks[end][1]))
                cv2.line(frame, pt1, pt2, line_color, 2)
            
            # Draw palm center
            palm = hand.palm_center
            palm_color = (200, 100, 200) if is_predicted else (255, 100, 255)
            cv2.circle(frame, (int(palm[0]), int(palm[1])), 8, palm_color, 2)
            
            # Show hand label with prediction indicator
            wrist = hand.wrist
            label = f"{hand.handedness}"
            if is_predicted:
                label += " (interp)"  # Interpolated from history
                conf_pct = int(hand.confidence * 100)
                label += f" {conf_pct}%"
            cv2.putText(frame, label, (int(wrist[0]) - 40, int(wrist[1]) + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Draw contact indicators
        for contact_state in self.current_contacts:
            hand = contact_state.hand
            
            # Draw contact lines from fingertips to objects
            for obj_id, contact_point in contact_state.contact_points:
                cp = (int(contact_point[0]), int(contact_point[1]))
                
                # Find the object bbox center
                for track_id, bbox, _ in objects_for_contact:
                    if track_id == obj_id:
                        obj_center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                        cv2.line(frame, cp, obj_center, COLORS["contact"], 3)
                        break
                
                # Draw contact point
                cv2.circle(frame, cp, 12, COLORS["contact"], 3)
            
            # Show contact/grab status
            if contact_state.is_grabbing:
                palm = hand.palm_center
                cv2.putText(frame, f"GRABBING! ({contact_state.grab_confidence:.0%})",
                           (int(palm[0]) - 50, int(palm[1]) - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Draw grab indicator circle
                cv2.circle(frame, (int(palm[0]), int(palm[1])), 30, (0, 0, 255), 3)
            elif contact_state.contacted_objects:
                palm = hand.palm_center
                cv2.putText(frame, "CONTACT",
                           (int(palm[0]) - 35, int(palm[1]) - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return frame

    def _draw_3d_view(self) -> np.ndarray:
        """Draw a simple 3D pose visualization."""
        # Create blank canvas
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark gray background

        if not self.pose_history:
            return canvas

        pose_frame = self.pose_history[-1]
        coords_3d = pose_frame.coords_3d
        conf_3d = pose_frame.conf_3d

        # Project 3D to 2D with simple orthographic projection
        center_x, center_y = 200, 200
        scale = 150

        points_2d = []
        for j in range(NUM_JOINTS):
            if coords_3d[j] != [-1.0, -1.0, -1.0] and conf_3d[j] > 0.1:
                x = center_x + int(coords_3d[j][0] * scale)
                y = center_y + int(coords_3d[j][1] * scale)
                z = coords_3d[j][2]
                points_2d.append((x, y, z, conf_3d[j]))
            else:
                points_2d.append(None)

        # Draw skeleton lines
        for joint1_name, joint2_name in SKELETON_CONNECTIONS:
            idx1 = JOINT_IDX[joint1_name]
            idx2 = JOINT_IDX[joint2_name]

            if points_2d[idx1] is not None and points_2d[idx2] is not None:
                pt1 = (points_2d[idx1][0], points_2d[idx1][1])
                pt2 = (points_2d[idx2][0], points_2d[idx2][1])
                avg_z = (points_2d[idx1][2] + points_2d[idx2][2]) / 2
                color_val = int(128 + 127 * avg_z)
                color = (color_val, color_val, 0)
                cv2.line(canvas, pt1, pt2, color, 2)

        # Draw keypoints
        for j, pt in enumerate(points_2d):
            if pt is not None:
                radius = int(4 + 3 * (1 + pt[2]))
                color_val = int(128 + 127 * pt[2])
                color = (0, color_val, color_val)
                cv2.circle(canvas, (pt[0], pt[1]), radius, color, -1)

        # Draw axes
        cv2.arrowedLine(canvas, (30, 380), (80, 380), (0, 0, 255), 2)
        cv2.arrowedLine(canvas, (30, 380), (30, 330), (0, 255, 0), 2)
        cv2.putText(canvas, "X", (85, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(canvas, "Y", (15, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(canvas, "3D Pose (root-relative)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return canvas

    def _draw_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw info overlay."""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        box_height = 180 if (self.lite_mode or self.skip_frames > 0) else 160
        cv2.rectangle(overlay, (5, 5), (250, box_height), COLORS["text_bg"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        y = 22
        
        # FPS with color coding
        fps_color = (0, 255, 0) if self.fps_display >= 20 else ((0, 255, 255) if self.fps_display >= 10 else (0, 0, 255))
        cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1)
        y += 22
        cv2.putText(frame, f"Frame: {self.frame_idx}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)
        y += 20
        
        # Mode indicator
        if self.lite_mode or self.skip_frames > 0:
            mode_parts = []
            if self.lite_mode:
                mode_parts.append("LITE")
            if self.skip_frames > 0:
                mode_parts.append(f"skip:{self.skip_frames}")
            mode_str = " | ".join(mode_parts)
            cv2.putText(frame, f"Mode: {mode_str}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y += 18
        
        pose_status = f"Pose ({self.pose_backend}): {'ON' if self.show_pose else 'OFF'}"
        cv2.putText(frame, pose_status, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   (0, 255, 0) if self.show_pose else (100, 100, 100), 1)
        y += 20
        obj_status = f"Objects ({self.object_backend}): {'ON' if self.show_objects else 'OFF'}"
        cv2.putText(frame, obj_status, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   (0, 255, 0) if self.show_objects else (100, 100, 100), 1)
        y += 20
        
        # Hand detection info
        num_hands = len(self.current_hands)
        num_predicted = sum(1 for h in self.current_hands if getattr(h, 'is_interpolated', False))
        num_detected = num_hands - num_predicted
        
        hand_color = (0, 255, 255) if num_detected > 0 else ((0, 165, 255) if num_predicted > 0 else (100, 100, 100))
        hand_text = f"Hands: {num_detected}"
        if num_predicted > 0:
            hand_text += f" (+{num_predicted} interp)"
        if num_hands > 0:
            hand_text += " [3D]"
        cv2.putText(frame, hand_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1)
        y += 20
        
        # Contact info
        num_contacts = sum(len(c.contacted_objects) for c in self.current_contacts)
        grabbing = any(c.is_grabbing for c in self.current_contacts)
        contact_color = (0, 0, 255) if grabbing else ((0, 165, 255) if num_contacts > 0 else (100, 100, 100))
        contact_text = "GRABBING" if grabbing else (f"Contacts: {num_contacts}" if num_contacts else "No contact")
        cv2.putText(frame, contact_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, contact_color, 1)

        return frame

    def _reset(self):
        """Reset all trackers and state."""
        self.object_tracker.reset()
        self.contact_detector.reset()
        self.contact_detector_improved.reset()
        self.hand_detector.reset()  # Reset hand tracking history
        self.pose_history.clear()
        self.object_tracks.clear()
        self.current_hands = []
        self.current_contacts = []
        self.frame_idx = 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Webcam test bench with REAL pose and object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webcam_test.py                    # Run with defaults
  python webcam_test.py --fast             # Fast mode (skip frames + lite models)
  python webcam_test.py --lite             # Use lighter/faster models
  python webcam_test.py --skip 2           # Process every 3rd frame for YOLO
  python webcam_test.py --stub             # Use fake/stub backends
  python webcam_test.py --camera 1         # Use different camera

Performance tips:
  --fast is equivalent to --lite --skip 2 (good balance of speed/accuracy)
  --skip N skips N frames between YOLO detections (pose/hands still run every frame)
  --lite uses lighter model variants
""",
    )

    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--width", "-W", type=int, default=640,
                        help="Frame width (default: 640)")
    parser.add_argument("--height", "-H", type=int, default=480,
                        help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (default: 30)")
    parser.add_argument("--no-pose", action="store_true",
                        help="Disable pose visualization")
    parser.add_argument("--no-objects", action="store_true",
                        help="Disable object visualization")
    parser.add_argument("--no-contacts", action="store_true",
                        help="Disable contact visualization")
    parser.add_argument("--show-3d", action="store_true",
                        help="Show 3D pose view")
    parser.add_argument("--show-ego", action="store_true",
                        help="Show egocentric (first-person) view overlay")
    parser.add_argument("--stub", action="store_true",
                        help="Use stub/fake backends instead of real detection")
    
    # Performance options
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: lite models + skip frames (best FPS)")
    parser.add_argument("--lite", action="store_true",
                        help="Use lighter/faster model variants")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip N frames between YOLO detections (0=none, 2=every 3rd)")

    args = parser.parse_args()

    # Choose backends
    pose_backend = "stub" if args.stub else "mediapipe"
    object_backend = "stub" if args.stub else "yolo"
    
    # Performance settings
    # Default: optimized for speed (lite models + skip 1 frame)
    # Fast: even faster (skip 2 frames)
    lite_mode = args.lite or args.fast
    skip_frames = args.skip if args.skip > 0 else (2 if args.fast else 1)

    bench = WebcamTestBench(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show_pose=not args.no_pose,
        show_objects=not args.no_objects,
        show_contacts=not args.no_contacts,
        show_3d=args.show_3d,
        show_ego=args.show_ego,
        pose_backend=pose_backend,
        object_backend=object_backend,
        skip_frames=skip_frames,
        lite_mode=lite_mode,
    )

    bench.run()


if __name__ == "__main__":
    main()
