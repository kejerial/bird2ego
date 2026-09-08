"""Hand detection using MediaPipe Hand Landmarker with occlusion handling.

Provides detailed hand tracking with 21 landmarks per hand including fingertips.
Includes temporal interpolation and geometric prediction for occluded landmarks.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import deque

import cv2
import numpy as np

# Hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# All fingertips
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Finger chains (base to tip)
FINGER_CHAINS = {
    'thumb': [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    'index': [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    'middle': [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    'ring': [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    'pinky': [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
}

# Palm landmarks for center calculation
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def get_hand_model_path() -> str:
    """Download hand model if needed and return path."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "hand_landmarker.task"
    
    if not model_path.exists():
        print(f"Downloading MediaPipe hand model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete!")
    
    return str(model_path)


@dataclass
class HandLandmarks:
    """Hand landmarks for a single hand with 2D and 3D coordinates."""
    handedness: str  # "Left" or "Right"
    landmarks: np.ndarray  # (21, 2) pixel coordinates
    landmarks_3d: np.ndarray = None  # (21, 3) normalized 3D coordinates (x, y, z relative to wrist)
    confidence: float = 0.9
    landmark_confidence: np.ndarray = None  # (21,) per-landmark confidence
    is_interpolated: bool = False  # True if landmarks were predicted/interpolated
    
    def __post_init__(self):
        if self.landmark_confidence is None:
            self.landmark_confidence = np.ones(21, dtype=np.float32) * self.confidence
        else:
            # Ensure it's a proper float array with no None values
            self.landmark_confidence = np.array([
                float(c) if c is not None else self.confidence 
                for c in self.landmark_confidence
            ], dtype=np.float32)
        
        # Initialize 3D landmarks if not provided
        if self.landmarks_3d is None:
            # Create placeholder 3D coordinates (will be filled by detector)
            self.landmarks_3d = np.zeros((21, 3), dtype=np.float32)
    
    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[WRIST]
    
    @property
    def wrist_3d(self) -> np.ndarray:
        """Get 3D wrist position."""
        return self.landmarks_3d[WRIST]
    
    @property
    def fingertips(self) -> np.ndarray:
        """Get all 5 fingertip positions (2D)."""
        return self.landmarks[FINGERTIPS]
    
    @property
    def fingertips_3d(self) -> np.ndarray:
        """Get all 5 fingertip positions (3D)."""
        return self.landmarks_3d[FINGERTIPS]
    
    @property
    def index_tip(self) -> np.ndarray:
        return self.landmarks[INDEX_TIP]
    
    @property
    def index_tip_3d(self) -> np.ndarray:
        """Get index fingertip in 3D."""
        return self.landmarks_3d[INDEX_TIP]
    
    @property
    def thumb_tip(self) -> np.ndarray:
        return self.landmarks[THUMB_TIP]
    
    @property
    def thumb_tip_3d(self) -> np.ndarray:
        """Get thumb tip in 3D."""
        return self.landmarks_3d[THUMB_TIP]
    
    @property
    def palm_center(self) -> np.ndarray:
        """Calculate palm center from palm landmarks (2D)."""
        return np.mean(self.landmarks[PALM_LANDMARKS], axis=0)
    
    @property
    def palm_center_3d(self) -> np.ndarray:
        """Calculate palm center from palm landmarks (3D)."""
        return np.mean(self.landmarks_3d[PALM_LANDMARKS], axis=0)
    
    def get_pinch_distance(self) -> float:
        """Distance between thumb and index finger tips (2D)."""
        return float(np.linalg.norm(self.thumb_tip - self.index_tip))
    
    def get_pinch_distance_3d(self) -> float:
        """Distance between thumb and index finger tips (3D)."""
        return float(np.linalg.norm(self.thumb_tip_3d - self.index_tip_3d))
    
    def is_hand_open(self, threshold: float = 50.0) -> bool:
        """Check if hand is open (fingers spread) - 2D."""
        palm = self.palm_center
        avg_dist = np.mean([np.linalg.norm(self.landmarks[ft] - palm) for ft in FINGERTIPS])
        return avg_dist > threshold
    
    def is_grabbing(self, threshold: float = 40.0) -> bool:
        """Check if hand is in grabbing pose (fingers curled) - 2D."""
        return self.get_pinch_distance() < threshold


@dataclass
class HandContactState:
    """Contact state between a hand and objects."""
    hand: HandLandmarks
    contacted_objects: List[int] = field(default_factory=list)  # object IDs
    contact_points: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (obj_id, point)
    is_grabbing: bool = False
    grab_confidence: float = 0.0


class HandDetector:
    """MediaPipe Hand Landmarker - simple, fast detection only.
    
    No interpolation or prediction - just raw MediaPipe detection for maximum speed.
    """
    
    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize hand detector.
        
        Args:
            num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum detection confidence.
            min_tracking_confidence: Minimum tracking confidence.
        """
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        self.mp = mp
        
        model_path = get_hand_model_path()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp = 0
        
        print("  MediaPipe Hand Landmarker loaded (fast mode, no interpolation)")
    
    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """Detect hands - simple, fast, no interpolation.
        
        Args:
            frame: BGR image.
            
        Returns:
            List of HandLandmarks for detected hands only.
        """
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        
        self.frame_timestamp += 33
        results = self.detector.detect_for_video(mp_image, self.frame_timestamp)
        
        detected_hands = []
        
        # Process detected hands - no interpolation, just raw detection
        if results.hand_landmarks:
            for i, hand_lms in enumerate(results.hand_landmarks):
                # Extract 2D pixel coordinates
                landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in hand_lms], dtype=np.float32)
                
                # Extract 3D normalized coordinates (x, y normalized 0-1, z is depth relative to wrist)
                # MediaPipe provides normalized coordinates where z is depth
                landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                
                handedness = "Right"
                if results.handedness and i < len(results.handedness):
                    handedness = results.handedness[i][0].category_name
                
                # Calculate per-landmark confidence from visibility
                landmark_conf = np.array([
                    float(getattr(lm, 'visibility', 0.9) or 0.9)
                    for lm in hand_lms
                ], dtype=np.float32)
                
                hand = HandLandmarks(
                    handedness=handedness,
                    landmarks=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    confidence=0.9,
                    landmark_confidence=landmark_conf,
                    is_interpolated=False,
                )
                
                detected_hands.append(hand)
        
        return detected_hands
    
    def reset(self):
        """Reset detector (no-op for simple detector)."""
        pass
    
    def __del__(self):
        if hasattr(self, 'detector'):
            self.detector.close()


class ImprovedContactDetector:
    """Improved contact detection using hand landmarks and spatial reasoning.
    
    Features:
    - Fingertip-based contact detection
    - Grab detection based on hand pose
    - Temporal smoothing to reduce flickering
    """
    
    def __init__(
        self,
        contact_distance: float = 30.0,
        grab_distance: float = 50.0,
        overlap_threshold: float = 0.3,
        temporal_smoothing: int = 5,
    ):
        """Initialize contact detector.
        
        Args:
            contact_distance: Distance threshold for fingertip-object contact (pixels).
            grab_distance: Distance threshold for grab detection (pixels).
            overlap_threshold: Minimum overlap ratio for "inside object" detection.
            temporal_smoothing: Number of frames for temporal smoothing.
        """
        self.contact_distance = contact_distance
        self.grab_distance = grab_distance
        self.overlap_threshold = overlap_threshold
        self.temporal_smoothing = temporal_smoothing
        
        # Contact history for smoothing (key: hand + object)
        self.contact_history: Dict[str, deque] = {}
        self.grab_history: Dict[str, deque] = {}
    
    def detect_contacts(
        self,
        hands: List[HandLandmarks],
        objects: List[Tuple[int, List[float], str]],  # [(id, bbox_xyxy, class_name), ...]
    ) -> List[HandContactState]:
        """Detect contacts between hands and objects.
        
        Args:
            hands: List of detected hands (may include interpolated ones).
            objects: List of (object_id, bbox_xyxy, class_name) tuples.
            
        Returns:
            List of HandContactState for each hand.
        """
        contact_states = []
        
        for hand in hands:
            state = HandContactState(hand=hand)
            
            # For each object, check for contact
            for obj_id, bbox, class_name in objects:
                contact_info = self._check_hand_object_contact(hand, bbox)
                
                if contact_info['is_contact']:
                    state.contacted_objects.append(obj_id)
                    state.contact_points.append((obj_id, contact_info['contact_point']))
                    
                    if contact_info['is_grab']:
                        state.is_grabbing = True
                        state.grab_confidence = max(state.grab_confidence, contact_info['grab_confidence'])
            
            # Apply temporal smoothing
            state = self._smooth_contact_state(state, objects)
            contact_states.append(state)
        
        return contact_states
    
    def _check_hand_object_contact(
        self,
        hand: HandLandmarks,
        bbox: List[float],
    ) -> dict:
        """Check if a hand is contacting an object.
        
        Returns dict with:
            - is_contact: bool
            - contact_point: np.ndarray or None
            - is_grab: bool
            - grab_confidence: float
        """
        x1, y1, x2, y2 = bbox
        
        result = {
            'is_contact': False,
            'contact_point': None,
            'is_grab': False,
            'grab_confidence': 0.0,
        }
        
        # Check each fingertip with confidence weighting
        min_dist = float('inf')
        closest_point = None
        fingertips_near = 0
        fingertips_inside = 0
        weighted_fingertip_score = 0.0
        
        for ft_idx in FINGERTIPS:
            fingertip = hand.landmarks[ft_idx]
            conf = float(hand.landmark_confidence[ft_idx]) if (hand.landmark_confidence is not None and hand.landmark_confidence[ft_idx] is not None) else 1.0
            
            # Distance to bbox
            dist = self._point_to_bbox_distance(fingertip, bbox)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = fingertip.copy()
            
            if dist < self.contact_distance:
                fingertips_near += 1
                weighted_fingertip_score += conf
            
            # Check if inside bbox
            if x1 <= fingertip[0] <= x2 and y1 <= fingertip[1] <= y2:
                fingertips_inside += 1
                weighted_fingertip_score += conf * 0.5  # Bonus for being inside
        
        # Also check palm center
        palm = hand.palm_center
        palm_dist = self._point_to_bbox_distance(palm, bbox)
        palm_inside = x1 <= palm[0] <= x2 and y1 <= palm[1] <= y2
        
        # Contact detection: account for interpolated hands
        contact_threshold = self.contact_distance
        if hand.is_interpolated:
            contact_threshold *= 1.5  # More lenient for predicted positions
        
        if min_dist < contact_threshold or fingertips_inside >= 1:
            result['is_contact'] = True
            result['contact_point'] = closest_point
        
        # Also count as contact if palm is very close (hand wrapping around)
        if palm_dist < self.contact_distance * 0.7:
            result['is_contact'] = True
            if closest_point is None:
                result['contact_point'] = palm.copy()
        
        # Grab detection: multiple fingertips near/inside + hand in grab pose
        if fingertips_near >= 2 or fingertips_inside >= 2 or (palm_inside and fingertips_near >= 1):
            pinch_dist = hand.get_pinch_distance()
            
            # Calculate grab confidence
            fingertip_score = min(weighted_fingertip_score / 2.5, 1.0)
            pinch_score = max(0, 1.0 - pinch_dist / 100.0)
            palm_score = 1.0 if palm_inside else max(0, 1.0 - palm_dist / self.grab_distance)
            
            # Reduce confidence for interpolated hands
            interpolation_penalty = 0.8 if hand.is_interpolated else 1.0
            
            grab_conf = (fingertip_score * 0.4 + pinch_score * 0.3 + palm_score * 0.3) * interpolation_penalty
            
            if grab_conf > 0.35:
                result['is_grab'] = True
                result['grab_confidence'] = grab_conf
        
        return result
    
    def _point_to_bbox_distance(self, point: np.ndarray, bbox: List[float]) -> float:
        """Calculate distance from point to bounding box edge."""
        x1, y1, x2, y2 = bbox
        nearest_x = max(x1, min(point[0], x2))
        nearest_y = max(y1, min(point[1], y2))
        return float(np.sqrt((point[0] - nearest_x)**2 + (point[1] - nearest_y)**2))
    
    def _smooth_contact_state(
        self,
        state: HandContactState,
        objects: List[Tuple[int, List[float], str]],
    ) -> HandContactState:
        """Apply temporal smoothing to contact state."""
        hand_key = state.hand.handedness
        
        # Update contact history
        for obj_id in state.contacted_objects:
            key = f"{hand_key}_{obj_id}"
            if key not in self.contact_history:
                self.contact_history[key] = deque(maxlen=self.temporal_smoothing)
            self.contact_history[key].append(True)
        
        # Update grab history
        grab_key = f"{hand_key}_grab"
        if grab_key not in self.grab_history:
            self.grab_history[grab_key] = deque(maxlen=self.temporal_smoothing)
        self.grab_history[grab_key].append(state.is_grabbing)
        
        # Mark contacts as gone for objects no longer contacted
        for key in list(self.contact_history.keys()):
            if key.startswith(hand_key) and key != grab_key:
                obj_id_str = key.split('_')[-1]
                try:
                    obj_id = int(obj_id_str)
                    if obj_id not in state.contacted_objects:
                        self.contact_history[key].append(False)
                except ValueError:
                    pass
        
        # Apply hysteresis: maintain contact if majority of recent frames had contact
        smoothed_contacts = []
        for obj_id, bbox, _ in objects:
            key = f"{hand_key}_{obj_id}"
            if key in self.contact_history:
                history = self.contact_history[key]
                if sum(history) > len(history) * 0.4:  # 40% threshold to maintain
                    if obj_id not in state.contacted_objects:
                        smoothed_contacts.append(obj_id)
                        # Add contact point estimate
                        center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                        state.contact_points.append((obj_id, center))
        
        state.contacted_objects.extend(smoothed_contacts)
        
        # Apply hysteresis to grab state
        if grab_key in self.grab_history:
            history = self.grab_history[grab_key]
            if sum(history) > len(history) * 0.5:  # 50% threshold for grab
                state.is_grabbing = True
                if state.grab_confidence < 0.5:
                    state.grab_confidence = 0.5
        
        return state
    
    def reset(self):
        """Reset contact history."""
        self.contact_history.clear()
        self.grab_history.clear()


# ---------------------------------------------------------------------------
# Helper functions for timeline integration
# ---------------------------------------------------------------------------

def hand_landmarks_to_hand_frame(
    hand: HandLandmarks,
    frame_idx: int,
) -> 'HandFrame':  # type: ignore
    """Convert HandLandmarks to HandFrame for timeline storage.
    
    Args:
        hand: HandLandmarks from detector
        frame_idx: Current frame index
        
    Returns:
        HandFrame for timeline
    """
    from ..utils.timeline import HandFrame
    
    return HandFrame(
        frame_idx=frame_idx,
        handedness=hand.handedness,
        landmarks_2d=hand.landmarks.tolist(),
        landmarks_3d=hand.landmarks_3d.tolist(),
        conf=hand.landmark_confidence.tolist(),
        overall_confidence=hand.confidence,
        is_detected=not hand.is_interpolated,
    )
