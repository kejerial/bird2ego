"""Export egocentric data for VLA training.

Saves egocentric time-series data in formats suitable for training
Vision-Language-Action models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import asdict

import numpy as np

from .transformer import EgocentricFrame, EgocentricTimeSeries


def export_egocentric_json(
    ego_series: EgocentricTimeSeries,
    output_path: str,
    video_info: Optional[Dict[str, Any]] = None,
    action_labels: Optional[List[Dict]] = None,
) -> None:
    """Export egocentric data to JSON for VLA training.
    
    Args:
        ego_series: Egocentric time-series data
        output_path: Path to output JSON file
        video_info: Optional video metadata
        action_labels: Optional list of action labels with frame ranges
    """
    data = {
        "format_version": "1.0",
        "type": "egocentric_trajectory",
        "video_info": video_info or {},
        "num_frames": ego_series.num_frames,
        "frames": [],
        "action_labels": action_labels or [],
    }
    
    for frame in ego_series.frames:
        frame_data = {
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "head": {
                "position": frame.head_position.tolist(),
                "forward": frame.head_forward.tolist(),
                "up": frame.head_up.tolist(),
                "right": frame.head_right.tolist(),
                "confidence": frame.head_confidence,
            },
            "left_hand": None,
            "right_hand": None,
            "left_arm": None,
            "right_arm": None,
            "objects": {},
        }
        
        # Left hand
        if frame.left_hand_ego is not None:
            frame_data["left_hand"] = {
                "landmarks_3d": frame.left_hand_ego.tolist(),
                "pinching": frame.left_hand_pinching,
                "pinch_distance": frame.left_pinch_distance,
                "confidence": frame.left_hand_confidence,
            }
        
        # Right hand
        if frame.right_hand_ego is not None:
            frame_data["right_hand"] = {
                "landmarks_3d": frame.right_hand_ego.tolist(),
                "pinching": frame.right_hand_pinching,
                "pinch_distance": frame.right_pinch_distance,
                "confidence": frame.right_hand_confidence,
            }
        
        # Arms
        if frame.left_arm_ego is not None:
            frame_data["left_arm"] = {
                "keypoints_3d": frame.left_arm_ego.tolist(),  # shoulder, elbow, wrist
            }
        if frame.right_arm_ego is not None:
            frame_data["right_arm"] = {
                "keypoints_3d": frame.right_arm_ego.tolist(),
            }
        
        # Objects
        for obj_id, obj_pos in frame.objects_ego.items():
            frame_data["objects"][str(obj_id)] = {
                "position_3d": obj_pos.tolist(),
            }
        
        data["frames"].append(frame_data)
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported egocentric data to {output_path}")


def export_action_dataset(
    ego_series: EgocentricTimeSeries,
    action_labels: List[Dict],
    output_dir: str,
    dataset_name: str = "ego_actions",
) -> None:
    """Export egocentric data as action-labeled dataset.
    
    Suitable for training action recognition or VLA models.
    
    Args:
        ego_series: Egocentric time-series data
        action_labels: List of dicts with {frame_start, frame_end, action, object_id}
        output_dir: Output directory
        dataset_name: Name for the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create action segments
    segments = []
    for i, label in enumerate(action_labels):
        frame_start = label.get("frame_start", 0)
        frame_end = label.get("frame_end", ego_series.num_frames)
        action = label.get("action", "unknown")
        object_id = label.get("object_id", None)
        
        segment = {
            "segment_id": i,
            "action": action,
            "object_id": object_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "duration_frames": frame_end - frame_start,
            "trajectory": [],
        }
        
        # Extract trajectory for this segment
        for j in range(frame_start, min(frame_end, ego_series.num_frames)):
            frame = ego_series.frames[j]
            
            # Get the primary hand for this action (default to right)
            hand_data = None
            if frame.right_hand_ego is not None:
                hand_data = {
                    "landmarks_3d": frame.right_hand_ego.tolist(),
                    "pinching": frame.right_hand_pinching,
                }
            elif frame.left_hand_ego is not None:
                hand_data = {
                    "landmarks_3d": frame.left_hand_ego.tolist(),
                    "pinching": frame.left_hand_pinching,
                }
            
            trajectory_point = {
                "frame_idx": j,
                "relative_frame": j - frame_start,
                "hand": hand_data,
                "object_position": frame.objects_ego.get(object_id, np.zeros(3)).tolist() if object_id else None,
            }
            segment["trajectory"].append(trajectory_point)
        
        segments.append(segment)
    
    # Save dataset
    dataset = {
        "format_version": "1.0",
        "dataset_name": dataset_name,
        "num_segments": len(segments),
        "action_vocabulary": list(set(s["action"] for s in segments)),
        "segments": segments,
    }
    
    dataset_file = output_path / f"{dataset_name}.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Exported {len(segments)} action segments to {dataset_file}")


class EgocentricRecorder:
    """Record egocentric data during live capture."""
    
    def __init__(self):
        self.frames: List[EgocentricFrame] = []
        self.action_annotations: List[Dict] = []
        self.current_action: Optional[Dict] = None
    
    def add_frame(self, ego_frame: EgocentricFrame):
        """Add a frame to the recording."""
        self.frames.append(ego_frame)
    
    def start_action(self, action: str, object_id: Optional[int] = None):
        """Start recording an action annotation."""
        if self.current_action is not None:
            self.end_action()
        
        self.current_action = {
            "frame_start": len(self.frames),
            "action": action,
            "object_id": object_id,
        }
    
    def end_action(self):
        """End the current action annotation."""
        if self.current_action is not None:
            self.current_action["frame_end"] = len(self.frames)
            self.action_annotations.append(self.current_action)
            self.current_action = None
    
    def get_time_series(self) -> EgocentricTimeSeries:
        """Get the recorded time series."""
        return EgocentricTimeSeries(frames=self.frames.copy())
    
    def save(self, output_path: str, video_info: Optional[Dict] = None):
        """Save the recording to JSON."""
        # End any in-progress action
        if self.current_action is not None:
            self.end_action()
        
        ego_series = self.get_time_series()
        export_egocentric_json(
            ego_series,
            output_path,
            video_info=video_info,
            action_labels=self.action_annotations,
        )
    
    def clear(self):
        """Clear the recording."""
        self.frames.clear()
        self.action_annotations.clear()
        self.current_action = None
