"""JSONExporter: exports timeline data to JSON files."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.timeline import (
    COCO17_JOINT_NAMES,
    ActionSegment,
    ContactEvent,
    ObjectTrack,
    PersonPose,
    Timeline,
)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class JSONExporter:
    """Exports timeline data to JSON files.

    Produces:
    - timeline.json: Canonical timeline reference
    - pose.json: Human skeleton over time
    - objects.json: Object tracks and states
    - contacts.json: Contact events
    - actions.json: Action segments
    """

    def __init__(
        self,
        output_dir: str,
        indent: int = 2,
        run_id: Optional[str] = None,
    ):
        """Initialize JSONExporter.

        Args:
            output_dir: Directory to write output files.
            indent: JSON indentation level.
            run_id: Optional run identifier.
        """
        self.output_dir = Path(output_dir)
        self.indent = indent
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, timeline: Timeline) -> Dict[str, str]:
        """Export all timeline data to JSON files.

        Args:
            timeline: Timeline to export.

        Returns:
            Dictionary mapping output type to file path.
        """
        paths = {}

        paths["timeline"] = self.export_timeline(timeline)
        paths["pose"] = self.export_pose(timeline)
        paths["objects"] = self.export_objects(timeline)
        paths["contacts"] = self.export_contacts(timeline)
        paths["actions"] = self.export_actions(timeline)

        logger.info(f"Exported all JSON files to {self.output_dir}")
        return paths

    def export_timeline(self, timeline: Timeline) -> str:
        """Export timeline metadata.

        Args:
            timeline: Timeline to export.

        Returns:
            Path to output file.
        """
        data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "fps_extracted": timeline.fps_extracted,
            "t0": timeline.t0,
            "timestamp_source": timeline.timestamp_source,
            "frame_to_time_rule": timeline.frame_to_time_rule,
            "num_frames": timeline.num_frames,
            "duration": timeline.duration,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "t": f.t,
                    "width": f.width,
                    "height": f.height,
                }
                for f in timeline.frames
            ],
        }

        path = self.output_dir / "timeline.json"
        self._write_json(data, path)
        return str(path)

    def export_pose(self, timeline: Timeline) -> str:
        """Export pose data following the exact schema.

        Args:
            timeline: Timeline with pose data.

        Returns:
            Path to output file.
        """
        pose = timeline.person_pose

        if pose is None or pose.num_frames == 0:
            # Empty pose data
            data = {
                "schema_version": "1.0",
                "run": {"run_id": self.run_id},
                "image": {"width": 0, "height": 0, "space": "processed_frames"},
                "person": None,
            }
        else:
            # Get image dimensions from timeline
            width = timeline.frames[0].width if timeline.frames else 0
            height = timeline.frames[0].height if timeline.frames else 0

            # Build arrays
            T = pose.num_frames
            bbox_xyxy = [f.bbox_xyxy for f in pose.frames]
            frame_idx = [f.frame_idx for f in pose.frames]
            keypoints_2d_px = [f.keypoints_2d_px for f in pose.frames]
            conf_2d = [f.conf_2d for f in pose.frames]
            coords_3d = [f.coords_3d for f in pose.frames]
            conf_3d = [f.conf_3d for f in pose.frames]
            joint_visible = [f.joint_visible for f in pose.frames]
            joint_occluded = [f.joint_occluded for f in pose.frames]
            joint_in_frame = [f.joint_in_frame for f in pose.frames]

            data = {
                "schema_version": "1.0",
                "run": {"run_id": self.run_id},
                "image": {
                    "width": width,
                    "height": height,
                    "space": "processed_frames",
                },
                "person": {
                    "bbox_xyxy": bbox_xyxy,
                    "skeleton_type": pose.skeleton_type,
                    "joint_names": pose.joint_names,
                    "root_joint_index": pose.root_joint_index,
                    "root_joint_name": pose.root_joint_name,
                    "root_definition": pose.root_definition,
                    "frame_idx": frame_idx,
                    "keypoints_2d_px": keypoints_2d_px,
                    "conf_2d": conf_2d,
                    "pose_3d_rootrel": {
                        "coords": coords_3d,
                        "units": "arb",
                        "coord_frame": "root-relative-camera",
                        "axis_convention": "x=right, y=down, z=forward (camera)",
                        "conf_3d": conf_3d,
                    },
                    "visibility": {
                        "joint_visible": joint_visible,
                        "joint_occluded": joint_occluded,
                        "joint_in_frame": joint_in_frame,
                    },
                    "missing_joint_convention": {
                        "coord_sentinel_2d": [-1.0, -1.0],
                        "coord_sentinel_3d": [-1.0, -1.0, -1.0],
                        "conf_sentinel": 0.0,
                        "missing_rules": "keypoints_2d_px[t][j]=[-1,-1], pose_3d_rootrel.coords[t][j]=[-1,-1,-1], conf_*=0.0 when missing",
                        "visibility_rules": "joint_in_frame=false, joint_visible=false, joint_occluded=false when out of frame",
                    },
                    "quality": {
                        "overall_pose_conf": self._compute_avg_conf(conf_2d),
                        "warnings": [],
                    },
                },
                "camera": {
                    "status": "unknown",
                    "model": "pinhole",
                    "intrinsics": {
                        "fx": None,
                        "fy": None,
                        "cx": None,
                        "cy": None,
                    },
                    "notes": "Camera parameters not estimated",
                },
                "verification": {
                    "reprojection_error_px": None,
                    "notes": "",
                },
            }

        path = self.output_dir / "pose.json"
        self._write_json(data, path)
        return str(path)

    def export_objects(self, timeline: Timeline) -> str:
        """Export object tracks.

        Args:
            timeline: Timeline with object tracks.

        Returns:
            Path to output file.
        """
        objects_data = []

        for obj_id, track in timeline.objects.items():
            # Build arrays aligned to timeline
            T = track.num_frames
            bbox_xyxy = [f.bbox_xyxy for f in track.frames]
            conf = [f.conf for f in track.frames]
            states = [
                {
                    "support_relation": f.state.support_relation.value,
                    "motion_state": f.state.motion_state.value,
                    "interaction_state": f.state.interaction_state.value,
                    "containment": f.state.containment.value,
                }
                for f in track.frames
            ]

            objects_data.append({
                "object_id": obj_id,
                "class_name": track.class_name,
                "class_id": track.class_id,
                "num_frames": T,
                "bbox_xyxy": bbox_xyxy,
                "conf": conf,
                "states": states,
            })

        data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "num_objects": len(objects_data),
            "objects": objects_data,
            "sentinel_convention": {
                "bbox_sentinel": [-1.0, -1.0, -1.0, -1.0],
                "conf_sentinel": 0.0,
            },
        }

        path = self.output_dir / "objects.json"
        self._write_json(data, path)
        return str(path)

    def export_contacts(self, timeline: Timeline) -> str:
        """Export contact events.

        Args:
            timeline: Timeline with contact events.

        Returns:
            Path to output file.
        """
        events_data = []

        for event in timeline.events:
            events_data.append({
                "event_id": event.event_id,
                "type": event.event_type,
                "t_start": event.t_start,
                "t_end": event.t_end,
                "frame_start": event.frame_start,
                "frame_end": event.frame_end,
                "hand": event.hand.value,
                "object_id": event.object_id,
                "label": event.label,
                "conf": event.conf,
                "evidence": event.evidence,
            })

        data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "num_events": len(events_data),
            "event_scope": "contact_events_only_v1",
            "events": events_data,
        }

        path = self.output_dir / "contacts.json"
        self._write_json(data, path)
        return str(path)

    def export_actions(self, timeline: Timeline) -> str:
        """Export action segments.

        Args:
            timeline: Timeline with action segments.

        Returns:
            Path to output file.
        """
        segments_data = []

        for segment in timeline.segments:
            segments_data.append({
                "seg_id": segment.seg_id,
                "t_start": segment.t_start,
                "t_end": segment.t_end,
                "frame_start": segment.frame_start,
                "frame_end": segment.frame_end,
                "label": segment.label,
                "conf": segment.conf,
                "objects_involved": segment.objects_involved,
                "evidence": segment.evidence,
            })

        data = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "num_segments": len(segments_data),
            "segments": segments_data,
        }

        path = self.output_dir / "actions.json"
        self._write_json(data, path)
        return str(path)

    def _write_json(self, data: Dict, path: Path) -> None:
        """Write data to JSON file.

        Args:
            data: Data to write.
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=self.indent, cls=NumpyEncoder)
        logger.debug(f"Wrote {path}")

    def _compute_avg_conf(self, conf_2d: List[List[float]]) -> float:
        """Compute average confidence across all frames and joints.

        Args:
            conf_2d: (T, 17) confidence values.

        Returns:
            Average confidence.
        """
        if not conf_2d:
            return 0.0
        flat = [c for frame in conf_2d for c in frame if c > 0]
        return sum(flat) / len(flat) if flat else 0.0
