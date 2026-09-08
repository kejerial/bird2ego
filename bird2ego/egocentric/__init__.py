"""Egocentric view transformation module.

Transforms third-person camera footage into first-person (egocentric) representations
suitable for VLA (Vision-Language-Action) training.
"""

from .exporter import EgocentricRecorder, export_egocentric_json
from .renderer import EgocentricRenderer
from .stage import EgocentricStage, fit_image_to_pose_3d, summarize_series
from .transformer import EgocentricFrame, EgocentricTimeSeries, EgocentricTransformer

__all__ = [
    "EgocentricTransformer",
    "EgocentricFrame",
    "EgocentricTimeSeries",
    "EgocentricRenderer",
    "export_egocentric_json",
    "EgocentricRecorder",
    "EgocentricStage",
    "fit_image_to_pose_3d",
    "summarize_series",
]
