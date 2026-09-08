"""Egocentric view transformation module.

Transforms third-person camera footage into first-person (egocentric) representations
suitable for VLA (Vision-Language-Action) training.
"""
from .transformer import EgocentricTransformer, EgocentricFrame, EgocentricTimeSeries
from .renderer import EgocentricRenderer
from .exporter import export_egocentric_json, EgocentricRecorder
from .stage import EgocentricStage, fit_image_to_pose_3d, summarize_series

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
