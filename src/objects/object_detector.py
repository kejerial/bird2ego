"""ObjectDetector: detects objects in frames."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection."""
    bbox_xyxy: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None  # optional segmentation mask


class ObjectDetectorBase(ABC):
    """Abstract base class for object detection."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            List of Detection objects.
        """
        pass

    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in a batch of frames.

        Args:
            frames: List of input frames.

        Returns:
            List of detection lists, one per frame.
        """
        pass


class StubObjectDetector(ObjectDetectorBase):
    """Stub object detector that generates random plausible detections.

    Used for testing the pipeline without heavy ML dependencies.
    """

    # Common industrial object classes
    DEFAULT_CLASSES = [
        "box",
        "container",
        "tool",
        "part",
        "assembly",
        "fixture",
        "bin",
    ]

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        num_objects: int = 3,
        detection_prob: float = 0.9,
        seed: Optional[int] = None,
    ):
        """Initialize stub detector.

        Args:
            classes: List of class names. Uses default if None.
            num_objects: Number of objects to generate per frame.
            detection_prob: Probability of detecting each object.
            seed: Random seed for reproducibility.
        """
        self.classes = classes or self.DEFAULT_CLASSES
        self.num_objects = num_objects
        self.detection_prob = detection_prob
        self.rng = np.random.default_rng(seed)

        # Generate consistent object positions for tracking
        self._object_templates: Optional[List[dict]] = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate stub object detections."""
        h, w = frame.shape[:2]

        # Initialize object templates on first call
        if self._object_templates is None:
            self._object_templates = self._generate_templates(w, h)

        detections = []
        for template in self._object_templates:
            # Probabilistically detect
            if self.rng.random() > self.detection_prob:
                continue

            # Add noise to position
            cx = template["cx"] + self.rng.normal(0, w * 0.01)
            cy = template["cy"] + self.rng.normal(0, h * 0.01)
            bw = template["bw"] * (1 + self.rng.normal(0, 0.02))
            bh = template["bh"] * (1 + self.rng.normal(0, 0.02))

            x1 = max(0, cx - bw / 2)
            y1 = max(0, cy - bh / 2)
            x2 = min(w, cx + bw / 2)
            y2 = min(h, cy + bh / 2)

            conf = 0.7 + 0.3 * self.rng.random()

            detections.append(
                Detection(
                    bbox_xyxy=[x1, y1, x2, y2],
                    class_id=template["class_id"],
                    class_name=template["class_name"],
                    confidence=conf,
                )
            )

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in a batch of frames."""
        return [self.detect(f) for f in frames]

    def _generate_templates(self, width: int, height: int) -> List[dict]:
        """Generate consistent object templates."""
        templates = []
        for i in range(self.num_objects):
            class_id = i % len(self.classes)
            class_name = self.classes[class_id]

            # Position objects in different areas
            if i == 0:
                # Object on table (bottom center)
                cx = width * 0.5
                cy = height * 0.7
            elif i == 1:
                # Object on left
                cx = width * 0.25
                cy = height * 0.6
            elif i == 2:
                # Object on right
                cx = width * 0.75
                cy = height * 0.6
            else:
                # Random position
                cx = width * (0.2 + 0.6 * self.rng.random())
                cy = height * (0.4 + 0.4 * self.rng.random())

            # Size varies by class
            if class_name in ["box", "container", "bin"]:
                bw = width * (0.1 + 0.1 * self.rng.random())
                bh = height * (0.1 + 0.15 * self.rng.random())
            elif class_name in ["tool", "part"]:
                bw = width * (0.05 + 0.05 * self.rng.random())
                bh = height * (0.05 + 0.1 * self.rng.random())
            else:
                bw = width * (0.08 + 0.08 * self.rng.random())
                bh = height * (0.08 + 0.12 * self.rng.random())

            templates.append({
                "cx": cx,
                "cy": cy,
                "bw": bw,
                "bh": bh,
                "class_id": class_id,
                "class_name": class_name,
            })

        return templates


class ObjectDetector:
    """Object detector with configurable backend."""

    def __init__(
        self,
        backend: str = "stub",
        classes: Optional[List] = None,
        confidence_threshold: float = 0.5,
        **kwargs,
    ):
        """Initialize ObjectDetector.

        Args:
            backend: Backend to use ("stub" for testing).
            classes: List of class names (stub) or class IDs (yolo) to detect.
            confidence_threshold: Minimum confidence threshold.
            **kwargs: Additional arguments for backend.
        """
        self.confidence_threshold = confidence_threshold

        if backend == "stub":
            self._detector = StubObjectDetector(classes=classes, **kwargs)
        elif backend == "yolo":
            from .yolo_detector import YOLODetector
            self._detector = YOLODetector(classes=classes, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'stub' or 'yolo'")

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            List of Detection objects above confidence threshold.
        """
        detections = self._detector.detect(frame)
        return [d for d in detections if d.confidence >= self.confidence_threshold]

    def detect_video(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in all frames.

        Args:
            frames: List of video frames.

        Returns:
            List of detection lists, one per frame.
        """
        return [self.detect_frame(f) for f in frames]
