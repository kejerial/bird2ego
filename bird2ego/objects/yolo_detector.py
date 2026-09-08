"""YOLO-based object detector using ultralytics.

Works from any camera angle.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .object_detector import Detection, ObjectDetectorBase

logger = logging.getLogger(__name__)


class YOLODetector(ObjectDetectorBase):
    """YOLO object detector using ultralytics.

    Pre-trained on 80 COCO classes including:
    - person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
    - traffic light, fire hydrant, stop sign, parking meter, bench
    - bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    - backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard
    - sports ball, kite, baseball bat, baseball glove, skateboard, surfboard
    - tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl
    - banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza
    - donut, cake, chair, couch, potted plant, bed, dining table, toilet
    - tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven
    - toaster, sink, refrigerator, book, clock, vase, scissors
    - teddy bear, hair drier, toothbrush
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        classes: Optional[List[int]] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        verbose: bool = False,
    ):
        """Initialize YOLO detector.

        Args:
            model_name: YOLO model to use. Options:
                - "yolov8n.pt" (nano, fastest, 6MB)
                - "yolov8s.pt" (small, fast, 22MB)
                - "yolov8m.pt" (medium, balanced, 52MB)
                - "yolov8l.pt" (large, accurate, 87MB)
                - "yolov8x.pt" (xlarge, most accurate, 137MB)
            classes: List of class IDs to detect. None = all 80 classes.
            device: "cpu", "cuda", "mps", or "auto".
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            verbose: Show YOLO output logs.
        """
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.classes = classes
        self.device = device if device != "auto" else None  # None = auto
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose

        logger.info(
            f"YOLO detector initialized: {model_name}, classes={classes or 'all'}, device={device}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame.

        Args:
            frame: Input frame (BGR, uint8).

        Returns:
            List of Detection objects.
        """
        # Run inference
        results = self.model(
            frame,
            classes=self.classes,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=self.verbose,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = self.model.names[cls_id]

                detections.append(
                    Detection(
                        bbox_xyxy=bbox,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                    )
                )

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in multiple frames."""
        return [self.detect(f) for f in frames]


# Useful COCO class IDs for quick reference
COCO_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "book": 73,
    "clock": 74,
    "scissors": 76,
    "teddy bear": 77,
    "toothbrush": 79,
}
