"""
detector.py

Same abstraction as before: something with a .detect(frame) method
returning a list of Detection(bbox_xyxy, confidence). Swap MockDetector
for a real YOLODetector once trained weights exist.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable
import numpy as np


@dataclass
class Detection:
    bbox_xyxy: tuple
    confidence: float


class VehicleDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...


class YOLODetector(VehicleDetector):
    def __init__(self, weights_path: str, conf: float = 0.25, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        return [Detection(tuple(box.xyxy[0].tolist()), float(box.conf[0])) for box in results.boxes]


class MockDetector(VehicleDetector):
    def __init__(self, script: Callable[[int], List[Detection]]):
        self.script = script
        self.frame_index = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        detections = self.script(self.frame_index)
        self.frame_index += 1
        return detections
