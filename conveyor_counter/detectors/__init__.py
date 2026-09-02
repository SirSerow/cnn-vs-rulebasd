"""Detector backends."""

from .opencv import OpenCVDetector
from .yolo_onnx import YoloOnnxDetector

__all__ = ["OpenCVDetector", "YoloOnnxDetector"]
