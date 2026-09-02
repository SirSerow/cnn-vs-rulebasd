"""The small interface implemented by detector backends."""

from typing import Protocol

from ..models import Detection, ImageSample


class Detector(Protocol):
    def detect(self, sample: ImageSample) -> list[Detection]: ...
