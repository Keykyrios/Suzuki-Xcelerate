"""Frame acquisition for the cabin pipeline.

Frames live in memory only and are discarded immediately after inference;
nothing here writes images to disk. A prerecorded sample stands in for a cabin
camera when no device is available, so the demonstrator keeps working offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.config.loader import ConfigBundle, get_config


class CameraUnavailable(RuntimeError):
    """The requested frame source cannot deliver frames."""


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None:
        """Return one RGB frame as a uint8 array, or None when unavailable."""

    def close(self) -> None: ...


class LocalCameraSource:
    """Webcam capture through OpenCV; frames are handed to the caller in memory."""

    def __init__(self, device_index: int) -> None:
        import cv2

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            raise CameraUnavailable(f"camera device {device_index} did not open")

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._capture.release()


class PrerecordedSource:
    """Loops over a directory of sample frames; the directory is read-only input."""

    def __init__(self, directory: Path) -> None:
        if not directory.is_dir():
            raise CameraUnavailable(f"prerecorded frame directory not found: {directory}")
        self._frames = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self._frames:
            raise CameraUnavailable(f"no sample frames found in {directory}")
        self._index = 0

    def read(self) -> np.ndarray | None:
        from PIL import Image

        path = self._frames[self._index % len(self._frames)]
        self._index += 1
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def close(self) -> None:
        return None


def open_frame_source(cfg: ConfigBundle | None = None) -> FrameSource:
    """Build the configured frame source; falls back to the sample when a
    local camera cannot be opened."""
    c = cfg or get_config()
    kind = str(c.op("vision.camera_source"))
    if kind == "local":
        try:
            return LocalCameraSource(int(c.op("vision.camera_device_index")))
        except CameraUnavailable:
            return PrerecordedSource(Path(str(c.op("vision.prerecorded_frames_path"))))
    if kind == "prerecorded":
        return PrerecordedSource(Path(str(c.op("vision.prerecorded_frames_path"))))
    raise CameraUnavailable(f"unknown camera_source '{kind}'")
