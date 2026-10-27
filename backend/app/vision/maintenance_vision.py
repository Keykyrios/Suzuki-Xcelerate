"""Optional maintenance cue classifier for visible tyre/component cues.

The classifier is a small convolutional network trained on procedurally
generated cue imagery; its provenance is synthetic and uncalibrated, and its
output is a coarse visual cue (inspect / no_obvious_issue_detected / uncertain)
- never a safety or roadworthiness judgement. The weights file, input size and
the confidence floor below which the label degrades to "uncertain" all come
from configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import EvidenceLabel

LABEL_INSPECT = "inspect"
LABEL_NO_ISSUE = "no_obvious_issue_detected"
LABEL_UNCERTAIN = "uncertain"
TRAINED_LABELS = (LABEL_NO_ISSUE, LABEL_INSPECT)


class MaintenanceCueNet(nn.Module):
    """Two-convolution cue classifier over grayscale crops."""

    def __init__(self, num_classes: int = len(TRAINED_LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@dataclass(frozen=True)
class MaintenanceCue:
    label: str
    confidence: float
    evidence_status: EvidenceLabel
    disclaimer: str


class MaintenanceCueClassifier:
    """Loads the locally trained cue model and reports coarse labels."""

    def __init__(self, cfg: ConfigBundle | None = None) -> None:
        c = cfg or get_config()
        self.cfg = c
        self.input_px = int(c.op("vision.maintenance_input_px"))
        self.uncertain_below = float(c.op("vision.maintenance_uncertain_below"))
        self.disclaimer = str(c.op("vision.maintenance_disclaimer"))
        weights_path = Path(str(c.op("vision.maintenance_model_path")))
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"maintenance cue weights not found at {weights_path}; "
                "run experiments/tools/train_maintenance_cue.py to produce them"
            )
        self.model = MaintenanceCueNet()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["weights"] if "weights" in state else state)
        self.model.eval()
        self.training_accuracy = float(state.get("held_out_accuracy", float("nan"))) if isinstance(state, dict) else float("nan")

    @torch.no_grad()
    def classify(self, frame_rgb: np.ndarray) -> MaintenanceCue:
        """Classify visible cues in one in-memory frame; the frame is dropped."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB frame, got shape {frame_rgb.shape}")
        tensor = self._preprocess(frame_rgb)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        best = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[best].item())
        if confidence < self.uncertain_below:
            label = LABEL_UNCERTAIN
        else:
            label = TRAINED_LABELS[best]
        return MaintenanceCue(
            label=label,
            confidence=round(confidence, 3),
            evidence_status=EvidenceLabel.SYNTHETIC,
            disclaimer=self.disclaimer,
        )

    def _preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        from PIL import Image

        image = Image.fromarray(frame_rgb).convert("L").resize(
            (self.input_px, self.input_px)
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
