"""Train the maintenance cue classifier on procedurally generated imagery.

The training distribution is synthetic: rendered tyre-like discs in good
condition versus tyres with visible cues (deflated contact patch, surface
cracks). That makes the model honest about what it can and cannot say - it is
a demonstrator classifier over a controlled distribution, labelled SYNTHETIC
and UNCALIBRATED, and it never outputs a roadworthiness verdict.

Usage:
    python experiments/tools/train_maintenance_cue.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.vision.maintenance_vision import (  # noqa: E402
    LABEL_INSPECT,
    LABEL_NO_ISSUE,
    MaintenanceCueNet,
)

SEED = 20260101
EPOCHS = 14
BATCH = 64
TRAIN_PER_CLASS = 2000
TEST_PER_CLASS = 500
INPUT_PX = 64


def render_sample(rng: np.random.Generator, inspect: bool) -> np.ndarray:
    """One 64x64 grayscale tyre-cue image."""
    size = INPUT_PX
    image = rng.normal(0.45, 0.06, (size, size)).clip(0.0, 1.0)
    cy = size / 2 + rng.normal(0, 1.0)
    cx = size / 2 + rng.normal(0, 1.0)
    radius = size * 0.34 * float(rng.uniform(0.9, 1.05))

    yy, xx = np.mgrid[0:size, 0:size]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    image[inside] = rng.normal(0.18, 0.03, inside.sum()).clip(0.0, 1.0)

    # Sidewall ring.
    ring = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius * 0.82) < 1.2
    image[ring] = rng.normal(0.32, 0.03, ring.sum()).clip(0.0, 1.0)

    if inspect:
        if rng.random() < 0.5:
            # Deflated contact patch: flatten the bottom of the disc.
            flat_y = cy + radius * float(rng.uniform(0.55, 0.8))
            patch = inside & (yy > flat_y)
            image[patch] = rng.normal(0.45, 0.05, patch.sum()).clip(0.0, 1.0)
        else:
            # Surface cracks: bright jagged streaks inside the disc.
            for _ in range(int(rng.integers(1, 4))):
                start_x = float(rng.uniform(cx - radius * 0.5, cx + radius * 0.5))
                start_y = float(rng.uniform(cy - radius * 0.5, cy + radius * 0.5))
                for step in range(int(rng.integers(6, 14))):
                    x = int(start_x + rng.normal(0, 1.5) + step * rng.choice([-1, 1]))
                    y = int(start_y + rng.normal(0, 1.5) + step)
                    if 0 <= x < size and 0 <= y < size and inside[y, x]:
                        image[y, x] = float(rng.uniform(0.7, 0.95))
    # Lighting and sensor noise variations.
    image = image * float(rng.uniform(0.8, 1.2)) + rng.normal(0, 0.02, image.shape)
    return image.clip(0.0, 1.0).astype(np.float32)


def build_dataset(rng: np.random.Generator, per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for label_index, inspect in enumerate([False, True]):
        for _ in range(per_class):
            images.append(render_sample(rng, inspect))
            labels.append(label_index)
    order = rng.permutation(len(labels))
    x = torch.from_numpy(np.stack(images)[order]).unsqueeze(1)
    y = torch.tensor(np.array(labels)[order], dtype=torch.long)
    return x, y


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    x_train, y_train = build_dataset(rng, TRAIN_PER_CLASS)
    x_test, y_test = build_dataset(rng, TEST_PER_CLASS)

    model = MaintenanceCueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for start in range(0, len(x_train), BATCH):
            xb = x_train[start : start + BATCH]
            yb = y_train[start : start + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        print(f"epoch {epoch + 1}/{EPOCHS} loss {total_loss / len(x_train):.4f}")

    model.eval()
    confusion = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        logits = model(x_test)
        predicted = logits.argmax(dim=1)
        for truth, guess in zip(y_test.tolist(), predicted.tolist()):
            confusion[truth][guess] += 1
    accuracy = float(np.trace(confusion) / confusion.sum())
    per_class_recall = [
        float(confusion[i, i] / max(confusion[i].sum(), 1)) for i in range(2)
    ]
    print(f"held-out accuracy {accuracy:.3f}")
    print(f"recall no_issue {per_class_recall[0]:.3f} | recall inspect {per_class_recall[1]:.3f}")

    out_path = ROOT / "data" / "models" / "maintenance_cue_cnn.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": model.state_dict(),
            "held_out_accuracy": accuracy,
            "seed": SEED,
            "input_px": INPUT_PX,
        },
        out_path,
    )
    report = {
        "model": "MaintenanceCueNet",
        "provenance": "SYNTHETIC procedural training imagery",
        "seed": SEED,
        "epochs": EPOCHS,
        "train_samples": len(x_train),
        "held_out_samples": len(x_test),
        "held_out_accuracy": round(accuracy, 4),
        "per_class_recall": {
            LABEL_NO_ISSUE: round(per_class_recall[0], 4),
            LABEL_INSPECT: round(per_class_recall[1], 4),
        },
        "confusion_matrix": confusion.tolist(),
        "weights_path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
    }
    report_path = ROOT / "experiments" / "results" / "maintenance_cue_training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved weights to {out_path}")
    print(f"saved training report to {report_path}")


if __name__ == "__main__":
    main()
