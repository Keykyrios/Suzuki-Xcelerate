"""Validation of the cabin detection pipeline on the labeled evaluation set.

Runs the local detector over every labeled frame (base plus low-light,
occlusion and clutter variants), matches detections to ground-truth boxes at
the configured IoU threshold, and reports class-wise precision/recall/F1,
false positives, false negatives, occupancy-count confusion and example
failures. Everything is written to experiments/results/.

Usage:
    python experiments/tools/run_vision_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config.loader import get_config  # noqa: E402
from app.vision.detector import CabinDetector  # noqa: E402

CONDITIONS = ("base", "low_light", "occlusion", "clutter")
CLASSES = ("person", "small", "large")
Buckets = ("EMPTY", "P2", "P4", "P6", "OTHER")


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def apply_condition(image: np.ndarray, condition: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    if condition == "low_light":
        out = (out.astype(np.float32) * 0.35).astype(np.uint8)
    elif condition == "occlusion":
        h, w = out.shape[:2]
        y0, x0 = int(h * 0.3), int(w * 0.3)
        out[y0 : y0 + int(h * 0.4), x0 : x0 + int(w * 0.4)] = 128
    elif condition == "clutter":
        h, w = out.shape[:2]
        for _ in range(6):
            ph, pw = int(h * 0.1), int(w * 0.1)
            y = int(rng.uniform(0, h - ph))
            x = int(rng.uniform(0, w - pw))
            out[y : y + ph, x : x + pw] = rng.integers(0, 255, (ph, pw, 3), dtype=np.uint8)
    return out


def bucket(count: int) -> str:
    if count == 0:
        return "EMPTY"
    if count == 2:
        return "P2"
    if count == 4:
        return "P4"
    if count == 6:
        return "P6"
    return "OTHER"


def main() -> None:
    cfg = get_config()
    detector = CabinDetector(cfg)
    iou_threshold = float(cfg.op("vision.eval_iou_threshold"))
    eval_dir = Path(str(cfg.op("vision.eval_set_path")))
    labels = json.loads((eval_dir / "labels.json").read_text(encoding="utf-8"))

    category_to_class = {
        name: ("person" if name == "person" else "small" if name in ("backpack", "handbag") else "large")
        for name in labels["categories"].values()
    }
    gt_counts = {img_id: counts for img_id, counts in labels["images"].items()}
    gt_boxes = {img_id: entries for img_id, entries in labels.get("boxes", {}).items()}

    from PIL import Image

    results: dict[str, dict] = {}
    failures: dict[str, list[dict]] = {}
    confusion_total = np.zeros((len(Buckets), len(Buckets)), dtype=int)

    for condition in CONDITIONS:
        tp = {c: 0 for c in CLASSES}
        fp = {c: 0 for c in CLASSES}
        fn = {c: 0 for c in CLASSES}
        condition_failures: list[dict] = []

        for img_key in sorted(gt_counts, key=int):
            image_id = int(img_key)
            frame = np.asarray(Image.open(eval_dir / f"{image_id:012d}.jpg").convert("RGB"))
            frame = apply_condition(frame, condition, seed=image_id)
            detections = detector.detect(frame)

            truth: dict[str, list[tuple[float, float, float, float]]] = {c: [] for c in CLASSES}
            for entry in gt_boxes.get(img_key, []):
                cls = category_to_class[entry["category"]]
                truth[cls].append(xywh_to_xyxy(entry["bbox_xywh"]))

            per_class_ok = True
            for cls in CLASSES:
                candidates = sorted(
                    (d for d in detections if d.label == cls), key=lambda d: -d.score
                )
                remaining = list(truth[cls])
                for detection in candidates:
                    best_index, best_iou = None, 0.0
                    for index, box in enumerate(remaining):
                        value = iou(detection.bbox_xyxy, box)
                        if value > best_iou:
                            best_index, best_iou = index, value
                    if best_index is not None and best_iou >= iou_threshold:
                        tp[cls] += 1
                        remaining.pop(best_index)
                    else:
                        fp[cls] += 1
                        per_class_ok = False
                fn[cls] += len(remaining)
                if remaining:
                    per_class_ok = False

            true_pax = int(gt_counts[img_key].get(
                next(k for k, v in labels["categories"].items() if v == "person"), 0
            ))
            predicted_pax = sum(1 for d in detections if d.label == "person")
            confusion_total[Buckets.index(bucket(true_pax)), Buckets.index(bucket(predicted_pax))] += 1
            if not per_class_ok:
                condition_failures.append(
                    {
                        "image_id": image_id,
                        "true_persons": true_pax,
                        "predicted_persons": predicted_pax,
                        "detections": len(detections),
                    }
                )

        def prf(class_tp: int, class_fp: int, class_fn: int) -> dict:
            precision = class_tp / max(class_tp + class_fp, 1)
            recall = class_tp / max(class_tp + class_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            return {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": class_tp,
                "fp": class_fp,
                "fn": class_fn,
            }

        per_class = {cls: prf(tp[cls], fp[cls], fn[cls]) for cls in CLASSES}
        results[condition] = {
            "per_class": per_class,
            "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in CLASSES])), 4),
            "images_evaluated": len(gt_counts),
            "images_with_errors": len(condition_failures),
        }
        failures[condition] = condition_failures[:12]

    occupancy_f1 = []
    for index, name in enumerate(Buckets):
        true_positive = confusion_total[index, index]
        predicted = confusion_total[:, index].sum()
        actual = confusion_total[index, :].sum()
        precision = true_positive / max(predicted, 1)
        recall = true_positive / max(actual, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        occupancy_f1.append(round(float(f1), 4))

    report = {
        "model_version": detector.model_version,
        "weights_sha256": detector.weights_sha256,
        "config_hash": cfg.hash,
        "detection_score_threshold": detector.score_threshold,
        "input_size_px": detector.input_size,
        "iou_threshold": iou_threshold,
        "eval_set": str(eval_dir),
        "eval_set_source": labels.get("source"),
        "conditions": results,
        "occupancy_confusion_matrix": {
            "labels": list(Buckets),
            "matrix": confusion_total.tolist(),
            "macro_f1": round(float(np.mean(occupancy_f1)), 4),
            "per_bucket_f1": dict(zip(Buckets, occupancy_f1)),
        },
        "failure_examples": failures,
        "notes": (
            "Evaluation frames are open-licensed COCO 2017 val images selected by "
            "person/luggage counts; low_light, occlusion and clutter apply seeded "
            "transformations with unchanged labels. This measures the cabin "
            "detection pipeline on this controlled set, not field performance in "
            "a vehicle cabin."
        ),
    }
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vision_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Cabin detection validation", ""]
    lines.append(f"Model: `{detector.model_version}` · IoU {iou_threshold} · score gate {detector.score_threshold}")
    lines.append("")
    for condition in CONDITIONS:
        block = results[condition]
        lines.append(f"## {condition} (macro-F1 {block['macro_f1']})")
        lines.append("")
        lines.append("| class | precision | recall | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        for cls in CLASSES:
            row = block["per_class"][cls]
            lines.append(
                f"| {cls} | {row['precision']} | {row['recall']} | {row['f1']} "
                f"| {row['tp']} | {row['fp']} | {row['fn']} |"
            )
        lines.append("")
    lines.append("## Occupancy count confusion (base+variants pooled)")
    lines.append("")
    lines.append("| true \\ predicted | " + " | ".join(Buckets) + " |")
    lines.append("|" + "---|" * (len(Buckets) + 1))
    for index, name in enumerate(Buckets):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in confusion_total[index].tolist()) + " |")
    lines.append("")
    lines.append(f"Occupancy macro-F1: {report['occupancy_confusion_matrix']['macro_f1']}")
    lines.append("")
    lines.append("Failure examples are stored in `vision_validation.json` under `failure_examples`.")
    (out_dir / "vision_validation.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({c: results[c]["macro_f1"] for c in CONDITIONS}, indent=2))
    print("occupancy macro-F1:", report["occupancy_confusion_matrix"]["macro_f1"])
    print(f"wrote {out_dir / 'vision_validation.json'}")
    print(f"wrote {out_dir / 'vision_validation.md'}")


if __name__ == "__main__":
    main()
