# Cabin detection validation

Model: `yolov8s:1f47a78bf100` · IoU 0.5 · score gate 0.25

## base (macro-F1 0.6205)

| class | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| person | 0.7935 | 0.7122 | 0.7506 | 146 | 38 | 59 |
| small | 0.6 | 0.1731 | 0.2687 | 9 | 6 | 43 |
| large | 0.7273 | 1.0 | 0.8421 | 8 | 3 | 0 |

## low_light (macro-F1 0.6033)

| class | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| person | 0.8286 | 0.7073 | 0.7632 | 145 | 30 | 60 |
| small | 0.5789 | 0.2115 | 0.3099 | 11 | 8 | 41 |
| large | 0.6364 | 0.875 | 0.7368 | 7 | 4 | 1 |

## occlusion (macro-F1 0.3532)

| class | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| person | 0.5279 | 0.5073 | 0.5174 | 104 | 93 | 101 |
| small | 0.4667 | 0.1346 | 0.209 | 7 | 8 | 45 |
| large | 0.3 | 0.375 | 0.3333 | 3 | 7 | 5 |

## clutter (macro-F1 0.5081)

| class | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| person | 0.7582 | 0.6732 | 0.7132 | 138 | 44 | 67 |
| small | 0.3333 | 0.1731 | 0.2278 | 9 | 18 | 43 |
| large | 0.4375 | 0.875 | 0.5833 | 7 | 9 | 1 |

## Occupancy count confusion (base+variants pooled)

| true \ predicted | EMPTY | P2 | P4 | P6 | OTHER |
|---|---|---|---|---|---|
| EMPTY | 80 | 0 | 0 | 0 | 0 |
| P2 | 6 | 17 | 3 | 0 | 14 |
| P4 | 0 | 4 | 13 | 5 | 18 |
| P6 | 2 | 2 | 2 | 8 | 18 |
| OTHER | 1 | 8 | 6 | 0 | 33 |

Occupancy macro-F1: 0.5382

Failure examples are stored in `vision_validation.json` under `failure_examples`.