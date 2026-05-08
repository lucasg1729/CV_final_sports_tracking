# Sports Tracker

Multi-object tracking in sports footage using Kalman filtering and the Hungarian
algorithm, evaluated on the SportsMOT dataset.

This is a from-scratch implementation of the SORT framework (Bewley et al.,
2016), tested on sports (specifically basketball) footage rather than the
pedestrian benchmarks that dominate the tracking literature.

## Project structure

```
sports_tracker/
├── sports_tracker/        # The package itself
│   ├── __init__.py
│   ├── config.py          # YAML config loader
│   ├── kalman.py          # Constant-velocity Kalman filter for bbox tracking
│   ├── association.py     # IOU + Hungarian algorithm for detection->track matching
│   ├── tracker.py         # The SORT tracker: track lifecycle management
│   ├── detector.py        # YOLO detection wrapper (cached to disk)
│   ├── evaluation.py      # MOTA / ID switches via py-motmetrics
│   └── mot_io.py          # MOT-format I/O
├── tests/                 # Unit tests (run before trusting any module)
├── scripts/               # End-to-end runners + verifiers + plotting
├── configs/               # YAML configs for experiments / ablations
├── cache/                 # Cached YOLO detections (gitignored)
├── results/               # Tracker outputs and figures (gitignored)
└── data/                  # Local data dir (gitignored). SportsMOT clips live here.
```

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow

The pipeline is split so that detection (slow, GPU-friendly) and tracking
(fast, CPU-only) can be run independently. This is essential for ablations:
you cache detections once, then sweep tracker hyperparameters in seconds.

1. Run YOLO once per clip, cache detections to disk:
   `python scripts/run_detector.py --clip v_00HRwkvvjtQ_c001`
2. Run the tracker on cached detections:
   `python scripts/run_tracker.py --clip v_00HRwkvvjtQ_c001 --tag maxage-5`
3. Evaluate against ground truth:
   `python scripts/evaluate.py --clip v_00HRwkvvjtQ_c001 --tag maxage-5`

For comparison against YOLO's built-in trackers:

```
python scripts/run_yolo_tracker.py --all --tracker bytetrack
python scripts/run_yolo_tracker.py --all --tracker botsort
```

## Running tests

```
python -m pytest tests/ -v
```
