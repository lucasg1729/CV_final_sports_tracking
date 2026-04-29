# CV_final_sports_tracking
sports_tracker/
├── sports_tracker/        # The package itself
│   ├── __init__.py
│   ├── kalman.py          # Constant-velocity Kalman filter for bbox tracking
│   ├── association.py     # IOU + Hungarian algorithm for detection->track matching
│   ├── tracker.py         # The SORT tracker: track lifecycle management
│   ├── detector.py        # YOLO detection wrapper (cached to disk)
│   ├── evaluation.py      # MOTA / ID switches via py-motmetrics
│   └── io.py              # SportsMOT dataset loading + MOT-format I/O
├── tests/                 # Unit tests (run before trusting any module)
├── scripts/               # End-to-end runners (detect, track, evaluate)
├── configs/               # YAML configs for experiments / ablations
└── data/                  # Local data dir (gitignored). SportsMOT clips live here.
