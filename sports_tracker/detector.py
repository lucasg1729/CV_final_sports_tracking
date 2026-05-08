"""Detection pipeline.

Two pieces:

1. YoloDetector: a thin wrapper around ultralytics YOLOv8 that runs
   inference on one frame at a time and returns person-class detections
   as numpy arrays

2. A cache format and helpers (save_detections, load_detections)
   that store all detections for an entire clip in a single .npz file

YOLO inference is slow on CPU (a few seconds per frame for v8x, sub-second 
per frame for v8n) so we us a cache instead. A typical SportsMOT clip
is ~750 frames, so a single inference pass is several minutes. 
The tracker, by contrast, runs in milliseconds. We never
want to re-run YOLO when sweeping tracker hyperparameters, so we run
detection once, cache it, and feed the cache into every tracker
experiment.

The cache also fixes the inputs to the tracker for reproducibility since
two reproducers running on different hardware will get bit-identical
tracking results as long as they read the same cached detections

Cache format (one .npz per clip):
    frame_indices : (N,) int32  - 1-indexed frame numbers (matches MOT)
    detections    : (N, 5) float32  - columns: x1, y1, x2, y2, conf
    meta          : (1,) object array containing a dict with clip name,
                    YOLO model, confidence threshold used during run

Detections are stored sorted by frame index, then by row order from YOLO
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# COCO class index for "person". Used to filter YOLO output to only
# people since we don't track balls/equipment in this project
PERSON_CLASS_ID = 0

# Default confidence threshold for keeping a detection. The tracker
# applies its own filtering at load time too, so this is just the
# floor below which we don't even bother caching (saves disk)
DEFAULT_CACHE_CONF = 0.1


@dataclass
class FrameDetections:
    """All detections from one frame

    Attributes:
        frame_idx: 1-indexed frame number (matches MOT/SportsMOT convention)
        boxes: (M, 4) array of [x1, y1, x2, y2] boxes (may be empty)
        scores: (M,) array of confidence scores in [0, 1]
    """

    frame_idx: int
    boxes: np.ndarray
    scores: np.ndarray

    def filter(self, min_conf: float) -> "FrameDetections":
        """Return a new FrameDetections containing only boxes above min_conf"""
        mask = self.scores >= min_conf
        return FrameDetections(
            frame_idx=self.frame_idx,
            boxes=self.boxes[mask],
            scores=self.scores[mask],
        )


class YoloDetector:
    """A thin wrapper around ultralytics YOLOv8

    Loads the model lazily on first use so that import doesn't pull in
    PyTorch (which is heavy and slow). This means tests that don't
    actually run YOLO can still import this module fine
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: Optional[str] = None,
    ):
        """Initialize the detector

        Args:
            model_name: ultralytics model identifier, e.g. ``yolov8n.pt``
                (nano), ``yolov8s.pt`` (small), ``yolov8x.pt`` (extra
                large). Larger models are more accurate but slower.
                Files are downloaded automatically on first use
            device: ``cpu``, ``cuda``, or ``mps``. Pass ``None`` to let
                ultralytics auto-detect
        """
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy

    def _ensure_loaded(self):
        if self._model is None:
            # Defer the import so users without ultralytics can still
            # import this module to use the cache helpers
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)

    def detect_frame(
        self, image: np.ndarray, conf: float = DEFAULT_CACHE_CONF
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run YOLO on a single image and return person-class boxes

        Args:
            image: (H, W, 3) BGR or RGB image. ultralytics handles either
            conf: minimum confidence to return a detection

        Returns:
            (boxes, scores):
                boxes:  (M, 4) float32 array, [x1, y1, x2, y2] in pixels
                scores: (M,)   float32 array of confidence scores
        """
        self._ensure_loaded()

        # ultralytics returns a list of Results (one per input image,
        # but we pass one image at a time so we take the only entry)
        # verbose=False suppresses the per-frame stdout spam
        results = self._model.predict(
            image,
            classes=[PERSON_CLASS_ID],
            conf=conf,
            device=self.device,
            verbose=False,
        )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        # .xyxy is a tensor of [x1, y1, x2, y2]; .conf is a tensor of scores
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        return boxes, scores


# Cache I/O


def save_detections(
    out_path: Path,
    frame_dets: List[FrameDetections],
    meta: dict,
) -> None:
    """Save a clip's detections to a single .npz file

    Args:
        out_path: where to write. Parent directory will be created
        frame_dets: list of per-frame detections, in any order. They get
            sorted by frame_idx before saving
        meta: free-form dict to embed (clip name, model name, conf, etc.).
            Stored in the npz as a 1-element object array, retrievable
            via load_detections
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten per-frame detections into two parallel arrays
    frame_indices = []
    rows = []  # each row: [x1, y1, x2, y2, conf]
    for fd in sorted(frame_dets, key=lambda f: f.frame_idx):
        for box, score in zip(fd.boxes, fd.scores):
            frame_indices.append(fd.frame_idx)
            rows.append([box[0], box[1], box[2], box[3], score])

    if rows:
        frame_indices_arr = np.array(frame_indices, dtype=np.int32)
        detections_arr = np.array(rows, dtype=np.float32)
    else:
        frame_indices_arr = np.empty(0, dtype=np.int32)
        detections_arr = np.empty((0, 5), dtype=np.float32)

    np.savez_compressed(
        out_path,
        frame_indices=frame_indices_arr,
        detections=detections_arr,
        meta=np.array([meta], dtype=object),
    )


def load_detections(
    cache_path: Path,
    min_conf: float = 0.0,
    n_frames: Optional[int] = None,
) -> tuple[List[FrameDetections], dict]:
    """Load a cached clip's detections

    Args:
        cache_path: path to the .npz produced by save_detections
        min_conf: drop detections below this confidence at load time
            This is how the tracker pipeline tunes the conf threshold
            without re-running YOLO
        n_frames: if given, the returned list will have exactly n_frames
            entries (one per frame, indexed 1..n_frames), with empty
            FrameDetections for frames that had zero detections. If None,
            the list contains only frames that had at least one detection

    Returns:
        (frame_detections_list, meta_dict)
    """
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=True)
    frame_indices = data["frame_indices"]
    detections = data["detections"]
    meta = data["meta"][0] if data["meta"].size > 0 else {}

    # Apply the confidence cutoff up front
    if len(detections) and min_conf > 0.0:
        keep = detections[:, 4] >= min_conf
        frame_indices = frame_indices[keep]
        detections = detections[keep]

    # Group rows by frame index
    by_frame: dict[int, list[np.ndarray]] = {}
    by_frame_scores: dict[int, list[float]] = {}
    for fi, row in zip(frame_indices, detections):
        fi_int = int(fi)
        by_frame.setdefault(fi_int, []).append(row[:4])
        by_frame_scores.setdefault(fi_int, []).append(row[4])

    if n_frames is None:
        # Return just the frames with detections, sorted
        result = []
        for fi in sorted(by_frame):
            result.append(
                FrameDetections(
                    frame_idx=fi,
                    boxes=np.array(by_frame[fi], dtype=np.float32).reshape(-1, 4),
                    scores=np.array(by_frame_scores[fi], dtype=np.float32),
                )
            )
        return result, meta

    # Dense mode: one entry per frame from 1...n_frames, empty if none
    result = []
    for fi in range(1, n_frames + 1):
        if fi in by_frame:
            result.append(
                FrameDetections(
                    frame_idx=fi,
                    boxes=np.array(by_frame[fi], dtype=np.float32).reshape(-1, 4),
                    scores=np.array(by_frame_scores[fi], dtype=np.float32),
                )
            )
        else:
            result.append(
                FrameDetections(
                    frame_idx=fi,
                    boxes=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty(0, dtype=np.float32),
                )
            )
    return result, meta
