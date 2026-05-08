"""The SORT tracker

Wires the Kalman filter and the IOU+Hungarian association module into
a complete multi-object tracker, following Bewley et al. 2016.

A Tracker holds a list of Track objects, where each Track wraps a
Kalman filter and a unique integer ID. Each frame, the tracker:

  1. Predicts every track's position one frame forward
  2. Associates new detections to predicted tracks via IOU + Hungarian
  3. Updates matched tracks with their assigned detection
  4. Increments time-since-update on unmatched tracks; deletes any that
     have been unmatched for more than `max_age` frames
  5. Creates a new tentative track for each unmatched detection

Track lifecycle: a track is tentative until it has accumulated
min_hits consecutive successful updates, at which point it becomes
confirmed. Only confirmed tracks are reported in the per-frame
output, which prevents transient detector noise from generating short
ghost tracks.

A confirmed track that gets briefly unmatched (e.g., during a one-frame
occlusion) is still reported in the output, using the Kalman filter's
predicted position. This is the entire point of having a motion model:
the track persists through brief gaps in detections.

Parameters and their relationship to SORT defaults:

  max_age   SORT default: 1. We default to 30 (about 1 second at 30 fps),
            which is much more permissive. The article review for this
            project identified max_age=1 as poorly suited to sports
            where brief occlusions are constant.
  min_hits  SORT default: 3. We use the same. Probably not worth changing
            unless the detector is very noisy.
  iou_thr   SORT default: 0.3. Tunable here for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from sports_tracker.association import DEFAULT_IOU_THRESHOLD, associate
from sports_tracker.kalman import BoxKalmanFilter


# Defaults differ from SORT's reference values (max_age=1, min_hits=3)
# because the SORT defaults are tuned for pedestrian sequences. We
# loosen max_age substantially for sports
DEFAULT_MAX_AGE = 30
DEFAULT_MIN_HITS = 3


@dataclass
class Track:
    """A single tracked object: a Kalman filter plus identity bookkeeping

    Attributes:
        id: A globally unique positive integer assigned at creation time
        kf: The Kalman filter holding this track's state
        hits: Total number of successful updates over this track's life
        confirmed: Whether the track has crossed the min_hits threshold
            and is being reported in output. Once confirmed, a track
            stays confirmed until it's deleted
    """

    id: int
    kf: BoxKalmanFilter
    confirmed: bool = False

    @property
    def time_since_update(self) -> int:
        return self.kf.time_since_update

    @property
    def hits(self) -> int:
        return self.kf.hits

    @property
    def hit_streak(self) -> int:
        return self.kf.hit_streak

    def predict(self) -> np.ndarray:
        return self.kf.predict()

    def update(self, bbox: np.ndarray) -> None:
        self.kf.update(bbox)

    def get_state(self) -> np.ndarray:
        return self.kf.get_state()


@dataclass
class TrackOutput:
    """One row of output from the tracker for a given frame

    Attributes:
        id: The track's unique ID
        bbox: Current bounding box estimate as [x1, y1, x2, y2]
        time_since_update: 0 if this track was matched this frame, otherwise
            the number of frames since the last successful match. Useful
            for downstream code that wants to filter out tracks that are
            being predicted-only (no detection support)
    """

    id: int
    bbox: np.ndarray
    time_since_update: int


class Tracker:
    """A SORT-style multi-object tracker"""

    def __init__(
        self,
        max_age: int = DEFAULT_MAX_AGE,
        min_hits: int = DEFAULT_MIN_HITS,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        """Initialize an empty tracker

        Args:
            max_age: Delete a track after this many consecutive frames
                without a matching detection. Higher values tolerate
                longer occlusions but increase the chance of identity
                drift onto a different object
            min_hits: A new track must accumulate this many consecutive
                successful updates before it appears in output
            iou_threshold: Detection-to-track pairs with IOU below this
                are not matched (handed off to spawn/delete logic)
        """
        if max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {max_age}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: List[Track] = []
        self._next_id = 1  # Track IDs start at 1; 0 is reserved for "no track"
        self._frame_count = 0

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def step(self, detections: np.ndarray) -> List[TrackOutput]:
        """Process one frame of detections and return the active tracks

        Args:
            detections: shape (M, 4) array of detection boxes [x1, y1, x2, y2].
                May be empty (shape (0, 4)) if the detector found nothing

        Returns:
            A list of TrackOutput for every track that should be reported
            this frame. A track is reported if either:
              - it was matched this frame and has crossed min_hits, OR
              - it was already confirmed previously and is still alive
                (i.e., within its grace period)
            New tentative tracks (just created, hits < min_hits) are not
            reported
        """
        self._frame_count += 1
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 4)

        # 1. Predict every existing track forward by one frame
        predicted_boxes = np.array(
            [t.predict() for t in self.tracks], dtype=np.float64
        ).reshape(-1, 4)

        # Any track whose predicted box has NaN values is broken
        # Drop it before association rather than poisoning the IOU matrix
        valid_mask = np.all(np.isfinite(predicted_boxes), axis=1) if len(predicted_boxes) else np.array([], dtype=bool)
        if len(predicted_boxes) and not valid_mask.all():
            self.tracks = [t for t, ok in zip(self.tracks, valid_mask) if ok]
            predicted_boxes = predicted_boxes[valid_mask]

        # 2. Associate detections to tracks
        result = associate(
            predicted_boxes, detections, iou_threshold=self.iou_threshold
        )

        # 3. Update matched tracks
        for track_idx, det_idx in result.matches:
            self.tracks[track_idx].update(detections[det_idx])

        # 4. Spawn new tracks for unmatched detections
        for det_idx in result.unmatched_detections:
            kf = BoxKalmanFilter(detections[det_idx])
            self.tracks.append(Track(id=self._new_id(), kf=kf))

        # 5. Build the per-frame output and promote tentative tracks
        #
        # A track is reported this frame if:
        #   (a) it is already confirmed and still alive (within max_age), OR
        #   (b) it is tentative, was matched this frame, AND has accumulated
        #       enough consecutive hits to be promoted now
        #
        # The "promote on this frame's match" case in (b) handles the very
        # first frames of the tracker. Without it, a track born in frame 1
        # would not appear in output until frame min_hits + 1, even though
        # it has been matched every frame since birth. The compromise is
        # to allow promotion as soon as hit_streak >= min_hits, but only
        # if the track was actually matched this frame.
        outputs: List[TrackOutput] = []
        for t in self.tracks:
            # Tracks beyond max_age are about to be deleted in step 6;
            # don't emit a final output row for them. This makes max_age
            # the literal upper bound on time_since_update in output
            if t.time_since_update > self.max_age:
                continue
            if t.confirmed:
                outputs.append(
                    TrackOutput(
                        id=t.id,
                        bbox=t.get_state(),
                        time_since_update=t.time_since_update,
                    )
                )
            elif t.time_since_update == 0 and t.hit_streak >= self.min_hits:
                t.confirmed = True
                outputs.append(
                    TrackOutput(
                        id=t.id,
                        bbox=t.get_state(),
                        time_since_update=0,
                    )
                )

        # 6. Delete tracks that have been unmatched for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return outputs

    def reset(self) -> None:
        """Clear all tracks and reset internal counters

        Call between clips when running on multiple sequences in a row,
        otherwise track IDs and frame counts will leak from one clip
        into the next
        """
        self.tracks = []
        self._next_id = 1
        self._frame_count = 0
