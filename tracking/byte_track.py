"""
tracking/byte_track.py — Lightweight ByteTrack implementation (ZCP-05)

Pure Python implementation using filterpy (Kalman filter) and scipy
(Hungarian algorithm for IoU matching). No external ByteTrack package needed.

This implements the core ByteTrack logic:
  1. High-confidence detections matched to existing tracks via IoU
  2. Unmatched tracks kept alive for track_buffer frames (handles occlusion)
  3. Low-confidence detections used as second-pass matches
  4. New tracks created for unmatched high-conf detections

Reference: ByteTrack — Multi-Object Tracking by Associating Every Detection Box
           Zhang et al., ECCV 2022
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    Kalman filter tracker for a single bounding box.
    State: [x, y, w, h, dx, dy, dw, dh] where x,y = center, w,h = size.
    """
    _count = 0

    def __init__(self, bbox: list[float]) -> None:
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State transition matrix
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=float)
        # Measurement matrix
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P         *= 10.
        self.kf.Q[-1,-1]  *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4]     = _xyxy_to_xywh(bbox).reshape((4,1))

        KalmanBoxTracker._count += 1
        self.id           = KalmanBoxTracker._count
        self.hits         = 1
        self.hit_streak   = 1
        self.age          = 0
        self.time_since_update = 0
        self.score        = bbox[4] if len(bbox) > 4 else 1.0

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return _xywh_to_xyxy(self.kf.x[:4].flatten())

    def update(self, bbox: list[float]) -> None:
        self.time_since_update = 0
        self.hits             += 1
        self.hit_streak       += 1
        self.score             = bbox[4] if len(bbox) > 4 else self.score
        self.kf.update(_xyxy_to_xywh(bbox).reshape((4,1)))

    def get_state(self) -> np.ndarray:
        return _xywh_to_xyxy(self.kf.x[:4].flatten())


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Args:
        track_thresh:  Min confidence to create a new track.
        track_buffer:  Frames to keep a lost track alive.
        match_thresh:  IoU threshold for track-detection matching.
        frame_rate:    Camera FPS (affects track buffer duration).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int   = 30,
        match_thresh: float = 0.8,
        frame_rate:   float = 10.0,
    ) -> None:
        self.track_thresh  = track_thresh
        self.match_thresh  = match_thresh
        self.max_age       = int(frame_rate * track_buffer / 30)
        self._trackers:    list[KalmanBoxTracker] = []
        KalmanBoxTracker._count = 0   # reset IDs on new tracker

    def update(
        self,
        detections: np.ndarray,
        _frame:     np.ndarray = None,
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: Nx6 array [[x1,y1,x2,y2,conf,cls], ...]
            _frame:     Unused (kept for API compatibility with boxmot)

        Returns:
            Mx8 array [[x1,y1,x2,y2,track_id,conf,cls,det_idx], ...]
        """
        # Predict all existing trackers
        predicted = []
        for t in self._trackers:
            predicted.append(t.predict())

        # Split detections by confidence
        if len(detections) == 0:
            # Age out lost tracks
            self._trackers = [
                t for t in self._trackers
                if t.time_since_update <= self.max_age
            ]
            return np.empty((0, 8))

        high_mask  = detections[:, 4] >= self.track_thresh
        high_dets  = detections[high_mask]
        low_dets   = detections[~high_mask]

        # --- First association: high-conf dets vs all tracks ---
        matched_h, unmatched_t, unmatched_d = self._match(
            predicted, high_dets, thresh=1 - self.match_thresh
        )

        for ti, di in matched_h:
            self._trackers[ti].update(high_dets[di].tolist())

        # --- Second association: low-conf dets vs unmatched tracks ---
        if len(low_dets) > 0 and len(unmatched_t) > 0:
            pred_unmatched = [predicted[i] for i in unmatched_t]
            matched_l, still_unmatched_t, _ = self._match(
                pred_unmatched, low_dets, thresh=0.5
            )
            for ti_local, di in matched_l:
                ti = unmatched_t[ti_local]
                self._trackers[ti].update(low_dets[di].tolist())
            unmatched_t = [unmatched_t[i] for i in still_unmatched_t]

        # --- Create new tracks for unmatched high-conf detections ---
        for di in unmatched_d:
            if high_dets[di, 4] >= self.track_thresh:
                self._trackers.append(KalmanBoxTracker(high_dets[di].tolist()))

        # --- Remove dead tracks ---
        self._trackers = [
            t for t in self._trackers
            if t.time_since_update <= self.max_age
        ]

        # --- Build output ---
        results = []
        for t in self._trackers:
            if t.time_since_update > 0:
                continue   # only return confirmed active tracks
            box = t.get_state()
            results.append([
                box[0], box[1], box[2], box[3],
                float(t.id), t.score, 0.0, 0.0
            ])

        return np.array(results, dtype=float) if results else np.empty((0, 8))

    def _match(
        self,
        predicted:  list[np.ndarray],
        detections: np.ndarray,
        thresh:     float,
    ) -> tuple[list, list, list]:
        """IoU-based Hungarian matching."""
        if not predicted or len(detections) == 0:
            return [], list(range(len(predicted))), list(range(len(detections)))

        iou_matrix = np.zeros((len(predicted), len(detections)))
        for ti, pred in enumerate(predicted):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = 1 - _iou(pred, det[:4])

        row_ind, col_ind = linear_sum_assignment(iou_matrix)

        matched, unmatched_t, unmatched_d = [], [], []

        matched_t = set()
        matched_d = set()
        for ti, di in zip(row_ind, col_ind):
            if iou_matrix[ti, di] <= thresh:
                matched.append((ti, di))
                matched_t.add(ti)
                matched_d.add(di)

        unmatched_t = [i for i in range(len(predicted))  if i not in matched_t]
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]

        return matched, unmatched_t, unmatched_d


# ── Helpers ────────────────────────────────────────────────────────────────────

def _xyxy_to_xywh(box: list | np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → [cx,cy,w,h]"""
    x1, y1, x2, y2 = box[:4]
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=float)


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """[cx,cy,w,h] → [x1,y1,x2,y2]"""
    cx, cy, w, h = box[:4]
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=float)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0., ix2-ix1) * max(0., iy2-iy1)
    if inter == 0:
        return 0.
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)