"""
ingestion/preprocessor.py — Preprocessing pipeline for low-quality CCTV frames.

Pipeline (in order):
    1. Gamma correction       — fix dark/overexposed scenes
    2. Bilateral filtering    — edge-preserving noise reduction (replaces NLM;
                                ~10ms/frame vs ~200ms for NLM on CPU)
    3. CLAHE                  — adaptive contrast enhancement for mixed lighting
    4. Resize to 640x640      — standard YOLO input resolution

Why bilateral over NLM:
    NLM is the highest quality denoiser but is O(n²) in search window size,
    making it too slow for real-time use on CPU (200ms+/frame). Bilateral
    filter achieves similar perceptual quality for CCTV footage at ~5-10ms/frame
    by using spatial + colour proximity weights rather than exhaustive patch search.

All parameters are read from config.yaml (preprocessing section).
No hardcoded values.

Usage:
    from ingestion.preprocessor import Preprocessor
    pre = Preprocessor()
    processed_frame = pre.process(raw_frame)
"""

import cv2
import numpy as np

from config.config_loader import cfg
from utils.logger import get_logger

log = get_logger("system")


class Preprocessor:
    """
    Applies the full preprocessing chain to a raw BGR frame.

    The CLAHE object is created once at init time (not per frame) since
    creating it repeatedly is expensive.

    Args:
        gamma:              Gamma correction value. >1 brightens, <1 darkens.
        clahe_clip_limit:   CLAHE contrast cap. Higher = more aggressive.
        clahe_tile_grid:    Tuple (w, h) tile grid size for CLAHE.
        bilateral_d:        Bilateral filter neighbourhood diameter.
        bilateral_sigma_color: Bilateral sigma in colour space.
        bilateral_sigma_space: Bilateral sigma in coordinate space.
        target_size:        Output (width, height) after resize.

    All args default to values from config.yaml if not provided.
    """

    def __init__(
        self,
        gamma:                 float            = None,
        clahe_clip_limit:      float            = None,
        clahe_tile_grid:       tuple            = None,
        bilateral_d:           int              = None,
        bilateral_sigma_color: float            = None,
        bilateral_sigma_space: float            = None,
        target_size:           tuple[int, int]  = None,
    ) -> None:
        p = cfg.preprocessing

        self.gamma                 = gamma                 or p.gamma
        self.clahe_clip            = clahe_clip_limit      or p.clahe_clip_limit
        self.tile_grid             = tuple(clahe_tile_grid or p.clahe_tile_grid_size)
        self.bilateral_d           = bilateral_d           or p.bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color or p.bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space or p.bilateral_sigma_space
        self.target_size           = target_size           or (p.target_width, p.target_height)

        # Build gamma lookup table once (256-entry uint8 LUT)
        self._gamma_lut = self._build_gamma_lut(self.gamma)

        # Build CLAHE object once — reused across all frames
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.tile_grid,
        )

        log.info(
            f"Preprocessor ready | gamma={self.gamma} "
            f"bilateral_d={self.bilateral_d} "
            f"sigma_color={self.bilateral_sigma_color} "
            f"sigma_space={self.bilateral_sigma_space} "
            f"clahe_clip={self.clahe_clip} "
            f"tile_grid={self.tile_grid} "
            f"target={self.target_size}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing chain on a single BGR frame.

        Steps:
            gamma → bilateral denoise → CLAHE → resize

        Args:
            frame: Raw BGR frame from VideoCapture (H x W x 3, uint8).

        Returns:
            Preprocessed BGR frame of shape (target_height, target_width, 3).
        """
        frame = self._apply_gamma(frame)
        frame = self._apply_bilateral(frame)
        frame = self._apply_clahe(frame)
        frame = self._apply_resize(frame)
        return frame

    # ── Pipeline steps ─────────────────────────────────────────────────────────

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction via a precomputed 256-entry LUT.

        LUT lookup is O(1) per pixel — much faster than per-pixel pow().
        """
        return cv2.LUT(frame, self._gamma_lut)

    def _apply_bilateral(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving noise reduction.

        Bilateral filter smooths flat regions while preserving edges by
        weighting neighbours by both spatial distance AND colour similarity.
        This is critical for CCTV footage where seat edges must remain sharp
        for accurate IoU computation against ROI polygons.

        Args:
            frame: BGR frame.

        Returns:
            Filtered BGR frame.
        """
        return cv2.bilateralFilter(
            frame,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel of the LAB colour space.

        Applying CLAHE only to the luminance channel (L) avoids distorting
        colour information — saturation and hue (A, B channels) are unchanged.

        Steps:
            BGR → LAB → CLAHE on L → LAB → BGR
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _apply_resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target_size using INTER_LINEAR interpolation.

        INTER_LINEAR is the best tradeoff between speed and quality for
        downscaling to 640x640 from typical CCTV resolutions.
        """
        w, h = self.target_size
        if frame.shape[1] == w and frame.shape[0] == h:
            return frame
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """
        Build a 256-entry uint8 LUT for gamma correction.

        Formula: output = (input / 255) ^ (1 / gamma) * 255

        A gamma > 1 brightens the image (useful for dark CCTV feeds).
        A gamma < 1 darkens it.

        Returns:
            NumPy array of shape (256,) and dtype uint8.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return table