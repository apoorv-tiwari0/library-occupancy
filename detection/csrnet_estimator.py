"""
detection/csrnet_estimator.py — CSRNet far zone density estimator (ZCP-09)

Uses CSRNet (fine-tuned on Indian metro crowd data) for far-zone headcount.
Architecture: VGG-16 frontend + dilated conv backend + 1x1 output layer.

Why CSRNet over DM-Count for this project:
  - Fine-tuned on Indian demographic (Delhi/Hyderabad/Mumbai metro)
  - MAE=12.36 on Indian crowd data vs ~50 for vanilla CSRNet
  - Faster inference (<0.5s CPU, ~80ms GPU)
  - Simpler architecture — weights load reliably

Usage:
    from detection.csrnet_estimator import CSRNetFarZoneEstimator
    estimator = CSRNetFarZoneEstimator(zone_config["g_huss"], "g_huss")
    count, density_map = estimator.estimate(frame)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from detection.zone_detector import get_zone_crop
from config.config_loader import cfg

log = get_logger("system")

# ── Transforms ─────────────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ── CSRNet architecture (matches model.py exactly) ────────────────────────────

def _make_layers(cfg_list, in_channels=3, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg_list:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v,
                kernel_size=3,
                padding=d_rate,
                dilation=d_rate,
            )
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Dense Crowd Counting.
    Architecture exactly matches the downloaded checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0
        frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend     = _make_layers(frontend_feat, in_channels=3,   dilation=False)
        self.backend      = _make_layers(backend_feat,  in_channels=512,  dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# ── Far zone estimator ────────────────────────────────────────────────────────

class CSRNetFarZoneEstimator:
    """
    Estimates headcount in the FAR zone using CSRNet.

    The far zone crop is passed through CSRNet which produces a density map.
    Summing the density map × 100 (CSRNet convention) gives the count.
    A per-section scale_factor corrects for perspective/scene differences.

    Args:
        zone_config:   Zone polygons {"near":..., "mid":..., "far":...}
        section_id:    Section identifier.
        weights_path:  Path to CSRNet .pth weights.
        device:        'cuda' or 'cpu'.
        scale_factor:  Correction factor. If None, reads from config.yaml.
    """

    def __init__(
        self,
        zone_config:  dict[str, list[list[int]]],
        section_id:   str,
        weights_path: str   = "models/csrnet_v3_best.pth",
        device:       str   = "cuda" if torch.cuda.is_available() else "cpu",
        scale_factor: float | None = None,
    ) -> None:
        self.zone_config = zone_config
        self.section_id  = section_id
        self.far_poly    = zone_config.get("far", [])
        self.device      = device

        # Read scale factor from config if not provided
        if scale_factor is not None:
            self.scale_factor = scale_factor
        else:
            self.scale_factor = next(
                (getattr(c, "far_zone_scale", 1.0)
                 for c in cfg.cameras if c.section_id == section_id),
                1.0
            )

        self._model = self._load_model(weights_path)

        log.info(
            f"CSRNetFarZoneEstimator | section={section_id} "
            f"far_polygon_vertices={len(self.far_poly)} "
            f"device={device} scale_factor={self.scale_factor}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame: np.ndarray,
    ) -> tuple[int, np.ndarray | None]:
        """
        Estimate headcount in the far zone.

        Returns:
            count:       Estimated integer headcount.
            density_map: Raw density map (H×W float32) or None.
        """
        if not self.far_poly:
            log.warning(f"[{self.section_id}] Far zone polygon not configured")
            return 0, None

        crop, x1, y1, x2, y2 = get_zone_crop(frame, self.far_poly)
        if crop.size == 0:
            return 0, None

        density_map = self._run_model(crop)

        # Clamp: the output conv has no ReLU, so tiny negatives can appear
        density_map = np.maximum(density_map, 0.0)

        # ── Fix 1: Output-space polygon masking ────────────────────────────
        # get_zone_crop returns a rectangular crop — the actual far-zone
        # polygon may not fill the whole rectangle.  Corners outside the
        # polygon (empty chairs, bookshelves) contribute spurious density.
        # Zero out density pixels that map back to outside-polygon locations
        # in the original frame.  We operate on the density map (not the
        # input) so CSRNet's receptive field is never disrupted.
        density_map = self._mask_density_to_polygon(density_map, x1, y1, crop.shape[:2])

        # ── Fix 2: Density thresholding ────────────────────────────────────
        # CSRNet produces diffuse low-amplitude activations on empty chairs
        # and book-spine textures.  Suppress any density below a per-pixel
        # threshold — genuine head responses are typically much stronger.
        density_map = self._apply_density_threshold(density_map)

        # For this checkpoint sum(density_map) ≈ person count directly.
        # (The ×100 convention only applies to models trained with a 1/100-
        # normalised GT density — this one was not.)
        raw_count = float(density_map.sum())
        count     = max(0, round(raw_count * self.scale_factor))

        log.info(
            f"[{self.section_id}] CSRNet far zone | "
            f"raw={raw_count:.2f} scale={self.scale_factor} count={count}"
        )

        return count, density_map


    def estimate_with_visualization(
        self,
        frame: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Returns annotated frame with density heatmap overlay."""
        count, density_map = self.estimate(frame)
        vis = frame.copy()

        if self.far_poly and density_map is not None:
            poly    = np.array(self.far_poly, dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.polylines(vis, [poly], isClosed=True,
                          color=(0, 0, 255), thickness=2)

            crop, x1, y1, x2, y2 = get_zone_crop(frame, self.far_poly)
            if crop.size > 0:
                dm_resized = cv2.resize(
                    density_map, (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LINEAR
                )
                dm_norm = cv2.normalize(
                    dm_resized, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                heatmap = cv2.applyColorMap(dm_norm, cv2.COLORMAP_JET)
                roi     = vis[y1:y2, x1:x2]
                vis[y1:y2, x1:x2] = cv2.addWeighted(heatmap, 0.5, roi, 0.5, 0)

            cx = int(sum(p[0] for p in self.far_poly) / len(self.far_poly))
            cy = int(sum(p[1] for p in self.far_poly) / len(self.far_poly))
            cv2.putText(vis, f"FAR: {count}",
                        (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(vis, (0, 0), (380, 28), (0, 0, 0), -1)
        cv2.putText(vis,
                    f"FAR zone (CSRNet) | count={count} scale={self.scale_factor}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return count, vis

    # ── Output-space polygon masking ───────────────────────────────────────

    def _mask_density_to_polygon(
        self,
        density_map:  np.ndarray,
        x1:           int,
        y1:           int,
        crop_hw:      tuple[int, int],
    ) -> np.ndarray:
        """
        Zero out density map values that correspond to pixels OUTSIDE the
        actual far-zone polygon.

        CSRNet's VGG-16 frontend has 3 max-pool layers (stride 8 total), so
        the density map is 1/8 the size of the input crop.  We scale the
        polygon vertices accordingly before drawing the mask.

        Operating on the density map (rather than zeroing the input) avoids
        disrupting CSRNet's receptive field and spatial context, which caused
        the model to increase counts when the input was black-filled.

        Args:
            density_map: CSRNet output (dh x dw), already >= 0.
            x1:          Left offset of the crop in full-frame coordinates.
            y1:          Top  offset of the crop in full-frame coordinates.
            crop_hw:     (crop_h, crop_w) of the input crop before resizing.

        Returns:
            Density map with out-of-polygon cells zeroed.
        """
        dh, dw = density_map.shape[:2]
        ch, cw = crop_hw

        # Scale factor from crop pixels → density map cells
        sx = dw / cw
        sy = dh / ch

        # Translate polygon from full-frame coords → density-map coords
        poly_dm = np.array(
            [[(p[0] - x1) * sx, (p[1] - y1) * sy] for p in self.far_poly],
            dtype=np.int32,
        )

        mask = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_dm], 255)

        masked = density_map.copy()
        masked[mask == 0] = 0.0
        return masked

    # ── Density thresholding ───────────────────────────────────────────────

    def _apply_density_threshold(
        self,
        density_map: np.ndarray,
        threshold:   float = 0.01,
    ) -> np.ndarray:
        """
        Zero out density values below `threshold`.

        CSRNet produces a diffuse, low-amplitude response on empty chairs
        and repetitive textures (book spines, uniform backgrounds).  Real
        head responses are at least an order of magnitude stronger.  Hard-
        zeroing weak activations suppresses false chair/shelf counts while
        preserving genuine head peaks.

        Args:
            density_map: Raw density map (H×W float32), already >= 0.
            threshold:   Per-pixel density below which values are zeroed.
                         Default 0.01 is conservative — tune up if needed.

        Returns:
            Thresholded density map.
        """
        threshold = next(
            (getattr(c, "far_zone_density_threshold", 0.01)
            for c in cfg.cameras if c.section_id == self.section_id),
            0.01
        )
        return np.where(density_map >= threshold, density_map, 0.0).astype(np.float32)

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self, weights_path: str) -> CSRNet:
        model = CSRNet().to(self.device)
        wp    = Path(weights_path)

        if not wp.exists():
            log.warning(f"CSRNet weights not found at {weights_path}")
            model.eval()
            return model

        try:
            state = torch.load(wp, map_location=self.device, weights_only=False)
            # Handle various checkpoint formats
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]
            model.load_state_dict(state, strict=True)
            log.info(f"CSRNet weights loaded | {weights_path}")
        except RuntimeError as e:
            # Try strict=False as fallback
            log.warning(f"strict=True failed: {e} — trying strict=False")
            try:
                model.load_state_dict(state, strict=False)
                log.info(f"CSRNet weights loaded (strict=False) | {weights_path}")
            except Exception as e2:
                log.error(f"Could not load CSRNet weights: {e2}")

        model.eval()
        return model

    def _run_model(self, crop: np.ndarray) -> np.ndarray:
        """Run CSRNet on a BGR crop. Returns density map as float32 numpy array."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Resize to multiple of 16 (required by 3 max-pool layers in VGG-16)
        h, w   = rgb.shape[:2]
        new_h  = max(16, (h // 16) * 16)
        new_w  = max(16, (w // 16) * 16)
        if new_h != h or new_w != w:
            rgb = cv2.resize(rgb, (new_w, new_h))

        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            density = self._model(tensor)

        dm = density.squeeze().cpu().numpy().astype(np.float32)
        # Clamp at source so callers never see negative density values
        return np.maximum(dm, 0.0)