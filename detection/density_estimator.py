"""
detection/density_estimator.py — DM-Count density estimator for far zone (ZCP-08)

DM-Count (Distribution Matching for Crowd Counting) estimates the number
of people in a region by predicting a density map, then summing it.

Why density estimation for the far zone:
  - People in the far zone are too small for YOLO to detect reliably
    (often only a few pixels tall)
  - Density estimators are trained to count from texture/pattern cues
    rather than individual bounding boxes
  - DM-Count achieves state-of-the-art accuracy on standard benchmarks

Model: Pretrained on ShanghaiTech Part B (indoor/outdoor scenes)
Weights: models/dm_count_shb.pth

Usage:
    from detection.density_estimator import FarZoneEstimator
    estimator = FarZoneEstimator(zone_config["cad_lab"])
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

log = get_logger("system")

# ── DM-Count model architecture ───────────────────────────────────────────────

class DMCountModel(nn.Module):
    """
    DM-Count density estimation network.
    VGG-19 backbone + regression head → density map.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as models

        # Use torchvision VGG-19 — more stable feature extraction than timm
        vgg = models.vgg19(weights=None)

        # Frontend: first 28 layers of VGG-19 features (up to pool4, 512ch output)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:28])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return torch.abs(x)

# ── Transform ─────────────────────────────────────────────────────────────────

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


# ── Far zone estimator ────────────────────────────────────────────────────────

class FarZoneEstimator:
    """
    Estimates headcount in the FAR zone using DM-Count.

    The far zone crop is passed through DM-Count which produces a
    density map. Summing the density map gives the estimated count.

    A per-section scale_factor corrects for perspective distortion —
    cameras looking at different angles produce different density scales.
    Calibrated in ZCP-10.

    Args:
        zone_config:   Zone polygons {"near":..., "mid":..., "far":...}
        section_id:    Section identifier.
        weights_path:  Path to DM-Count .pth weights.
        device:        'cuda' or 'cpu'.
        scale_factor:  Multiplier applied to raw DM-Count output.
                       Default 1.0 — calibrate per section in ZCP-10.
    """

    def __init__(
        self,
        zone_config:  dict[str, list[list[int]]],
        section_id:   str,
        weights_path: str   = "models/dm_count_shb.pth",
        device:       str   = "cuda" if torch.cuda.is_available() else "cpu",
        scale_factor: float | None = None,   # if None, reads from config.yaml
    ) -> None:
        self.zone_config = zone_config
        self.section_id  = section_id
        self.far_poly    = zone_config.get("far", [])
        self.device      = device

        # Read scale factor from config if not provided explicitly
        if scale_factor is not None:
            self.scale_factor = scale_factor
        else:
            from config.config_loader import cfg
            self.scale_factor = next(
                (getattr(c, "far_zone_scale", 1.0)
                for c in cfg.cameras if c.section_id == section_id),
                1.0
            )

        self._model = self._load_model(weights_path)

        log.info(
            f"FarZoneEstimator | section={section_id} "
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

        Args:
            frame: Full BGR frame.

        Returns:
            count:       Estimated integer headcount in far zone.
            density_map: Raw density map (H×W float32) for visualization.
                         None if far zone is empty/invalid.
        """
        if not self.far_poly:
            log.warning(f"[{self.section_id}] Far zone polygon not configured")
            return 0, None

        # Crop to far zone bounding box
        crop, x1, y1, x2, y2 = get_zone_crop(frame, self.far_poly)
        if crop.size == 0:
            return 0, None

        # Convert BGR → RGB, run through model
        density_map = self._run_model(crop)

        # Raw count = sum of density map
        raw_count   = float(density_map.sum())
        count       = max(0, round(raw_count * self.scale_factor))

        log.info(
            f"[{self.section_id}] Far zone | "
            f"raw_count={raw_count:.2f} "
            f"scale={self.scale_factor} "
            f"count={count}"
        )

        return count, density_map

    def estimate_with_visualization(
        self,
        frame: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Same as estimate() but returns an annotated full frame showing:
        - Far zone polygon overlay
        - Density map heatmap overlaid on the far zone crop
        - Estimated count in HUD
        """
        count, density_map = self.estimate(frame)
        vis = frame.copy()

        if self.far_poly and density_map is not None:
            # Draw far zone polygon
            poly    = np.array(self.far_poly, dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.polylines(vis, [poly], isClosed=True,
                          color=(0, 0, 255), thickness=2)

            # Overlay density map as heatmap on the far zone crop region
            crop, x1, y1, x2, y2 = get_zone_crop(frame, self.far_poly)
            if crop.size > 0 and density_map is not None:
                # Resize density map to crop size
                dm_resized = cv2.resize(
                    density_map,
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LINEAR
                )
                # Normalize and apply colormap
                dm_norm = cv2.normalize(
                    dm_resized, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                heatmap = cv2.applyColorMap(dm_norm, cv2.COLORMAP_JET)
                # Blend heatmap onto vis
                roi = vis[y1:y2, x1:x2]
                blended = cv2.addWeighted(heatmap, 0.5, roi, 0.5, 0)
                vis[y1:y2, x1:x2] = blended

            # Label
            cx = int(sum(p[0] for p in self.far_poly) / len(self.far_poly))
            cy = int(sum(p[1] for p in self.far_poly) / len(self.far_poly))
            cv2.putText(vis, f"FAR: {count}",
                        (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # HUD
        cv2.rectangle(vis, (0, 0), (320, 28), (0, 0, 0), -1)
        cv2.putText(vis,
                    f"FAR zone | estimated={count} scale={self.scale_factor}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        return count, vis

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str) -> DMCountModel:
        model = DMCountModel().to(self.device)
        wp    = Path(weights_path)

        if not wp.exists():
            log.warning(
                f"DM-Count weights not found at {weights_path}. "
                f"Model will run with random weights — calibrate after downloading."
            )
            model.eval()
            return model

        try:
            state = torch.load(wp, map_location=self.device, weights_only=False)
            # Handle various checkpoint formats
            if isinstance(state, dict):
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            log.info(f"DM-Count weights loaded from {weights_path}")
        except Exception as e:
            log.warning(f"Could not load DM-Count weights: {e}")

        model.eval()
        return model

    def _run_model(self, crop: np.ndarray) -> np.ndarray:
        """Run DM-Count on a BGR crop. Returns density map as numpy array."""
        # BGR → RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Resize to multiple of 8 (required by VGG architecture)
        h, w  = rgb.shape[:2]
        new_h = max(8, (h // 8) * 8)
        new_w = max(8, (w // 8) * 8)
        if new_h != h or new_w != w:
            rgb = cv2.resize(rgb, (new_w, new_h))

        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            density = self._model(tensor)

        return density.squeeze().cpu().numpy()