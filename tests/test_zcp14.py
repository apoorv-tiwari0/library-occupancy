"""
tests/test_zcp14.py -- Head-to-head: Zone pipeline vs SAHI-only (ZCP-14)

Runs both SectionPipeline (raw SAHI) and ZoneSectionPipeline (zone fusion)
on the held-out test frame (-1) for all 11 sections and compares against
manually counted ground truth.

HOW TO FILL IN GT
-----------------
Open each  data/test_frames/<section>-1.*  image and count every visible
person in the FULL FRAME (near + mid + far zones combined).  Then set the
corresponding value in TOTAL_GT below.  Sections with None are skipped.

Pass criterion:
    Zone pipeline achieves lower (or equal) MAE than SAHI-only
    AND wins >= 50% of sections (requires >= 6 sections with GT).

Run from project root:
    python tests/test_zcp14.py
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.pipeline       import SectionPipeline, ZoneSectionPipeline
from detection.yolo_inference import YOLOInference
from ingestion.preprocessor   import Preprocessor

# -- Paths --------------------------------------------------------------------

ZONE_PATH = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
TEST_DIR  = Path("data/test_frames")

# -- Manual total GT ----------------------------------------------------------
# Count every person visible in the full frame of <section>-1.*.
# near + mid + far combined.  None = not counted yet (section is skipped).
#
# weeding_out_area = 1  <- confirmed by user (ZCP-12 showed 3 = double-count bug)
# Fill the rest in after manual counting.

TOTAL_GT: dict[str, int | None] = {
    "cad_lab":              6,   # <- fill in after counting
    "focused_reading_area": 9,
    "g_hall_2":             46,
    "g_huss":               68,
    "hindi_section":        30,
    "ip_camera_19":         7,
    "ipc":                  10,
    "main_computer_room":   7,
    "reference_2":          10,
    "reference_area":       9,
    "weeding_out_area":     1,      # confirmed
}

SECTIONS = list(TOTAL_GT.keys())


# -- Helpers ------------------------------------------------------------------

def _find_frame(section_id: str) -> Path | None:
    """Return the -1 test frame path for a section (any supported extension)."""
    for ext in [".png", ".jpg", ".PNG", ".JPG"]:
        p = TEST_DIR / f"{section_id}-1{ext}"
        if p.exists():
            return p
    return None


def _signed_pct(est: int, gt: int) -> str:
    """Signed percentage error string, e.g. '+20.0%'."""
    if gt == 0:
        return "0.0%" if est == 0 else "+inf"
    return f"{(est - gt) / gt * 100:+.1f}%"


# -- Main ---------------------------------------------------------------------

def main() -> int:
    print("=" * 80)
    print("ZCP-14 -- Zone Pipeline vs SAHI-Only Head-to-Head Benchmark")
    print("=" * 80)

    if not ZONE_PATH.exists():
        print(f"  [ERROR] Zone config not found: {ZONE_PATH}")
        return 1

    zone_data = json.loads(ZONE_PATH.read_text())

    print("\nStep 1 -- Loading YOLO model (shared across both pipelines)...")
    yolo         = YOLOInference()
    preprocessor = Preprocessor()

    hdr = (
        f"  {'Section':<28} {'GT':>4}"
        f" {'SAHI':>6} {'Zone':>6}"
        f" {'SAHI_err':>9} {'Zone_err':>9}"
        f" {'Winner':>8}"
    )
    sep = f"  {'-' * 74}"
    import logging
    logging.getLogger("system").setLevel(logging.WARNING)

    print(f"\nStep 2 -- Per-section comparison (frame -1 per section):\n")
    print(hdr)
    print(sep)

    rows: list[dict]  = []
    skipped_no_gt:  list[str] = []
    skipped_no_img: list[str] = []

    for sid in SECTIONS:
        gt       = TOTAL_GT.get(sid)
        img_path = _find_frame(sid)

        if img_path is None:
            skipped_no_img.append(sid)
            print(f"  {sid:<28} {'NO IMAGE':>52}")
            continue

        if gt is None:
            skipped_no_gt.append(sid)
            print(f"  {sid:<28} {'NO GT -- fill TOTAL_GT':>52}")
            continue

        if sid not in zone_data:
            print(f"  {sid:<28} {'NO ZONE CONFIG':>52}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  {sid:<28} {'UNREADABLE':>52}")
            continue

        # SAHI-only -----------------------------------------------------------
        try:
            sahi_pl    = SectionPipeline(
                section_id   = sid,
                yolo         = yolo,
                preprocessor = preprocessor,
            )
            sahi_count = sahi_pl.run(frame).headcount
        except Exception as exc:
            print(f"  {sid:<28}  SAHI ERROR: {exc}")
            continue

        # Zone pipeline -------------------------------------------------------
        try:
            zone_pl = ZoneSectionPipeline(
                section_id   = sid,
                yolo         = yolo,
                zone_config  = zone_data[sid],
                preprocessor = preprocessor,
            )
            zone_count = zone_pl.run(frame, frame_id=1).headcount
        except Exception as exc:
            print(f"  {sid:<28}  ZONE ERROR: {exc}")
            continue

        sahi_abs = abs(sahi_count - gt)
        zone_abs = abs(zone_count - gt)

        if zone_abs < sahi_abs:
            winner = "Zone"
        elif sahi_abs < zone_abs:
            winner = "SAHI"
        else:
            winner = "Tie"

        print(
            f"  {sid:<28} {gt:>4}"
            f" {sahi_count:>6} {zone_count:>6}"
            f" {_signed_pct(sahi_count, gt):>9} {_signed_pct(zone_count, gt):>9}"
            f" {winner:>8}"
        )

        rows.append(dict(
            section  = sid,
            gt       = gt,
            sahi     = sahi_count,
            zone     = zone_count,
            sahi_abs = sahi_abs,
            zone_abs = zone_abs,
            winner   = winner,
        ))

    # -- Summary --------------------------------------------------------------
    print(sep)

    if not rows:
        print("\n  [!] No sections evaluated.")
        print("      Fill in TOTAL_GT at the top of this file and re-run.")
        return 0

    n         = len(rows)
    sahi_mae  = sum(r["sahi_abs"] for r in rows) / n
    zone_mae  = sum(r["zone_abs"] for r in rows) / n
    nonzero   = [r for r in rows if r["gt"] > 0]
    sahi_mape = (
        sum(r["sahi_abs"] / r["gt"] * 100 for r in nonzero) / len(nonzero)
        if nonzero else float("nan")
    )
    zone_mape = (
        sum(r["zone_abs"] / r["gt"] * 100 for r in nonzero) / len(nonzero)
        if nonzero else float("nan")
    )
    zone_wins = sum(1 for r in rows if r["winner"] == "Zone")
    sahi_wins = sum(1 for r in rows if r["winner"] == "SAHI")
    ties      = sum(1 for r in rows if r["winner"] == "Tie")

    print(
        f"\n  Sections evaluated : {n}"
        f"  (skipped -- no GT: {len(skipped_no_gt)}, no image: {len(skipped_no_img)})"
    )
    print(f"\n  {'Metric':<24} {'SAHI-only':>12} {'Zone':>12}  {'Better':>6}")
    print(f"  {'-' * 58}")
    better_mae  = "Zone" if zone_mae  <= sahi_mae  else "SAHI"
    better_mape = "Zone" if zone_mape <= sahi_mape else "SAHI"
    print(f"  {'MAE (persons)':<24} {sahi_mae:>12.2f} {zone_mae:>12.2f}  {better_mae:>6}")
    print(f"  {'MAPE (%)':<24} {sahi_mape:>11.1f}% {zone_mape:>11.1f}%  {better_mape:>6}")
    print(f"  {'Section wins':<24} {sahi_wins:>12} {zone_wins:>12}  {ties} tie(s)")

    if skipped_no_gt:
        print(f"\n  [REMIND] Sections still needing manual GT count: {skipped_no_gt}")
        print(  "           Open data/test_frames/<section>-1.* and count all visible people,")
        print(  "           then update TOTAL_GT at the top of this file.")

    print()
    if n < 6:
        print(f"  [SKIP] ZCP-14 -- Only {n} section(s) with GT. Add more GT and re-run.")
        return 0
    elif zone_mae <= sahi_mae and zone_wins >= n * 0.5:
        print(f"  [PASS] ZCP-14 -- Zone pipeline outperforms SAHI-only")
        print(f"         MAE: Zone={zone_mae:.2f} vs SAHI={sahi_mae:.2f}"
              f"  |  Wins: Zone={zone_wins}, SAHI={sahi_wins}")
        return 0
    else:
        print(f"  [FAIL] ZCP-14 -- Zone pipeline does NOT outperform SAHI-only")
        print(f"         MAE: Zone={zone_mae:.2f} vs SAHI={sahi_mae:.2f}")
        print(  "         Review zone_config polygon boundaries and far_zone_scale values.")
        return 1


if __name__ == "__main__":
    code = main()
    print("\n" + "=" * 80)
    print("ZCP-14 complete")
    print("=" * 80)
    sys.exit(code)
