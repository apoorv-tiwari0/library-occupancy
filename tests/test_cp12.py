"""
test_cp12.py — Validates seat occupancy classifier (CP-12)

Tests:
  1. Classifier initialises from config
  2. FREE — no person, no object near seat
  3. OCCUPIED — person centroid inside seat polygon
  4. OCCUPIED — person bbox overlaps seat bbox (IoU check)
  5. RESERVED — no person but object inside seat polygon
  6. RESERVED does not fire when person also present (person wins)
  7. Multiple seats classified correctly in one call
  8. Headcount matches number of person_boxes passed in
  9. classify_with_capacity — vacancy calculation
  10. Real frame — cad_lab.jpg end-to-end with YOLO + classifier

Run from project root:
    python test_cp12.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detection.classifier import SeatClassifier, SeatState, _box_iou
from detection.yolo_inference import YOLOInference


ROI_PATH = Path(r"C:\IITD_Internship\library-occupancy\data\roi\roi_polygons.json")


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_seat(x1=100, y1=100, x2=300, y2=300) -> list[list[int]]:
    """Rectangle seat polygon."""
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


def person_inside(seat_pts) -> list[float]:
    """Person box centred inside the seat polygon."""
    xs = [p[0] for p in seat_pts]; ys = [p[1] for p in seat_pts]
    cx = int(sum(xs)/len(xs)); cy = int(sum(ys)/len(ys))
    return [cx-30, cy-60, cx+30, cy+10, 0.92]


def person_outside(seat_pts) -> list[float]:
    """Person box far away from the seat polygon."""
    xs = [p[0] for p in seat_pts]
    return [max(xs)+200, 100, max(xs)+280, 300, 0.88]


def object_inside(seat_pts) -> list[float]:
    """Laptop box centred inside the seat polygon (class_id=63=laptop)."""
    xs = [p[0] for p in seat_pts]; ys = [p[1] for p in seat_pts]
    cx = int(sum(xs)/len(xs)); cy = int(sum(ys)/len(ys))
    return [cx-20, cy-20, cx+20, cy+20, 0.81, 63.0]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_init():
    clf = SeatClassifier()
    assert clf.iou_threshold > 0
    print(f"  ✅ SeatClassifier init | iou_threshold={clf.iou_threshold}")
    return clf


def test_free(clf):
    seat = make_seat()
    states, hc = clf.classify(
        seat_polygons = {"seat_1": seat},
        person_boxes  = [],
        object_boxes  = [],
    )
    assert states["seat_1"] == SeatState.FREE
    assert hc == 0
    print("  ✅ FREE — no person, no object")


def test_occupied_centroid(clf):
    seat = make_seat()
    states, hc = clf.classify(
        seat_polygons = {"seat_1": seat},
        person_boxes  = [person_inside(seat)],
        object_boxes  = [],
    )
    assert states["seat_1"] == SeatState.OCCUPIED
    assert hc == 1
    print("  ✅ OCCUPIED — person centroid inside polygon")


def test_occupied_iou(clf):
    """Person box heavily overlaps seat bbox even if centroid is slightly outside."""
    seat = make_seat(100, 100, 300, 300)
    # Person box overlaps top half of seat significantly
    person = [80.0, 80.0, 280.0, 220.0, 0.88]
    states, hc = clf.classify(
        seat_polygons = {"seat_1": seat},
        person_boxes  = [person],
        object_boxes  = [],
    )
    assert states["seat_1"] == SeatState.OCCUPIED
    print("  ✅ OCCUPIED — person bbox IoU overlap")


def test_reserved(clf):
    seat = make_seat()
    states, hc = clf.classify(
        seat_polygons = {"seat_1": seat},
        person_boxes  = [],
        object_boxes  = [object_inside(seat)],
    )
    assert states["seat_1"] == SeatState.RESERVED
    assert hc == 0
    print("  ✅ RESERVED — object inside polygon, no person")


def test_person_beats_object(clf):
    """When both person and object are present, seat must be OCCUPIED not RESERVED."""
    seat = make_seat()
    states, hc = clf.classify(
        seat_polygons = {"seat_1": seat},
        person_boxes  = [person_inside(seat)],
        object_boxes  = [object_inside(seat)],
    )
    assert states["seat_1"] == SeatState.OCCUPIED
    print("  ✅ OCCUPIED beats RESERVED when person present")


def test_outside_person_is_free(clf):
    seat = make_seat()
    states, _ = clf.classify(
        seat_polygons = {"seat_1": seat},
        person_boxes  = [person_outside(seat)],
        object_boxes  = [],
    )
    assert states["seat_1"] == SeatState.FREE
    print("  ✅ FREE — person outside polygon boundary")


def test_multiple_seats(clf):
    seats = {
        "seat_occ":  make_seat(0,   0,   200, 200),
        "seat_res":  make_seat(300, 0,   500, 200),
        "seat_free": make_seat(600, 0,   800, 200),
    }
    person  = person_inside(seats["seat_occ"])
    obj     = object_inside(seats["seat_res"])

    states, hc = clf.classify(
        seat_polygons = seats,
        person_boxes  = [person],
        object_boxes  = [obj],
    )
    assert states["seat_occ"]  == SeatState.OCCUPIED
    assert states["seat_res"]  == SeatState.RESERVED
    assert states["seat_free"] == SeatState.FREE
    assert hc == 1
    print("  ✅ Multiple seats: occupied + reserved + free all correct")


def test_headcount(clf):
    seat = make_seat()
    _, hc = clf.classify(
        seat_polygons = {"s1": seat, "s2": seat},
        person_boxes  = [person_inside(seat)] * 5,
        object_boxes  = [],
    )
    assert hc == 5, f"Expected headcount=5, got {hc}"
    print(f"  ✅ Headcount: 5 persons detected correctly")


def test_vacancy(clf):
    seat = make_seat()
    result = clf.classify_with_capacity(
        seat_polygons = {"s1": seat},
        person_boxes  = [person_inside(seat)] * 3,
        object_boxes  = [],
        max_capacity  = 10,
    )
    assert result["headcount"]   == 3
    assert result["max_capacity"] == 10
    assert result["vacancy"]     == 7
    print(f"  ✅ Vacancy: headcount=3 max=10 → vacancy=7")


def test_vacancy_cannot_go_negative(clf):
    seat = make_seat()
    result = clf.classify_with_capacity(
        seat_polygons = {"s1": seat},
        person_boxes  = [person_inside(seat)] * 15,
        object_boxes  = [],
        max_capacity  = 10,
    )
    assert result["vacancy"] == 0, \
        f"Vacancy should be 0 when overcrowded, got {result['vacancy']}"
    print(f"  ✅ Vacancy clamps to 0 when headcount > max_capacity")


def test_real_frame(clf):
    """End-to-end: YOLO detection → classifier → draw states on real frame."""
    img_path = Path("data/sample_frames/reference_2.jpg")
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return
    if not ROI_PATH.exists():
        print(f"  ⚠️  Skipped — ROI file not found at {ROI_PATH}")
        return

    roi_data     = json.loads(ROI_PATH.read_text())
    seat_polys   = roi_data.get("reference_2", {})
    max_capacity = next(
        (c.max_capacity for c in __import__("config.config_loader",
        fromlist=["cfg"]).cfg.cameras if c.section_id == "reference_2"), None
    )

    frame = cv2.imread(str(img_path))

    # Run YOLO
    yolo = YOLOInference()
    persons, objects = yolo.run_inference(frame, section_id="reference_2")

    # Run classifier
    result = clf.classify_with_capacity(
        seat_polygons = seat_polys,
        person_boxes  = persons,
        object_boxes  = objects,
        max_capacity  = max_capacity,
    )

    # Draw and save
    vis = clf.draw_states(
        frame, seat_polys, result["seat_states"],
        result["headcount"], max_capacity
    )
    out = Path("data/roi_previews/cp12_classified.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)

    print(f"  ✅ Real frame — reference_2 end-to-end:")
    print(f"     Seats total   : {result['counts']['total']}")
    print(f"     Occupied      : {result['counts']['occupied']}")
    print(f"     Reserved      : {result['counts']['reserved']}")
    print(f"     Free          : {result['counts']['free']}")
    print(f"     Headcount     : {result['headcount']}")
    print(f"     Max capacity  : {result['max_capacity']}")
    print(f"     Vacancy       : {result['vacancy']}")
    print(f"     → cp12_classified.jpg saved")

    assert result["counts"]["total"] == len(seat_polys)
    assert result["headcount"] >= 0


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CP-12 — Seat Occupancy Classifier")
    print("=" * 55)

    print("\n[1] Classifier init:")
    clf = test_init()

    print("\n[2] FREE state:")
    test_free(clf)

    print("\n[3] OCCUPIED — centroid check:")
    test_occupied_centroid(clf)

    print("\n[4] OCCUPIED — IoU check:")
    test_occupied_iou(clf)

    print("\n[5] RESERVED state:")
    test_reserved(clf)

    print("\n[6] Person beats object:")
    test_person_beats_object(clf)

    print("\n[7] Outside person → FREE:")
    test_outside_person_is_free(clf)

    print("\n[8] Multiple seats:")
    test_multiple_seats(clf)

    print("\n[9] Headcount:")
    test_headcount(clf)

    print("\n[10] Vacancy calculation:")
    test_vacancy(clf)

    print("\n[11] Vacancy clamps to zero:")
    test_vacancy_cannot_go_negative(clf)

    print("\n[12] Real frame end-to-end:")
    test_real_frame(clf)

    print("\n" + "=" * 55)
    print("🎉 CP-12 PASSED — seat classifier ready")
    print("=" * 55)