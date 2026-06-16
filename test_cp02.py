"""CP-02 verification test."""

from config.config_loader import cfg
from config.constants import SeatStatus, CocoClass, RESERVATION_CLASSES

def test_config():
    # Config loading
    assert len(cfg.cameras) > 0, "No cameras defined"
    for cam in cfg.cameras:
        assert hasattr(cam, "camera_id")
        assert hasattr(cam, "section_id")
        assert hasattr(cam, "stream_url")
    assert cfg.classifier.person_iou_threshold > 0
    assert cfg.classifier.object_iou_threshold > 0
    assert cfg.smoothing.window_size > 0
    assert cfg.redis.port == 6379
    assert cfg.api.port == 8000
    print(" Config loaded successfully")
    print(f"   Cameras defined: {len(cfg.cameras)}")
    for cam in cfg.cameras:
        print(f"   - {cam.camera_id} ({cam.section_id}): {cam.stream_url}")

def test_constants():
    # SeatStatus
    assert SeatStatus.FREE == "free"
    assert SeatStatus.OCCUPIED == "occupied"
    assert SeatStatus.RESERVED == "reserved"
    # CocoClass
    assert CocoClass.PERSON == 0
    assert CocoClass.LAPTOP == 63
    # Reservation classes
    assert CocoClass.PERSON not in RESERVATION_CLASSES
    assert CocoClass.LAPTOP in RESERVATION_CLASSES
    assert CocoClass.BACKPACK in RESERVATION_CLASSES
    print(" Constants verified")
    print(f"   Reservation class IDs: {sorted(RESERVATION_CLASSES)}")

if __name__ == "__main__":
    test_config()
    test_constants()
    print("\n CP-02 PASSED — config & constants ready")   