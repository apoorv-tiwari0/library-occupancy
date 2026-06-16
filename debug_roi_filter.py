# debug_roi_filter.py — paste and run from project root
import json
import cv2
import numpy as np
from pathlib import Path
from detection.yolo_inference import YOLOInference

ROI_PATH = Path(r"C:\IITD_Internship\library-occupancy\data\roi\roi_polygons.json")
IMG_PATH = Path("data/sample_frames/cad_lab.jpg")
SECTION  = "cad_lab"

roi_data = json.loads(ROI_PATH.read_text())
seats    = roi_data[SECTION]
frame    = cv2.imread(str(IMG_PATH))
h, w     = frame.shape[:2]

# Print the envelope
all_pts = np.array([pt for pts in seats.values() for pt in pts], dtype=np.float32)
env_x1, env_y1 = all_pts[:,0].min(), all_pts[:,1].min()
env_x2, env_y2 = all_pts[:,0].max(), all_pts[:,1].max()
pad_x = (env_x2-env_x1)*0.15
pad_y = (env_y2-env_y1)*0.15
print(f"Frame size: {w}x{h}")
print(f"ROI envelope (before pad): x={env_x1:.0f}-{env_x2:.0f}  y={env_y1:.0f}-{env_y2:.0f}")
print(f"ROI envelope (after 15% pad): x={env_x1-pad_x:.0f}-{env_x2+pad_x:.0f}  y={env_y1-pad_y:.0f}-{env_y2+pad_y:.0f}")
print(f"Seats annotated: {list(seats.keys())}")

# Draw envelope on frame and save
debug = frame.copy()
cv2.rectangle(debug,
    (int(env_x1-pad_x), int(env_y1-pad_y)),
    (int(env_x2+pad_x), int(env_y2+pad_y)),
    (0, 255, 255), 3)
for seat_id, pts in seats.items():
    poly = np.array(pts, dtype=np.int32)
    cv2.polylines(debug, [poly], True, (0, 255, 0), 2)
    cx = int(sum(p[0] for p in pts)/len(pts))
    cy = int(sum(p[1] for p in pts)/len(pts))
    cv2.putText(debug, seat_id, (cx-20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2.imwrite("data/roi_previews/debug_envelope.jpg", debug)
print("Saved → data/roi_previews/debug_envelope.jpg")