# visualize_detections.py
import json, cv2
from pathlib import Path
from detection.yolo_inference import YOLOInference

yolo  = YOLOInference()
frame = cv2.imread("data/sample_frames/hindi_section.jpg")

persons, objects = yolo.run_inference(frame, section_id="hindi_section")

vis = frame.copy()
h, w = vis.shape[:2]

for i, box in enumerate(persons):
    x1, y1, x2, y2, conf = box
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    # Green box + number label
    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.circle(vis, (cx,cy), 6, (0,255,0), -1)
    cv2.putText(vis, f"#{i+1} {conf:.2f}", (int(x1), max(0,int(y1)-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

for box in objects:
    x1, y1, x2, y2, conf, cls_id = box
    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,165,255), 2)
    cv2.putText(vis, f"obj {conf:.2f}", (int(x1), max(0,int(y1)-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)

# HUD
cv2.rectangle(vis, (0,0), (340,32), (0,0,0), -1)
cv2.putText(vis, f"persons={len(persons)}  objects={len(objects)}  conf>0.50",
            (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

out = Path("data/roi_previews/cp11_verify.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), vis)
print(f"Saved → {out}")
print(f"Persons detected: {len(persons)}")
for i, box in enumerate(persons):
    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
    print(f"  #{i+1} conf={box[4]:.3f}  centroid=({cx},{cy})")