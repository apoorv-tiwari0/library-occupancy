# debug_conf.py
import json, cv2
from pathlib import Path
from detection.yolo_inference import YOLOInference

yolo     = YOLOInference()
frame    = cv2.imread("data/sample_frames/reference_2.jpg")
roi_data = json.loads(Path(r"C:\IITD_Internship\library-occupancy\data\roi\roi_polygons.json").read_text())

# Run WITHOUT ROI filter to see all detections + their confidences
persons, objects = yolo.run_inference(frame, section_id="reference_2", roi_data=None)

print(f"Total detections: {len(persons)}")
print("\nAll person confidences (sorted high→low):")
for i, box in enumerate(sorted(persons, key=lambda b: b[4], reverse=True)):
    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
    print(f"  [{i+1}] conf={box[4]:.3f}  centroid=({cx},{cy})")