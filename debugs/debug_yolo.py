import cv2, sys
from pathlib import Path
sys.path.insert(0, '.')
from ultralytics import YOLO

model  = YOLO('models/yolov10x.pt')
frames = list(Path('data/sample_frames').glob('*.jpg'))

for img_path in frames:
    frame   = cv2.imread(str(img_path))
    results = model(frame, conf=0.25, verbose=False)
    persons = []
    others  = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            name   = result.names[cls_id]
            if name == 'person':
                persons.append(conf)
            else:
                others.append((name, conf))
    print(f"\n{img_path.name}:")
    print(f"  persons ({len(persons)}): {[round(c,2) for c in persons]}")
    print(f"  others  ({len(others)}):  {others}")