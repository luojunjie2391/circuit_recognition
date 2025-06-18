import cv2
from ultralytics import YOLO


def elements_recognition(img):
    model = YOLO('best_model/best.pt')
    original = img
    img = cv2.resize(original, (1000, int(original.shape[0] * 1000 / original.shape[1])))
    results = model(img)[0]
    components = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        components.append({
            "label": label,
            "bbox": [x1, y1, x2, y2]
        })

    return components



