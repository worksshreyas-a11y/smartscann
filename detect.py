# detect.py
from ultralytics import YOLO
import os
import cv2
import numpy as np

# Available model options
AVAILABLE_MODELS = {
    "COCO (80 classes)": "yolov8s.pt",
    "Open Images (600+ classes)": "yolov8x-openimages.pt",
    "World Model (900+ classes)": "yolov8x-world.pt"
}

def load_model(model_key="COCO (80 classes)"):
    """Load a YOLO model based on selected dataset."""
    model_path = AVAILABLE_MODELS.get(model_key, "yolov8s.pt")
    print(f"🔍 Loading YOLO model: {model_key} -> {model_path}")
    return YOLO(model_path)

def detect_objects(image_input, model_key="COCO (80 classes)", save_dir="results", conf_threshold=0.4):
    """
    Detect objects in an image using YOLOv8 and return detailed results.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "detected_image.jpg")

    # Load selected model
    model = load_model(model_key)

    # Handle both file paths and numpy arrays
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input

    # Run YOLO inference
    results = model(img)

    # Filter detections
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        if conf >= conf_threshold:
            obj_name = model.names.get(cls, f"Unknown({cls})")
            detections.append({"object": obj_name, "confidence": round(conf, 2)})

    # Save annotated image
    annotated = results[0].plot()
    cv2.imwrite(save_path, annotated)

    # Unique object list
    unique_objects = list({d["object"] for d in detections})

    return save_path, detections, unique_objects


# Example usage
if __name__ == "__main__":
    image_path = "sample.jpg"
    output_path, detections, objects = detect_objects(
        image_input=image_path,
        model_key="Open Images (600+ classes)",
        conf_threshold=0.45
    )
    print("✅ Annotated image saved at:", output_path)
    print("Detected objects:", objects)
    print("Detailed detections:", detections)
