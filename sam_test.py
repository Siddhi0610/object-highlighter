from ultralytics import YOLO, SAM

# Load models
yolo = YOLO("yolov8n.pt")
sam = SAM("sam2_b.pt")

# YOLO detection
results = yolo("images/test.jpg")

result = results[0]

for box in result.boxes:

    class_id = int(box.cls[0])
    class_name = result.names[class_id]

    confidence = float(box.conf[0])

    # YOLO bounding box
    coordinates = box.xyxy[0].tolist()

    print(f"Segmenting {class_name}...")

    # Give YOLO's bounding box to SAM
    sam_results = sam(
        "images/test.jpg",
        bboxes=[coordinates]
    )

    # Save SAM result
    sam_results[0].save(
        f"output/{class_name.replace(' ', '_')}.jpg"
    )

print("Done!")
