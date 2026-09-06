from ultralytics import YOLO
import cv2

# Load image
image = cv2.imread("images/test.jpg")

# Load YOLO
model = YOLO("yolov8n.pt")

# Run detection
results = model(image)

result = results[0]

# Draw every detection
for box in result.boxes:

    class_id = int(box.cls[0])
    class_name = result.names[class_id]

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Convert coordinates to integers
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    # Draw rectangle
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Create label
    label = f"{class_name} {confidence:.2f}"

    # Draw label
    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Save result
cv2.imwrite("output/yolo_opencv.jpg", image)

print("Saved output/yolo_opencv.jpg")