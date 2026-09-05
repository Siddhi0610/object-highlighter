from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Run detection
results = model("images/test.jpg")

# Get the first result
result = results[0]

# Create image with bounding boxes
annotated_image = result.plot()

# Save the result
cv2.imwrite("output/yolo_result.jpg", annotated_image)

print("Detection complete!")
print("Saved to output/yolo_result.jpg")