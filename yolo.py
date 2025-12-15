from ultralytics import YOLO

# Load a model (it downloads automatically)
model = YOLO('yolov8n-pose.pt') 

# Run inference on camera source
results = model(source=0, show=True)