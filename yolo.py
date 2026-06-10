from ultralytics import YOLO

# Load a model (it downloads automatically)
model = YOLO('yolo11n-pose.pt') 

# Export to ONNX format with input size 320
model.export(format='ncnn', imgsz=320)