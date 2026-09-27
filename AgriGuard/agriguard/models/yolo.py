from typing import Any, Dict

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class YoloDetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        if YOLO is None:
            raise ImportError("ultralytics library is required. Run 'pip install ultralytics'")
        self.model = YOLO(model_path)
        
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Returns detection results."""
        results = self.model(image_path)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": conf,
                    "bbox": xyxy
                })
                
        return {"detections": detections}
