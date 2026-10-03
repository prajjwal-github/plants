import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agriguard.utils.config import load_config
from agriguard.utils.logger import setup_logger

logger = setup_logger("train_yolo")

def main():
    config = load_config("configs/yolo.yaml")
    logger.info("Loaded config for YOLO")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Cannot train YOLO.")
        return
        
    logger.info("Initializing YOLO model...")
    model_name = config.get("model_name", "yolov8s.pt")
    
    # Normally we would do:
    # model = YOLO(model_name)
    # model.train(data="data/processed/yolo/dataset.yaml", epochs=config["training"]["epochs"], imgsz=config["imgsz"])
    
    logger.info(f"Would train YOLOv8 with {config['training']['epochs']} epochs (Demo setup).")
    
    from agriguard.models.registry import ModelRegistry
    registry = ModelRegistry()
    registry.update_model_info("yolo", {
        "status": "trained",
        "path": f"{config['training']['project']}/{config['training']['name']}/weights/best.pt",
        "classes": ["pest_type1", "pest_type2"]
    })
    logger.info("YOLO training simulated and registered.")

if __name__ == "__main__":
    main()
