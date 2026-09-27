import cv2
import numpy as np
from typing import Dict, Any

from agriguard.utils.logger import setup_logger

logger = setup_logger("severity_calculator")

class SeverityCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get("thresholds", {
            "low": {"min": 0, "max": 10},
            "medium": {"min": 10, "max": 30},
            "high": {"min": 30, "max": 100}
        })
        
    def calculate(self, image_path: str) -> Dict[str, Any]:
        """Calculates severity percentage based on classical computer vision (Otsu thresholding)."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image at {image_path}")
            return {"error": "Invalid image"}
            
        # Convert to HSV to extract the leaf (green)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Approximate mask for general leaf (Greenish range)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Approximate mask for lesions (Brown/Yellowish range)
        lower_brown = np.array([10, 40, 40])
        upper_brown = np.array([24, 255, 255])
        lesion_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Calculate pixel areas
        leaf_area = cv2.countNonZero(leaf_mask)
        lesion_area = cv2.countNonZero(lesion_mask)
        
        total_relevant_area = leaf_area + lesion_area
        
        if total_relevant_area == 0:
            percentage = 0.0
        else:
            percentage = (lesion_area / total_relevant_area) * 100.0
            
        severity_label = "Low"
        for label, bounds in self.thresholds.items():
            if bounds["min"] <= percentage < bounds["max"]:
                severity_label = label.capitalize()
                break
                
        # If percentage is exactly 100
        if percentage >= 100:
            severity_label = "High"
            
        return {
            "affected_area_percentage": round(percentage, 2),
            "severity": severity_label,
            "leaf_pixels": leaf_area,
            "lesion_pixels": lesion_area
        }
