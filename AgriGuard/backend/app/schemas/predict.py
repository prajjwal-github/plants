from pydantic import BaseModel
from typing import List, Optional, Any

class PestDetection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]

class AnalysisResponse(BaseModel):
    crop: str
    disease: str
    disease_confidence: float
    pests: List[PestDetection]
    severity: str
    affected_area_percentage: float
    recommendation: str
    explainability_image: Optional[str]
    processing_time_ms: float
