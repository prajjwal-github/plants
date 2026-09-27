import json
from pathlib import Path
from typing import Dict, Any

from agriguard.utils.logger import setup_logger

logger = setup_logger("model_registry")

class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._create_empty_registry()
            
    def _create_empty_registry(self):
        default = {
            "efficientnet": {"status": "missing", "path": "", "classes": []},
            "mobilevit": {"status": "missing", "path": "", "classes": []},
            "yolo": {"status": "missing", "path": "", "classes": []}
        }
        self.save(default)
        
    def load(self) -> Dict[str, Any]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def save(self, data: Dict[str, Any]):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        data = self.load()
        return data.get(model_id, {})
        
    def update_model_info(self, model_id: str, info: Dict[str, Any]):
        data = self.load()
        if model_id not in data:
            data[model_id] = {}
        data[model_id].update(info)
        self.save(data)
