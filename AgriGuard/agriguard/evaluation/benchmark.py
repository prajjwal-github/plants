import time
import os
import psutil
from typing import Dict, Any

try:
    import torch
except ImportError:
    torch = None

from agriguard.utils.logger import setup_logger

logger = setup_logger("benchmarker")

class ModelBenchmarker:
    def __init__(self, model, input_shape=(1, 3, 224, 224), device="cpu"):
        self.model = model
        self.input_shape = input_shape
        self.device = device
        
    def count_parameters(self) -> int:
        if not torch: return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def benchmark_inference(self, num_iterations=100) -> Dict[str, Any]:
        if not torch:
            return {"error": "PyTorch not found"}
            
        self.model.to(self.device)
        self.model.eval()
        dummy_input = torch.randn(*self.input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
            
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
                
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_iterations) * 1000
        fps = num_iterations / total_time
        
        params = self.count_parameters()
        
        # Estimate size (naive)
        size_mb = params * 4 / (1024 * 1024)
        
        return {
            "parameters": params,
            "estimated_size_mb": round(size_mb, 2),
            "avg_inference_time_ms": round(avg_time_ms, 2),
            "fps": round(fps, 2),
            "device": self.device
        }
