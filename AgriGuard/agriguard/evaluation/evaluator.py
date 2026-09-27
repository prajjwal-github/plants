import time
import json
from pathlib import Path
from typing import Dict, Any

from agriguard.utils.logger import setup_logger

try:
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except ImportError:
    torch = None

logger = setup_logger("cross_dataset_evaluator")

class CrossDatasetEvaluator:
    def __init__(self, model, dataloader_in_domain, dataloader_cross_domain, classes, device="cpu"):
        self.model = model
        self.dataloader_in_domain = dataloader_in_domain
        self.dataloader_cross_domain = dataloader_cross_domain
        self.classes = classes
        self.device = device
        
    def _evaluate_loader(self, dataloader) -> Dict[str, Any]:
        if not torch:
            return {"error": "PyTorch/sklearn required."}
            
        self.model.eval()
        self.model.to(self.device)
        
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        inference_time = time.time() - start_time
        
        acc = accuracy_score(all_labels, all_preds)
        p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()
        
        return {
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "inference_time_s": round(inference_time, 2),
            "confusion_matrix": cm
        }
        
    def run_evaluation(self, output_path: str):
        logger.info("Evaluating on In-Domain dataset...")
        in_domain_metrics = self._evaluate_loader(self.dataloader_in_domain)
        
        logger.info("Evaluating on Cross-Domain dataset...")
        cross_domain_metrics = self._evaluate_loader(self.dataloader_cross_domain)
        
        if "error" not in in_domain_metrics and "error" not in cross_domain_metrics:
            perf_drop = in_domain_metrics["accuracy"] - cross_domain_metrics["accuracy"]
        else:
            perf_drop = 0.0
            
        report = {
            "in_domain": in_domain_metrics,
            "cross_domain": cross_domain_metrics,
            "performance_drop": round(perf_drop, 4)
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Evaluation report saved to {output_path}")
        return report
