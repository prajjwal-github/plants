import os
import json
import csv
import random
from pathlib import Path
from typing import Dict, Any

from agriguard.utils.logger import setup_logger

logger = setup_logger("splitter")

class DatasetSplitter:
    def __init__(self, config: Dict[str, Any]):
        self.interim_dir = Path(config.get("interim_dir", "data/interim"))
        self.processed_dir = Path(config.get("processed_dir", "data/processed"))
        self.metadata_dir = Path(config.get("metadata_dir", "data/metadata"))
        
        splits = config.get("splits", {})
        self.train_ratio = splits.get("train_ratio", 0.8)
        self.val_ratio = splits.get("val_ratio", 0.1)
        self.test_ratio = splits.get("test_ratio", 0.1)
        self.seed = splits.get("random_seed", 42)
        
        random.seed(self.seed)
        
    def generate_classification_splits(self, source_name: str):
        """Splits an image folder structure into train/val/test."""
        src_dir = self.interim_dir / "cleaned" / source_name
        dest_dir = self.processed_dir / "classification"
        
        if not src_dir.exists():
            logger.warning(f"Cleaned directory {src_dir} does not exist.")
            return
            
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.metadata_dir / f"{source_name}_manifest.csv"
        
        logger.info(f"Splitting dataset {source_name}...")
        
        manifest_rows = [["image_path", "class", "split", "source"]]
        
        classes = [d for d in os.listdir(src_dir) if (src_dir / d).is_dir()]
        
        for cls in classes:
            cls_dir = src_dir / cls
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            
            n = len(images)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train + n_val]
            test_imgs = images[n_train + n_val:]
            
            splits_map = {
                "train": train_imgs,
                "val": val_imgs,
                "test": test_imgs
            }
            
            for split_name, imgs in splits_map.items():
                split_dir = dest_dir / split_name / cls
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for img in imgs:
                    # Normally we'd copy or symlink here. For simplicity in demo, we'll symlink if supported, or just keep track.
                    src_img = cls_dir / img
                    dst_img = split_dir / f"{source_name}_{img}"
                    
                    try:
                        # Hardlink to save space, fallback to copy
                        os.link(src_img, dst_img)
                    except Exception:
                        import shutil
                        shutil.copy2(src_img, dst_img)
                        
                    manifest_rows.append([str(dst_img), cls, split_name, source_name])
                    
        # Write manifest
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(manifest_rows)
            
        logger.info(f"Split completed for {source_name}. Manifest saved to {manifest_path}")
