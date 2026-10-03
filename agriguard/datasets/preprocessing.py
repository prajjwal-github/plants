import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np

from agriguard.utils.logger import setup_logger

logger = setup_logger("preprocessor")

class DatasetPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.raw_dir = Path(config.get("raw_dir", "data/raw"))
        self.interim_dir = Path(config.get("interim_dir", "data/interim"))
        self.processed_dir = Path(config.get("processed_dir", "data/processed"))
        
    def _compute_image_hash(self, image_path: Path) -> str | None:
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def _is_valid_image(self, image_path: Path) -> bool:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            # Check dimensions (arbitrary minimum size for validity)
            if img.shape[0] < 32 or img.shape[1] < 32:
                return False
            return True
        except Exception:
            return False

    def clean_dataset(self, source_name: str):
        """Removes corrupted and duplicate images."""
        src_dir = self.raw_dir / source_name
        dest_dir = self.interim_dir / "cleaned" / source_name
        
        if not src_dir.exists():
            logger.warning(f"Source directory {src_dir} does not exist.")
            return
            
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        seen_hashes = set()
        valid_count = 0
        duplicate_count = 0
        corrupt_count = 0
        
        logger.info(f"Cleaning dataset {source_name}...")
        
        for root, _, files in os.walk(src_dir):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                file_path = Path(root) / file
                
                # Check validity
                if not self._is_valid_image(file_path):
                    corrupt_count += 1
                    continue
                    
                # Check duplicate
                img_hash = self._compute_image_hash(file_path)
                if img_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                    
                seen_hashes.add(img_hash)
                
                # Copy to interim
                # Maintain relative structure
                rel_path = file_path.relative_to(src_dir)
                out_path = dest_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Read and write via cv2 to normalize formats
                img = cv2.imread(str(file_path))
                cv2.imwrite(str(out_path), img)
                valid_count += 1
                
        logger.info(f"Cleaned {source_name}: {valid_count} valid, {duplicate_count} duplicates, {corrupt_count} corrupt.")
