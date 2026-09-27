import os
from pathlib import Path
from typing import Dict, List, Any

from agriguard.utils.logger import setup_logger

logger = setup_logger("downloader")

class DatasetDownloader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_dir = Path(config.get("raw_dir", "data/raw"))
        self.sources = config.get("sources", [])
        
    def download_all(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for source in self.sources:
            if source["type"] == "kaggle":
                self._download_kaggle(source)
            elif source["type"] == "roboflow":
                self._download_roboflow(source)
            else:
                logger.warning(f"Unknown source type: {source['type']}")
                
    def _download_kaggle(self, source: Dict[str, str]):
        name = source["name"]
        url = source["url"]
        dest = self.raw_dir / name
        
        if dest.exists():
            logger.info(f"Kaggle dataset {name} already exists. Skipping download.")
            return
            
        logger.info(f"Downloading Kaggle dataset {name} from {url}...")
        # Check Kaggle credentials
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            logger.warning("KAGGLE_USERNAME or KAGGLE_KEY not found in environment. Using Demo Mode or manual download required.")
            logger.info(f"Please manually download {url} and extract to {dest}")
            return
            
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(url, path=str(dest), unzip=True)
            logger.info(f"Successfully downloaded {name}")
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset {name}: {e}")
            
    def _download_roboflow(self, source: Dict[str, str]):
        name = source["name"]
        workspace = source.get("workspace")
        project = source.get("project")
        dest = self.raw_dir / name
        
        if dest.exists():
            logger.info(f"Roboflow dataset {name} already exists.")
            return
            
        logger.info(f"Downloading Roboflow dataset {name}...")
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            logger.warning("ROBOFLOW_API_KEY not found. Using Demo Mode or manual download required.")
            return
            
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            project_rf = rf.workspace(workspace).project(project)
            dataset = project_rf.version(1).download("yolov8", location=str(dest))
            logger.info(f"Successfully downloaded {name}")
        except ImportError:
            logger.error("roboflow package not installed. Run 'pip install roboflow'")
        except Exception as e:
            logger.error(f"Failed to download Roboflow dataset {name}: {e}")
