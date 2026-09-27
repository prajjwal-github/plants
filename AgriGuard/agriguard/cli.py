import argparse
import sys
from pathlib import Path

from agriguard.utils.config import load_config
from agriguard.utils.logger import setup_logger

logger = setup_logger("cli")

def predict(args):
    logger.info(f"Predicting on image: {args.image}")
    # Pipeline loading logic will go here
    logger.info("Demo mode: Predictions not yet implemented.")

def train(args):
    logger.info(f"Starting training for model: {args.model}")
    if args.model == "efficientnet":
        logger.info("Training EfficientNet...")
    elif args.model == "mobilevit":
        logger.info("Training MobileViT...")
    elif args.model == "yolo":
        logger.info("Training YOLO...")
    else:
        logger.error(f"Unknown model {args.model}")

def evaluate(args):
    logger.info("Starting evaluation...")

def prepare_data(args):
    logger.info("Preparing data...")
    config = load_config("configs/dataset.yaml")
    
    from agriguard.datasets.downloader import DatasetDownloader
    from agriguard.datasets.preprocessing import DatasetPreprocessor
    from agriguard.datasets.splitter import DatasetSplitter
    
    downloader = DatasetDownloader(config)
    downloader.download_all()
    
    preprocessor = DatasetPreprocessor(config)
    splitter = DatasetSplitter(config)
    
    for source in config.get("sources", []):
        name = source["name"]
        if source["type"] == "kaggle":
            preprocessor.clean_dataset(name)
            splitter.generate_classification_splits(name)
        elif source["type"] == "roboflow":
            logger.info(f"Skipping custom splitting for {name}, Roboflow provides pre-split YOLO format.")
    
    logger.info("Data preparation complete.")

def main():
    parser = argparse.ArgumentParser(description="AgriGuard CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Run inference on an image")
    predict_parser.add_argument("image", type=str, help="Path to image file")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, required=True, choices=["efficientnet", "mobilevit", "yolo"])

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    
    # Prepare data
    prep_parser = subparsers.add_parser("prepare-data", help="Prepare datasets")

    args = parser.parse_args()

    if args.command == "predict":
        predict(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "prepare-data":
        prepare_data(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
