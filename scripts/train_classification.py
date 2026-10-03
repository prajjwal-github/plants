import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from agriguard.utils.config import load_config
from agriguard.utils.logger import setup_logger
from agriguard.models.efficientnet import EfficientNetModel

logger = setup_logger("train_real")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="efficientnet")
    args = parser.parse_args()
    
    config = load_config(f"configs/{args.model}.yaml")
    data_dir = "data/processed/classification/train"
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        logger.error(f"CRITICAL ERROR: No data found in {data_dir}!")
        logger.error("I cannot train the model because the dataset is missing.")
        logger.error("Please add your Kaggle keys to the .env file so I can download the data!")
        return

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model = EfficientNetModel(num_classes=len(train_dataset.classes)).to(device)
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = config["training"].get("epochs", 5)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        acc = 100. * correct / total
        logger.info(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")
        
    # Save Model
    os.makedirs(f"models/{args.model}", exist_ok=True)
    model_path = f"models/{args.model}/best_model.pt"
    torch.save(model.state_dict(), model_path)
    
    from agriguard.models.registry import ModelRegistry
    registry = ModelRegistry()
    registry.update_model_info(args.model, {
        "status": "trained",
        "path": model_path,
        "classes": train_dataset.classes
    })
    logger.info(f"Training complete! Real model saved to {model_path}.")

if __name__ == "__main__":
    main()
