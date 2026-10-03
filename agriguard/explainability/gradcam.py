import cv2
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
except ImportError:
    torch = None

from agriguard.utils.logger import setup_logger

logger = setup_logger("gradcam")

class GradCAMWrapper:
    def __init__(self, model, target_layer, img_size=(224, 224)):
        self.model = model
        self.target_layer = target_layer
        self.img_size = img_size
        self.gradients = None
        self.activations = None
        
        if torch is None:
            raise ImportError("PyTorch is required for Grad-CAM.")
            
        # Hook registration
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, class_idx=None):
        """Generates Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        if np.max(heatmap) == 0:
            heatmap = np.zeros_like(heatmap)
        else:
            heatmap /= np.max(heatmap)
            
        heatmap = cv2.resize(heatmap, self.img_size)
        heatmap = np.uint8(255 * heatmap)
        return heatmap
        
    def overlay(self, img_path, heatmap, alpha=0.5):
        """Overlays heatmap onto original image."""
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)
        return overlay
