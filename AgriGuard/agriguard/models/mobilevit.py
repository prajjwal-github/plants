import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

class MobileViTModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm library is required for MobileViT. Run 'pip install timm'")
            
        # Using mobilevit_s or similar available in timm
        self.model = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
