import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18FER(nn.Module):
    """Pretrained ResNet18 (ImageNet) adapted for FER2013.
    Grayscale is repeated to 3 channels and ImageNet-normalized so the pretrained
    conv filters apply; the final fc is replaced with a 7-class head.
    freeze=True -> feature extraction (train head only);
    freeze=False -> fine-tune the whole network;
    pretrained=False -> same architecture trained from scratch (control)."""
    def __init__(self, dropout=0.0, freeze=False, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_feat = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 7),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)          # [B,1,48,48] -> [B,3,48,48]
        x = (x - self.mean) / self.std    # match ImageNet preprocessing
        return self.backbone(x)
