import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class ResNet34FER(nn.Module):
    """Pretrained ResNet34 (ImageNet) adapted for FER2013 — deeper sibling of ResNet18FER.
    Same adaptations: grayscale repeated to 3 channels, ImageNet-normalized,
    final fc replaced with a 7-class head. Lets us test 18 vs 34 layers directly."""
    def __init__(self, dropout=0.0, freeze=False, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet34(weights=weights)

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
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.backbone(x)
