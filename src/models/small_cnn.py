import torch.nn as nn

class SmallCNN(nn.Module):
    """Two conv blocks + classifier head. Optional dropout in the head
    to regularize the ~1.2M-param Linear layer that drove the overfitting."""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 48 -> 24
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7),
        )
    def forward(self, x):
        return self.classifier(self.features(x))
