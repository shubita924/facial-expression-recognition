import torch.nn as nn

class BaselineMLP(nn.Module):
    """Flattens the image — no spatial structure. Our deliberately weak starting point."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 256), nn.ReLU(),
            nn.Linear(256, 7),
        )
    def forward(self, x):
        return self.net(x)
