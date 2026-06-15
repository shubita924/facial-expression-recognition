import torch.nn as nn

class SmallCNN(nn.Module):
    """Two conv blocks + classifier head.
    Optional dropout (head) and batchnorm (conv blocks), each toggleable
    so their effects can be isolated in separate runs."""
    def __init__(self, dropout=0.0, batchnorm=False):
        super().__init__()

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            return layers

        self.features = nn.Sequential(
            *conv_block(1, 32),    # 48 -> 24
            *conv_block(32, 64),   # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
