import torch.nn as nn

class DeeperCNN(nn.Module):
    """Three conv blocks (64->128->256) + classifier head.
    Deeper than SmallCNN to test whether added capacity breaks the ~0.53 ceiling.
    dropout (head) and batchnorm (all blocks) are toggleable to isolate effects."""
    def __init__(self, dropout=0.0, batchnorm=False):
        super().__init__()

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(),
                       nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            return layers

        self.features = nn.Sequential(
            *conv_block(1, 64),     # 48 -> 24
            *conv_block(64, 128),   # 24 -> 12
            *conv_block(128, 256),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
