import torch.nn as nn

class DeeperCNNGAP(nn.Module):
    """DeeperCNN but with Global Average Pooling instead of a large flatten+linear head.
    Collapses the ~2.4M-param classifier to ~1.8K, attacking the train/val gap
    structurally: averaging spatial maps prevents memorizing pixel arrangements."""
    def __init__(self, dropout=0.0, batchnorm=False):
        super().__init__()

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if batchnorm: layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)]
            if batchnorm: layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            return layers

        self.features = nn.Sequential(
            *conv_block(1, 64),     # 48 -> 24
            *conv_block(64, 128),   # 24 -> 12
            *conv_block(128, 256),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 256 x 6 x 6 -> 256 x 1 x 1
            nn.Flatten(),              # -> 256
            nn.Dropout(dropout),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
