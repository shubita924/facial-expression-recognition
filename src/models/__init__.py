from .baseline_mlp import BaselineMLP
from .small_cnn import SmallCNN
from .deeper_cnn import DeeperCNN
from .deeper_cnn_gap import DeeperCNNGAP
from .resnet18_fer import ResNet18FER

REGISTRY = {
    "baseline_mlp": BaselineMLP,
    "small_cnn": SmallCNN,
    "deeper_cnn": DeeperCNN,
    "deeper_cnn_gap": DeeperCNNGAP,
    "resnet18": ResNet18FER,
}

def get_model(name, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
