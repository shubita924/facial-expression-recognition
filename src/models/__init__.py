from .baseline_mlp import BaselineMLP
from .small_cnn import SmallCNN
from .deeper_cnn import DeeperCNN

REGISTRY = {
    "baseline_mlp": BaselineMLP,
    "small_cnn": SmallCNN,
    "deeper_cnn": DeeperCNN,
}

def get_model(name, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
