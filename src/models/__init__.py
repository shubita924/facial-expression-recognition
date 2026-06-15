from .baseline_mlp import BaselineMLP
from .small_cnn import SmallCNN

REGISTRY = {
    "baseline_mlp": BaselineMLP,
    "small_cnn": SmallCNN,
}

def get_model(name, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
