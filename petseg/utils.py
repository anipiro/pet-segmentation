import json
import random
from pathlib import Path

import numpy as np

# torch and yaml are imported inside the functions that need them, so the
# serving image (ONNX Runtime only) can import this module without PyTorch.

ROOT = Path(__file__).resolve().parent.parent

# ImageNet statistics: every pretrained backbone we use expects them, and using
# the same normalisation for the scratch U-Net keeps the comparison fair.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def use_system_certs():
    """Make Python trust the OS certificate store.

    Needed on networks with a TLS-inspecting proxy, where pretrained-weight
    downloads otherwise fail with CERTIFICATE_VERIFY_FAILED. Harmless elsewhere.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path) -> dict:
    """Load a YAML config, merging it over configs/base.yaml."""
    import yaml

    with open(ROOT / "configs" / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        cfg.update(yaml.safe_load(f) or {})
    return cfg


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
