from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(run_dir: str) -> logging.Logger:
    """Configure root logger to log to stdout and a file in run_dir."""
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger("nic")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def generate_run_id(prefix: str = "run") -> str:
    """Generate a simple timestamp-based run id."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
