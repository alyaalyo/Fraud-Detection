from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import train_model  # noqa: E402
from src.utils import generate_run_id, seed_everything  # noqa: E402


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline MLP on IEEE-CIS")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    seed_everything(seed)

    run_cfg = config.setdefault("run", {})
    if not run_cfg.get("run_id"):
        run_cfg["run_id"] = generate_run_id("baseline")
    if "run_dir" not in run_cfg:
        output_dir = run_cfg.get("output_dir", "runs")
        run_cfg["run_dir"] = os.path.join(output_dir, run_cfg["run_id"])

    metrics = train_model(config)
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
