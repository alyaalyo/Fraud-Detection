from __future__ import annotations

from typing import List

import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        dropout: float,
        activation: str,
        batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_get_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(input_dim: int, model_cfg: dict) -> nn.Module:
    return MLP(
        input_dim=input_dim,
        hidden_sizes=list(model_cfg.get("hidden_sizes", [256, 128, 64])),
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=str(model_cfg.get("activation", "relu")),
        batchnorm=bool(model_cfg.get("batchnorm", True)),
    )
