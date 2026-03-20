from __future__ import annotations

from typing import List, Sequence, Union

import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


def _expand_to_list(
    value: Union[str, float, Sequence],
    n: int,
) -> List:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Expected list of length {n}, got {len(value)}")
        return list(value)
    return [value for _ in range(n)]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        activation: Union[str, Sequence[str]],
        dropout: Union[float, Sequence[float]],
        batchnorm: bool,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        activations = _expand_to_list(activation, len(hidden_sizes))
        dropouts = _expand_to_list(dropout, len(hidden_sizes))

        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            act_name = activations[i]
            layers.append(_get_activation(str(act_name)))
            drop_val = float(dropouts[i]) if dropouts is not None else 0.0
            if drop_val > 0:
                layers.append(nn.Dropout(drop_val))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(input_dim: int, model_cfg: dict) -> nn.Module:
    return MLP(
        input_dim=input_dim,
        hidden_sizes=list(model_cfg.get("hidden_sizes", [256, 128, 64])),
        activation=model_cfg.get("activation", "relu"),
        dropout=model_cfg.get("dropout", 0.0),
        batchnorm=bool(model_cfg.get("batchnorm", True)),
    )
