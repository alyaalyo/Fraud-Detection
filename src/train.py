from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .data import build_dataloaders
from .model import build_model
from .utils import save_json, save_yaml, setup_logging


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos)


def _eval_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    all_logits = []
    all_targets = []
    losses = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))

    try:
        roc = float(roc_auc_score(targets, probs))
    except ValueError:
        roc = float("nan")

    preds = (probs >= 0.5).astype(np.int32)
    f1 = float(f1_score(targets, preds, zero_division=0))
    loss = float(np.mean(losses)) if losses else float("nan")

    return roc, f1, loss


def train_model(config: Dict) -> Dict[str, float]:
    run_cfg = config.get("run", {})
    run_dir = run_cfg.get("run_dir")
    if run_dir is None:
        output_dir = run_cfg.get("output_dir", "runs")
        run_id = run_cfg.get("run_id", "run")
        run_dir = os.path.join(output_dir, run_id)

    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(run_dir)

    save_yaml(os.path.join(run_dir, "config.yaml"), config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, val_loader, input_dim, _ = build_dataloaders(config)
    model = build_model(input_dim, config.get("model", {})).to(device)

    train_targets = train_loader.dataset.y.numpy()
    pos_weight = _compute_pos_weight(train_targets).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_cfg = config.get("train", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 5))
    early_stopping = bool(train_cfg.get("early_stopping", False))
    patience = int(train_cfg.get("patience", 3))

    best_roc = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_roc, val_f1, val_loss = _eval_epoch(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_roc_auc={val_roc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_roc > best_roc:
            best_roc = val_roc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    if best_state is not None:
        torch.save(best_state, os.path.join(run_dir, "model.pt"))

    metrics = {
        "val_roc_auc": float(best_roc),
        "val_f1": float(val_f1),
        "val_loss": float(val_loss),
    }
    save_json(os.path.join(run_dir, "metrics.json"), metrics)

    return metrics
