from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Preprocessor:
    num_cols: List[str]
    cat_cols: List[str]
    num_medians: Dict[str, float]
    cat_fill: str
    cat_maps: Dict[str, Dict[str, int]]
    scaler: Optional[StandardScaler]

    @staticmethod
    def _is_cat(series: pd.Series) -> bool:
        return series.dtype == object or str(series.dtype).startswith("category")

    @classmethod
    def fit(cls, df: pd.DataFrame, target_col: str) -> "Preprocessor":
        features = df.drop(columns=[target_col])
        cat_cols = [c for c in features.columns if cls._is_cat(features[c])]
        num_cols = [c for c in features.columns if c not in cat_cols]

        num_medians = {c: float(features[c].median()) for c in num_cols}
        cat_fill = "missing"

        cat_maps: Dict[str, Dict[str, int]] = {}
        for c in cat_cols:
            series = features[c].fillna(cat_fill).astype(str)
            uniques = pd.unique(series)
            cat_maps[c] = {v: i for i, v in enumerate(uniques)}

        scaler = StandardScaler() if num_cols else None
        if scaler is not None:
            num_data = features[num_cols].fillna(pd.Series(num_medians))
            scaler.fit(num_data)

        return cls(
            num_cols=num_cols,
            cat_cols=cat_cols,
            num_medians=num_medians,
            cat_fill=cat_fill,
            cat_maps=cat_maps,
            scaler=scaler,
        )

    def transform(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        features = df.drop(columns=[target_col])
        y = df[target_col].values.astype(np.float32)

        if self.num_cols:
            num_data = features[self.num_cols].copy()
            for c in self.num_cols:
                num_data[c] = num_data[c].fillna(self.num_medians[c])
            num_arr = num_data.values.astype(np.float32)
            if self.scaler is not None:
                num_arr = self.scaler.transform(num_arr)
        else:
            num_arr = np.zeros((len(df), 0), dtype=np.float32)

        if self.cat_cols:
            cat_data = features[self.cat_cols].copy()
            for c in self.cat_cols:
                cat_data[c] = cat_data[c].fillna(self.cat_fill).astype(str)
                mapping = self.cat_maps[c]
                cat_data[c] = cat_data[c].map(mapping).fillna(-1).astype(np.int64)
            cat_arr = cat_data.values.astype(np.float32)
        else:
            cat_arr = np.zeros((len(df), 0), dtype=np.float32)

        X = np.concatenate([num_arr, cat_arr], axis=1).astype(np.float32)
        return X, y


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_dataframe(
    train_transaction_path: str, train_identity_path: Optional[str]
) -> pd.DataFrame:
    df = pd.read_csv(train_transaction_path)

    if train_identity_path and os.path.exists(train_identity_path):
        df_id = pd.read_csv(train_identity_path)
        if "TransactionID" in df.columns and "TransactionID" in df_id.columns:
            df = df.merge(df_id, on="TransactionID", how="left")

    return df


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, int, Preprocessor]:
    data_cfg = config["data"]
    df = load_dataframe(
        data_cfg["train_transaction_path"],
        data_cfg.get("train_identity_path"),
    )

    target_col = data_cfg.get("target_col", "isFraud")

    train_df, val_df = train_test_split(
        df,
        test_size=float(data_cfg.get("val_size", 0.2)),
        random_state=int(data_cfg.get("random_state", 42)),
        stratify=df[target_col],
    )

    pre = Preprocessor.fit(train_df, target_col)
    X_train, y_train = pre.transform(train_df, target_col)
    X_val, y_val = pre.transform(val_df, target_col)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    input_dim = X_train.shape[1]
    return train_loader, val_loader, input_dim, pre
