import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path


# directories initialization
HOME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HOME_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _resolve_data_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env_dir = os.environ.get("NIC_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    # Backward-compatible fallback
    fallback = HOME_DIR / "data"
    if fallback.exists():
        return fallback
    # Default to project-root data path for clearer error messages
    return DEFAULT_DATA_DIR


# the whole preprocessing pipeline:
# merges transactions with identities
# preprocesses it
# splits based on a time of transaction
# returns X_train, X_test, y_train, y_test for model training
def process_data(
    data_dir: Path | None = None,
    train_transaction_path: str | None = None,
    train_identity_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    trans, id = load_data(
        data_dir=data_dir,
        train_transaction_path=train_transaction_path,
        train_identity_path=train_identity_path,
    )
    trans_id_merged = pd.merge(trans, id, on="TransactionID", how="left")
    df = preprocess(trans_id_merged)
    data_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
    # Create time-based split (80-20)
    split_idx = int(0.8 * len(data_sorted))
    train, test = split(data_sorted, split_idx)
    X_train = train.drop("isFraud", axis=1)
    y_train = train["isFraud"]
    X_test = test.drop("isFraud", axis=1)
    y_test = test["isFraud"]
    return X_train, X_test, y_train, y_test


# reads transation and identitu csv files from the dataset
# retruns a separate dataframe for each 
def load_data(
    data_dir: Path | None = None,
    train_transaction_path: str | None = None,
    train_identity_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = _resolve_data_dir(data_dir)
    train_trans_path = (
        Path(train_transaction_path)
        if train_transaction_path
        else base_dir / "train_transaction.csv"
    )
    train_id_path = (
        Path(train_identity_path)
        if train_identity_path
        else base_dir / "train_identity.csv"
    )

    if not train_trans_path.exists():
        raise FileNotFoundError(
            f"Missing train_transaction.csv at '{train_trans_path}'. "
            "Place the Kaggle files under 'data/' at project root or set NIC_DATA_DIR."
        )
    if not train_id_path.exists():
        raise FileNotFoundError(
            f"Missing train_identity.csv at '{train_id_path}'. "
            "Place the Kaggle files under 'data/' at project root or set NIC_DATA_DIR."
        )

    train_trans = pd.read_csv(str(train_trans_path))
    train_id = pd.read_csv(str(train_id_path))

    return train_trans, train_id


# returns a preprocessed dataframe
def preprocess(df : pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    
    # Fill missing values simply
    for col in df_new.select_dtypes(include=[np.number]).columns:
        if col != 'isFraud':
            df_new[col] = df_new[col].fillna(0)
    
    for col in df_new.select_dtypes(include=['object', 'str']).columns:
        df_new[col] = df_new[col].fillna('unknown')
    

    new_features = []
    # Add basic time features
    if 'TransactionDT' in df_new.columns:
        time_features = pd.DataFrame({
        'hour' : (df['TransactionDT'] / 3600) % 24,
        'day' : (df['TransactionDT'] / (24*3600)) % 7
        })
        new_features.append(time_features)
    
    # Simple amount features
    if 'TransactionAmt' in df_new.columns:
        amount_feature = pd.DataFrame({
            'amt_log' : np.log1p(df['TransactionAmt'])
        })
        new_features.append(amount_feature)
        
    
    if new_features:
        new_features = pd.concat(new_features, axis=1)
        df_new = pd.concat([df_new, new_features], axis=1)

    return df_new

# creates a train test split based on the split_idx
def split(df : pd.DataFrame, split_idx: int) -> tuple [pd.DataFrame, pd.DataFrame]:
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return train, test
