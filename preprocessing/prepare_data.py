import pandas as pd
import numpy as np
import kaggle
import os
import zipfile
from pathlib import Path


# directories initialization
HOME_DIR = Path(__file__).resolve().parent
DATA_DIR = HOME_DIR / "data"
IEE_DIR = DATA_DIR / "ieee_fraud_detection"


# the whole preprocessing pipeline:
# downloads the dataset from kaggle
# merges transactions with identities
# preprocesses it
# splits based on a time of transaction
# returns X_train, X_test, y_train, y_test for model training
def process_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None :
    if check_download():
        trans, id = load_data()
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


# doqnloads the dataset from kaggle into a folder
# Fraud-Detection/preprocessing/data/ieee_fraud_detection
def download_dataset() -> bool:
    try:
        print("downloading the dataset")
        competition = "ieee-fraud-detection"
        kaggle.api.competition_download_files(competition, path=str(IEE_DIR))
        zip_file = os.path.join(str(IEE_DIR), f"{competition}.zip")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(str(IEE_DIR))

        os.remove(zip_file)
        competition_specific = os.path.join(str(IEE_DIR), "sample_submission.csv")
        os.remove(competition_specific)
        return True
    except Exception as e:
        print(f"An error downloading the dataset: {e}")
    return False


# checks if dataset was already downloaded
def check_download() -> bool:
    DATA_DIR.mkdir(exist_ok=True)
    files_required = [
        str(IEE_DIR)  + "/train_identity.csv",
        str(IEE_DIR)  + "/train_transaction.csv"
    ]
    if all(os.path.exists(p) for p in files_required):
        print("The dataset is already downloaded")
        return True
    return download_dataset()


# reads transation and identitu csv files from the dataset
# retruns a separate dataframe for each 
def load_data() -> tuple [pd.DataFrame, pd.DataFrame]:
    base_path = str(IEE_DIR)

    train_trans = pd.read_csv(base_path + "/train_transaction.csv")
    train_id = pd.read_csv(base_path + "/train_identity.csv")

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

