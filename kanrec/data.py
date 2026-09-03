"""
KANRecDataModule — loads Parquet splits exported from Databricks Delta Lake
and wraps them in PyTorch DataLoaders.
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CriteoDataset(Dataset):
    def __init__(self, parquet_path: str, numerical_cols: list[str],
                 categorical_cols: list[str]):
        df = pd.read_parquet(parquet_path)
        self.x_num = torch.tensor(
            df[numerical_cols].fillna(0).values, dtype=torch.float32
        )
        self.x_cat = torch.tensor(
            df[categorical_cols].fillna(0).values, dtype=torch.long
        )
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class KANRecDataModule:
    """
    Args:
        train_path / val_path / test_path: paths to Parquet directories.
        feature_selection_path: path to feature_selection.json from Databricks.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker processes.
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        feature_selection_path: str = "data/feature_selection.json",
        batch_size: int = 4096,
        num_workers: int = 4,
    ):
        self.batch_size  = batch_size
        self.num_workers = num_workers

        with open(feature_selection_path) as f:
            sel = json.load(f)
        self.numerical_cols   = sel["selected"]
        self.categorical_cols = [f"C{i}" for i in range(1, 27)]

        self._train = CriteoDataset(train_path, self.numerical_cols, self.categorical_cols)
        self._val   = CriteoDataset(val_path,   self.numerical_cols, self.categorical_cols)
        self._test  = CriteoDataset(test_path,  self.numerical_cols, self.categorical_cols)

        # Build vocabulary sizes for categorical fields
        train_df = pd.read_parquet(train_path, columns=self.categorical_cols)
        self.cat_cardinalities = [
            int(train_df[c].max()) + 1 for c in self.categorical_cols
        ]

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self._val,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test,  batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
