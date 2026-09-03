"""Tests for KANRecDataModule using synthetic Parquet files."""
import json
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from kanrec.data import KANRecDataModule, CriteoDataset


NUM_COLS = [f"I{i}" for i in range(1, 6)]   # 5 synthetic numerical fields
CAT_COLS = [f"C{i}" for i in range(1, 4)]   # 3 synthetic categorical fields


def _make_parquet(path: str, n: int = 200):
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame({
        "label": np.random.randint(0, 2, n),
        **{c: np.random.randn(n).astype("float32") for c in NUM_COLS},
        **{c: np.random.randint(0, 10, n) for c in CAT_COLS},
    })
    df.to_parquet(os.path.join(path, "part-0.parquet"), index=False)
    return df


@pytest.fixture
def tmp_data(tmp_path):
    """Creates minimal train/val/test Parquet directories and feature_selection.json."""
    for split in ["train", "val", "test"]:
        _make_parquet(str(tmp_path / split))

    sel = {"selected": NUM_COLS, "excluded": [], "spearman_scores": {}}
    sel_path = str(tmp_path / "feature_selection.json")
    with open(sel_path, "w") as f:
        json.dump(sel, f)

    return tmp_path, sel_path


def test_dataset_length(tmp_data):
    tmp_path, _ = tmp_data
    ds = CriteoDataset(str(tmp_path / "train"), NUM_COLS, CAT_COLS)
    assert len(ds) == 200


def test_dataset_item_shapes(tmp_data):
    tmp_path, _ = tmp_data
    ds = CriteoDataset(str(tmp_path / "train"), NUM_COLS, CAT_COLS)
    x_num, x_cat, y = ds[0]
    assert x_num.shape == (len(NUM_COLS),)
    assert x_cat.shape == (len(CAT_COLS),)
    assert y.shape     == ()


def test_datamodule_dataloaders(tmp_data):
    tmp_path, sel_path = tmp_data
    dm = KANRecDataModule(
        train_path=str(tmp_path / "train"),
        val_path=str(tmp_path / "val"),
        test_path=str(tmp_path / "test"),
        feature_selection_path=sel_path,
        batch_size=64,
        num_workers=0,
    )
    batch = next(iter(dm.train_dataloader()))
    x_num, x_cat, y = batch
    assert x_num.shape[1] == len(NUM_COLS)
    assert x_cat.shape[1] == len(CAT_COLS)
    assert y.shape[0]     == x_num.shape[0]


def test_datamodule_cat_cardinalities(tmp_data):
    tmp_path, sel_path = tmp_data
    dm = KANRecDataModule(
        train_path=str(tmp_path / "train"),
        val_path=str(tmp_path / "val"),
        test_path=str(tmp_path / "test"),
        feature_selection_path=sel_path,
        num_workers=0,
    )
    assert len(dm.cat_cardinalities) == len(CAT_COLS)
    assert all(c > 0 for c in dm.cat_cardinalities)
