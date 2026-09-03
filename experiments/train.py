"""
Main training script for KAN-REC.

Usage:
    python experiments/train.py
    python experiments/train.py --encoder autodis --seed 123
    python experiments/train.py --dataset avazu --grid-size 5
"""
import argparse
import json
import os
import torch
import mlflow
from sklearn.metrics import roc_auc_score, log_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kanrec.model import KANRecModel
from kanrec.data import KANRecDataModule

os.makedirs("checkpoints", exist_ok=True)


def train(config: dict) -> float:
    torch.manual_seed(config["seed"])
    mlflow.set_experiment("kanrec-ctr")

    with mlflow.start_run(run_name=f"{config['encoder']}-{config['dataset']}-s{config['seed']}") as run:
        mlflow.log_params(config)
        run_id = run.info.run_id

        dm = KANRecDataModule(
            train_path=config["train_path"],
            val_path=config["val_path"],
            test_path=config["test_path"],
            feature_selection_path=config["feature_selection"],
            batch_size=config["batch_size"],
        )

        model = KANRecModel(
            num_numerical=len(dm.numerical_cols),
            cat_cardinalities=dm.cat_cardinalities,
            embedding_dim=config["embedding_dim"],
            kan_grid_size=config["grid_size"],
            kan_spline_order=config["spline_order"],
            monotone_fields=config.get("monotone_fields"),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training on {device} | encoder={config['encoder']} | "
              f"numerical fields={len(dm.numerical_cols)}")

        optimizer  = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        scheduler  = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)
        criterion  = torch.nn.BCELoss()

        best_val_auc, patience_ctr = 0.0, 0
        ckpt_path = f"checkpoints/best_{config['encoder']}_{config['dataset']}_s{config['seed']}.pt"

        for epoch in range(config["max_epochs"]):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for x_num, x_cat, y in dm.train_dataloader():
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x_num, x_cat).squeeze()
                loss   = criterion(y_pred, y) + model.entropy_regularization_loss()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for x_num, x_cat, y in dm.val_dataloader():
                    p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                    val_preds.extend(p)
                    val_labels.extend(y.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)
            val_ll  = log_loss(val_labels, val_preds)
            scheduler.step(1 - val_auc)
            mlflow.log_metrics({"train_loss": train_loss, "val_auc": val_auc,
                                 "val_logloss": val_ll}, step=epoch)
            print(f"Epoch {epoch+1:02d} | loss={train_loss:.4f} | "
                  f"val_auc={val_auc:.4f} | val_ll={val_ll:.4f}")

            if val_auc > best_val_auc:
                best_val_auc, patience_ctr = val_auc, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # ── Test ──────────────────────────────────────────────────────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, y in dm.test_dataloader():
                p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                test_preds.extend(p)
                test_labels.extend(y.numpy())

        test_auc = roc_auc_score(test_labels, test_preds)
        test_ll  = log_loss(test_labels, test_preds)
        mlflow.log_metrics({"test_auc": test_auc, "test_logloss": test_ll})
        mlflow.pytorch.log_model(model, "model")
        print(f"\nTest AUC={test_auc:.4f} | Log-loss={test_ll:.4f}")

        # Save run_id for symbolic extraction
        with open(f"checkpoints/run_id_{config['encoder']}_{config['dataset']}_s{config['seed']}.txt", "w") as f:
            f.write(run_id)

    return test_auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder",   default="kan-bspline",
                   choices=["kan-bspline", "kan-rbf", "autodis", "raw"])
    p.add_argument("--dataset",   default="criteo")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "encoder":          args.encoder,
        "dataset":          args.dataset,
        "seed":             args.seed,
        "train_path":       f"data/delta_parquet/{args.dataset}/train",
        "val_path":         f"data/delta_parquet/{args.dataset}/val",
        "test_path":        f"data/delta_parquet/{args.dataset}/test",
        "feature_selection": "data/feature_selection.json",
        "batch_size":       4096,
        "embedding_dim":    args.embedding_dim,
        "grid_size":        args.grid_size,
        "spline_order":     3,
        "lr":               1e-3,
        "max_epochs":       30,
        "patience":         3,
        "monotone_fields":  [0, 1, 2],  # I1, I2, I3: expected monotone
    }
    train(config)
