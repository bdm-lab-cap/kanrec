"""
Main training script for KAN-REC.

Usage:
    pip install -e ".[train]"   # mlflow no es dependencia base de kanrec (ver setup.py):
                                  # solo se necesita para ejecutar ESTE script localmente
    python experiments/train.py
    python experiments/train.py --encoder autodis --seed 123
    python experiments/train.py --encoder raw --seed 256
    python experiments/train.py --dataset avazu --grid-size 5

Fix applied (2026-09, auditoría de tribunal — hallazgo B6)
--------------------------------------------------------------
Every branch of this script used to build a KANRecModel regardless of
--encoder: the flag only changed the checkpoint's file name, so
experiments/run_all.sh's three "different" runs were the same model saved
under three names. --encoder now goes through kanrec.baselines.build_model,
which actually selects the numerical encoder (see that module for raw /
autodis / kan-bspline).

Note: --encoder kan-rbf was listed as a choice but never implemented; it has
been removed rather than left as a silent no-op.
"""
import argparse
import json
import os

import mlflow
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from kanrec.baselines import build_model
from kanrec.data import KANRecDataModule

os.makedirs("checkpoints", exist_ok=True)

# Learning rate por encoder, no unico para los tres (verificado empiricamente:
# con el mismo lr, AutoDis queda infraentrenado por su capa de discretizacion
# mas parametrizada; raw y kan-bspline son estables en el mismo rango). Ver
# fabric/04_model_comparison.py para el detalle de la verificacion.
DEFAULT_LR = {"raw": 1e-3, "autodis": 1e-2, "kan-bspline": 1e-3}


def train(config: dict) -> float:
    torch.manual_seed(config["seed"])
    # mlflow>=3.0 dejo el backend de fichero ("./mlruns") en modo
    # mantenimiento y lo bloquea por defecto, exigiendo un backend de base
    # de datos. sqlite es la via soportada hacia delante (no depende de
    # activar MLFLOW_ALLOW_FILE_STORE, que el propio mensaje de mlflow dice
    # que no recibira mas actualizaciones).
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

        model = build_model(
            encoder=config["encoder"],
            num_numerical=len(dm.numerical_cols),
            cat_cardinalities=dm.cat_cardinalities,
            embedding_dim=config["embedding_dim"],
            kan_grid_size=config["grid_size"],
            kan_spline_order=config["spline_order"],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(
            f"Training on {device} | encoder={config['encoder']} "
            f"({type(model.numerical_encoder).__name__}) | "
            f"numerical fields={len(dm.numerical_cols)}"
        )

        # Grid calibration (hallazgo A3): only kan-bspline needs it. Uses a
        # sample of *normalised* training data drawn before any weight
        # update, so the spline grid matches the real distribution of each
        # field from the very first training step.
        #
        # 50k rows, not a fixed batch count (hallazgo verificado on the real
        # 8M-row Criteo run in Fabric): I6 and I12 reach ~690 standard
        # deviations after StandardScaler -- extreme tails. A ~5-10k row
        # sample had low odds of including those outliers, and the
        # calibrated grid could leave them uncovered (the same A3 problem,
        # for those specific rows). Accumulating by ROW COUNT rather than
        # batch count is also robust to whatever batch_size is configured.
        if hasattr(model, "calibrate"):
            CALIB_ROWS = 50_000
            calib_batches, calib_n = [], 0
            for x_num, _, _ in dm.train_dataloader():
                calib_batches.append(x_num)
                calib_n += x_num.size(0)
                if calib_n >= CALIB_ROWS:
                    break
            calib_sample = torch.cat(calib_batches, dim=0).to(device)
            model.calibrate(calib_sample)
            print(f"Grid calibrated on {calib_sample.size(0)} rows.")

        # parameter_groups da al spline un lr 25x mayor: arranca ~50x mas pequeno
        # que la ruta base por la inicializacion de efficient-kan y, con un lr
        # compartido, nunca despega (las curvas phi salian rectas siempre, ver
        # KANRecModel.parameter_groups). Los encoders raw/autodis no tienen
        # splines, asi que para ellos el helper no existe y se usa el Adam normal.
        if hasattr(model, "parameter_groups"):
            optimizer = Adam(model.parameter_groups(base_lr=config["lr"]), weight_decay=1e-5)
        else:
            optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = torch.nn.BCELoss()

        best_val_auc, patience_ctr = 0.0, 0
        # grid_size va en el nombre: evita que un barrido local de
        # --grid-size sobre el mismo encoder/seed se sobreescriba entre si
        # (mismo motivo que en fabric/04_model_comparison.py).
        ckpt_path = f"checkpoints/best_{config['encoder']}_{config['dataset']}_gs{config['grid_size']}_s{config['seed']}.pt"

        for epoch in range(config["max_epochs"]):
            # -- Train ------------------------------------------------------
            model.train()
            train_loss = 0.0
            for x_num, x_cat, y in dm.train_dataloader():
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x_num, x_cat).squeeze()
                loss = criterion(y_pred, y)
                if hasattr(model, "entropy_regularization_loss"):
                    loss = loss + model.entropy_regularization_loss()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # -- Validate -----------------------------------------------------
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for x_num, x_cat, y in dm.val_dataloader():
                    p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                    val_preds.extend(p)
                    val_labels.extend(y.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)
            val_ll = log_loss(val_labels, val_preds)
            scheduler.step(1 - val_auc)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_auc": val_auc, "val_logloss": val_ll}, step=epoch
            )
            print(f"Epoch {epoch+1:02d} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_ll={val_ll:.4f}")

            if val_auc > best_val_auc:
                best_val_auc, patience_ctr = val_auc, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # -- Test -------------------------------------------------------------
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, y in dm.test_dataloader():
                p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                test_preds.extend(p)
                test_labels.extend(y.numpy())

        test_auc = roc_auc_score(test_labels, test_preds)
        test_ll = log_loss(test_labels, test_preds)
        mlflow.log_metrics({"test_auc": test_auc, "test_logloss": test_ll})
        mlflow.pytorch.log_model(model, "model")
        print(f"\nTest AUC={test_auc:.4f} | Log-loss={test_ll:.4f}")

        with open(f"checkpoints/run_id_{config['encoder']}_{config['dataset']}_s{config['seed']}.txt", "w") as f:
            f.write(run_id)

    return test_auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="kan-bspline", choices=["kan-bspline", "autodis", "raw"])
    p.add_argument("--dataset", default="criteo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "encoder": args.encoder,
        "dataset": args.dataset,
        "seed": args.seed,
        "train_path": f"data/delta_parquet/{args.dataset}/train",
        "val_path": f"data/delta_parquet/{args.dataset}/val",
        "test_path": f"data/delta_parquet/{args.dataset}/test",
        "feature_selection": "data/feature_selection.json",
        "batch_size": 4096,
        "embedding_dim": args.embedding_dim,
        "grid_size": args.grid_size,
        "spline_order": 3,
        "lr": DEFAULT_LR[args.encoder],
        "max_epochs": 30,
        "patience": 3,
    }
    train(config)
