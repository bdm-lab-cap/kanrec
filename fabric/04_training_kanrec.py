# Microsoft Fabric Notebook — 04_training_kanrec
# Trains the KAN-REC model using PyTorch + EfficientKAN on Fabric.
# Attach kanrec_lakehouse before running.

# CELL 1: Install dependencies
# %pip install torch==2.2.2 torchvision==0.17.2
# %pip install git+https://github.com/Blealtan/efficient-kan.git@7b6ce1c --no-deps
# %pip install scikit-learn scipy numpy==1.26.4

# CELL 2: Imports
import os, json, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import mlflow
from sklearn.metrics import roc_auc_score, log_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from efficient_kan import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__} | Device: {device}")

with open("/lakehouse/default/Files/config/feature_selection.json") as f:
    sel = json.load(f)
NUMERICAL_COLS   = sel["selected"]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
CKPT_PATH = "/lakehouse/default/Files/checkpoints"
os.makedirs(CKPT_PATH, exist_ok=True)
print(f"Numerical fields: {NUMERICAL_COLS}")

# CELL 3: Dataset
class CriteoDataset(Dataset):
    def __init__(self, table_name, num_cols, cat_cols, n_rows=100000):
        print(f"  Loading {table_name}...")
        n_half = n_rows // 2
        df_pos = spark.read.table(table_name).filter("label = 1").limit(n_half).toPandas()
        df_neg = spark.read.table(table_name).filter("label = 0").limit(n_half).toPandas()
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  {table_name}: {len(df):,} rows | pos={len(df_pos):,} | neg={len(df_neg):,}")
        self.x_num = torch.tensor(df[num_cols].fillna(0).values.astype("float32"), dtype=torch.float32)
        idx_cols = [f"{c}_idx" for c in cat_cols if f"{c}_idx" in df.columns]
        self.x_cat = torch.tensor(df[idx_cols].fillna(0).values.astype("int64"), dtype=torch.long) if idx_cols \
                     else torch.zeros(len(df), len(cat_cols), dtype=torch.long)
        self.y = torch.tensor(df["label"].values.astype("float32"), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x_num[idx], self.x_cat[idx], self.y[idx]

train_ds = CriteoDataset("train", NUMERICAL_COLS, CATEGORICAL_COLS, 100000)
val_ds   = CriteoDataset("val",   NUMERICAL_COLS, CATEGORICAL_COLS, 20000)
test_ds  = CriteoDataset("test",  NUMERICAL_COLS, CATEGORICAL_COLS, 20000)

n_cat = train_ds.x_cat.shape[1]
cat_cardinalities = [max(int(train_ds.x_cat[:, i].max()), int(val_ds.x_cat[:, i].max()),
                         int(test_ds.x_cat[:, i].max())) + 10 for i in range(n_cat)]

BATCH_SIZE = 1024
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# CELL 4: Model
class KANNumericalEncoder(nn.Module):
    def __init__(self, num_fields, embedding_dim=16, grid_size=5, spline_order=3):
        super().__init__()
        self.field_kans = nn.ModuleList([
            KAN(layers_hidden=[1, embedding_dim], grid_size=grid_size, spline_order=spline_order)
            for _ in range(num_fields)
        ])
    def forward(self, x):
        return torch.cat([kan(x[:, j:j+1]).unsqueeze(1) for j, kan in enumerate(self.field_kans)], dim=1)
    def get_spline_curves(self, field_idx, n_points=300):
        x_grid = torch.linspace(-3.0, 3.0, n_points).unsqueeze(1)
        with torch.no_grad(): y = self.field_kans[field_idx](x_grid)
        return x_grid.squeeze(), y
    def get_edge_norms(self):
        return [sum(p.abs().sum().item() for p in kan.parameters()) for kan in self.field_kans]

class KANRecModel(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim=16, grid_size=5, spline_order=3):
        super().__init__()
        self.numerical_encoder = KANNumericalEncoder(num_numerical, embedding_dim, grid_size, spline_order)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim, padding_idx=0) for card in cat_cardinalities])
        total_fields = num_numerical + len(cat_cardinalities)
        self.interaction = nn.Sequential(
            nn.Linear(total_fields * embedding_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x_num, x_cat):
        num_emb = self.numerical_encoder(x_num)
        cat_embs = [emb(x_cat[:, i].clamp(0, emb.num_embeddings-1)) for i, emb in enumerate(self.cat_embeddings)]
        all_emb = torch.cat([num_emb, torch.stack(cat_embs, dim=1)], dim=1)
        return self.head(self.interaction(all_emb.view(all_emb.size(0), -1)))
    def entropy_reg_loss(self):
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for kan in self.numerical_encoder.field_kans:
            if hasattr(kan, "spline_weight"):
                w = kan.spline_weight.abs(); w_norm = w / (w.sum() + 1e-8)
                reg += -(w_norm * (w_norm + 1e-8).log()).sum()
        return 1e-3 * reg

# CELL 5: Train all seeds
for SEED in [42, 123, 256]:
    torch.manual_seed(SEED)
    model = KANRecModel(len(NUMERICAL_COLS), cat_cardinalities, 16, 5, 3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.BCELoss()
    best_val_auc, patience_ctr = 0.0, 0

    print(f"\n{'='*50}\nTraining seed {SEED}...")
    with mlflow.start_run(run_name=f"kan-bspline-criteo-s{SEED}"):
        mlflow.log_params({"encoder":"kan-bspline","dataset":"criteo","seed":SEED,"embedding_dim":16,"grid_size":5})
        for epoch in range(10):
            model.train(); train_loss = 0.0
            for x_num, x_cat, y in train_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_num, x_cat).squeeze(), y) + model.entropy_reg_loss()
                loss.backward(); optimizer.step(); train_loss += loss.item()

            model.eval(); val_preds, val_labels = [], []
            with torch.no_grad():
                for x_num, x_cat, y in val_loader:
                    p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                    val_preds.extend(p); val_labels.extend(y.numpy())
            val_auc = roc_auc_score(val_labels, val_preds)
            scheduler.step(1 - val_auc)
            mlflow.log_metrics({"train_loss": train_loss, "val_auc": val_auc}, step=epoch)
            print(f"  Epoch {epoch+1:02d} | val_auc={val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc; patience_ctr = 0
                torch.save(model.state_dict(), f"{CKPT_PATH}/best_kan-bspline_criteo_s{SEED}.pt")
            else:
                patience_ctr += 1
                if patience_ctr >= 3: print(f"  Early stopping at epoch {epoch+1}"); break

        model.load_state_dict(torch.load(f"{CKPT_PATH}/best_kan-bspline_criteo_s{SEED}.pt", map_location=device))
        model.eval(); test_preds, test_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                test_preds.extend(p); test_labels.extend(y.numpy())
        test_auc = roc_auc_score(test_labels, test_preds)
        test_ll  = log_loss(test_labels, test_preds)
        mlflow.log_metrics({"test_auc": test_auc, "test_logloss": test_ll})
        print(f"  TEST AUC={test_auc:.4f} | Log-loss={test_ll:.4f}")

print("\nAll seeds trained. Checkpoints saved in Files/checkpoints/")
