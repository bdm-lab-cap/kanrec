# Microsoft Fabric Notebook — 08_comparativa_encoders
# Trains Raw, AutoDis and KAN-REC under identical conditions (100K balanced rows).
# Attach kanrec_lakehouse before running.

# CELL 1: Install
# %pip install torch==2.2.2
# %pip install git+https://github.com/Blealtan/efficient-kan.git@7b6ce1c --no-deps
# %pip install scikit-learn scipy numpy==1.26.4

import os, json, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from sklearn.metrics import roc_auc_score, log_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from efficient_kan import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("/lakehouse/default/Files/config/feature_selection.json") as f:
    sel = json.load(f)
NUMERICAL_COLS   = sel["selected"]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]
CKPT_PATH = "/lakehouse/default/Files/checkpoints"
os.makedirs(CKPT_PATH, exist_ok=True)

class CriteoDataset(Dataset):
    def __init__(self, table_name, num_cols, cat_cols, n_rows=100000):
        n_half   = n_rows // 2
        idx_cols = [f"{c}_idx" for c in cat_cols]
        df_pos   = spark.read.table(table_name).filter("label = 1").limit(n_half).toPandas()
        df_neg   = spark.read.table(table_name).filter("label = 0").limit(n_half).toPandas()
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  {table_name}: {len(df):,} rows | CTR={df['label'].mean():.4f}")
        self.x_num = torch.tensor(df[num_cols].fillna(0).values.astype("float32"), dtype=torch.float32)
        avail      = [c for c in idx_cols if c in df.columns]
        self.x_cat = torch.tensor(df[avail].fillna(0).values.astype("int64"), dtype=torch.long) \
                     if avail else torch.zeros(len(df), len(cat_cols), dtype=torch.long)
        self.y = torch.tensor(df["label"].values.astype("float32"), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x_num[i], self.x_cat[i], self.y[i]

train_ds = CriteoDataset("train", NUMERICAL_COLS, CATEGORICAL_COLS, 100000)
val_ds   = CriteoDataset("val",   NUMERICAL_COLS, CATEGORICAL_COLS, 20000)
test_ds  = CriteoDataset("test",  NUMERICAL_COLS, CATEGORICAL_COLS, 20000)
sample   = spark.read.table("train").limit(100000).toPandas()
idx_cols = [f"C{i}_idx" for i in range(1, 27) if f"C{i}_idx" in sample.columns]
cat_cardinalities = [int(sample[c].max()) + 10 for c in idx_cols]

BATCH_SIZE   = 1024
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class RawModel(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim=16):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim, padding_idx=0) for c in cat_cardinalities])
        input_dim = num_numerical + len(cat_cardinalities) * embedding_dim
        self.interaction = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x_num, x_cat):
        cat_flat = torch.stack([emb(x_cat[:, i].clamp(0, emb.num_embeddings-1)) for i, emb in enumerate(self.cat_embeddings)], dim=1).view(x_cat.size(0), -1)
        return self.head(self.interaction(torch.cat([x_num, cat_flat], dim=1)))
    def entropy_reg_loss(self): return torch.tensor(0.0)

class AutoDisEncoder(nn.Module):
    def __init__(self, num_fields, embedding_dim=16, num_buckets=16, temperature=1.0):
        super().__init__()
        self.num_fields = num_fields; self.num_buckets = num_buckets; self.temperature = temperature
        self.bucket_boundaries = nn.Parameter(torch.randn(num_fields, num_buckets-1)*0.1)
        self.meta_embeddings   = nn.Parameter(torch.randn(num_fields, num_buckets, embedding_dim)*0.1)
    def forward(self, x):
        embeddings = []
        for j in range(self.num_fields):
            dist = (x[:, j:j+1] - self.bucket_boundaries[j].unsqueeze(0)) / self.temperature
            w    = F.softmax(torch.cat([torch.sigmoid(dist), torch.ones(x.size(0),1,device=x.device)], dim=1), dim=-1)
            embeddings.append(torch.einsum("bh,hd->bd", w, self.meta_embeddings[j]).unsqueeze(1))
        return torch.cat(embeddings, dim=1)

class AutoDisModel(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim=16, num_buckets=16):
        super().__init__()
        self.numerical_encoder = AutoDisEncoder(num_numerical, embedding_dim, num_buckets)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim, padding_idx=0) for c in cat_cardinalities])
        total_fields = num_numerical + len(cat_cardinalities)
        self.interaction = nn.Sequential(nn.Linear(total_fields*embedding_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x_num, x_cat):
        num_emb  = self.numerical_encoder(x_num)
        cat_embs = [emb(x_cat[:, i].clamp(0, emb.num_embeddings-1)) for i, emb in enumerate(self.cat_embeddings)]
        all_emb  = torch.cat([num_emb, torch.stack(cat_embs, dim=1)], dim=1)
        return self.head(self.interaction(all_emb.view(all_emb.size(0), -1)))
    def entropy_reg_loss(self): return torch.tensor(0.0)

class KANNumericalEncoder(nn.Module):
    def __init__(self, num_fields, embedding_dim=16, grid_size=5, spline_order=3):
        super().__init__()
        self.field_kans = nn.ModuleList([KAN(layers_hidden=[1, embedding_dim], grid_size=grid_size, spline_order=spline_order) for _ in range(num_fields)])
    def forward(self, x):
        return torch.cat([kan(x[:, j:j+1]).unsqueeze(1) for j, kan in enumerate(self.field_kans)], dim=1)

class KANRecModel(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embedding_dim=16, grid_size=5, spline_order=3):
        super().__init__()
        self.numerical_encoder = KANNumericalEncoder(num_numerical, embedding_dim, grid_size, spline_order)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim, padding_idx=0) for c in cat_cardinalities])
        total_fields = num_numerical + len(cat_cardinalities)
        self.interaction = nn.Sequential(nn.Linear(total_fields*embedding_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x_num, x_cat):
        num_emb  = self.numerical_encoder(x_num)
        cat_embs = [emb(x_cat[:, i].clamp(0, emb.num_embeddings-1)) for i, emb in enumerate(self.cat_embeddings)]
        all_emb  = torch.cat([num_emb, torch.stack(cat_embs, dim=1)], dim=1)
        return self.head(self.interaction(all_emb.view(all_emb.size(0), -1)))
    def entropy_reg_loss(self):
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for kan in self.numerical_encoder.field_kans:
            if hasattr(kan, "spline_weight"):
                w = kan.spline_weight.abs(); w_norm = w / (w.sum() + 1e-8)
                reg += -(w_norm * (w_norm + 1e-8).log()).sum()
        return 1e-3 * reg

def train_encoder(name, model, seed=42, max_epochs=10, patience=3):
    torch.manual_seed(seed); model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.BCELoss()
    best_val_auc, patience_ctr = 0.0, 0
    ckpt_file = f"{CKPT_PATH}/best_{name}_criteo_100K_s{seed}.pt"
    print(f"\n{'='*55}\nTraining: {name.upper()} | seed={seed}")
    with mlflow.start_run(run_name=f"{name}-criteo-100K-s{seed}"):
        mlflow.log_params({"encoder": name, "dataset": "criteo-100K", "seed": seed})
        for epoch in range(max_epochs):
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
            mlflow.log_metrics({"val_auc": val_auc}, step=epoch)
            print(f"  Epoch {epoch+1:02d} | val_auc={val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc; patience_ctr = 0
                torch.save(model.state_dict(), ckpt_file)
            else:
                patience_ctr += 1
                if patience_ctr >= patience: print(f"  Early stopping epoch {epoch+1}"); break
        model.load_state_dict(torch.load(ckpt_file, map_location=device)); model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                p = model(x_num.to(device), x_cat.to(device)).squeeze().cpu().numpy()
                test_preds.extend(p); test_labels.extend(y.numpy())
        test_auc = roc_auc_score(test_labels, test_preds)
        test_ll  = log_loss(test_labels, test_preds)
        mlflow.log_metrics({"test_auc": test_auc, "test_logloss": test_ll})
        print(f"\n  TEST AUC={test_auc:.4f} | Log-loss={test_ll:.4f}")
        return {"encoder": name, "seed": seed, "test_auc": test_auc, "test_logloss": test_ll}

mlflow.set_experiment("kanrec-comparativa-100K")
all_results = []
all_results.append(train_encoder("raw",         RawModel(len(NUMERICAL_COLS), cat_cardinalities)))
all_results.append(train_encoder("autodis",     AutoDisModel(len(NUMERICAL_COLS), cat_cardinalities)))
all_results.append(train_encoder("kan-bspline", KANRecModel(len(NUMERICAL_COLS), cat_cardinalities)))

print(f"\n{'='*55}\nFINAL COMPARISON — 100K balanced (seed=42)")
print(f"{'Encoder':<15} {'AUC':>10} {'Log-loss':>10}")
for r in sorted(all_results, key=lambda x: x['test_auc']):
    print(f"  {r['encoder']:<13} {r['test_auc']:>10.4f} {r['test_logloss']:>10.4f}")

kanrec_auc  = next(r['test_auc'] for r in all_results if r['encoder']=='kan-bspline')
autodis_auc = next(r['test_auc'] for r in all_results if r['encoder']=='autodis')
print(f"\n  KAN-REC vs AutoDis: +{kanrec_auc-autodis_auc:.4f} AUC")

spark.createDataFrame(pd.DataFrame(all_results)).write.format("delta").mode("overwrite").save("Tables/experiment_results")
print("✓ experiment_results updated in OneLake")
