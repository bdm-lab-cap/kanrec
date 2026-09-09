# Microsoft Fabric Notebook — 09_mongodb_vector_search
# Generates KAN embeddings and uploads to MongoDB Atlas Vector Search.
# Run AFTER 04_training_kanrec. Attach kanrec_lakehouse before running.

# CELL 1: Install
# %pip install "pymongo[srv]"
# %pip install git+https://github.com/Blealtan/efficient-kan.git@7b6ce1c --no-deps

import os, json, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from pymongo.mongo_client import MongoClient
from efficient_kan import KAN

ATLAS_URI  = "mongodb+srv://***USER-ROTATED***:***CREDENTIAL-ROTATED***@clusterkanrec.wnwt3ze.mongodb.net/?appName=ClusterKanrec"
CKPT_PATH  = "/lakehouse/default/Files/checkpoints/best_kan-bspline_criteo_s42.pt"
NUMERICAL_COLS   = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]

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

state_dict = torch.load(CKPT_PATH, map_location="cpu")
cat_cards  = [state_dict[f"cat_embeddings.{i}.weight"].shape[0] for i in range(26)]
model = KANRecModel(len(NUMERICAL_COLS), cat_cards, 16, 5, 3)
model.load_state_dict(state_dict); model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Load raw data from CSV
raw = (spark.read.option("sep","\t").option("inferSchema","true")
       .csv("Files/raw/criteo_10m.tsv")
       .toDF("label", *NUMERICAL_COLS, *CATEGORICAL_COLS)
       .limit(5000).toPandas())
for c in NUMERICAL_COLS:
    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).clip(lower=0).astype("float32")
for c in [f"I{i}" for i in range(1,6)]:
    raw[c] = np.log1p(raw[c])

idx_cols = [f"C{i}_idx" for i in range(1, 27) if f"C{i}_idx" in spark.read.table("test").columns]
df_idx   = spark.read.table("test").limit(5000).toPandas()

x_num = torch.tensor(raw[NUMERICAL_COLS].values.astype("float32"), dtype=torch.float32)
x_cat = torch.tensor(df_idx[idx_cols].fillna(0).values.astype("int64"), dtype=torch.long)

print("Generating KAN embeddings...")
all_embeddings = []
BATCH = 256
with torch.no_grad():
    for i in range(0, len(x_num), BATCH):
        num_emb  = model.numerical_encoder(x_num[i:i+BATCH])  # [batch, 13, 16]
        emb_mean = num_emb.mean(dim=-1)                        # [batch, 13]
        norms    = emb_mean.norm(dim=-1, keepdim=True)
        emb_norm = (emb_mean / (norms + 1e-8)).numpy()
        all_embeddings.extend(emb_norm.tolist())

print(f"✓ {len(all_embeddings):,} embeddings generated | dims=13 | mean={np.array(all_embeddings).mean():.4f}")

client = MongoClient(ATLAS_URI)
col    = client["kanrec"]["item_embeddings"]
col.delete_many({})

BATCH_SIZE = 500
docs = []
for i in range(len(all_embeddings)):
    docs.append({"item_id": f"criteo_{i:06d}", "embedding": all_embeddings[i],
                 "label": int(raw.iloc[i]["label"]),
                 "I1": float(raw.iloc[i]["I1"]), "I3": float(raw.iloc[i]["I3"]),
                 "I11": float(raw.iloc[i]["I11"]), "source": "kan-encoder-real"})
    if len(docs) == BATCH_SIZE:
        col.insert_many(docs); docs = []
        print(f"  {i+1:,} embeddings uploaded...")
if docs: col.insert_many(docs)

print(f"\n✓ {col.count_documents({}):,} KAN embeddings in MongoDB Atlas")

# Demo Vector Search
query_emb = all_embeddings[0]
pipeline = [
    {"$vectorSearch": {"index": "kanrec_vector_index", "path": "embedding",
                       "queryVector": query_emb, "numCandidates": 50, "limit": 5}},
    {"$project": {"item_id":1,"label":1,"I1":1,"I3":1,"I11":1,"score":{"$meta":"vectorSearchScore"},"_id":0}}
]
print("\n── Top 5 similar items ($vectorSearch) ────────────────────────")
print(f"{'Item ID':<15} {'Score':>8} {'Label':>6} {'I1':>6} {'I3':>6} {'I11':>6}")
for r in col.aggregate(pipeline):
    print(f"  {r['item_id']:<13} {r['score']:>8.4f} {r['label']:>6} {r.get('I1',0):>6.2f} {r.get('I3',0):>6.2f} {r.get('I11',0):>6.2f}")
client.close()
print("\n✓ KAN Vector Search operational from Microsoft Fabric")
