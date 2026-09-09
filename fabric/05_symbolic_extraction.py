# Microsoft Fabric Notebook — 05_symbolic_extraction
# Extracts symbolic scoring rules from trained KAN-REC models.
# Run AFTER 04_training_kanrec. Attach kanrec_lakehouse before running.

# CELL 1: Install
# %pip install torch==2.2.2
# %pip install git+https://github.com/Blealtan/efficient-kan.git@7b6ce1c --no-deps
# %pip install scipy numpy==1.26.4

import json, os, torch
import numpy as np
import torch.nn as nn
import pandas as pd
from scipy.optimize import curve_fit
from collections import Counter
from efficient_kan import KAN

CKPT_PATH = "/lakehouse/default/Files/checkpoints"
with open("/lakehouse/default/Files/config/feature_selection.json") as f:
    sel = json.load(f)
NUMERICAL_COLS   = sel["selected"]
CATEGORICAL_COLS = [f"C{i}" for i in range(1, 27)]

# Operator library
OPERATOR_LIBRARY = {
    "log":     lambda x, a, b: a * np.log(np.abs(x) + 1) + b,
    "exp":     lambda x, a, b: a * np.exp(np.clip(x, -10, 10)) + b,
    "square":  lambda x, a, b: a * x**2 + b,
    "sqrt":    lambda x, a, b: a * np.sqrt(np.abs(x)) + b,
    "inverse": lambda x, a, b: a / (np.abs(x) + 1e-6) + b,
    "sigmoid": lambda x, a, b: a / (1 + np.exp(-x)) + b,
    "linear":  lambda x, a, b: a * x + b,
}
FORMULA_TEMPLATES = {
    "log": "{a}·log(|x|+1) + {b}", "exp": "{a}·exp(x) + {b}",
    "square": "{a}·x² + {b}", "sqrt": "{a}·√|x| + {b}",
    "inverse": "{a}/(|x|+ε) + {b}", "sigmoid": "{a}·σ(x) + {b}", "linear": "{a}·x + {b}",
}

# Model definition (same as training)
class KANNumericalEncoder(nn.Module):
    def __init__(self, num_fields, embedding_dim=16, grid_size=5, spline_order=3):
        super().__init__()
        self.field_kans = nn.ModuleList([
            KAN(layers_hidden=[1, embedding_dim], grid_size=grid_size, spline_order=spline_order)
            for _ in range(num_fields)])
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
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, embedding_dim, padding_idx=0) for card in cat_cardinalities])
        total_fields = num_numerical + len(cat_cardinalities)
        self.interaction = nn.Sequential(nn.Linear(total_fields * embedding_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x_num, x_cat):
        num_emb = self.numerical_encoder(x_num)
        cat_embs = [emb(x_cat[:, i].clamp(0, emb.num_embeddings-1)) for i, emb in enumerate(self.cat_embeddings)]
        all_emb = torch.cat([num_emb, torch.stack(cat_embs, dim=1)], dim=1)
        return self.head(self.interaction(all_emb.view(all_emb.size(0), -1)))
    def entropy_reg_loss(self): return torch.tensor(0.0)

def fit_field(model, field_idx, r2_threshold=0.90):
    x_grid, y_curves = model.numerical_encoder.get_spline_curves(field_idx)
    x_np, y_np = x_grid.numpy(), y_curves[:, 0].numpy()
    best = {"operator": None, "r2": -1.0, "params": None, "formula": "?", "accepted": False}
    for name, fn in OPERATOR_LIBRARY.items():
        try:
            params, _ = curve_fit(fn, x_np, y_np, maxfev=5000)
            y_pred = fn(x_np, *params)
            r2 = float(1 - np.sum((y_np - y_pred)**2) / (np.sum((y_np - y_np.mean())**2) + 1e-10))
            if r2 > best["r2"]:
                a, b = round(float(params[0]), 4), round(float(params[1]), 4)
                best = {"operator": name, "r2": r2, "params": [float(params[0]), float(params[1])],
                        "formula": FORMULA_TEMPLATES[name].format(a=a, b=b), "accepted": r2 >= r2_threshold}
        except: continue
    return best

# Extract for all seeds
results_all_seeds = {}
for seed in [42, 123, 256]:
    ckpt = f"{CKPT_PATH}/best_kan-bspline_criteo_s{seed}.pt"
    if not os.path.exists(ckpt): print(f"Missing: {ckpt}"); continue
    print(f"\n{'='*50}\nSeed {seed}")
    state_dict = torch.load(ckpt, map_location="cpu")
    cat_cardinalities = [state_dict[f"cat_embeddings.{i}.weight"].shape[0] for i in range(26)]
    model = KANRecModel(len(NUMERICAL_COLS), cat_cardinalities, 16, 5, 3)
    model.load_state_dict(state_dict); model.eval()
    norms = model.numerical_encoder.get_edge_norms()
    surviving = [j for j, n in enumerate(norms) if n >= np.percentile(norms, 20)]
    print(f"L1 pruning: {len(surviving)}/{len(norms)} fields survive")
    seed_results = {}
    for j in surviving:
        r = fit_field(model, j)
        field = NUMERICAL_COLS[j]
        seed_results[field] = r
        print(f"  {'✓' if r['accepted'] else '✗'} {field}: {r['operator']:8s} R²={r['r2']:.4f}  {r['formula']}")
    results_all_seeds[seed] = seed_results

# Stability report
print("\nSTABILITY REPORT")
print("="*50)
field_ops = {}
for seed, results in results_all_seeds.items():
    for field, r in results.items():
        if r["accepted"]: field_ops.setdefault(field, []).append(r["operator"])

for field, ops in sorted(field_ops.items()):
    dominant, count = Counter(ops).most_common(1)[0]
    print(f"  {field:<6}: {dominant:<10} {count}/3 seeds ({count/3:.0%})")

# Save results
os.makedirs("/lakehouse/default/Files/results", exist_ok=True)
with open("/lakehouse/default/Files/results/symbolic_results.json", "w") as f:
    json.dump({"results_by_seed": {str(k): {fld: {kk: vv for kk, vv in v.items() if kk != "params"} | {"params": v["params"]}
               for fld, v in res.items()} for k, res in results_all_seeds.items()},
               "stability": {f: {"dominant": Counter(ops).most_common(1)[0][0], "stability": Counter(ops).most_common(1)[0][1]/3}
                             for f, ops in field_ops.items()}}, f, indent=2)
print("\nsymbolic_results.json saved to Files/results/")
