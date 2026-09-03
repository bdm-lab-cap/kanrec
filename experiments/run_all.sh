#!/bin/bash
# Reproduces the full results table in a single command.
# Requires: GPU (optional), datasets in data/delta_parquet/, MongoDB running.
set -e

echo "══════════════════════════════════════════════════════"
echo "  KAN-REC — Full experiment suite"
echo "══════════════════════════════════════════════════════"
echo ""

SEEDS="42 123 256"
DATASET="criteo"

# ── 1. Baselines ──────────────────────────────────────────────────────────────
echo "[1/5] Raw normalisation baseline..."
for s in $SEEDS; do
    python experiments/train.py --encoder raw --dataset $DATASET --seed $s
done

echo "[2/5] AutoDis baseline..."
for s in $SEEDS; do
    python experiments/train.py --encoder autodis --dataset $DATASET --seed $s
done

# ── 2. KAN encoder (primary) ──────────────────────────────────────────────────
echo "[3/5] KAN encoder (B-spline, EfficientKAN)..."
for s in $SEEDS; do
    python experiments/train.py --encoder kan-bspline --dataset $DATASET --seed $s
done

# ── 3. Ablation: basis ────────────────────────────────────────────────────────
echo "[4/5] KAN encoder (RBF/FastKAN) — basis ablation..."
python experiments/train.py --encoder kan-rbf --dataset $DATASET --seed 42

# ── 4. Symbolic extraction (all seeds) ───────────────────────────────────────
echo "[5/5] Symbolic extraction + MongoDB persistence..."
for s in $SEEDS; do
    python experiments/symbolic_extraction.py \
        --checkpoint checkpoints/best_kan-bspline_${DATASET}_s${s}.pt \
        --dataset $DATASET --seed $s
done

# ── 5. Stability report from MongoDB ─────────────────────────────────────────
echo ""
echo "══ Stability report (MongoDB) ══════════════════════"
python - <<'PYEOF'
from kanrec.mongo_store import MongoSymbolicStore
store = MongoSymbolicStore()
report = store.stability_report("criteo", seeds=[42, 123, 256])
print(f"{'Field':<8} {'Operator':<12} {'Seeds':<7} {'Stability':<12} Avg R²")
print("-" * 50)
seen = set()
for r in report:
    if r["field"] not in seen:
        seen.add(r["field"])
        print(f"{r['field']:<8} {r['operator']:<12} {r['seeds_count']:<7} "
              f"{r['stability_pct']:.0%}{'':8} {r['avg_r2']:.4f}")
print()
print(store.scoring_formula_summary("criteo", seeds=[42, 123, 256]))
store.close()
PYEOF

echo ""
echo "══ Done. View MLflow UI: mlflow ui --port 5000 ══════"
echo "══ View dashboard:       streamlit run dashboard/app.py ══"
