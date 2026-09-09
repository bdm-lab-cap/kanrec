#!/bin/bash
# Reproduces the full results table in a single command.
# Requires: GPU (optional), datasets in data/delta_parquet/, MongoDB running.
#
# Fix applied (2026-09, auditoría de tribunal — hallazgo B6):
# --encoder raw / autodis / kan-bspline now build three DIFFERENT models
# (see kanrec/baselines.py::build_model). Before this fix, train.py always
# built a KANRecModel regardless of --encoder, so this script trained the
# same model three times under three checkpoint names.
#
# The --encoder kan-rbf ablation and the `streamlit run dashboard/app.py`
# hint below were removed: neither exists in this repository.
set -e

echo "══════════════════════════════════════════════════════"
echo "  KAN-REC — Full experiment suite"
echo "══════════════════════════════════════════════════════"
echo ""

SEEDS="42 123 256"
DATASET="criteo"

# ── 1. Baselines ──────────────────────────────────────────────────────────────
echo "[1/4] Raw normalisation baseline..."
for s in $SEEDS; do
    python experiments/train.py --encoder raw --dataset $DATASET --seed $s
done

echo "[2/4] AutoDis baseline..."
for s in $SEEDS; do
    python experiments/train.py --encoder autodis --dataset $DATASET --seed $s
done

# ── 2. KAN encoder (primary contribution) ──────────────────────────────────────
echo "[3/4] KAN encoder (B-spline, EfficientKAN)..."
for s in $SEEDS; do
    python experiments/train.py --encoder kan-bspline --dataset $DATASET --seed $s
done

# ── 3. Symbolic extraction (all seeds) ─────────────────────────────────────────
echo "[4/4] Symbolic extraction + MongoDB persistence..."
for s in $SEEDS; do
    python experiments/symbolic_extraction.py \
        --checkpoint checkpoints/best_kan-bspline_${DATASET}_gs10_s${s}.pt \
        --dataset $DATASET --seed $s
done

# ── 4. Stability report from MongoDB ───────────────────────────────────────────
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
