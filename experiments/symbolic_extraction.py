"""
Symbolic extraction pipeline: loads a trained model, extracts formulas,
persists to MongoDB, and evaluates faithfulness.

Usage:
    python experiments/symbolic_extraction.py \
        --checkpoint checkpoints/best_kan-bspline_criteo_s42.pt \
        --dataset criteo --seed 42
"""
import argparse
import json
import torch

from kanrec.model import KANRecModel
from kanrec.data import KANRecDataModule
from kanrec.symbolic import SymbolicExtractor
from kanrec.faithfulness import FaithfulnessEvaluator


def run_extraction(args):
    with open("data/feature_selection.json") as f:
        sel = json.load(f)
    field_names = sel["selected"]

    dm = KANRecDataModule(
        train_path=f"data/delta_parquet/{args.dataset}/train",
        val_path=f"data/delta_parquet/{args.dataset}/val",
        test_path=f"data/delta_parquet/{args.dataset}/test",
    )

    model = KANRecModel(
        num_numerical=len(field_names),
        cat_cardinalities=dm.cat_cardinalities,
        embedding_dim=16,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Read MLflow run_id if available
    try:
        run_id_file = args.checkpoint.replace(".pt", "").replace("best_", "run_id_") + ".txt"
        run_id = open(run_id_file).read().strip()
    except FileNotFoundError:
        run_id = "no-mlflow"

    # Symbolic extraction + persist to MongoDB
    extractor = SymbolicExtractor(model, r2_threshold=0.95, l1_percentile=20)
    extractor.extract_and_persist(
        dataset=args.dataset,
        seed=args.seed,
        field_names=field_names,
        run_id=run_id,
    )
    extractor.print_scoring_formula()

    # Faithfulness evaluation
    evaluator = FaithfulnessEvaluator(model, extractor, dm.test_dataloader())
    gap = evaluator.compute_gap_rmse()
    evaluator.update_gap_in_mongo(args.dataset, args.seed, gap)

    # Monotonicity audit: I1 (counter, increasing), I2 (recency, decreasing)
    evaluator.monotonicity_audit({0: "increasing", 1: "decreasing"})

    extractor.close()
    evaluator.close()
    print("\nExtraction complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset",    default="criteo")
    p.add_argument("--seed",       type=int, default=42)
    run_extraction(p.parse_args())
