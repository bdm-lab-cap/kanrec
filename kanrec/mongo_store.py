"""
MongoSymbolicStore — persists and queries symbolic extraction results.

Collection: kanrec.symbolic_results
Schema:
    run_id, dataset, seed, field_name, field_idx,
    operator, param_a, param_b, formula_str,
    r2, gap_rmse, is_monotone, is_accepted,
    importance_chisq_rank, importance_spearman,
    timestamp
"""
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection


class MongoSymbolicStore:
    """
    Args:
        uri:         MongoDB connection string (default: localhost:27017).
        db_name:     Database name (default: kanrec).
        collection:  Collection name (default: symbolic_results).
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "kanrec",
        collection: str = "symbolic_results",
    ):
        self.client = MongoClient(uri)
        self.db     = self.client[db_name]
        self.col: Collection = self.db[collection]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.col.create_index([("dataset", ASCENDING), ("seed", ASCENDING)])
        self.col.create_index([("field_name", ASCENDING), ("operator", ASCENDING)])
        self.col.create_index([("timestamp", DESCENDING)])
        self.col.create_index([("run_id", ASCENDING)])

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_result(
        self,
        run_id: str,
        dataset: str,
        seed: int,
        field_name: str,
        field_idx: int,
        operator: Optional[str],
        param_a: Optional[float],
        param_b: Optional[float],
        formula_str: str,
        r2: float,
        gap_rmse: float,
        is_monotone: bool,
        is_accepted: bool,
        importance_chisq_rank: Optional[int] = None,
        importance_spearman: Optional[float] = None,
    ) -> str:
        doc = {
            "run_id":                run_id,
            "dataset":               dataset,
            "seed":                  seed,
            "field_name":            field_name,
            "field_idx":             field_idx,
            "operator":              operator,
            "param_a":               param_a,
            "param_b":               param_b,
            "formula_str":           formula_str,
            "r2":                    r2,
            "gap_rmse":              gap_rmse,
            "is_monotone":           is_monotone,
            "is_accepted":           is_accepted,
            "importance_chisq_rank": importance_chisq_rank,
            "importance_spearman":   importance_spearman,
            "timestamp":             datetime.now(timezone.utc),
        }
        result = self.col.insert_one(doc)
        return str(result.inserted_id)

    # ── Read / aggregate ──────────────────────────────────────────────────────

    def stability_report(self, dataset: str, seeds: list[int]) -> list[dict]:
        """Aggregates operator frequency per field across seeds."""
        pipeline = [
            {"$match": {"dataset": dataset, "seed": {"$in": seeds},
                        "is_accepted": True}},
            {"$group": {
                "_id":          {"field": "$field_name", "op": "$operator"},
                "count":        {"$sum": 1},
                "avg_r2":       {"$avg": "$r2"},
                "avg_gap_rmse": {"$avg": "$gap_rmse"},
            }},
            {"$sort": {"_id.field": 1, "count": -1}},
        ]
        rows = []
        for r in self.col.aggregate(pipeline):
            rows.append({
                "field":        r["_id"]["field"],
                "operator":     r["_id"]["op"],
                "seeds_count":  r["count"],
                "stability_pct": r["count"] / len(seeds),
                "avg_r2":       round(r["avg_r2"], 4),
                "avg_gap_rmse": round(r["avg_gap_rmse"], 6),
            })
        return rows

    def cross_dataset_comparison(self, field_name: str) -> list[dict]:
        """Compares the dominant operator for a field across datasets."""
        pipeline = [
            {"$match": {"field_name": field_name, "is_accepted": True}},
            {"$group": {
                "_id":    {"dataset": "$dataset", "op": "$operator"},
                "count":  {"$sum": 1},
                "avg_r2": {"$avg": "$r2"},
            }},
            {"$sort": {"_id.dataset": 1, "count": -1}},
        ]
        return list(self.col.aggregate(pipeline))

    def get_best_formula(
        self, dataset: str, field_name: str
    ) -> Optional[dict]:
        """Returns the document with the highest R² for a field+dataset."""
        return self.col.find_one(
            {"dataset": dataset, "field_name": field_name, "is_accepted": True},
            sort=[("r2", DESCENDING)],
        )

    def scoring_formula_summary(
        self, dataset: str, seeds: list[int]
    ) -> str:
        """Builds the human-readable scoring formula from MongoDB results."""
        report = self.stability_report(dataset, seeds)
        seen, lines = set(), []
        for r in report:
            if r["field"] in seen:
                continue
            seen.add(r["field"])
            best = self.get_best_formula(dataset, r["field"])
            if best:
                lines.append(
                    f"  φ_{r['field']}(x) ≈ {best['formula_str']}"
                    f"   [stability {r['seeds_count']}/{len(seeds)}  R²={r['avg_r2']}]"
                )
        fields = sorted(seen)
        return f"ŷ ≈ f({', '.join(fields)}) where:\n" + "\n".join(lines)

    def close(self):
        self.client.close()
