"""
Tests for MongoSymbolicStore.
Requires MongoDB running on localhost:27017 (provided by CI service container).
Uses a separate test collection that is dropped after each test.
"""
import os
import pytest
from kanrec.mongo_store import MongoSymbolicStore

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")


@pytest.fixture
def store():
    s = MongoSymbolicStore(uri=MONGO_URI)
    s.col = s.db["symbolic_results_test"]   # isolated test collection
    s._ensure_indexes()
    yield s
    s.col.drop()
    s.close()


def _insert(store, **kwargs):
    defaults = dict(
        run_id="run-test", dataset="criteo", seed=42,
        field_name="I1", field_idx=0, operator="log",
        param_a=0.7, param_b=0.0, formula_str="0.7·log(|x|+1)",
        r2=0.97, gap_rmse=0.005, is_monotone=True, is_accepted=True,
    )
    defaults.update(kwargs)
    return store.save_result(**defaults)


def test_save_returns_string_id(store):
    doc_id = _insert(store)
    assert isinstance(doc_id, str)
    assert len(doc_id) == 24   # MongoDB ObjectId hex length


def test_get_best_formula_returns_highest_r2(store):
    _insert(store, seed=42, r2=0.97)
    _insert(store, seed=123, r2=0.99)
    best = store.get_best_formula("criteo", "I1")
    assert best is not None
    assert abs(best["r2"] - 0.99) < 1e-6


def test_stability_report_dominant_operator(store):
    for seed in [42, 123, 256]:
        _insert(store, seed=seed, operator="log",  r2=0.97)
    _insert(store, seed=512, operator="sqrt", r2=0.93)

    report = store.stability_report("criteo", seeds=[42, 123, 256, 512])
    top = report[0]
    assert top["operator"]     == "log"
    assert top["seeds_count"]  == 3
    assert abs(top["stability_pct"] - 0.75) < 0.01


def test_cross_dataset_comparison_detects_divergence(store):
    _insert(store, dataset="criteo", field_name="I3", operator="log")
    _insert(store, dataset="avazu",  field_name="I3", operator="sqrt")

    results = store.cross_dataset_comparison("I3")
    datasets = {r["_id"]["dataset"] for r in results}
    assert "criteo" in datasets
    assert "avazu"  in datasets


def test_scoring_formula_summary_format(store):
    for field, op in [("I1", "log"), ("I3", "linear")]:
        for seed in [42, 123]:
            _insert(store, field_name=field, field_idx=0, seed=seed,
                    operator=op, formula_str=f"{op}(x)")
    formula = store.scoring_formula_summary("criteo", seeds=[42, 123])
    assert "I1" in formula
    assert "log" in formula
    assert "where:" in formula


def test_update_gap_rmse(store):
    _insert(store, seed=42, gap_rmse=0.0)
    store.col.update_many(
        {"dataset": "criteo", "seed": 42},
        {"$set": {"gap_rmse": 0.0073}}
    )
    doc = store.get_best_formula("criteo", "I1")
    assert abs(doc["gap_rmse"] - 0.0073) < 1e-6


def test_empty_collection_returns_empty_report(store):
    report = store.stability_report("criteo", seeds=[42])
    assert report == []
