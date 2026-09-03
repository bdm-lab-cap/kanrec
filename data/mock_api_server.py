"""
Mock Item Metadata API — simulates a real product catalogue REST service.

Usage:
    pip install fastapi uvicorn
    uvicorn data.mock_api_server:app --port 8000 --reload

Endpoints:
    GET  /items/{item_id}        → single item metadata
    POST /items/batch            → list of item metadata
    GET  /health                 → health check
"""
import hashlib
import random
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Item Metadata API", version="1.0.0")

CATEGORIES = ["electronics", "clothing", "books", "sports",
               "home", "beauty", "food", "toys", "automotive"]


def _metadata(item_id: str) -> dict:
    """Deterministic metadata based on item_id hash — same input always returns same output."""
    seed = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    return {
        "item_id": item_id,
        "price_usd": round(rng.uniform(0.99, 999.99), 2),
        "category": rng.choice(CATEGORIES),
        "popularity_score": round(rng.uniform(0.0, 1.0), 4),
        "days_since_listing": rng.randint(1, 730),
        "avg_rating": round(rng.uniform(1.0, 5.0), 2),
    }


class ItemMetadata(BaseModel):
    item_id: str
    price_usd: float
    category: str
    popularity_score: float
    days_since_listing: int
    avg_rating: float


@app.get("/items/{item_id}", response_model=ItemMetadata)
def get_item(item_id: str):
    return _metadata(item_id)


@app.post("/items/batch", response_model=list[ItemMetadata])
def get_batch(item_ids: list[str]):
    return [_metadata(iid) for iid in item_ids]


@app.get("/health")
def health():
    return {"status": "ok"}
