# conftest.py — shared pytest fixtures and configuration
# MongoDB tests are skipped automatically if the server is not reachable.

import pytest
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_mongo: mark test as requiring a running MongoDB instance"
    )


@pytest.fixture(scope="session")
def mongo_available():
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1000)
        client.admin.command("ping")
        client.close()
        return True
    except ConnectionFailure:
        return False
