from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pyrag.models.base import Base


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_markdown() -> str:
    return """# Introduction

This is a test document about RAG systems.

## Section 1

Retrieval-Augmented Generation combines search with generation.

## Section 2

The system uses vector embeddings and BM25 for hybrid retrieval.
"""


@pytest.fixture
def fake_llm_factory():
    class FakeLLM:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        async def chat(self, messages, model=None, temperature=0.3, max_tokens=4096, purpose=""):
            self.calls.append({"messages": messages, "model": model, "purpose": purpose})
            if not self.responses:
                raise AssertionError(f"FakeLLM exhausted, purpose={purpose}")
            return self.responses.pop(0)

    return FakeLLM


@pytest.fixture
def fake_storage():
    class FakeLock:
        async def acquire(self, key, ttl_s=0):
            return True

        async def release(self, key):
            pass

    class FakeStorage:
        lock = FakeLock()

    return FakeStorage()
