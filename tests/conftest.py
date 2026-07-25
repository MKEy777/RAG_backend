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
