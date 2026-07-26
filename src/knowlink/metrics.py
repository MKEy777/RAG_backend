from __future__ import annotations

import time
from functools import wraps

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "knowlink_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "knowlink_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

LLM_CALL_COUNT = Counter(
    "knowlink_llm_calls_total",
    "Total LLM API calls",
    ["endpoint", "status"],
)

LLM_CALL_DURATION = Histogram(
    "knowlink_llm_call_duration_seconds",
    "LLM API call duration",
    ["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

EMBED_BATCH_SIZE = Histogram(
    "knowlink_embed_batch_size",
    "Embedding batch size",
    buckets=(1, 5, 10, 20, 50, 100),
)

RETRIEVAL_HITS = Histogram(
    "knowlink_retrieval_hits",
    "Number of retrieval hits per query",
    buckets=(0, 1, 3, 5, 10, 20, 50),
)

PIPELINE_DURATION = Histogram(
    "knowlink_pipeline_duration_seconds",
    "Indexing pipeline duration",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

CIRCUIT_BREAKER_STATE = Counter(
    "knowlink_circuit_breaker_transitions_total",
    "Circuit breaker state transitions",
    ["breaker", "to_state"],
)


def metrics_response():
    return generate_latest()


def metrics_content_type():
    return CONTENT_TYPE_LATEST
