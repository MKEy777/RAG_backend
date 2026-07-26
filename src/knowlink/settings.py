from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "KNOWLINK_", "env_file": ".env"}

    # database
    database_url: str = "sqlite:///./knowlink.db"

    # redis
    redis_url: str = "redis://localhost:6379/0"

    # celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # storage profile: lite | full
    storage_profile: str = "lite"

    # object storage
    object_storage_root: Path = Path("./data/objects")

    # llm - chat/vlm (opencode.ai)
    llm_base_url: str = "https://opencode.ai/zen/go/v1"
    llm_api_key: str = ""
    llm_chat_model: str = "deepseek-v4-flash"
    llm_vlm_model: str = "mimo-v2.5"
    llm_rerank_model: str = "deepseek-v4-flash"
    llm_rerank_base_url: str = "https://opencode.ai/zen/go/v1"

    # llm - embedding (siliconflow)
    llm_embed_base_url: str = "https://api.siliconflow.cn/v1"
    llm_embed_api_key: str = ""
    llm_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # embedding
    embedding_dims: int = 1024
    batch_embed_size: int = 5  # detail.md §嵌入
    embed_max_chars: int = 20000  # detail.md §嵌入
    embed_retry_count: int = 5  # detail.md §嵌入
    embed_retry_base_ms: int = 200  # detail.md §嵌入

    # chunking
    chunk_size: int = 512  # detail.md §切块
    chunk_overlap: int = 80  # detail.md §切块
    parent_chunk_size: int = 4096  # detail.md §父子分块
    child_chunk_size: int = 384  # detail.md §父子分块
    protect_block_limit: int = 7500  # detail.md §第三层

    # retrieval
    rrf_k: int = 60  # detail.md §混合检索
    rrf_vector_weight: float = 0.7  # detail.md §混合检索
    rrf_keyword_weight: float = 0.3  # detail.md §混合检索
    embedding_top_k: int = 50  # detail.md §混合检索
    vector_threshold: float = 0.15  # detail.md §混合检索
    keyword_threshold: float = 0.3  # detail.md §混合检索
    rerank_top_k: int = 10  # detail.md §混合检索
    rerank_threshold: float = 0.2  # detail.md §混合检索
    wiki_boost: float = 1.3  # detail.md §混合检索

    # wiki
    wiki_debounce_ingest_s: int = 30  # detail.md Wiki §步骤1
    wiki_debounce_followup_s: int = 5  # detail.md Wiki §步骤1
    wiki_debounce_finalize_s: int = 20  # detail.md Wiki §步骤1
    wiki_batch_max_docs: int = 5  # detail.md Wiki §步骤2
    wiki_map_concurrency: int = 10  # detail.md Wiki §步骤3
    wiki_reduce_concurrency: int = 10  # detail.md Wiki §步骤4
    wiki_max_inflight: int = 4  # detail.md Wiki §步骤2
    wiki_slug_lock_ttl_s: int = 300  # detail.md Wiki §步骤4
    wiki_slug_lock_wait_s: int = 120  # detail.md Wiki §步骤4
    wiki_slug_lock_poll_ms: int = 50  # detail.md Wiki §步骤4
    wiki_finalize_lock_ttl_s: int = 60  # detail.md Wiki §步骤6
    wiki_finalize_lock_renew_s: int = 20  # detail.md Wiki §步骤6
    wiki_citation_batch_chars: int = 12000  # detail.md Wiki §步骤3
    wiki_citation_concurrency: int = 4  # detail.md Wiki §步骤3
    wiki_dead_letter_threshold: int = 5  # detail.md Wiki §步骤2
    wiki_llm_retry_count: int = 3  # detail.md Wiki §步骤2
    wiki_llm_retry_base_s: float = 2.0  # detail.md Wiki §步骤2
    wiki_max_source_chars: int = 32768  # detail.md Wiki §步骤3
    wiki_dedup_jaccard: float = 0.08  # detail.md Wiki §步骤4
    wiki_dedup_top_k: int = 5  # detail.md Wiki §步骤4
    wiki_dedup_bypass: int = 25  # detail.md Wiki §步骤4

    # vlm
    vlm_concurrency: int = 3

    # graph
    graph_entity_concurrency: int = 4  # detail.md §知识图谱
    graph_relation_concurrency: int = 4  # detail.md §知识图谱
    graph_relation_batch_size: int = 5  # detail.md §知识图谱
    graph_pmi_weight: float = 0.6  # detail.md §知识图谱
    graph_strength_weight: float = 0.4  # detail.md §知识图谱
    graph_two_hop_decay: float = 0.5  # detail.md §知识图谱
    graph_profile: str = "lite"  # lite | neo4j

    # faq
    faq_max_iterations: int = 5  # detail.md §FAQ
    faq_initial_top_k_multiplier: int = 3  # detail.md §FAQ
    faq_question_concurrency: int = 4  # detail.md §FAQ

    # indexing
    index_batch_size: int = 40  # detail.md §嵌入
    index_concurrency: int = 5  # detail.md §嵌入

    # security
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = [
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".md", ".txt", ".csv", ".zip",
    ]
    rate_limit_per_minute: int = 60
    max_kb_ids_per_query: int = 20


settings = Settings()
