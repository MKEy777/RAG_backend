"""对已入库 KB 跑端到端检索评估：hit@k 与 MRR。

用法: uv run python scripts/evaluate_retrieval.py --kb <kb_id> [--dataset tests/golden/retrieval_eval.json] [--top-k 10]
命中判定: top_k 内任一 hit 的 document_id 在 expect_doc_ids，或 content 含全部 expect_keywords。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knowlink.models.database import SessionLocal  # noqa: E402
from knowlink.retrieval.pipeline import retrieve_candidates  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser(description="KnowLink 检索评估")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parents[1] / "tests" / "golden" / "retrieval_eval.json"))
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    session = SessionLocal()
    hits_at_k = 0
    mrr_total = 0.0
    try:
        for case in cases:
            result = await retrieve_candidates(session, [args.kb], case["query"], top_k=args.top_k)
            rank = 0
            for i, c in enumerate(result.hits, start=1):
                doc_ok = c.document_id in case.get("expect_doc_ids", [])
                kws = case.get("expect_keywords", [])
                kw_ok = bool(kws) and all(k in (c.content or "") for k in kws)
                if doc_ok or kw_ok:
                    rank = i
                    break
            if rank:
                hits_at_k += 1
                mrr_total += 1.0 / rank
            print(f"[{'HIT' if rank else 'MISS'}] rank={rank or '-'} {case['query']}")
        n = len(cases) or 1
        print(f"\nhit@{args.top_k}: {hits_at_k}/{len(cases)} = {hits_at_k / n:.2%}  MRR: {mrr_total / n:.4f}")
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
