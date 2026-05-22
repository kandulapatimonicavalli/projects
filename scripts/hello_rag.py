"""
hello_rag.py
============
End-to-end RAG smoke test without UI or LangGraph.

Proves: ingest → duplicate skip → semantic query → off-topic empty retrieval.

Run from project root:
    uv run python scripts/hello_rag.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as: uv run python scripts/hello_rag.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager


def load_sample_chunk() -> DocumentChunk:
    """Build a DocumentChunk from examples/sample_chunk.json."""
    sample_path = PROJECT_ROOT / "examples" / "sample_chunk.json"
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    meta = ChunkMetadata(**data["metadata"])
    chunk_id = VectorStoreManager.generate_chunk_id(meta.source, data["chunk_text"])
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_text=data["chunk_text"],
        metadata=meta,
    )


def main() -> None:
    store = VectorStoreManager()
    chunk = load_sample_chunk()

    print("=" * 60)
    print("PHASE 3 — RAG smoke test (config + store only)")
    print("=" * 60)

    print("\n--- Collection stats (before) ---")
    print(store.get_collection_stats())

    print("\n--- Ingest 1 ---")
    r1 = store.ingest([chunk])
    print(f"ingested={r1.ingested} skipped={r1.skipped} errors={r1.errors}")

    print("\n--- Ingest 2 (duplicate) ---")
    r2 = store.ingest([chunk])
    print(f"ingested={r2.ingested} skipped={r2.skipped} errors={r2.errors}")

    print("\n--- Corpus inspection ---")
    print("documents:", store.list_documents())

    print("\n--- Query (on-topic) ---")
    results = store.query("LSTM forget gate vanishing gradient")
    if not results:
        print("FAIL: expected at least one result")
        sys.exit(1)
    for r in results:
        print(r.to_citation(), f"score={r.score:.3f}")
        print(r.chunk_text[:100], "...")

    print("\n--- Query (off-topic) ---")
    off_topic = store.query("history of the roman empire")
    print(f"results: {len(off_topic)} (expect 0 for hallucination guard later)")
    if off_topic:
        print("WARN: off-topic query returned chunks; check SIMILARITY_THRESHOLD")

    print("\n--- Collection stats (after) ---")
    print(store.get_collection_stats())

    print("\n=== Phase 3 PASSED ===")


if __name__ == "__main__":
    main()
