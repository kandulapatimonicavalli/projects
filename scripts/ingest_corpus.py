"""
ingest_corpus.py
================
Chunk all markdown/PDF files in data/corpus/ and ingest into ChromaDB.

Run from project root:
    uv run python scripts/ingest_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


def main() -> None:
    settings = get_settings()
    corpus_dir = Path(settings.corpus_dir)
    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    files = sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.pdf"))
    if not files:
        print(f"No .md or .pdf files in {corpus_dir}")
        sys.exit(1)

    chunker = DocumentChunker(settings)
    store = VectorStoreManager(settings)

    print("=" * 60)
    print("PHASE 5 — Ingest corpus")
    print("=" * 60)

    total_ingested = 0
    total_skipped = 0

    for file_path in files:
        # Replace prior chunks for this file so re-ingest stays idempotent.
        store.delete_document(file_path.name)
        chunks = chunker.chunk_file(file_path)
        result = store.ingest(chunks)
        total_ingested += result.ingested
        total_skipped += result.skipped
        print(
            f"{file_path.name}: {len(chunks)} chunks, "
            f"ingested={result.ingested}, skipped={result.skipped}"
        )

    print("\n--- Collection stats ---")
    print(store.get_collection_stats())
    print(f"\nTotal ingested this run: {total_ingested}, skipped: {total_skipped}")
    print("=== Phase 5 ingest complete ===")


if __name__ == "__main__":
    main()
