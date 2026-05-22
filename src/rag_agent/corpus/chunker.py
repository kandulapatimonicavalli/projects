"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager

# Filename stem (lowercase) → canonical topic label
_TOPIC_FROM_STEM: dict[str, str] = {
    "ann": "ANN",
    "cnn": "CNN",
    "rnn": "RNN",
    "lstm": "LSTM",
    "seq2seq": "Seq2Seq",
    "autoencoder": "Autoencoder",
    "gan": "GAN",
    "som": "SOM",
    "boltzmann": "BoltzmannMachine",
    "boltzmannmachine": "BoltzmannMachine",
    "alexnet": "CNN",
    "lenet": "CNN",
}

_BONUS_TOPICS = {"GAN", "SOM", "BoltzmannMachine"}

_RELATED_TOPICS: dict[str, list[str]] = {
    "ANN": ["backpropagation", "activation_functions", "gradient_descent"],
    "CNN": ["convolution", "pooling", "image_classification"],
    "RNN": ["LSTM", "BPTT", "vanishing_gradient", "sequences"],
    "LSTM": ["RNN", "vanishing_gradient", "Seq2Seq"],
    "Seq2Seq": ["RNN", "LSTM", "attention"],
    "Autoencoder": ["dimensionality_reduction", "representation_learning"],
    "GAN": ["generative_models", "autoencoder"],
    "SOM": ["unsupervised_learning", "clustering"],
    "BoltzmannMachine": ["energy_based_models", "unsupervised_learning"],
}

_VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
_MIN_CHUNK_CHARS = 80


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    _MD_HEADERS = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """Load a file and split it into DocumentChunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".md":
            raw_pieces = self._chunk_markdown(path, chunk_size, chunk_overlap)
        elif suffix == ".pdf":
            raw_pieces = self._chunk_pdf(path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .md or .pdf")

        metadata = self._infer_metadata(path, metadata_overrides)
        chunks: list[DocumentChunk] = []

        for piece in raw_pieces:
            body = piece["text"].strip()
            if len(body) < _MIN_CHUNK_CHARS:
                continue

            text = self._embed_chunk_text(
                body,
                metadata,
                header=piece.get("header"),
                page=piece.get("page"),
            )
            chunk_id = VectorStoreManager.generate_chunk_id(metadata.source, text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info("Chunked '{}' → {} chunks", path.name, len(chunks))
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """Chunk multiple files; continues on per-file errors."""
        all_chunks: list[DocumentChunk] = []

        for file_path in file_paths:
            try:
                all_chunks.extend(
                    self.chunk_file(file_path, metadata_overrides=metadata_overrides)
                )
            except Exception as exc:
                logger.error("Failed to chunk {}: {}", file_path, exc)

        return all_chunks

    def _embed_chunk_text(
        self,
        body: str,
        metadata: ChunkMetadata,
        header: dict | None = None,
        page: int | None = None,
    ) -> str:
        """
        Prepend document context so embeddings match short topic queries.

        Standard contextual-retrieval pattern — same treatment for every
        topic (ANN, CNN, RNN, PDF uploads, etc.), not query-specific logic.
        """
        labels: list[str] = [f"Topic: {metadata.topic}"]

        if header:
            doc_title = (header.get("Header 1") or "").strip()
            section = (header.get("Header 2") or header.get("Header 3") or "").strip()
            if doc_title:
                labels.append(f"Document: {doc_title}")
            if section:
                labels.append(f"Section: {section}")
        else:
            labels.append(f"Source: {metadata.source}")

        if page is not None:
            labels.append(f"Page: {page + 1}")

        prefix = "[" + " | ".join(labels) + "]"
        return f"{prefix}\n\n{body}"

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Load and chunk a PDF file."""
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        pieces: list[dict] = []
        for page in pages:
            text = page.page_content.strip()
            if not text:
                continue
            page_num = page.metadata.get("page", 0)
            if len(text) <= chunk_size:
                pieces.append({"text": text, "page": page_num})
            else:
                for sub in splitter.split_text(text):
                    if sub.strip():
                        pieces.append({"text": sub.strip(), "page": page_num})
        return pieces

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Load and chunk a Markdown file using headers then character splits."""
        content = file_path.read_text(encoding="utf-8")
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self._MD_HEADERS)
        sections = md_splitter.split_text(content)

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        pieces: list[dict] = []
        for section in sections:
            text = section.page_content.strip()
            if not text:
                continue
            header_meta = dict(section.metadata) if section.metadata else {}

            if len(text) <= chunk_size:
                pieces.append({"text": text, "header": header_meta})
            else:
                for sub in char_splitter.split_text(text):
                    if sub.strip():
                        pieces.append({"text": sub.strip(), "header": header_meta})

        return pieces

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """Infer metadata from filename <topic>_<difficulty>.ext plus overrides."""
        overrides = overrides or {}
        stem = file_path.stem.lower()
        parts = stem.split("_")

        topic_key = parts[0] if parts else stem
        topic = _TOPIC_FROM_STEM.get(topic_key, topic_key.upper())

        difficulty = "intermediate"
        if len(parts) >= 2 and parts[1] in _VALID_DIFFICULTIES:
            difficulty = parts[1]

        doc_type = overrides.get("type", "concept_explanation")
        source = overrides.get("source", file_path.name)
        related = overrides.get(
            "related_topics", _RELATED_TOPICS.get(topic, [])
        )
        is_bonus = overrides.get("is_bonus", topic in _BONUS_TOPICS)

        if "topic" in overrides:
            topic = overrides["topic"]
        if "difficulty" in overrides:
            difficulty = overrides["difficulty"]

        return ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type=doc_type,
            source=source,
            related_topics=list(related),
            is_bonus=bool(is_bonus),
        )
