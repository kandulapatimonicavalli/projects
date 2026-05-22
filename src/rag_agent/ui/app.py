"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.config import get_chat_model, get_embedding_model
from rag_agent.vectorstore.store import VectorStoreManager, get_vector_store

_TOPIC_FILTER_OPTIONS = ["All", "ANN", "CNN", "RNN", "LSTM", "Seq2Seq", "Autoencoder"]
_DIFFICULTY_FILTER_OPTIONS = ["All", "beginner", "intermediate", "advanced"]


@st.cache_resource
def _load_heavy_resources() -> tuple[VectorStoreManager, object, object]:
    """
    Load embedding model, vector store, and LLM once per Streamlit process.

    First load can take 15–60s while the local embedding model is read from disk.
    """
    return get_vector_store(), get_embedding_model(), get_chat_model()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


def initialise_session_state() -> None:
    """Initialise session state keys on first run."""
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": f"session-{uuid.uuid4().hex[:8]}",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _refresh_document_list(store: VectorStoreManager) -> None:
    """Sync sidebar document list from ChromaDB."""
    st.session_state.ingested_documents = store.list_documents()


def _upload_dir() -> Path:
    """Directory for uploaded corpus files."""
    settings = get_settings()
    path = Path(settings.corpus_dir) / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Sidebar: upload, ingest, list documents, remove."""
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Ingest documents",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        upload_path = _upload_dir()
        saved_paths: list[Path] = []

        for uploaded in uploaded_files:
            dest = upload_path / uploaded.name
            dest.write_bytes(uploaded.getvalue())
            saved_paths.append(dest)

        with st.spinner("Chunking and embedding..."):
            chunks = chunker.chunk_files(saved_paths)
            result = store.ingest(chunks)

        st.session_state.last_ingestion_result = result
        _refresh_document_list(store)

        if result.errors:
            st.sidebar.error(
                f"Ingested {result.ingested}, skipped {result.skipped}, "
                f"errors {len(result.errors)}"
            )
            for err in result.errors:
                st.sidebar.caption(err)
        elif result.ingested > 0:
            st.sidebar.success(
                f"{result.ingested} chunk(s) added, {result.skipped} duplicate(s) skipped."
            )
        elif result.skipped > 0:
            st.sidebar.warning(
                f"No new chunks — {result.skipped} duplicate(s) skipped."
            )
        else:
            st.sidebar.info("No chunks produced from uploaded files.")

    if st.session_state.last_ingestion_result is not None:
        r = st.session_state.last_ingestion_result
        st.sidebar.caption(
            f"Last run: ingested={r.ingested}, skipped={r.skipped}"
        )

    st.sidebar.divider()
    st.sidebar.subheader("Ingested documents")

    docs = st.session_state.ingested_documents
    if not docs:
        st.sidebar.info("No documents in the corpus yet.")
        return

    for doc in docs:
        col_info, col_del = st.sidebar.columns([3, 1])
        with col_info:
            st.markdown(
                f"**{doc['source']}**  \n"
                f"{doc['topic']} · {doc['chunk_count']} chunk(s)"
            )
        with col_del:
            if st.button("🗑", key=f"delete_{doc['source']}", help="Remove document"):
                deleted = store.delete_document(doc["source"])
                st.sidebar.success(f"Removed {deleted} chunk(s).")
                _refresh_document_list(store)
                if st.session_state.selected_document == doc["source"]:
                    st.session_state.selected_document = None
                st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """Sidebar: corpus health metrics."""
    st.sidebar.divider()
    st.sidebar.subheader("📊 Corpus health")

    stats = store.get_collection_stats()
    st.sidebar.metric("Total chunks", stats["total_chunks"])

    if stats["topics"]:
        st.sidebar.write("**Topics:**", ", ".join(stats["topics"]))
    else:
        st.sidebar.write("**Topics:** none yet")

    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus topics present")
    else:
        st.sidebar.caption("No bonus topics yet")


def render_document_viewer(store: VectorStoreManager) -> None:
    """Centre column: browse chunks for a selected document."""
    st.subheader("📄 Document Viewer")

    docs = st.session_state.ingested_documents
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in docs]
    default_index = 0
    if st.session_state.selected_document in sources:
        default_index = sources.index(st.session_state.selected_document)

    selected = st.selectbox(
        "Select document",
        options=sources,
        index=default_index,
        format_func=lambda s: next(
            (f"{d['source']} ({d['topic']}, {d['chunk_count']} chunks)" for d in docs if d["source"] == s),
            s,
        ),
    )
    st.session_state.selected_document = selected

    chunks = store.get_document_chunks(selected)
    st.caption(f"Showing **{len(chunks)}** chunk(s) from `{selected}`")

    container = st.container(height=420)
    with container:
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            st.markdown(
                f"**Chunk {i}** · `{meta.topic}` · {meta.difficulty} · {meta.type}"
            )
            st.markdown(chunk.chunk_text)
            st.divider()


def _run_agent(graph, query: str) -> AgentResponse:
    """Invoke LangGraph and return structured response."""
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "topic_filter": st.session_state.topic_filter,
            "difficulty_filter": st.session_state.difficulty_filter,
        },
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    response = result.get("final_response")
    if response is None:
        return AgentResponse(
            answer="No response was generated. Please try again.",
            sources=[],
            confidence=0.0,
            no_context_found=True,
        )
    return response


def render_chat_interface(graph) -> None:
    """Right column: filters, chat history, and input."""
    st.subheader("💬 Interview Prep Chat")

    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic_ix = (
            _TOPIC_FILTER_OPTIONS.index(st.session_state.topic_filter)
            if st.session_state.topic_filter in _TOPIC_FILTER_OPTIONS
            else 0
        )
        topic_choice = st.selectbox(
            "Topic filter",
            _TOPIC_FILTER_OPTIONS,
            index=topic_ix,
        )
        st.session_state.topic_filter = (
            None if topic_choice == "All" else topic_choice
        )

    with col_diff:
        diff_ix = (
            _DIFFICULTY_FILTER_OPTIONS.index(st.session_state.difficulty_filter)
            if st.session_state.difficulty_filter in _DIFFICULTY_FILTER_OPTIONS
            else 0
        )
        diff_choice = st.selectbox(
            "Difficulty filter",
            _DIFFICULTY_FILTER_OPTIONS,
            index=diff_ix,
        )
        st.session_state.difficulty_filter = (
            None if diff_choice == "All" else diff_choice
        )

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        st.rerun()

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("rewritten_query"):
                    with st.expander("🔍 Search query used"):
                        st.caption(message["rewritten_query"])
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    query = st.chat_input("Ask about a deep learning topic...")
    if query:
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        with st.spinner("Generating response..."):
            try:
                response = _run_agent(graph, query)
            except Exception as exc:
                st.error(
                    f"Agent error: {exc}\n\n"
                    "If this timed out: check your network can reach Groq, "
                    "confirm GROQ_API_KEY in `.env`, and restart Streamlit."
                )
                st.session_state.chat_history.pop()
                st.stop()

        assistant_entry = {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "no_context_found": response.no_context_found,
            "rewritten_query": response.rewritten_query,
        }
        if response.confidence > 0:
            assistant_entry["confidence"] = response.confidence

        st.session_state.chat_history.append(assistant_entry)
        st.rerun()


def main() -> None:
    """Application entry point."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    with st.spinner("Loading embedding model and agent (first start may take a minute)..."):
        store, _, _ = _load_heavy_resources()
    chunker = get_chunker()
    graph = get_graph()

    _refresh_document_list(store)

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
