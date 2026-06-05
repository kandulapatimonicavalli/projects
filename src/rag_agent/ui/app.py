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
from rag_agent.agent.state import (
    AgentMode,
    AgentResponse,
    AnswerEvaluationResult,
    QuestionGenerationResult,
)
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
        "last_question_result": None,
        "last_eval_result": None,
        "last_eval_no_context": False,
        "last_gen_no_context": False,
        "eval_question_text": "",
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


def _render_topic_difficulty_filters() -> None:
    """Shared topic and difficulty filters for all agent modes."""
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


def _invoke_graph(
    graph,
    mode: AgentMode,
    payload: dict,
    thread_suffix: str = "",
) -> dict:
    """Invoke LangGraph with mode-specific state and an isolated thread id."""
    thread_id = st.session_state.thread_id
    if thread_suffix:
        thread_id = f"{thread_id}-{thread_suffix}"

    invoke_payload = {
        "mode": mode,
        "topic_filter": st.session_state.topic_filter,
        "difficulty_filter": st.session_state.difficulty_filter,
        **payload,
    }
    return graph.invoke(
        invoke_payload,
        config={"configurable": {"thread_id": thread_id}},
    )


def _run_chat_agent(graph, query: str) -> AgentResponse:
    """Invoke LangGraph in chat mode and return structured response."""
    result = _invoke_graph(
        graph,
        mode="chat",
        payload={"messages": [HumanMessage(content=query)]},
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


def _run_question_generation(graph) -> tuple[QuestionGenerationResult | None, AgentResponse | None]:
    """Invoke LangGraph in generate_question mode."""
    result = _invoke_graph(graph, mode="generate_question", payload={}, thread_suffix="gen")
    return result.get("question_result"), result.get("final_response")


def _run_answer_evaluation(
    graph,
    question: str,
    candidate_answer: str,
) -> tuple[AnswerEvaluationResult | None, AgentResponse | None]:
    """Invoke LangGraph in evaluate_answer mode."""
    result = _invoke_graph(
        graph,
        mode="evaluate_answer",
        payload={
            "evaluation_question": question,
            "candidate_answer": candidate_answer,
        },
        thread_suffix="eval",
    )
    return result.get("eval_result"), result.get("final_response")


def _show_agent_error(exc: Exception) -> None:
    st.error(
        f"Agent error: {exc}\n\n"
        "If this timed out: check your network can reach Groq, "
        "confirm GROQ_API_KEY in `.env`, and restart Streamlit."
    )


def render_chat_tab(graph) -> None:
    """Chat tab: multi-turn Q&A with the study corpus."""
    if st.button("Clear chat history", use_container_width=True, key="clear_chat"):
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
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Generating response..."):
            try:
                response = _run_chat_agent(graph, query)
            except Exception as exc:
                _show_agent_error(exc)
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


def render_generate_question_tab(graph) -> None:
    """Generate tab: create an interview question from corpus material."""
    st.caption(
        "Uses topic and difficulty filters to retrieve study material, "
        "then generates one interview question."
    )

    if st.button("Generate interview question", type="primary", use_container_width=True):
        with st.spinner("Generating question..."):
            try:
                question_result, _final_response = _run_question_generation(graph)
            except Exception as exc:
                _show_agent_error(exc)
                st.stop()

        st.session_state.last_question_result = question_result
        st.session_state.last_gen_no_context = question_result is None
        if question_result:
            st.session_state.eval_question_text = question_result.question

    if st.session_state.last_gen_no_context and st.session_state.last_question_result is None:
        st.warning("⚠️ No relevant content found in corpus for the selected filters.")
        return

    result: QuestionGenerationResult | None = st.session_state.last_question_result
    if result is None:
        st.info("Click **Generate interview question** to create a question from the corpus.")
        return

    st.markdown(f"### {result.question}")
    st.caption(f"Topic: **{result.topic}** · Difficulty: **{result.difficulty}**")

    with st.expander("📝 Model answer"):
        st.markdown(result.model_answer)

    with st.expander("🔁 Follow-up question"):
        st.markdown(result.follow_up)

    if result.source_citations:
        with st.expander("📎 Sources"):
            for citation in result.source_citations:
                st.caption(citation)


def render_evaluate_answer_tab(graph) -> None:
    """Evaluate tab: grade a student answer against corpus material."""
    st.caption(
        "Paste an interview question and your answer. "
        "The agent retrieves ground-truth chunks and scores your response."
    )

    question = st.text_area(
        "Interview question",
        value=st.session_state.eval_question_text,
        height=100,
        placeholder="e.g. Explain how LSTM gates address the vanishing gradient problem.",
    )
    candidate_answer = st.text_area(
        "Your answer",
        height=160,
        placeholder="Write your answer here as you would in an interview...",
    )

    if st.button("Evaluate my answer", type="primary", use_container_width=True):
        if not question.strip() or not candidate_answer.strip():
            st.warning("Please provide both a question and your answer.")
            st.stop()

        st.session_state.eval_question_text = question

        with st.spinner("Evaluating answer..."):
            try:
                eval_result, final_response = _run_answer_evaluation(
                    graph, question.strip(), candidate_answer.strip()
                )
            except Exception as exc:
                _show_agent_error(exc)
                st.stop()

        st.session_state.last_eval_result = eval_result
        st.session_state.last_eval_no_context = bool(
            final_response and final_response.no_context_found
        )

    if st.session_state.last_eval_no_context and st.session_state.last_eval_result is None:
        st.warning("⚠️ No relevant content found in corpus for this question.")
        return

    result: AnswerEvaluationResult | None = st.session_state.last_eval_result
    if result is None:
        st.info("Submit a question and answer to receive a score and coaching feedback.")
        return

    st.metric("Score", f"{result.score} / 10")
    st.markdown(f"**Verdict:** {result.interview_verdict}")

    with st.expander("✅ What you got right"):
        st.markdown(result.what_was_correct or "_Nothing noted._")

    with st.expander("❌ What was missing"):
        st.markdown(result.what_was_missing or "_Nothing noted._")

    with st.expander("📖 Ideal answer"):
        st.markdown(result.ideal_answer)

    st.info(f"**Coaching tip:** {result.coaching_tip}")


def render_agent_interface(graph) -> None:
    """Right column: filters and tabbed agent modes."""
    st.subheader("🎯 Interview Prep Agent")
    _render_topic_difficulty_filters()

    tab_chat, tab_generate, tab_evaluate = st.tabs(
        ["💬 Chat", "❓ Generate Question", "✅ Grade Answer"]
    )

    with tab_chat:
        render_chat_tab(graph)
    with tab_generate:
        render_generate_question_tab(graph)
    with tab_evaluate:
        render_evaluate_answer_tab(graph)


def render_chat_interface(graph) -> None:
    """Deprecated alias — use render_agent_interface."""
    render_agent_interface(graph)


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
        render_agent_interface(graph)


if __name__ == "__main__":
    main()
