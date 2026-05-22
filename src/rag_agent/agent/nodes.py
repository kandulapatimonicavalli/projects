"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from loguru import logger

from rag_agent.agent.prompts import NO_CONTEXT_RESPONSE, QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import get_chat_model, get_settings
from rag_agent.vectorstore.store import get_vector_store


def _get_state_value(state: AgentState, key: str, default=None):
    """Read a field from AgentState whether LangGraph passes a dict or object."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _get_latest_human_query(messages: list) -> str:
    """Return the most recent user message text from conversation history."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _approx_message_tokens(messages: list[BaseMessage]) -> int:
    """Fast token estimate — avoids loading a GPT-2 tokenizer on every answer."""
    total = 0
    for message in messages:
        content = message.content
        text = content if isinstance(content, str) else str(content)
        total += max(1, len(text) // 4)
    return total


def _format_retrieved_context(
    chunks: list[RetrievedChunk],
) -> tuple[str, list[str], float]:
    """
    Build a context block, citation list, and average similarity score.

    Returns
    -------
    tuple[str, list[str], float]
        (context_text, citations, average_confidence)
    """
    if not chunks:
        return "", [], 0.0

    parts: list[str] = []
    citations: list[str] = []
    scores: list[float] = []

    for chunk in chunks:
        citation = chunk.to_citation()
        citations.append(citation)
        scores.append(chunk.score)
        parts.append(f"{citation}\n{chunk.chunk_text}")

    avg_confidence = sum(scores) / len(scores)
    return "\n\n".join(parts), citations, avg_confidence


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.
    """
    messages = _get_state_value(state, "messages", [])
    original_query = _get_latest_human_query(messages)
    if not original_query:
        return {"original_query": "", "rewritten_query": ""}

    llm = get_chat_model()
    rewritten_query = original_query

    try:
        prompt = QUERY_REWRITE_PROMPT.format(original_query=original_query)
        response = llm.invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
        rewritten_query = text.strip() or original_query
        logger.debug("Query rewrite: '{}' -> '{}'", original_query, rewritten_query)
    except Exception as exc:
        logger.warning("Query rewrite failed, using original query: {}", exc)

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
    }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.
    """
    query_text = _get_state_value(state, "rewritten_query", "") or _get_state_value(
        state, "original_query", ""
    )
    if not query_text:
        return {"retrieved_chunks": [], "no_context_found": True}

    store = get_vector_store()
    original_query = _get_state_value(state, "original_query", "")
    filters = {
        "topic_filter": _get_state_value(state, "topic_filter"),
        "difficulty_filter": _get_state_value(state, "difficulty_filter"),
    }

    chunks = store.query(query_text=query_text, **filters)

    # Rewrite can inject DL terms and cause false-positive retrieval (e.g. Rome → RNN).
    # If the user's original wording finds nothing, do not trust rewrite-only matches.
    if (
        chunks
        and original_query
        and original_query.strip().lower() != query_text.strip().lower()
    ):
        original_chunks = store.query(query_text=original_query, **filters)
        if not original_chunks:
            logger.info(
                "Rewrite matched corpus but original query did not — no context: {}",
                original_query,
            )
            return {"retrieved_chunks": [], "no_context_found": True}

    if not chunks:
        logger.info("No chunks met similarity threshold for query: {}", query_text)
        return {"retrieved_chunks": [], "no_context_found": True}

    logger.info("Retrieved {} chunks for query: {}", len(chunks), query_text)
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: No Context (Hallucination Guard)
# ---------------------------------------------------------------------------


def no_context_node(state: AgentState) -> dict:
    """
    Return a safe response when retrieval found no relevant chunks.

    Routed via conditional edge "end" — skips LLM generation entirely.
    """
    response = AgentResponse(
        answer=NO_CONTEXT_RESPONSE,
        sources=[],
        confidence=0.0,
        no_context_found=True,
        rewritten_query=_get_state_value(state, "rewritten_query", ""),
    )
    return {
        "final_response": response,
        "messages": [AIMessage(content=NO_CONTEXT_RESPONSE)],
    }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.
    """
    settings = get_settings()
    llm = get_chat_model()

    # Defensive guard (primary guard is no_context_node via graph routing)
    if _get_state_value(state, "no_context_found", False) or not _get_state_value(
        state, "retrieved_chunks", []
    ):
        return no_context_node(state)

    context_text, citations, confidence = _format_retrieved_context(
        _get_state_value(state, "retrieved_chunks", [])
    )
    messages = _get_state_value(state, "messages", [])
    user_query = _get_state_value(state, "original_query", "") or _get_latest_human_query(
        messages
    )

    context_message = SystemMessage(
        content=(
            "Use ONLY the following retrieved study material to answer.\n\n"
            f"{context_text}"
        )
    )

    history = trim_messages(
        messages,
        max_tokens=settings.max_context_tokens,
        strategy="last",
        token_counter=_approx_message_tokens,
    )

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        context_message,
        *history,
        HumanMessage(content=user_query),
    ]

    llm_response = llm.invoke(prompt_messages)
    answer = (
        llm_response.content
        if isinstance(llm_response.content, str)
        else str(llm_response.content)
    )

    response = AgentResponse(
        answer=answer,
        sources=citations,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=_get_state_value(state, "rewritten_query", ""),
    )

    return {
        "final_response": response,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge: generate with context, or end with hallucination guard.
    """
    if _get_state_value(state, "no_context_found", False):
        return "end"
    return "generate"
