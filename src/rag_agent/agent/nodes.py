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

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from loguru import logger

from rag_agent.agent.prompts import (
    ANSWER_EVALUATION_PROMPT,
    NO_CONTEXT_RESPONSE,
    QUESTION_GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import (
    AgentResponse,
    AgentState,
    AnswerEvaluationResult,
    QuestionGenerationResult,
    RetrievedChunk,
)
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


def _parse_json_from_llm(text: str) -> dict:
    """Extract and parse a JSON object from an LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Node: Prepare Retrieval (non-chat modes)
# ---------------------------------------------------------------------------


def prepare_retrieval_node(state: AgentState) -> dict:
    """
    Build a search query for question generation or answer evaluation.

    Skips query rewrite — uses topic/difficulty keywords or the evaluation
    question text directly for vector search.
    """
    mode = _get_state_value(state, "mode", "chat")

    if mode == "evaluate_answer":
        question = _get_state_value(state, "evaluation_question", "").strip()
        return {
            "original_query": question,
            "rewritten_query": question,
        }

    topic = _get_state_value(state, "topic_filter") or "deep learning"
    difficulty = _get_state_value(state, "difficulty_filter") or "intermediate"
    search_query = (
        f"{topic} {difficulty} technical interview concepts neural network"
    )
    logger.debug("Prepared retrieval query for {}: '{}'", mode, search_query)
    return {
        "original_query": search_query,
        "rewritten_query": search_query,
    }


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

    Routed via conditional edge — skips LLM generation entirely.
    """
    mode = _get_state_value(state, "mode", "chat")
    response = AgentResponse(
        answer=NO_CONTEXT_RESPONSE,
        sources=[],
        confidence=0.0,
        no_context_found=True,
        rewritten_query=_get_state_value(state, "rewritten_query", ""),
    )
    updates: dict = {
        "final_response": response,
        "question_result": None,
        "eval_result": None,
    }
    if mode == "chat":
        updates["messages"] = [AIMessage(content=NO_CONTEXT_RESPONSE)]
    return updates


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
# Node: Question Generator
# ---------------------------------------------------------------------------


def question_generation_node(state: AgentState) -> dict:
    """Generate an interview question from retrieved study material."""
    if _get_state_value(state, "no_context_found", False) or not _get_state_value(
        state, "retrieved_chunks", []
    ):
        return no_context_node(state)

    llm = get_chat_model()
    context_text, citations, confidence = _format_retrieved_context(
        _get_state_value(state, "retrieved_chunks", [])
    )
    difficulty = _get_state_value(state, "difficulty_filter") or "intermediate"

    prompt = QUESTION_GENERATION_PROMPT.format(
        context=context_text,
        difficulty=difficulty,
    )

    try:
        llm_response = llm.invoke(prompt)
        raw = (
            llm_response.content
            if isinstance(llm_response.content, str)
            else str(llm_response.content)
        )
        data = _parse_json_from_llm(raw)
        result = QuestionGenerationResult(
            question=data.get("question", ""),
            difficulty=data.get("difficulty", difficulty),
            topic=data.get("topic", _get_state_value(state, "topic_filter") or ""),
            model_answer=data.get("model_answer", ""),
            follow_up=data.get("follow_up", ""),
            source_citations=data.get("source_citations") or citations,
        )
    except Exception as exc:
        logger.error("Question generation failed: {}", exc)
        return no_context_node(state)

    summary = (
        f"**Interview question ({result.difficulty})**\n\n"
        f"{result.question}"
    )
    response = AgentResponse(
        answer=summary,
        sources=citations,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=_get_state_value(state, "rewritten_query", ""),
    )
    return {
        "question_result": result,
        "eval_result": None,
        "final_response": response,
    }


# ---------------------------------------------------------------------------
# Node: Answer Evaluator
# ---------------------------------------------------------------------------


def answer_evaluation_node(state: AgentState) -> dict:
    """Evaluate a candidate answer against retrieved source material."""
    if _get_state_value(state, "no_context_found", False) or not _get_state_value(
        state, "retrieved_chunks", []
    ):
        return no_context_node(state)

    question = _get_state_value(state, "evaluation_question", "").strip()
    candidate_answer = _get_state_value(state, "candidate_answer", "").strip()
    if not question or not candidate_answer:
        return no_context_node(state)

    llm = get_chat_model()
    context_text, citations, confidence = _format_retrieved_context(
        _get_state_value(state, "retrieved_chunks", [])
    )

    prompt = ANSWER_EVALUATION_PROMPT.format(
        question=question,
        candidate_answer=candidate_answer,
        context=context_text,
    )

    try:
        llm_response = llm.invoke(prompt)
        raw = (
            llm_response.content
            if isinstance(llm_response.content, str)
            else str(llm_response.content)
        )
        data = _parse_json_from_llm(raw)
        result = AnswerEvaluationResult(
            score=int(data.get("score", 0)),
            what_was_correct=data.get("what_was_correct", ""),
            what_was_missing=data.get("what_was_missing", ""),
            ideal_answer=data.get("ideal_answer", ""),
            interview_verdict=data.get("interview_verdict", ""),
            coaching_tip=data.get("coaching_tip", ""),
        )
    except Exception as exc:
        logger.error("Answer evaluation failed: {}", exc)
        return no_context_node(state)

    summary = (
        f"**Score: {result.score}/10** — {result.interview_verdict}\n\n"
        f"{result.coaching_tip}"
    )
    response = AgentResponse(
        answer=summary,
        sources=citations,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=_get_state_value(state, "rewritten_query", ""),
    )
    return {
        "eval_result": result,
        "question_result": None,
        "final_response": response,
    }


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------


def route_by_mode(state: AgentState) -> str:
    """Route from START: chat uses rewrite; other modes prepare retrieval."""
    mode = _get_state_value(state, "mode", "chat")
    if mode == "chat":
        return "rewrite"
    return "prepare"


def route_after_retrieval(state: AgentState) -> str:
    """Route after retrieval: no context guard or mode-specific generation."""
    if _get_state_value(state, "no_context_found", False):
        return "no_context"

    mode = _get_state_value(state, "mode", "chat")
    if mode == "generate_question":
        return "question_generation"
    if mode == "evaluate_answer":
        return "answer_evaluation"
    return "generation"


def should_retry_retrieval(state: AgentState) -> str:
    """Backward-compatible alias for route_after_retrieval."""
    return route_after_retrieval(state)
