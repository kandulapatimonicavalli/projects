"""Unit tests for LangGraph routing and JSON parsing helpers."""

from __future__ import annotations

import pytest

from rag_agent.agent.nodes import (
    _parse_json_from_llm,
    prepare_retrieval_node,
    route_after_retrieval,
    route_by_mode,
)


class TestRouteByMode:
    def test_chat_routes_to_rewrite(self):
        assert route_by_mode({"mode": "chat"}) == "rewrite"

    def test_generate_question_routes_to_prepare(self):
        assert route_by_mode({"mode": "generate_question"}) == "prepare"

    def test_evaluate_answer_routes_to_prepare(self):
        assert route_by_mode({"mode": "evaluate_answer"}) == "prepare"

    def test_default_mode_is_chat(self):
        assert route_by_mode({}) == "rewrite"


class TestRouteAfterRetrieval:
    def test_no_context_short_circuits(self):
        state = {"no_context_found": True, "mode": "chat"}
        assert route_after_retrieval(state) == "no_context"

    def test_chat_routes_to_generation(self):
        state = {"no_context_found": False, "mode": "chat"}
        assert route_after_retrieval(state) == "generation"

    def test_generate_routes_to_question_generation(self):
        state = {"no_context_found": False, "mode": "generate_question"}
        assert route_after_retrieval(state) == "question_generation"

    def test_evaluate_routes_to_answer_evaluation(self):
        state = {"no_context_found": False, "mode": "evaluate_answer"}
        assert route_after_retrieval(state) == "answer_evaluation"


class TestPrepareRetrievalNode:
    def test_evaluate_uses_evaluation_question(self):
        result = prepare_retrieval_node(
            {
                "mode": "evaluate_answer",
                "evaluation_question": "What is an LSTM?",
            }
        )
        assert result["original_query"] == "What is an LSTM?"
        assert result["rewritten_query"] == "What is an LSTM?"

    def test_generate_builds_topic_difficulty_query(self):
        result = prepare_retrieval_node(
            {
                "mode": "generate_question",
                "topic_filter": "LSTM",
                "difficulty_filter": "intermediate",
            }
        )
        assert "LSTM" in result["original_query"]
        assert "intermediate" in result["original_query"]
        assert result["original_query"] == result["rewritten_query"]


class TestParseJsonFromLlm:
    def test_parses_plain_json(self):
        data = _parse_json_from_llm('{"score": 8, "topic": "LSTM"}')
        assert data["score"] == 8

    def test_parses_fenced_json(self):
        raw = '```json\n{"question": "Explain CNNs?"}\n```'
        data = _parse_json_from_llm(raw)
        assert data["question"] == "Explain CNNs?"

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_json_from_llm("not json")
