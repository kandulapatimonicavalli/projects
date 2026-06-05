"""
graph.py
========
LangGraph agent graph definition and compilation.

Assembles the nodes from nodes.py into a directed state graph
and compiles it with a memory checkpointer for conversation persistence.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    answer_evaluation_node,
    generation_node,
    no_context_node,
    prepare_retrieval_node,
    question_generation_node,
    query_rewrite_node,
    retrieval_node,
    route_after_retrieval,
    route_by_mode,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    """
    Constructs and compiles the LangGraph agent state graph.

    The graph supports three modes via ``state.mode``:

        [START]
           │
           ▼
    route_by_mode
           │
     ┌─────┴──────────────────────────┐
     │                                │
  "rewrite"                      "prepare"
     │                                │
     ▼                                ▼
query_rewrite_node          prepare_retrieval_node
     │                                │
     └────────────┬───────────────────┘
                  ▼
           retrieval_node
                  │
                  ▼ (route_after_retrieval)
     ┌────────────┼────────────┬──────────────┐
     │            │            │              │
"generation" "question_gen" "answer_eval" "no_context"
     │            │            │              │
     ▼            ▼            ▼              ▼
generation   question_gen  answer_eval   no_context
     │            │            │              │
     └────────────┴────────────┴──────────────┘
                  │
                  ▼
                [END]

    Chat mode uses query rewrite; question generation and answer evaluation
    skip rewrite and build retrieval queries directly from filters or the
    evaluation question text.

    The checkpointer (MemorySaver) enables multi-turn conversation:
    each thread_id maintains its own message history and state,
    persisted in memory for the lifetime of the application session.
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        """
        Assemble nodes and edges, then compile the graph.

        Returns
        -------
        CompiledStateGraph
            A compiled LangGraph graph ready to invoke or stream.
        """
        graph = StateGraph(AgentState)

        graph.add_node("query_rewrite", query_rewrite_node)
        graph.add_node("prepare_retrieval", prepare_retrieval_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("generation", generation_node)
        graph.add_node("question_generation", question_generation_node)
        graph.add_node("answer_evaluation", answer_evaluation_node)
        graph.add_node("no_context", no_context_node)

        graph.add_conditional_edges(
            START,
            route_by_mode,
            {"rewrite": "query_rewrite", "prepare": "prepare_retrieval"},
        )
        graph.add_edge("query_rewrite", "retrieval")
        graph.add_edge("prepare_retrieval", "retrieval")
        graph.add_conditional_edges(
            "retrieval",
            route_after_retrieval,
            {
                "generation": "generation",
                "question_generation": "question_generation",
                "answer_evaluation": "answer_evaluation",
                "no_context": "no_context",
            },
        )
        graph.add_edge("generation", END)
        graph.add_edge("question_generation", END)
        graph.add_edge("answer_evaluation", END)
        graph.add_edge("no_context", END)

        return graph.compile(checkpointer=self._checkpointer)


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Return the singleton compiled graph.

    Uses lru_cache so the graph is built only once per process.
    In Streamlit, wrap with st.cache_resource instead:

        @st.cache_resource
        def get_graph():
            return AgentGraphBuilder().build()

    Returns
    -------
    CompiledStateGraph
    """
    return AgentGraphBuilder().build()
