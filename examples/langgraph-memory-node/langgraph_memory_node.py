"""Minimal LangGraph integration example."""

from __future__ import annotations

from typing import TypedDict

from consolidation_memory import MemoryClient
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    user_input: str
    memory_hits: list[str]
    response: str


def load_memory(state: AgentState) -> AgentState:
    with MemoryClient(auto_consolidate=False) as mem:
        result = mem.recall(
            state["user_input"],
            n_results=5,
            include_knowledge=True,
        )

    memory_hits = [episode.content for episode in result.episodes]
    return {
        **state,
        "memory_hits": memory_hits,
    }


def draft_response(state: AgentState) -> AgentState:
    hits = state["memory_hits"] or ["(no prior memory hits)"]
    bullets = "\n".join(f"- {item}" for item in hits)
    response = (
        "Use the recalled memory below when drafting the assistant reply.\n\n"
        f"{bullets}"
    )
    return {
        **state,
        "response": response,
    }


graph = StateGraph(AgentState)
graph.add_node("load_memory", load_memory)
graph.add_node("draft_response", draft_response)
graph.add_edge(START, "load_memory")
graph.add_edge("load_memory", "draft_response")
graph.add_edge("draft_response", END)
memory_graph = graph.compile()


if __name__ == "__main__":
    seed: AgentState = {
        "user_input": "How should I summarize pull requests for this user?",
        "memory_hits": [],
        "response": "",
    }
    result = memory_graph.invoke(seed)
    print(result["response"])
