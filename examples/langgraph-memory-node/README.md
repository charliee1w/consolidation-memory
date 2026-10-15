# LangGraph Memory Node

This example shows the smallest useful pattern for reading prior memory inside
LangGraph state.

Prerequisites:

```bash
pip install -e ".[fastembed]"
pip install langgraph
```

Run:

```bash
python examples/langgraph-memory-node/langgraph_memory_node.py
```

What it does:

- recalls the top 5 memory hits for a user prompt (`include_knowledge=True`)
- injects those episode contents into graph state
- drafts a response that includes the recalled context

It reads from the active project, so run `consolidation-memory init` first (or
seed a few episodes) or recall comes back empty. Set
`CONSOLIDATION_MEMORY_PROJECT` to point it at a specific project.

This is intentionally simple. It demonstrates where `MemoryClient` fits in the
graph, not a full production agent loop.
