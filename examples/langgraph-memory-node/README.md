# LangGraph Memory Node

This example shows the smallest useful pattern for reading prior memory inside
LangGraph state.

Prerequisites:

```bash
pip install -e .[fastembed]
pip install langgraph
```

Run:

```bash
python examples/langgraph-memory-node/langgraph_memory_node.py
```

What it does:

- recalls the top few memory hits for a user prompt
- injects those hits into graph state
- drafts a response that includes the recalled context

This is intentionally simple. It demonstrates where `MemoryClient` fits in the
graph, not a full production agent loop.
