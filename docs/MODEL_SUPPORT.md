# Model Support

`consolidation-memory` supports both local-first and hosted model backends.

The package itself does not force a hosted provider. You choose the embedding
and consolidation backends independently through config or environment
variables.

## Embedding Backends

| Backend | Default model | Extra install | Network target | Local-only | Notes |
| --- | --- | --- | --- | --- | --- |
| `fastembed` | `BAAI/bge-small-en-v1.5` | `consolidation-memory[fastembed]` | None beyond model download | Yes | Best default for zero-config local installs |
| `lmstudio` | `text-embedding-nomic-embed-text-v1.5` | none | OpenAI-compatible API, usually `http://127.0.0.1:1234/v1` | Yes | Good when LM Studio is already running |
| `openai` | `text-embedding-3-small` | `consolidation-memory[openai]` | OpenAI API | No | Hosted embedding path |
| `ollama` | `nomic-embed-text` | none | Ollama API, usually `http://localhost:11434` | Yes | Good for all-local setups with Ollama |

## LLM Backends

| Backend | Default model | Extra install | Network target | Local-only | Notes |
| --- | --- | --- | --- | --- | --- |
| `lmstudio` | `qwen2.5-7b-instruct` | none | OpenAI-compatible API, usually `http://localhost:1234/v1` | Yes | Default consolidation backend |
| `openai` | `gpt-4o-mini` | `consolidation-memory[openai]` | OpenAI API | No | Hosted option with SDK-based retries |
| `ollama` | `qwen2.5:7b` | none | Ollama API, usually `http://localhost:11434` | Yes | Local chat models via Ollama |
| `disabled` | none | none | none | Yes | Store and recall only; skips LLM consolidation |

## Recommended Pairings

| Use case | Embeddings | LLM |
| --- | --- | --- |
| Fastest zero-config local start | `fastembed` | `lmstudio` or `disabled` |
| Fully local with one runtime family | `ollama` | `ollama` |
| Fully local with separate GUI model host | `fastembed` | `lmstudio` |
| Hosted provider with minimum setup friction | `openai` | `openai` |
| Deterministic store and recall only | `fastembed` | `disabled` |

## Environment Variables

Common overrides:

```bash
CONSOLIDATION_MEMORY_EMBEDDING_BACKEND=fastembed
CONSOLIDATION_MEMORY_LLM_BACKEND=lmstudio
CONSOLIDATION_MEMORY_EMBEDDING_API_BASE=http://127.0.0.1:1234/v1
CONSOLIDATION_MEMORY_LLM_API_BASE=http://localhost:1234/v1
CONSOLIDATION_MEMORY_EMBEDDING_API_KEY=
CONSOLIDATION_MEMORY_LLM_API_KEY=
```

## Trust And Network Behavior

- `consolidation-memory` itself has no built-in telemetry.
- Network traffic only goes to the embedding and LLM backends you configure.
- A fully local setup is supported with `fastembed` plus `lmstudio`, `ollama`,
  or `disabled`.
