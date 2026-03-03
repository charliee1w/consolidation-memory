# Architecture

> A contributor-oriented overview of consolidation-memory internals.
> Read time: ~15 minutes.

## Table of Contents

- [High-Level Data Flow](#high-level-data-flow)
- [Threading Model](#threading-model)
- [Storage Layout](#storage-layout)
- [Consolidation Engine](#consolidation-engine)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Security Considerations](#security-considerations)

---

## High-Level Data Flow

```mermaid
flowchart TD
    subgraph Ingestion
        A[MCP / REST request] --> B[MemoryClient.store]
        B --> C[Embed content via backend]
        C --> D[Dedup check<br/>top-3 search, threshold 0.95]
        D -->|unique| E[Insert episode into SQLite]
        D -->|duplicate| Z[Reject silently]
        E --> F[Add vector to FAISS index]
    end

    subgraph Consolidation["Background Consolidation (every 6h)"]
        G[Fetch unconsolidated episodes] --> H[Compute embeddings pairwise similarity]
        H --> I[Hierarchical clustering<br/>scipy linkage, threshold 0.78]
        I --> J{Match existing topic?}
        J -->|yes| K[Merge into topic<br/>contradiction detection]
        J -->|no| L[Create new topic]
        K --> M[LLM extracts typed records]
        L --> M
        M --> N[Write knowledge markdown]
        N --> O[Insert records into SQLite]
        O --> P[Mark episodes consolidated]
        P --> Q[Prune old episodes<br/>after 30 days]
    end

    subgraph Retrieval
        R[recall query] --> S[Embed query]
        S --> T[FAISS ANN search]
        T --> U[Batch-fetch episodes from SQLite]
        U --> V[Apply filters + priority scoring]
        V --> W[Search knowledge topics<br/>cached embeddings]
        V --> X[Search knowledge records<br/>cached embeddings]
        W --> Y[Return ranked results]
        X --> Y
    end

    F -.->|vector index| T
    E -.->|rows| G
    O -.->|records| X
    N -.->|markdown files| W
```

**Key principle:** Episodes are the raw material. Consolidation distills them into
structured knowledge (topics + typed records). Recall searches both layers and
blends results with priority scoring.

---

## Threading Model

```mermaid
flowchart LR
    subgraph Main["Main Thread (MCP/REST)"]
        store[store]
        recall_fn[recall]
        compact[compact]
    end

    subgraph BG["Background Thread (daemon)"]
        loop["_consolidation_loop<br/>sleeps on Event.wait(interval)"]
        loop --> pool["ThreadPoolExecutor<br/>max_workers=1"]
        pool --> run["run_consolidation()"]
    end

    faiss[(FAISS Index)]
    sqlite[(SQLite WAL)]

    store -->|Lock| faiss
    recall_fn -->|Lock| faiss
    compact -->|Lock| faiss
    run -->|Lock| faiss

    store --> sqlite
    recall_fn --> sqlite
    run --> sqlite
```

### Synchronization Primitives

| Primitive | Location | Protects |
|-----------|----------|----------|
| `threading.Lock` | `VectorStore._lock` | All FAISS index reads/writes |
| `threading.Lock` | `MemoryClient._consolidation_lock` | Prevents concurrent consolidation runs |
| `threading.Event` | `MemoryClient._consolidation_stop` | Clean shutdown signal for background thread |
| `threading.local` | `database._local` | Thread-local SQLite connection cache |
| `threading.Lock` | `database._conn_list_lock` | Global connection list for shutdown cleanup |

### SQLite Concurrency

- **WAL mode** (`PRAGMA journal_mode=WAL`) enables readers to proceed without
  blocking writers and vice versa.
- Each thread gets its own `sqlite3.Connection` via `threading.local`, created
  on first access with a 10-second busy timeout.
- `get_connection()` is a context manager that commits on clean exit and rolls
  back on exception.

### FAISS Locking Strategy

Every public method on `VectorStore` (`add`, `search`, `remove`, `compact`,
`reconstruct_batch`) acquires `_lock` for the duration of the operation.
Cross-process coordination uses a signal file (`.faiss_reload`): after a write,
the writer calls `signal_reload()` which touches the file; readers check the
file's mtime and reload the index if it's newer than their load timestamp.

### Background Consolidation Lifecycle

1. Started as a daemon thread in `MemoryClient.__init__` if `CONSOLIDATION_AUTO_RUN=True`.
2. Sleeps via `Event.wait(timeout=interval_seconds)` — wakes on timeout or stop signal.
3. Acquires `_consolidation_lock` (non-blocking) — skips the run if already in progress.
4. Submits `run_consolidation()` to a single-worker `ThreadPoolExecutor` with a
   `CONSOLIDATION_MAX_DURATION + 60s` timeout.
5. On `MemoryClient.close()`, the stop event is set and the thread is joined with
   a 30-second timeout.

---

## Storage Layout

```
<DATA_DIR>/projects/<PROJECT_NAME>/
├── memory.db                    # SQLite database
├── faiss_index.bin              # FAISS binary index
├── faiss_id_map.json            # UUID ↔ FAISS position mapping
├── faiss_tombstones.json        # Soft-deleted episode UUIDs
├── .faiss_reload                # Signal file for cross-process reload
├── knowledge/                   # Markdown knowledge documents
│   ├── <topic_slug>.md
│   └── versions/                # Up to 5 historical versions per topic
│       └── <slug>.<ISO-timestamp>.md
├── backups/                     # JSON export snapshots
├── consolidation_logs/          # Per-run consolidation reports
└── logs/                        # Application logs
```

Config file: `~/.config/consolidation_memory/config.toml` (XDG), or
`%APPDATA%/consolidation_memory/config.toml` (Windows).

### SQLite Schema (v10)

```mermaid
erDiagram
    episodes {
        TEXT id PK
        TEXT created_at
        TEXT updated_at
        TEXT content
        TEXT content_type
        TEXT tags "JSON array"
        REAL surprise_score
        INTEGER access_count
        TEXT source_session
        INTEGER consolidated "0=no, 1=yes, 2=pruned"
        TEXT consolidated_at
        TEXT consolidated_to
        INTEGER deleted
        INTEGER consolidation_attempts
        TEXT last_consolidation_attempt
        INTEGER protected "0=no, 1=immune to pruning"
    }

    knowledge_topics {
        TEXT id PK
        TEXT filename "UNIQUE"
        TEXT title
        TEXT summary
        TEXT created_at
        TEXT updated_at
        TEXT source_episodes "JSON array"
        INTEGER fact_count
        INTEGER access_count
        REAL confidence
    }

    knowledge_records {
        TEXT id PK
        TEXT topic_id FK
        TEXT record_type "fact|solution|preference|procedure"
        TEXT content "JSON object"
        TEXT embedding_text
        TEXT source_episodes "JSON array"
        REAL confidence
        TEXT created_at
        TEXT updated_at
        INTEGER access_count
        INTEGER deleted
        TEXT valid_from
        TEXT valid_until
    }

    consolidation_runs {
        TEXT id PK
        TEXT started_at
        TEXT completed_at
        INTEGER episodes_processed
        INTEGER clusters_formed
        INTEGER topics_created
        INTEGER topics_updated
        INTEGER episodes_pruned
        TEXT status
        TEXT error_message
    }

    consolidation_metrics {
        TEXT id PK
        TEXT run_id
        TEXT timestamp
        INTEGER clusters_succeeded
        INTEGER clusters_failed
        REAL avg_confidence
        INTEGER episodes_processed
        REAL duration_seconds
        INTEGER api_calls
        INTEGER topics_created
        INTEGER topics_updated
        INTEGER episodes_pruned
    }

    contradiction_log {
        TEXT id PK
        TEXT topic_id FK
        TEXT old_record_id
        TEXT new_record_id
        TEXT old_content
        TEXT new_content
        TEXT resolution "expired_old"
        TEXT reason
        TEXT detected_at
    }

    tag_cooccurrence {
        TEXT tag_a
        TEXT tag_b
        INTEGER count
        TEXT last_seen
    }

    episodes_fts {
        TEXT content "FTS5 virtual table mirroring episode content for BM25 keyword search"
    }

    schema_version {
        INTEGER version
        TEXT applied_at
    }

    knowledge_topics ||--o{ knowledge_records : "topic_id"
    knowledge_topics ||--o{ contradiction_log : "topic_id"
    consolidation_runs ||--o{ consolidation_metrics : "run_id"
```

Notable indexes: `idx_episodes_consolidated`, `idx_episodes_created`,
`idx_episodes_type`, `idx_episodes_deleted`, `idx_episodes_consolidation_attempts`,
`idx_records_topic`, `idx_records_type`, `idx_records_deleted`,
`idx_records_valid_until`, `idx_contradiction_topic`, `idx_contradiction_detected`,
`idx_cooccurrence_tag_a`, `idx_cooccurrence_tag_b`.

### FAISS Index

- **Initial type:** `IndexFlatIP(dim)` — brute-force inner product on L2-normalized
  vectors (equivalent to cosine similarity).
- **Auto-upgrade:** When `ntotal >= FAISS_IVF_UPGRADE_THRESHOLD` (default 10,000),
  the index is rebuilt as `IndexIVFFlat` with `nlist = min(sqrt(n), 4096)` and
  `nprobe = min(nlist/4, 64)`.
- **Persistence:** Atomic writes via temp file + `os.replace()`.
- **Deletions:** Tombstone set in `faiss_tombstones.json`; vectors aren't removed
  from the index until `compact()` rebuilds it.
- **Consistency check:** On load, validates `ntotal == len(id_map)` and
  `index.d == EMBEDDING_DIMENSION`.

---

## Consolidation Engine

Located in `src/consolidation_memory/consolidation/` (a package split into
`clustering.py`, `prompting.py`, `scoring.py`, `engine.py`).

### Pipeline Overview

```mermaid
flowchart TD
    A[Fetch unconsolidated episodes<br/>limit=200, max_attempts=5] --> B[Encode all episode embeddings]
    B --> C[Build cosine similarity matrix]
    C --> D["Hierarchical clustering<br/>scipy linkage(method='average')<br/>fcluster(threshold=0.78)"]
    D --> E[Filter clusters by size<br/>min=2, max=20]
    E --> F{For each cluster}

    F --> G[Compute cluster confidence<br/>coherence×0.6 + surprise×0.4]
    G --> H["Find matching topic<br/>semantic ≥ 0.75 or word overlap > 50%"]
    H -->|match| I[Merge: detect contradictions<br/>expire stale records]
    H -->|no match| J[Create new topic]
    I --> K[LLM extraction prompt]
    J --> K
    K --> L[Parse + validate JSON output]
    L -->|invalid| M[Retry with validation feedback<br/>circuit breaker: 3 failures → open]
    L -->|valid| N[Render markdown, write file<br/>version old file, max 5 versions]
    N --> O[Insert records into SQLite]
    O --> P[Mark episodes consolidated=1]
    P --> Q["Prune: consolidated>30 days ago<br/>set consolidated=2, tombstone FAISS"]
    Q --> R[Adjust surprise scores<br/>boost high-access, decay inactive]
```

### Clustering Details

Uses `scipy.cluster.hierarchy.linkage` with `method='average'` (UPGMA) on a
cosine distance matrix (`1 - dot(a, b)` on normalized vectors). The dendrogram
is cut with `fcluster(Z, t=0.78, criterion='distance')`. Clusters smaller than
`MIN_CLUSTER_SIZE` (2) or larger than `MAX_CLUSTER_SIZE` (20) are skipped.

**Cluster confidence** is computed as:

```
confidence = clamp(coherence × 0.6 + source_quality × 0.4, 0.5, 0.95)
```

Where `coherence` is the mean intra-cluster pairwise similarity and
`source_quality` is the mean surprise score of cluster episodes.

### LLM Prompt Strategy

The system prompt establishes the LLM as a "precise knowledge extractor" and
explicitly instructs it to treat `<episode>` tag contents as raw data, never as
instructions. Each episode is wrapped as:

```
<episode>
[2025-02-28T12:00:00Z] [fact] {sanitized content}
</episode>
```

The extraction prompt requests JSON output with four record types:

| Type | Required Fields |
|------|-----------------|
| `fact` | `subject`, `info` |
| `solution` | `problem`, `fix`, `context` |
| `preference` | `key`, `value`, `context` |
| `procedure` | `trigger`, `steps`, `context` |

Validation checks all required fields. On failure, a retry prompt includes the
validation error. The circuit breaker opens after 3 consecutive LLM failures
with a 60-second cooldown.

### Contradiction Detection

1. Embed new and existing records.
2. Find candidate pairs with similarity >= 0.7.
3. If `CONTRADICTION_LLM_ENABLED`: send pairs to LLM for verdict
   (`CONTRADICTS` / `COMPATIBLE`).
4. Contradicting existing records are expired (`valid_until` set to now);
   new records replace them.

### Pruning

Episodes with `consolidated=1` older than `CONSOLIDATION_PRUNE_AFTER_DAYS`
(default 30) are marked `consolidated=2` and their FAISS vectors are
tombstoned. The SQLite rows remain for audit but are excluded from search.

---

## Retrieval Pipeline

Implemented in `context_assembler.py`. A single `recall()` call searches three
layers simultaneously and returns a unified result.

```mermaid
flowchart LR
    Q[Query] --> E[Embed query]
    E --> F[FAISS search<br/>k = n×3 or n×5 if filtered]
    F --> G[Batch-fetch episodes<br/>single SQL query]
    G --> H[Apply filters<br/>type, tags, date range]
    H --> I[Priority scoring]
    I --> J[Top-N episodes]

    E --> K[Topic cache<br/>semantic + keyword]
    K --> L[Top-5 topics]

    E --> M[Record cache<br/>semantic + keyword<br/>procedure boost]
    M --> N[Top-15 records]

    J --> O[Merged response]
    L --> O
    N --> O
```

### Priority Scoring Formula

```
score = similarity × metadata_boost
```

Where:

```
metadata_boost = surprise^w_s × recency^w_r × access_factor

recency        = exp(-age_days / 90)
access_factor  = 1.0 + log(1 + access_count) × w_a
```

Default weights: `w_s = 0.4`, `w_r = 0.35`, `w_a = 0.25`.

**Intuition:** Recent, surprising, frequently-accessed episodes rank higher.
The exponential decay gives a 90-day half-life — episodes from 3 months ago
score ~50% of the recency factor.

### Knowledge Search

Topics and records are searched using cached embedding matrices (rebuilt on
invalidation). Relevance is a weighted blend:

| Layer | Formula | Threshold |
|-------|---------|-----------|
| Topics | `semantic × 0.8 + keyword × 0.2` | 0.25 |
| Records | `semantic × 0.9 + keyword × 0.1` | 0.30 |

Procedure-type records receive a 1.15× relevance boost when the query
contains task-oriented words (`how`, `workflow`, `steps`, `deploy`, `test`,
etc.).

---

## Security Considerations

### Prompt Injection Defense

Episode content passes through an LLM during consolidation, creating a prompt
injection surface. Defenses are layered:

1. **Sanitization** (`_sanitize_for_prompt`): A regex strips common injection
   patterns — `system:`, `you are`, `ignore previous`, `override`,
   `[system]`, `<system>`, etc. — replacing them with `[REDACTED]`.

2. **Structural isolation**: Episodes are wrapped in `<episode>` XML tags.
   The system prompt explicitly states that tag contents are raw data.

3. **Output validation**: LLM output must parse as valid JSON with the
   expected schema. Free-text or instruction-like output fails validation
   and triggers a retry (bounded by the circuit breaker).

### Path Traversal Guards

- **Topic filenames** are produced by `_slugify()`, which strips all
  characters except lowercase alphanumeric and underscores, and caps length
  at 60 characters. No user-supplied path components reach the filesystem
  directly.
- **Project names** are validated against `^[a-z0-9][a-z0-9_-]{0,63}$`
  before any path derivation.
- All file operations target known subdirectories under `DATA_DIR`
  (`knowledge/`, `backups/`, `logs/`, etc.).

### Input Sanitization

- **Content types** are validated against an allowlist (`exchange`, `fact`,
  `solution`, `preference`); unrecognized values default to `exchange`.
- **Surprise scores** are clamped to `[0.0, 1.0]`.
- **Confidence values** are clamped to `[0.5, 0.95]`.
- **Tags** must parse as a JSON array.
- **All SQL queries** use parameterized placeholders (`?`) — no string
  interpolation.

### Error Isolation

- The **circuit breaker** (3 failures, 60s cooldown) prevents a misbehaving
  LLM from being called in a tight loop.
- **Consolidation timeouts** (`CONSOLIDATION_MAX_DURATION`, default 1800s)
  prevent runaway background work.
- **Race-safe upserts**: `upsert_knowledge_topic` catches `IntegrityError`
  on concurrent inserts and falls back to an update.
