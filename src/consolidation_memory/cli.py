"""CLI entry point for consolidation-memory.

Usage:
    consolidation-memory serve       # Start MCP server (default)
    consolidation-memory init        # Interactive first-run setup
    consolidation-memory test        # Verify installation end-to-end
    consolidation-memory consolidate # Run consolidation manually
    consolidation-memory status      # Show system stats
    consolidation-memory export      # Export to JSON
    consolidation-memory import PATH # Import from JSON export
    consolidation-memory reindex     # Re-embed all episodes with current backend
    consolidation-memory dashboard   # Launch TUI dashboard
"""

import argparse
import json
import os
import sys
import tempfile

from consolidation_memory import __version__


def cmd_serve(args):
    """Start the MCP or REST server."""
    if getattr(args, "rest", False):
        try:
            import uvicorn
            from consolidation_memory.rest import create_app
        except ImportError:
            print("REST API requires: pip install consolidation-memory[rest]")
            sys.exit(1)
        app = create_app()
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8080)
        print(f"Starting REST API on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        from consolidation_memory.server import run_server
        run_server()


def cmd_init():
    """Interactive first-run setup."""
    from consolidation_memory.config import get_default_config_dir, get_config_path

    print(f"consolidation-memory v{__version__} — first-run setup\n")

    existing = get_config_path()
    if existing:
        print(f"Config already exists at: {existing}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Keeping existing config.")
            return

    # Embedding backend
    print("\n--- Embedding Backend ---")
    print("1. fastembed (recommended, zero-config, downloads ~32MB model)")
    print("2. lmstudio (requires LM Studio running with embedding model)")
    print("3. openai (requires API key, uses text-embedding-3-small)")
    print("4. ollama (requires Ollama running with embedding model)")

    choice = input("\nChoice [1]: ").strip() or "1"
    backends = {"1": "fastembed", "2": "lmstudio", "3": "openai", "4": "ollama"}
    embed_backend = backends.get(choice, "fastembed")

    embed_config = f'backend = "{embed_backend}"'
    if embed_backend == "fastembed":
        try:
            import fastembed  # noqa: F401
            print("fastembed is installed.")
        except ImportError:
            print("\nfastembed not installed. Install with:")
            print("  pip install consolidation-memory[fastembed]")
    elif embed_backend == "lmstudio":
        api_base = input("LM Studio API base [http://127.0.0.1:1234/v1]: ").strip()
        if api_base:
            embed_config += f'\napi_base = "{api_base}"'
        model = input("Embedding model name [text-embedding-nomic-embed-text-v1.5]: ").strip()
        if model:
            embed_config += f'\nmodel = "{model}"'
        embed_config += '\ndimension = 768'
    elif embed_backend == "openai":
        api_key = input("OpenAI API key: ").strip()
        embed_config += f'\napi_key = "{api_key}"'
    elif embed_backend == "ollama":
        api_base = input("Ollama API base [http://localhost:11434]: ").strip()
        if api_base:
            embed_config += f'\napi_base = "{api_base}"'

    # LLM backend
    print("\n--- LLM Backend (for consolidation) ---")
    print("1. lmstudio (recommended if you have LM Studio)")
    print("2. openai")
    print("3. ollama")
    print("4. disabled (store/recall only, no consolidation)")

    llm_choice = input("\nChoice [1]: ").strip() or "1"
    llm_backends = {"1": "lmstudio", "2": "openai", "3": "ollama", "4": "disabled"}
    llm_backend = llm_backends.get(llm_choice, "lmstudio")

    llm_config = f'backend = "{llm_backend}"'
    if llm_backend == "lmstudio":
        api_base = input("LM Studio API base [http://localhost:1234/v1]: ").strip()
        if api_base:
            llm_config += f'\napi_base = "{api_base}"'
        model = input("LLM model [qwen2.5-7b-instruct]: ").strip()
        if model:
            llm_config += f'\nmodel = "{model}"'
    elif llm_backend == "openai":
        if embed_backend != "openai":
            api_key = input("OpenAI API key: ").strip()
            llm_config += f'\napi_key = "{api_key}"'
        model = input("LLM model [gpt-4o-mini]: ").strip()
        if model:
            llm_config += f'\nmodel = "{model}"'
    elif llm_backend == "ollama":
        api_base = input("Ollama API base [http://localhost:11434]: ").strip()
        if api_base:
            llm_config += f'\napi_base = "{api_base}"'
        model = input("LLM model [qwen2.5:7b]: ").strip()
        if model:
            llm_config += f'\nmodel = "{model}"'

    # Write config
    config_dir = get_default_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"

    config_content = f"""# Consolidation Memory configuration
# Generated by: consolidation-memory init

[embedding]
{embed_config}

[llm]
{llm_config}

[consolidation]
auto_run = true
interval_hours = 6

[dedup]
enabled = true
similarity_threshold = 0.95
"""

    config_path.write_text(config_content, encoding="utf-8")
    print(f"\nConfig written to: {config_path}")

    # Initialize data directory
    from consolidation_memory.config import get_config
    cfg = get_config()
    for d in [cfg.DATA_DIR, cfg.KNOWLEDGE_DIR, cfg.LOG_DIR, cfg.BACKUP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {cfg.DATA_DIR}")

    # Initialize DB
    from consolidation_memory.database import ensure_schema
    ensure_schema()
    print("Database initialized.")

    # Print Claude Desktop config snippet
    print("\n--- Add to Claude Desktop config ---")
    print(json.dumps({
        "mcpServers": {
            "consolidation_memory": {
                "command": "consolidation-memory"
            }
        }
    }, indent=2))

    print("\nSetup complete. Run 'consolidation-memory serve' to start.")


def cmd_test():
    """Verify installation works end-to-end."""
    import uuid
    from consolidation_memory.config import get_config
    from consolidation_memory.database import ensure_schema, insert_episode, soft_delete_episode

    cfg = get_config()

    # ANSI color support (respects NO_COLOR convention)
    use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    if use_color:
        GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
    else:
        GREEN = RED = YELLOW = DIM = RESET = ""

    PASS = f"{GREEN}\u2713{RESET}"
    FAIL = f"{RED}\u2717{RESET}"
    SKIP = f"{YELLOW}-{RESET}"

    print(f"consolidation-memory v{__version__} \u2014 verifying installation\n")

    checks: list[bool] = []
    test_episode_id: str | None = None
    forgotten = False
    vs = None

    def report(name: str, passed: bool, detail: str = "", skipped: bool = False):
        icon = SKIP if skipped else (PASS if passed else FAIL)
        suffix = f" {DIM}({detail}){RESET}" if detail else ""
        print(f"  {icon} {name}{suffix}")
        if not skipped:
            checks.append(passed)

    test_content = f"consolidation-memory-test {uuid.uuid4()}"

    try:
        # 1. Config
        report("Config loaded", True, f"{cfg.EMBEDDING_BACKEND} / {cfg.LLM_BACKEND}")

        # 2. Store test episode
        try:
            ensure_schema()
            test_episode_id = insert_episode(
                content=test_content,
                content_type="fact",
                tags=["_test"],
                surprise_score=0.0,
            )
            report("Store test episode", True, test_episode_id[:8])
        except Exception as e:
            report("Store test episode", False, str(e))

        # 3. Embedding backend connectivity
        embedding = None
        try:
            from consolidation_memory.backends import encode_documents
            embedding = encode_documents([test_content])
            report("Embedding backend", True, f"{cfg.EMBEDDING_BACKEND}, {embedding.shape[1]}-dim")
        except Exception as e:
            report("Embedding backend", False, str(e))

        # 4. Recall via semantic similarity
        if test_episode_id and embedding is not None:
            try:
                from consolidation_memory.backends import encode_query
                from consolidation_memory.vector_store import VectorStore
                vs = VectorStore()
                vs.add(test_episode_id, embedding[0])

                query_vec = encode_query(test_content)
                search_results = vs.search(query_vec, k=5)
                found = {r[0]: r[1] for r in search_results}
                if test_episode_id in found:
                    report("Recall test episode", True, f"similarity: {found[test_episode_id]:.2f}")
                else:
                    report("Recall test episode", False, "not found in search results")
            except Exception as e:
                report("Recall test episode", False, str(e))
        else:
            report("Recall test episode", False, "store or embed failed", skipped=True)

        # 5. Forget test episode
        if test_episode_id:
            try:
                deleted = soft_delete_episode(test_episode_id)
                if vs:
                    vs.remove(test_episode_id)
                forgotten = True
                report("Forget test episode", bool(deleted))
            except Exception as e:
                report("Forget test episode", False, str(e))
        else:
            report("Forget test episode", False, "no episode to forget", skipped=True)

        # 6. LLM backend connectivity
        if cfg.LLM_BACKEND != "disabled":
            try:
                from consolidation_memory.backends import get_llm_backend
                llm = get_llm_backend()
                if llm is None:
                    report("LLM backend", False, "returned None")
                else:
                    response = llm.generate("Reply with exactly: OK", "Say OK")
                    if response and response.strip():
                        report("LLM backend", True, f"{cfg.LLM_BACKEND}/{cfg.LLM_MODEL}")
                    else:
                        report("LLM backend", False, "empty response")
            except Exception as e:
                report("LLM backend", False, str(e))
        else:
            report("LLM backend", False, "disabled", skipped=True)

    finally:
        # Always clean up test episode, even if steps above failed
        if test_episode_id and not forgotten:
            try:
                soft_delete_episode(test_episode_id)
            except Exception:
                pass
            if vs:
                try:
                    vs.remove(test_episode_id)
                except Exception:
                    pass

    # 7. Summary
    passed = sum(checks)
    total = len(checks)
    print()
    if passed == total:
        print(f"  {GREEN}{passed}/{total} checks passed{RESET}")
    else:
        print(f"  {RED}{passed}/{total} checks passed{RESET}")
        sys.exit(1)


def cmd_consolidate():
    """Run consolidation manually."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    from consolidation_memory.consolidation import run_consolidation
    result = run_consolidation()
    print(json.dumps(result, indent=2))


def cmd_status():
    """Show system statistics."""
    from consolidation_memory.database import ensure_schema, get_stats, get_last_consolidation_run
    from consolidation_memory.config import get_config, get_active_project
    cfg = get_config()

    ensure_schema()
    stats = get_stats()
    last_run = get_last_consolidation_run()

    db_size_mb = 0.0
    if cfg.DB_PATH.exists():
        db_size_mb = round(cfg.DB_PATH.stat().st_size / (1024 * 1024), 2)

    print(f"consolidation-memory v{__version__}")
    print(f"Project:     {get_active_project()}")
    print(f"Data dir:    {cfg.DATA_DIR}")
    print(f"DB size:     {db_size_mb} MB")
    print(f"Embedding:   {cfg.EMBEDDING_BACKEND} ({cfg.EMBEDDING_MODEL_NAME}, {cfg.EMBEDDING_DIMENSION}-dim)")
    print(f"LLM:         {cfg.LLM_BACKEND} ({cfg.LLM_MODEL})")
    print()

    eb = stats["episodic_buffer"]
    print(f"Episodes:    {eb['total']} total, {eb['pending_consolidation']} pending, "
          f"{eb['consolidated']} consolidated, {eb['pruned']} pruned")

    kb = stats["knowledge_base"]
    print(f"Knowledge:   {kb['total_topics']} topics, {kb['total_facts']} facts")
    if "total_records" in kb:
        rbt = kb.get("records_by_type", {})
        print(f"Records:     {kb['total_records']} total "
              f"({rbt.get('facts', 0)} facts, {rbt.get('solutions', 0)} solutions, "
              f"{rbt.get('preferences', 0)} preferences)")

    if last_run:
        print(f"\nLast consolidation: {last_run['started_at']}")
        print(f"  Status: {last_run['status']}")
        if last_run.get("episodes_processed"):
            print(f"  Processed: {last_run['episodes_processed']} episodes")


def cmd_export():
    """Export to JSON."""
    from consolidation_memory.database import (
        ensure_schema, get_all_episodes, get_all_knowledge_topics, get_all_active_records,
    )
    from consolidation_memory.config import get_config
    from datetime import datetime, timezone

    cfg = get_config()
    ensure_schema()
    cfg.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    episodes = get_all_episodes(include_deleted=False)
    topics = get_all_knowledge_topics()
    knowledge = []
    for topic in topics:
        filepath = cfg.KNOWLEDGE_DIR / topic["filename"]
        content = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
        knowledge.append({**topic, "file_content": content})

    records = get_all_active_records()

    snapshot = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.1",
        "episodes": episodes,
        "knowledge_topics": knowledge,
        "knowledge_records": records,
        "stats": {
            "episode_count": len(episodes),
            "knowledge_count": len(knowledge),
            "record_count": len(records),
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_path = cfg.BACKUP_DIR / f"memory_export_{timestamp}.json"
    export_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    existing = sorted(cfg.BACKUP_DIR.glob("memory_export_*.json"), reverse=True)
    for old in existing[cfg.MAX_BACKUPS:]:
        old.unlink()

    print(f"Exported {len(episodes)} episodes + {len(knowledge)} topics + {len(records)} records to {export_path}")


def _validate_import(data: dict) -> list[str]:
    """Validate import JSON structure. Returns list of error strings (empty = valid)."""
    errors = []

    if not isinstance(data, dict):
        return ["Top-level value must be a JSON object"]

    for key in ("episodes", "knowledge_topics", "stats"):
        if key not in data:
            errors.append(f"Missing required key: {key!r}")

    if errors:
        return errors  # Can't validate deeper if structure is wrong

    if not isinstance(data["episodes"], list):
        errors.append("'episodes' must be a list")
    else:
        for i, ep in enumerate(data["episodes"]):
            if not isinstance(ep, dict):
                errors.append(f"episodes[{i}]: must be an object")
                continue
            for field in ("id", "content", "content_type"):
                if field not in ep:
                    errors.append(f"episodes[{i}]: missing required field {field!r}")

    if not isinstance(data["knowledge_topics"], list):
        errors.append("'knowledge_topics' must be a list")
    else:
        for i, topic in enumerate(data["knowledge_topics"]):
            if not isinstance(topic, dict):
                errors.append(f"knowledge_topics[{i}]: must be an object")
                continue
            for field in ("filename", "title", "summary"):
                if field not in topic:
                    errors.append(f"knowledge_topics[{i}]: missing required field {field!r}")

    if not isinstance(data.get("stats"), dict):
        errors.append("'stats' must be an object")

    # Cap error output to avoid flooding on completely wrong files
    if len(errors) > 20:
        errors = errors[:20]
        errors.append("... and more errors (showing first 20)")

    return errors


def cmd_import(path: str):
    """Import from JSON export."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    from pathlib import Path
    from consolidation_memory.database import (
        ensure_schema, insert_episode, upsert_knowledge_topic, get_episode,
        insert_knowledge_records,
    )
    from consolidation_memory.backends import encode_documents
    from consolidation_memory.vector_store import VectorStore
    from consolidation_memory.config import get_config

    cfg = get_config()
    export_path = Path(path)
    if not export_path.exists():
        print(f"File not found: {export_path}")
        sys.exit(1)

    try:
        data = json.loads(export_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)

    # Validate top-level structure
    errors = _validate_import(data)
    if errors:
        print("Import validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"Import file: {export_path}")
    print(f"  Episodes: {data['stats']['episode_count']}")
    print(f"  Knowledge: {data['stats']['knowledge_count']}")

    ensure_schema()
    vs = VectorStore()

    # Import episodes — collect new ones first, then batch-embed
    imported = 0
    skipped = 0
    new_episodes: list[dict] = []
    for ep in data.get("episodes", []):
        existing = get_episode(ep["id"])
        if existing:
            skipped += 1
            continue

        raw_tags = ep.get("tags")
        if raw_tags is None:
            tags = []
        elif isinstance(raw_tags, str):
            tags = json.loads(raw_tags)
        else:
            tags = raw_tags
        episode_id = insert_episode(
            content=ep["content"],
            content_type=ep.get("content_type", "exchange"),
            tags=tags,
            surprise_score=ep.get("surprise_score", 0.5),
            episode_id=ep["id"],
        )
        new_episodes.append({"id": episode_id, "content": ep["content"]})
        imported += 1

    # Batch-embed in chunks of 50
    BATCH_SIZE = 50
    for i in range(0, len(new_episodes), BATCH_SIZE):
        batch = new_episodes[i : i + BATCH_SIZE]
        texts = [ep["content"] for ep in batch]
        try:
            embeddings = encode_documents(texts)
            for ep, emb in zip(batch, embeddings):
                vs.add(ep["id"], emb)
        except Exception as e:
            print(f"  Warning: Failed to embed episode batch {i}-{i + len(batch)}: {e}")

    print(f"\nEpisodes: {imported} imported, {skipped} skipped (already exist)")

    # Import knowledge
    cfg.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    k_imported = 0
    knowledge_resolved = cfg.KNOWLEDGE_DIR.resolve()
    for topic in data.get("knowledge_topics", []):
        if topic.get("file_content"):
            filepath = (cfg.KNOWLEDGE_DIR / topic["filename"]).resolve()
            if not filepath.is_relative_to(knowledge_resolved):
                print(f"  Skipping {topic['filename']!r}: path traversal detected")
                continue
            filepath.write_text(topic["file_content"], encoding="utf-8")

        source_eps = json.loads(topic["source_episodes"]) if isinstance(topic["source_episodes"], str) else topic["source_episodes"]
        upsert_knowledge_topic(
            filename=topic["filename"],
            title=topic["title"],
            summary=topic["summary"],
            source_episodes=source_eps,
            fact_count=topic.get("fact_count", 0),
            confidence=topic.get("confidence", 0.8),
        )
        k_imported += 1

    print(f"Knowledge: {k_imported} topics imported")

    # Import knowledge records (v1.1+ exports)
    r_imported = 0
    for rec in data.get("knowledge_records", []):
        if not rec.get("topic_id") or not rec.get("record_type"):
            continue
        try:
            insert_knowledge_records(
                topic_id=rec["topic_id"],
                records=[{
                    "record_type": rec["record_type"],
                    "content": rec.get("content", "{}"),
                    "embedding_text": rec.get("embedding_text", ""),
                    "confidence": rec.get("confidence", 0.8),
                }],
                source_episodes=json.loads(rec["source_episodes"]) if isinstance(rec.get("source_episodes"), str) else rec.get("source_episodes", []),
            )
            r_imported += 1
        except Exception as e:
            print(f"  Warning: Failed to import record {rec.get('id', '?')}: {e}")

    if r_imported:
        print(f"Records: {r_imported} imported")

    VectorStore.signal_reload()
    print("\nImport complete.")


def cmd_reindex():
    """Re-embed all episodes with current backend."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    from consolidation_memory.database import ensure_schema, get_all_episodes
    from consolidation_memory.backends import encode_documents, get_dimension
    from consolidation_memory.config import get_config
    from consolidation_memory.vector_store import VectorStore
    import faiss

    cfg = get_config()
    ensure_schema()
    episodes = get_all_episodes(include_deleted=False)
    if not episodes:
        print("No episodes to reindex.")
        return

    print(f"Re-embedding {len(episodes)} episodes with current backend...")

    dim = get_dimension()
    print(f"Embedding dimension: {dim}")

    # Process in batches
    batch_size = 50
    all_ids = []
    all_vecs = []

    for i in range(0, len(episodes), batch_size):
        batch = episodes[i:i + batch_size]
        texts = [ep["content"] for ep in batch]
        try:
            vecs = encode_documents(texts)
            all_ids.extend(ep["id"] for ep in batch)
            all_vecs.append(vecs)
            print(f"  Batch {i // batch_size + 1}: {len(batch)} episodes embedded")
        except Exception as e:
            print(f"  Batch {i // batch_size + 1} failed: {e}")

    if not all_vecs:
        print("No embeddings produced. Aborting.")
        return

    import numpy as np
    all_vecs_arr = np.vstack(all_vecs)

    # Rebuild FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs_arr)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Atomic swap: write to temp files, then rename
    parent = str(cfg.FAISS_INDEX_PATH.parent)

    idx_fd, idx_tmp = tempfile.mkstemp(dir=parent, suffix=".faiss.tmp")
    os.close(idx_fd)
    try:
        faiss.write_index(index, idx_tmp)
    except Exception:
        os.unlink(idx_tmp)
        print("Failed to write new FAISS index. Original index unchanged.")
        return

    map_fd, map_tmp = tempfile.mkstemp(dir=parent, suffix=".json.tmp")
    try:
        with os.fdopen(map_fd, "w") as f:
            json.dump(all_ids, f)
    except Exception:
        os.unlink(idx_tmp)
        os.unlink(map_tmp)
        print("Failed to write id map. Original index unchanged.")
        return

    tomb_fd, tomb_tmp = tempfile.mkstemp(dir=parent, suffix=".json.tmp")
    try:
        with os.fdopen(tomb_fd, "w") as f:
            json.dump([], f)
    except Exception:
        os.unlink(idx_tmp)
        os.unlink(map_tmp)
        os.unlink(tomb_tmp)
        print("Failed to write tombstones. Original index unchanged.")
        return

    # All writes succeeded — atomic swap
    os.replace(idx_tmp, str(cfg.FAISS_INDEX_PATH))
    os.replace(map_tmp, str(cfg.FAISS_ID_MAP_PATH))
    os.replace(tomb_tmp, str(cfg.FAISS_TOMBSTONE_PATH))

    VectorStore.signal_reload()

    print(f"\nReindex complete: {len(all_ids)} vectors in {dim}-dim index")


def cmd_browse():
    """List knowledge topics interactively."""
    from consolidation_memory.database import ensure_schema, get_all_knowledge_topics, get_all_active_records
    from consolidation_memory.config import get_config

    cfg = get_config()
    ensure_schema()
    topics = get_all_knowledge_topics()

    if not topics:
        print("No knowledge topics yet. Run consolidation to generate knowledge.")
        return

    records = get_all_active_records(include_expired=False)
    records_by_topic: dict[str, dict[str, int]] = {}
    for rec in records:
        tid = rec["topic_id"]
        if tid not in records_by_topic:
            records_by_topic[tid] = {"facts": 0, "solutions": 0, "preferences": 0, "procedures": 0}
        rt = rec.get("record_type", "fact")
        if rt in records_by_topic[tid]:
            records_by_topic[tid][rt] += 1

    print(f"consolidation-memory v{__version__} — knowledge browser\n")
    print(f"Knowledge directory: {cfg.KNOWLEDGE_DIR}\n")

    for i, topic in enumerate(topics, 1):
        filepath = cfg.KNOWLEDGE_DIR / topic["filename"]
        rc = records_by_topic.get(topic["id"], {})
        exists = filepath.exists()

        parts = []
        for rtype in ("facts", "solutions", "preferences", "procedures"):
            count = rc.get(rtype, 0)
            if count > 0:
                parts.append(f"{count} {rtype}")
        records_str = ", ".join(parts) if parts else "no records"

        print(f"  {i}. {topic['title']}")
        print(f"     {topic['summary'][:100]}{'...' if len(topic.get('summary', '')) > 100 else ''}")
        print(f"     {records_str} | confidence: {topic.get('confidence', 0):.2f} | "
              f"updated: {topic.get('updated_at', 'unknown')}")
        print(f"     file: {topic['filename']} {'[exists]' if exists else '[missing]'}")
        print()


def cmd_setup_claude():
    """Append recommended CLAUDE.md snippet for proactive memory use."""
    from pathlib import Path

    snippet = """\
## Memory

**Recall**: At the start of every new conversation, call `memory_recall`
with a query matching the user's opening message topic. This is your
persistent memory — always check it before responding.

**Store**: Proactively call `memory_store` whenever you:
- Learn something new about the user's setup, environment, or projects
- Solve a non-trivial problem (store both the problem AND the solution)
- Discover a user preference or workflow pattern
- Complete a significant task (summarize what was done and where)
- Encounter something surprising or noteworthy

Write each memory as a self-contained note that future-you can understand
without context. Use appropriate `content_type` (fact, solution, preference,
exchange) and add `tags` for organization. Do NOT store trivial exchanges
like greetings or simple Q&A.
"""

    claude_md_path = Path.home() / ".claude" / "CLAUDE.md"

    if claude_md_path.exists():
        existing = claude_md_path.read_text(encoding="utf-8")
        if "memory_recall" in existing:
            print(f"Memory instructions already present in {claude_md_path}")
            print("No changes made.")
            return

        print(f"Found existing CLAUDE.md at {claude_md_path}")
        print("\nWill append this snippet:\n")
        print(snippet)
        resp = input("Append to existing CLAUDE.md? [y/N] ").strip().lower()
        if resp != "y":
            print("No changes made.")
            return

        with open(claude_md_path, "a", encoding="utf-8") as f:
            f.write("\n" + snippet)
        print(f"Snippet appended to {claude_md_path}")
    else:
        print(f"No CLAUDE.md found at {claude_md_path}")
        print("\nWill create it with this content:\n")
        print(snippet)
        resp = input("Create CLAUDE.md? [y/N] ").strip().lower()
        if resp != "y":
            print("No changes made.")
            return

        claude_md_path.parent.mkdir(parents=True, exist_ok=True)
        claude_md_path.write_text(snippet, encoding="utf-8")
        print(f"Created {claude_md_path}")

    print("\nClaude Code will now proactively use memory tools in every conversation.")


def cmd_dashboard():
    """Launch the TUI dashboard."""
    try:
        from consolidation_memory.dashboard import DashboardApp
    except ImportError:
        print("Dashboard requires textual. Install with: pip install consolidation-memory[dashboard]")
        sys.exit(1)
    app = DashboardApp()
    app.run()


def main():
    parser = argparse.ArgumentParser(
        prog="consolidation-memory",
        description="Persistent semantic memory for AI conversations",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--project", "-p",
        default=None,
        help="Project namespace (default: CONSOLIDATION_MEMORY_PROJECT env var or 'default')",
    )
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start server (MCP default, --rest for HTTP)")
    p_serve.add_argument("--rest", action="store_true", help="Start REST API instead of MCP")
    p_serve.add_argument("--host", default="127.0.0.1", help="REST host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8080, help="REST port (default: 8080)")
    sub.add_parser("init", help="Interactive first-run setup")
    sub.add_parser("test", help="Verify installation works end-to-end")
    sub.add_parser("consolidate", help="Run consolidation manually")
    sub.add_parser("status", help="Show system stats")
    sub.add_parser("export", help="Export to JSON")
    p_import = sub.add_parser("import", help="Import from JSON export")
    p_import.add_argument("path", help="Path to export JSON file")
    sub.add_parser("reindex", help="Re-embed all episodes with current backend")
    sub.add_parser("browse", help="Browse knowledge topics")
    sub.add_parser("setup-claude", help="Add memory instructions to CLAUDE.md")
    sub.add_parser("dashboard", help="Launch TUI dashboard")

    args = parser.parse_args()

    # Activate project namespace before any command
    from consolidation_memory.config import set_active_project
    set_active_project(args.project)

    if args.command is None or args.command == "serve":
        cmd_serve(args)
    elif args.command == "init":
        cmd_init()
    elif args.command == "test":
        cmd_test()
    elif args.command == "consolidate":
        cmd_consolidate()
    elif args.command == "status":
        cmd_status()
    elif args.command == "export":
        cmd_export()
    elif args.command == "import":
        cmd_import(args.path)
    elif args.command == "reindex":
        cmd_reindex()
    elif args.command == "browse":
        cmd_browse()
    elif args.command == "setup-claude":
        cmd_setup_claude()
    elif args.command == "dashboard":
        cmd_dashboard()


if __name__ == "__main__":
    main()
