"""Central configuration for the consolidation memory system.

Loads settings from a TOML config file, falling back to sensible defaults.
Config discovery order:
  1. CONSOLIDATION_MEMORY_CONFIG environment variable (path to .toml)
  2. Platform config directory (e.g. ~/.config/consolidation_memory/config.toml)
  3. Built-in defaults

All module-level attributes are populated at import time. Existing
`from consolidation_memory.config import X` patterns work unchanged.
Tests can patch these attributes normally with unittest.mock.patch.
"""

import logging as _logging
import os
import re as _re
import sys
from pathlib import Path

from platformdirs import user_data_dir, user_config_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

APP_NAME = "consolidation_memory"


def _find_config_path() -> Path | None:
    """Discover config file location."""
    # 1. Environment variable
    env = os.environ.get("CONSOLIDATION_MEMORY_CONFIG")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    # 2. Platform config directory
    platform_path = Path(user_config_dir(APP_NAME)) / "config.toml"
    if platform_path.exists():
        return platform_path

    return None


def _load_config() -> dict:
    """Load TOML config file, or return empty dict for defaults."""
    path = _find_config_path()
    if path:
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def get_config_path() -> Path | None:
    """Return the active config file path, or None if using defaults."""
    return _find_config_path()


def get_default_config_dir() -> Path:
    """Return the platform-specific config directory."""
    return Path(user_config_dir(APP_NAME))


_cfg = _load_config()

# ── Project name validation ──────────────────────────────────────────────────
_PROJECT_NAME_RE = _re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_project_name(name: str) -> str:
    """Validate and return a project name. Raises ValueError on invalid input."""
    if not name:
        raise ValueError("Project name must not be empty")
    if not _PROJECT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name {name!r}: must match {_PROJECT_NAME_RE.pattern} "
            f"(lowercase alphanumeric, hyphens, underscores; 1-64 chars; "
            f"must start with letter or digit)"
        )
    return name


# ── Paths ────────────────────────────────────────────────────────────────────
_default_data = Path(user_data_dir(APP_NAME))
_paths = _cfg.get("paths", {})

_data_dir_str = _paths.get("data_dir", "")
_base_data_dir = Path(_data_dir_str).expanduser().resolve() if _data_dir_str else _default_data

_active_project = validate_project_name(
    os.environ.get("CONSOLIDATION_MEMORY_PROJECT", "default")
)

DATA_DIR = _base_data_dir / "projects" / _active_project

DB_PATH = DATA_DIR / "memory.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.json"
FAISS_TOMBSTONE_PATH = DATA_DIR / "faiss_tombstones.json"
FAISS_RELOAD_SIGNAL = DATA_DIR / ".faiss_reload"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
CONSOLIDATION_LOG_DIR = DATA_DIR / "consolidation_logs"
LOG_DIR = _base_data_dir / "logs"
BACKUP_DIR = DATA_DIR / "backups"

# ── Embedding model ──────────────────────────────────────────────────────────
_embed = _cfg.get("embedding", {})

EMBEDDING_BACKEND = _embed.get("backend", "fastembed")

# Defaults vary by backend
_EMBED_DEFAULTS = {
    "fastembed": {"model": "BAAI/bge-small-en-v1.5", "dimension": 384},
    "lmstudio": {"model": "text-embedding-nomic-embed-text-v1.5", "dimension": 768},
    "openai": {"model": "text-embedding-3-small", "dimension": 1536},
    "ollama": {"model": "nomic-embed-text", "dimension": 768},
}
_embed_defs = _EMBED_DEFAULTS.get(EMBEDDING_BACKEND, _EMBED_DEFAULTS["fastembed"])

EMBEDDING_MODEL_NAME = _embed.get("model", _embed_defs["model"])
EMBEDDING_DIMENSION = int(_embed.get("dimension", _embed_defs["dimension"]))
EMBEDDING_API_BASE = _embed.get("api_base", "http://127.0.0.1:1234/v1")
EMBEDDING_API_KEY = _embed.get("api_key", "")

# ── Vector store ─────────────────────────────────────────────────────────────
FAISS_SIZE_WARNING_THRESHOLD = 10_000
FAISS_COMPACTION_THRESHOLD = 0.2

# ── FAISS tuning ─────────────────────────────────────────────────────────
_faiss = _cfg.get("faiss", {})
FAISS_SEARCH_FETCH_K_PADDING = int(_faiss.get("search_fetch_k_padding", 0))  # 0 = auto (tombstone count)

# ── LLM API (consolidation summarization) ────────────────────────────────────
_llm = _cfg.get("llm", {})

LLM_BACKEND = _llm.get("backend", "lmstudio")
LLM_API_BASE = _llm.get("api_base", "http://localhost:1234/v1")
LLM_MODEL = _llm.get("model", "qwen2.5-7b-instruct")
LLM_MAX_TOKENS = int(_llm.get("max_tokens", 2048))
LLM_TEMPERATURE = float(_llm.get("temperature", 0.3))
LLM_MIN_P = float(_llm.get("min_p", 0.05))
LLM_API_KEY = _llm.get("api_key", "")
LLM_CALL_TIMEOUT = float(_llm.get("call_timeout", 120))
LLM_CORRECTION_TIMEOUT = float(_llm.get("correction_timeout", 90))
LLM_VALIDATION_RETRY = bool(_cfg.get("consolidation", {}).get("validation_retry", True))

# ── Consolidation ────────────────────────────────────────────────────────────
_consol = _cfg.get("consolidation", {})

CONSOLIDATION_AUTO_RUN = bool(_consol.get("auto_run", True))
CONSOLIDATION_INTERVAL_HOURS = float(_consol.get("interval_hours", 6))
CONSOLIDATION_CLUSTER_THRESHOLD = float(_consol.get("cluster_threshold", 0.78))
CONSOLIDATION_MIN_CLUSTER_SIZE = int(_consol.get("min_cluster_size", 2))
CONSOLIDATION_MAX_CLUSTER_SIZE = int(_consol.get("max_cluster_size", 20))
CONSOLIDATION_PRUNE_ENABLED = bool(_consol.get("prune_enabled", False))
CONSOLIDATION_PRUNE_AFTER_DAYS = int(_consol.get("prune_after_days", 30))
CONSOLIDATION_MAX_EPISODES_PER_RUN = int(_consol.get("max_episodes_per_run", 200))
CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD = float(_consol.get("topic_semantic_match_threshold", 0.75))
CONSOLIDATION_CONFIDENCE_COHERENCE_W = float(_consol.get("cluster_confidence_coherence_weight", 0.6))
CONSOLIDATION_CONFIDENCE_SURPRISE_W = float(_consol.get("cluster_confidence_surprise_weight", 0.4))
CONSOLIDATION_MAX_DURATION = float(_consol.get("max_duration", 1800))
CONSOLIDATION_MAX_ATTEMPTS = int(_consol.get("max_attempts", 5))
CONTRADICTION_SIMILARITY_THRESHOLD = float(_consol.get("contradiction_similarity_threshold", 0.7))
CONTRADICTION_LLM_ENABLED = bool(_consol.get("contradiction_llm_enabled", True))
_DEFAULT_STOPWORDS = frozenset({"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "is", "at", "it"})
_extra_sw = _consol.get("extra_stopwords", [])
CONSOLIDATION_STOPWORDS = _DEFAULT_STOPWORDS | frozenset(_extra_sw)
CONSOLIDATION_PRIORITY_WEIGHTS = {
    "surprise": 0.4,
    "recency": 0.35,
    "access_frequency": 0.25,
}

# ── Knowledge versioning ─────────────────────────────────────────────────────
KNOWLEDGE_VERSIONS_DIR = KNOWLEDGE_DIR / "versions"
KNOWLEDGE_MAX_VERSIONS = 5


# ── Project switching ────────────────────────────────────────────────────────


def get_active_project() -> str:
    """Return the currently active project name."""
    return _active_project


def set_active_project(name: str | None = None) -> str:
    """Switch to a different project namespace.

    Validates the name, updates ``_active_project``, and recalculates every
    path global so that consumer modules accessing e.g. ``config.DB_PATH``
    dynamically will see the new values.

    Args:
        name: Project name to switch to.  If *None*, reads the
              ``CONSOLIDATION_MEMORY_PROJECT`` env var (default ``"default"``).

    Returns:
        The validated project name now active.

    Raises:
        ValueError: If *name* is invalid.
    """
    global _active_project, DATA_DIR, DB_PATH, FAISS_INDEX_PATH
    global FAISS_ID_MAP_PATH, FAISS_TOMBSTONE_PATH, FAISS_RELOAD_SIGNAL
    global KNOWLEDGE_DIR, KNOWLEDGE_VERSIONS_DIR, CONSOLIDATION_LOG_DIR
    global BACKUP_DIR

    if name is None:
        name = os.environ.get("CONSOLIDATION_MEMORY_PROJECT", "default")

    _active_project = validate_project_name(name)

    DATA_DIR = _base_data_dir / "projects" / _active_project
    DB_PATH = DATA_DIR / "memory.db"
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
    FAISS_ID_MAP_PATH = DATA_DIR / "faiss_id_map.json"
    FAISS_TOMBSTONE_PATH = DATA_DIR / "faiss_tombstones.json"
    FAISS_RELOAD_SIGNAL = DATA_DIR / ".faiss_reload"
    KNOWLEDGE_DIR = DATA_DIR / "knowledge"
    KNOWLEDGE_VERSIONS_DIR = KNOWLEDGE_DIR / "versions"
    CONSOLIDATION_LOG_DIR = DATA_DIR / "consolidation_logs"
    BACKUP_DIR = DATA_DIR / "backups"
    # NOTE: LOG_DIR intentionally NOT recalculated — logs are shared across projects.

    return _active_project


# ── Migration from flat layout ──────────────────────────────────────────────


def maybe_migrate_to_projects(base_dir: Path) -> bool:
    """Migrate flat DATA_DIR layout to projects/default/ structure.

    Returns True if migration was performed.
    """
    flat_db = base_dir / "memory.db"
    projects_dir = base_dir / "projects"

    if not flat_db.exists():
        return False  # nothing to migrate

    if projects_dir.exists():
        return False  # already migrated or manual setup

    import shutil

    default_dir = projects_dir / "default"
    default_dir.mkdir(parents=True)

    _FILES_TO_MOVE = [
        "memory.db", "faiss_index.bin", "faiss_id_map.json",
        "faiss_tombstones.json", ".faiss_reload",
    ]
    _DIRS_TO_MOVE = ["knowledge", "backups", "consolidation_logs"]

    try:
        for fname in _FILES_TO_MOVE:
            src = base_dir / fname
            if src.exists():
                shutil.move(str(src), str(default_dir / fname))

        for dname in _DIRS_TO_MOVE:
            src = base_dir / dname
            if src.exists():
                shutil.move(str(src), str(default_dir / dname))
    except OSError:
        # Rollback: move everything back so the next attempt can retry
        for item in default_dir.iterdir():
            shutil.move(str(item), str(base_dir / item.name))
        shutil.rmtree(str(projects_dir), ignore_errors=True)
        raise

    print(
        f"[consolidation-memory] Migrated data to {default_dir}",
        file=sys.stderr,
    )
    return True


# ── Deduplication ─────────────────────────────────────────────────────────────
_dedup = _cfg.get("dedup", {})
DEDUP_SIMILARITY_THRESHOLD = float(_dedup.get("similarity_threshold", 0.95))
DEDUP_ENABLED = bool(_dedup.get("enabled", True))

# ── Adaptive surprise scoring ─────────────────────────────────────────────────
_scoring = _cfg.get("scoring", {})
SURPRISE_BOOST_PER_ACCESS = float(_scoring.get("surprise_boost_per_access", 0.02))
SURPRISE_DECAY_INACTIVE_DAYS = int(_scoring.get("surprise_decay_inactive_days", 7))
SURPRISE_DECAY_RATE = float(_scoring.get("surprise_decay_rate", 0.05))
SURPRISE_MIN = float(_scoring.get("surprise_min", 0.1))
SURPRISE_MAX = float(_scoring.get("surprise_max", 1.0))

# ── Backup / export ──────────────────────────────────────────────────────────
MAX_BACKUPS = 5

# ── MCP server ────────────────────────────────────────────────────────────────
_recall = _cfg.get("recall", {})
RECALL_DEFAULT_N = int(_recall.get("default_n", 10))
RECALL_MAX_N = int(_recall.get("max_n", 50))

# ── Retrieval tuning ────────────────────────────────────────────────────
_retrieval = _cfg.get("retrieval", {})
RECENCY_HALF_LIFE_DAYS = float(_retrieval.get("recency_half_life_days", 90.0))
KNOWLEDGE_SEMANTIC_WEIGHT = float(_retrieval.get("knowledge_semantic_weight", 0.8))
KNOWLEDGE_KEYWORD_WEIGHT = float(_retrieval.get("knowledge_keyword_weight", 0.2))
KNOWLEDGE_RELEVANCE_THRESHOLD = float(_retrieval.get("knowledge_relevance_threshold", 0.25))
KNOWLEDGE_MAX_RESULTS = int(_retrieval.get("knowledge_max_results", 5))

# ── Knowledge records ──────────────────────────────────────────────────
RECORDS_SEMANTIC_WEIGHT = float(_retrieval.get("records_semantic_weight", 0.9))
RECORDS_KEYWORD_WEIGHT = float(_retrieval.get("records_keyword_weight", 0.1))
RECORDS_RELEVANCE_THRESHOLD = float(_retrieval.get("records_relevance_threshold", 0.3))
RECORDS_MAX_RESULTS = int(_retrieval.get("records_max_results", 15))
RENDER_MARKDOWN = bool(_consol.get("render_markdown", True))

# ── Circuit breaker ─────────────────────────────────────────────────────
_cb = _cfg.get("circuit_breaker", {})
CIRCUIT_BREAKER_THRESHOLD = int(_cb.get("threshold", 3))
CIRCUIT_BREAKER_COOLDOWN = float(_cb.get("cooldown", 60.0))


# ── Validation ───────────────────────────────────────────────────────────────

_KNOWN_EMBEDDING_BACKENDS = {"fastembed", "lmstudio", "openai", "ollama"}
_KNOWN_LLM_BACKENDS = {"lmstudio", "openai", "ollama", "disabled"}


def _validate_config() -> None:
    """Validate config values at load time. Raises ValueError on bad config."""
    errors = []

    if EMBEDDING_BACKEND not in _KNOWN_EMBEDDING_BACKENDS:
        errors.append(
            f"embedding.backend = {EMBEDDING_BACKEND!r}, "
            f"expected one of {_KNOWN_EMBEDDING_BACKENDS}"
        )
    if EMBEDDING_DIMENSION <= 0:
        errors.append(f"embedding.dimension = {EMBEDDING_DIMENSION}, must be > 0")
    if not EMBEDDING_MODEL_NAME:
        errors.append("embedding.model is empty")

    if LLM_BACKEND not in _KNOWN_LLM_BACKENDS:
        errors.append(
            f"llm.backend = {LLM_BACKEND!r}, "
            f"expected one of {_KNOWN_LLM_BACKENDS}"
        )
    if LLM_TEMPERATURE < 0:
        errors.append(f"llm.temperature = {LLM_TEMPERATURE}, must be >= 0")

    if not (0.0 < CONSOLIDATION_CLUSTER_THRESHOLD <= 1.0):
        errors.append(
            f"consolidation.cluster_threshold = {CONSOLIDATION_CLUSTER_THRESHOLD}, "
            f"must be in (0, 1]"
        )
    if not (0.0 < DEDUP_SIMILARITY_THRESHOLD <= 1.0):
        errors.append(
            f"dedup.similarity_threshold = {DEDUP_SIMILARITY_THRESHOLD}, "
            f"must be in (0, 1]"
        )

    if errors:
        raise ValueError(
            "Invalid consolidation_memory config:\n  " + "\n  ".join(errors)
        )


try:
    maybe_migrate_to_projects(_base_data_dir)
except OSError as _mig_err:
    print(
        f"[consolidation-memory] Migration skipped (file locked?): {_mig_err}",
        file=sys.stderr,
    )

_validate_config()

_config_logger = _logging.getLogger(__name__)
_config_path = _find_config_path()
_config_logger.info(
    "Config loaded: path=%s, embedding=%s/%s (dim=%d), llm=%s/%s",
    _config_path or "(defaults)",
    EMBEDDING_BACKEND, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    LLM_BACKEND, LLM_MODEL,
)
