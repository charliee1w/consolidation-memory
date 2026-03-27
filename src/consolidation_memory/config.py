"""Central configuration for the consolidation memory system.

Loads settings from a TOML config file, falling back to sensible defaults.
Config discovery order:
  1. CONSOLIDATION_MEMORY_CONFIG environment variable (path to .toml)
  2. Platform config directory (e.g. ~/.config/consolidation_memory/config.toml)
  3. Built-in defaults

After loading, every scalar Config field can be overridden by setting
``CONSOLIDATION_MEMORY_<FIELD_NAME>`` in the environment.  For example,
``CONSOLIDATION_MEMORY_LLM_BACKEND=openai`` overrides the LLM backend.
Priority order: defaults < TOML < env vars < ``reset_config()`` overrides.

Configuration is exposed via a ``Config`` dataclass singleton.  Call
``get_config()`` to obtain the current instance.  Tests should call
``reset_config()`` (optionally passing overrides) to get a fresh instance
without needing ``unittest.mock.patch``.  ``reset_config()`` does **not**
read env vars, keeping tests isolated.
"""

from __future__ import annotations

import logging as _logging
import os
import re as _re
import sys
import threading as _threading
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_dir, user_config_dir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

APP_NAME = "consolidation_memory"

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


# ── TOML discovery helpers (usable before Config exists) ─────────────────────


def _find_config_path() -> Path | None:
    """Discover config file location."""
    env = os.environ.get("CONSOLIDATION_MEMORY_CONFIG")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Config file specified by CONSOLIDATION_MEMORY_CONFIG does not exist: {p}"
        )

    platform_path = Path(user_config_dir(APP_NAME)) / "config.toml"
    if platform_path.exists():
        return platform_path

    return None


def _load_toml() -> dict:
    """Load TOML config file, or return empty dict for defaults."""
    path = _find_config_path()
    if path:
        with open(path, "rb") as f:
            loaded: dict = tomllib.load(f)
            return loaded
    return {}


def get_config_path() -> Path | None:
    """Return the active config file path, or None if using defaults."""
    return _find_config_path()


def get_default_config_dir() -> Path:
    """Return the platform-specific config directory."""
    return Path(user_config_dir(APP_NAME))


# ── Embedding backend defaults ───────────────────────────────────────────────

_EMBED_DEFAULTS: dict[str, dict[str, str | int]] = {
    "fastembed": {"model": "BAAI/bge-small-en-v1.5", "dimension": 384},
    "lmstudio": {"model": "text-embedding-nomic-embed-text-v1.5", "dimension": 768},
    "openai": {"model": "text-embedding-3-small", "dimension": 1536},
    "ollama": {"model": "nomic-embed-text", "dimension": 768},
}

_DEFAULT_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "is", "at", "it"}
)
_UTILITY_WEIGHT_KEYS = {
    "unconsolidated_backlog",
    "recall_miss_fallback",
    "contradiction_spike",
    "challenged_claim_backlog",
}

# ── Validation sets ──────────────────────────────────────────────────────────

_KNOWN_EMBEDDING_BACKENDS = {"fastembed", "lmstudio", "openai", "ollama"}
_KNOWN_LLM_BACKENDS = {"lmstudio", "openai", "ollama", "disabled"}

_ENV_PREFIX = "CONSOLIDATION_MEMORY_"

# Types eligible for env var override (primitives only)
_ENV_COERCIBLE_TYPES = (str, int, float, bool)


# ── Config dataclass ─────────────────────────────────────────────────────────


@dataclass
class Config:
    """All consolidation-memory settings in one place.

    Constructed from TOML + env vars by ``_build_config()``.  Access the
    singleton via ``get_config()``.
    """

    # ── internal bookkeeping (not user-facing) ────────────────────────────
    _base_data_dir: Path = field(default_factory=lambda: Path(user_data_dir(APP_NAME)))
    active_project: str = "default"

    # ── Paths (derived from _base_data_dir + active_project) ─────────────
    DATA_DIR: Path = field(default_factory=lambda: Path("."))
    DB_PATH: Path = field(default_factory=lambda: Path("."))
    FAISS_INDEX_PATH: Path = field(default_factory=lambda: Path("."))
    FAISS_ID_MAP_PATH: Path = field(default_factory=lambda: Path("."))
    FAISS_TOMBSTONE_PATH: Path = field(default_factory=lambda: Path("."))
    FAISS_RELOAD_SIGNAL: Path = field(default_factory=lambda: Path("."))
    FAISS_WRITE_LOCK_PATH: Path = field(default_factory=lambda: Path("."))
    KNOWLEDGE_DIR: Path = field(default_factory=lambda: Path("."))
    CONSOLIDATION_LOG_DIR: Path = field(default_factory=lambda: Path("."))
    LOG_DIR: Path = field(default_factory=lambda: Path("."))
    BACKUP_DIR: Path = field(default_factory=lambda: Path("."))
    KNOWLEDGE_VERSIONS_DIR: Path = field(default_factory=lambda: Path("."))

    # ── Embedding model ──────────────────────────────────────────────────
    EMBEDDING_BACKEND: str = "fastembed"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_API_BASE: str = "http://127.0.0.1:1234/v1"
    EMBEDDING_API_KEY: str = ""

    # ── Vector store ─────────────────────────────────────────────────────
    FAISS_SIZE_WARNING_THRESHOLD: int = 10_000
    FAISS_COMPACTION_THRESHOLD: float = 0.2
    FAISS_SEARCH_FETCH_K_PADDING: int = 0
    FAISS_IVF_UPGRADE_THRESHOLD: int = 10_000
    FAISS_WRITE_LOCK_TIMEOUT_SECONDS: float = 30.0
    FAISS_PLATFORM_REVIEW_THRESHOLD: int = 100_000

    # ── LLM API ──────────────────────────────────────────────────────────
    LLM_BACKEND: str = "lmstudio"
    LLM_API_BASE: str = "http://localhost:1234/v1"
    LLM_MODEL: str = "qwen2.5-7b-instruct"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.3
    LLM_MIN_P: float = 0.05
    LLM_API_KEY: str = ""
    LLM_CALL_TIMEOUT: float = 120
    LLM_CORRECTION_TIMEOUT: float = 90
    LLM_VALIDATION_RETRY: bool = True

    # ── Consolidation ────────────────────────────────────────────────────
    CONSOLIDATION_AUTO_RUN: bool = True
    CONSOLIDATION_INTERVAL_HOURS: float = 6
    CONSOLIDATION_CLUSTER_THRESHOLD: float = 0.78
    CONSOLIDATION_MIN_CLUSTER_SIZE: int = 2
    CONSOLIDATION_MAX_CLUSTER_SIZE: int = 20
    CONSOLIDATION_PRUNE_ENABLED: bool = False
    CONSOLIDATION_PRUNE_AFTER_DAYS: int = 30
    DECAY_POLICIES: dict[str, int] = field(default_factory=dict)  # tag -> days (-1=never)
    CONSOLIDATION_MAX_EPISODES_PER_RUN: int = 200
    CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD: float = 0.84
    CONSOLIDATION_TOPIC_TITLE_OVERLAP_THRESHOLD: float = 0.34
    CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD: float = 0.9
    CONSOLIDATION_CONFIDENCE_COHERENCE_W: float = 0.6
    CONSOLIDATION_CONFIDENCE_SURPRISE_W: float = 0.4
    CONSOLIDATION_MAX_DURATION: float = 1800
    CONSOLIDATION_MAX_ATTEMPTS: int = 5
    CONSOLIDATION_UTILITY_THRESHOLD: float = 0.6
    CONSOLIDATION_UTILITY_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "unconsolidated_backlog": 0.4,
            "recall_miss_fallback": 0.2,
            "contradiction_spike": 0.2,
            "challenged_claim_backlog": 0.2,
        }
    )
    CONTRADICTION_SIMILARITY_THRESHOLD: float = 0.7
    CONTRADICTION_LLM_ENABLED: bool = True
    CONTRADICTION_MAX_CANDIDATE_PAIRS: int = 24
    CONTRADICTION_LLM_BATCH_SIZE: int = 6
    CONTRADICTION_LLM_MAX_RETRIES: int = 1
    CONTRADICTION_PROMPT_RECORD_CHAR_LIMIT: int = 700
    MERGE_DROP_DETECTION_ENABLED: bool = True
    MERGE_DROP_SIMILARITY_THRESHOLD: float = 0.5
    CONSOLIDATION_STOPWORDS: frozenset[str] = field(default_factory=lambda: _DEFAULT_STOPWORDS)
    CONSOLIDATION_PRIORITY_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {"surprise": 0.4, "recency": 0.35, "access_frequency": 0.25}
    )
    RENDER_MARKDOWN: bool = True

    # ── Knowledge versioning ─────────────────────────────────────────────
    KNOWLEDGE_MAX_VERSIONS: int = 5

    # ── Deduplication ────────────────────────────────────────────────────
    DEDUP_SIMILARITY_THRESHOLD: float = 0.95
    DEDUP_ENABLED: bool = True

    # ── Adaptive surprise scoring ────────────────────────────────────────
    SURPRISE_BOOST_PER_ACCESS: float = 0.02
    SURPRISE_DECAY_INACTIVE_DAYS: int = 7
    SURPRISE_DECAY_RATE: float = 0.05
    SURPRISE_MIN: float = 0.1
    SURPRISE_MAX: float = 1.0

    # ── Backup / export ──────────────────────────────────────────────────
    MAX_BACKUPS: int = 5

    # ── MCP server ───────────────────────────────────────────────────────
    RECALL_DEFAULT_N: int = 10
    RECALL_MAX_N: int = 50

    # ── Hybrid search (BM25 + semantic) ─────────────────────────────────
    HYBRID_SEARCH_ENABLED: bool = True
    HYBRID_SEMANTIC_WEIGHT: float = 0.7
    HYBRID_KEYWORD_WEIGHT: float = 0.3
    HYBRID_FTS_CANDIDATES: int = 50

    # ── Retrieval tuning ─────────────────────────────────────────────────
    RECENCY_HALF_LIFE_DAYS: float = 90.0
    KNOWLEDGE_SEMANTIC_WEIGHT: float = 0.8
    KNOWLEDGE_KEYWORD_WEIGHT: float = 0.2
    KNOWLEDGE_RELEVANCE_THRESHOLD: float = 0.25
    KNOWLEDGE_MAX_RESULTS: int = 5

    # ── Knowledge records ────────────────────────────────────────────────
    RECORDS_SEMANTIC_WEIGHT: float = 0.9
    RECORDS_KEYWORD_WEIGHT: float = 0.1
    RECORDS_RELEVANCE_THRESHOLD: float = 0.3
    RECORDS_MAX_RESULTS: int = 15

    # ── Uncertainty signaling ────────────────────────────────────────────
    EVOLVING_TOPIC_LOOKBACK_DAYS: int = 30
    KNOWLEDGE_CONSISTENCY_THRESHOLD: float = 0.995

    # ── Recall deduplication ──────────────────────────────────────────────
    RECALL_DEDUP_ENABLED: bool = True

    # ── Plugins ───────────────────────────────────────────────────────────
    PLUGINS_ENABLED: list[str] = field(default_factory=list)

    # ── Circuit breaker ──────────────────────────────────────────────────
    CIRCUIT_BREAKER_THRESHOLD: int = 3
    CIRCUIT_BREAKER_COOLDOWN: float = 60.0

    _REDACT_FIELDS: frozenset[str] = field(
        default=frozenset({"EMBEDDING_API_KEY", "LLM_API_KEY"}), repr=False,
    )

    def __repr__(self) -> str:
        import dataclasses as _dc
        parts = []
        for f in _dc.fields(self):
            if not f.repr:
                continue
            val = getattr(self, f.name)
            if f.name in self._REDACT_FIELDS and val:
                val = "***"
            parts.append(f"{f.name}={val!r}")
        return f"Config({', '.join(parts)})"

    def _recompute_paths(self) -> None:
        """Recalculate all derived paths from _base_data_dir + active_project."""
        self.DATA_DIR = self._base_data_dir / "projects" / self.active_project
        self.DB_PATH = self.DATA_DIR / "memory.db"
        self.FAISS_INDEX_PATH = self.DATA_DIR / "faiss_index.bin"
        self.FAISS_ID_MAP_PATH = self.DATA_DIR / "faiss_id_map.json"
        self.FAISS_TOMBSTONE_PATH = self.DATA_DIR / "faiss_tombstones.json"
        self.FAISS_RELOAD_SIGNAL = self.DATA_DIR / ".faiss_reload"
        self.FAISS_WRITE_LOCK_PATH = self.DATA_DIR / ".faiss_write.lock"
        self.KNOWLEDGE_DIR = self.DATA_DIR / "knowledge"
        self.KNOWLEDGE_VERSIONS_DIR = self.KNOWLEDGE_DIR / "versions"
        self.CONSOLIDATION_LOG_DIR = self.DATA_DIR / "consolidation_logs"
        self.LOG_DIR = self._base_data_dir / "logs"
        self.BACKUP_DIR = self.DATA_DIR / "backups"


def _apply_env_overrides(c: Config) -> None:
    """Override Config fields from ``CONSOLIDATION_MEMORY_*`` env vars.

    Handles str, int, float, bool fields.  Skips private fields (``_``
    prefix), ``active_project`` (handled by ``CONSOLIDATION_MEMORY_PROJECT``),
    Path fields, and complex types (frozenset, dict).

    Special case: ``CONSOLIDATION_MEMORY_DATA_DIR`` sets ``_base_data_dir``.
    """
    import dataclasses as _dc

    # Special: DATA_DIR -> _base_data_dir (not a regular field loop candidate)
    data_dir_env = os.environ.get(f"{_ENV_PREFIX}DATA_DIR")
    if data_dir_env:
        c._base_data_dir = Path(data_dir_env).expanduser().resolve()

    for f in _dc.fields(c):
        if f.name.startswith("_"):
            continue
        if f.name == "active_project":
            continue  # handled by CONSOLIDATION_MEMORY_PROJECT

        env_key = f"{_ENV_PREFIX}{f.name}"
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue

        # Use the runtime type of the current value
        field_type = type(getattr(c, f.name))

        if field_type not in _ENV_COERCIBLE_TYPES:
            continue  # skip Path, frozenset, dict, etc.

        try:
            if field_type is bool:
                coerced: object = env_val.lower() in ("1", "true", "yes")
            else:
                coerced = field_type(env_val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{env_key}={env_val!r}: cannot convert to "
                f"{field_type.__name__}: {exc}"
            ) from exc

        object.__setattr__(c, f.name, coerced)


def _coerce_bool(val: object, default: bool = False) -> bool:
    """Coerce a value to bool, handling TOML booleans and string edge cases."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes")
    return bool(val) if val is not None else default


def _build_config(
    toml: dict | None = None,
    *,
    _load_env: bool = False,
    **overrides: object,
) -> Config:
    """Build a Config from TOML dict + optional field overrides.

    This is the single place where TOML keys are mapped to Config fields.
    When *_load_env* is True, ``CONSOLIDATION_MEMORY_*`` environment
    variables override TOML values (but are themselves overridden by
    explicit *overrides* kwargs, keeping tests isolated).
    """
    cfg = toml if toml is not None else {}

    _paths = cfg.get("paths", {})
    _embed = cfg.get("embedding", {})
    _faiss = cfg.get("faiss", {})
    _llm = cfg.get("llm", {})
    _consol = cfg.get("consolidation", {})
    _dedup = cfg.get("dedup", {})
    _scoring = cfg.get("scoring", {})
    _recall = cfg.get("recall", {})
    _retrieval = cfg.get("retrieval", {})
    _cb = cfg.get("circuit_breaker", {})
    _storage = cfg.get("storage", {})
    _plugins = cfg.get("plugins", {})

    # Base data dir
    _data_dir_str = _paths.get("data_dir", "")
    base_data_dir = (
        Path(_data_dir_str).expanduser().resolve()
        if _data_dir_str
        else Path(user_data_dir(APP_NAME))
    )

    # Active project
    active_project = validate_project_name(
        os.environ.get("CONSOLIDATION_MEMORY_PROJECT", "default")
    )

    # Embedding defaults based on backend
    embed_backend = _embed.get("backend", "fastembed")
    _edefs = _EMBED_DEFAULTS.get(embed_backend, _EMBED_DEFAULTS["fastembed"])

    # Extra stopwords
    extra_sw = _consol.get("extra_stopwords", [])
    stopwords = _DEFAULT_STOPWORDS | frozenset(extra_sw)

    c = Config(
        _base_data_dir=base_data_dir,
        active_project=active_project,
        # Embedding
        EMBEDDING_BACKEND=embed_backend,
        EMBEDDING_MODEL_NAME=_embed.get("model", _edefs["model"]),
        EMBEDDING_DIMENSION=int(_embed.get("dimension", _edefs["dimension"])),
        EMBEDDING_API_BASE=_embed.get("api_base", "http://127.0.0.1:1234/v1"),
        EMBEDDING_API_KEY=_embed.get("api_key", ""),
        # FAISS
        FAISS_SIZE_WARNING_THRESHOLD=int(_faiss.get("size_warning_threshold", 10_000)),
        FAISS_COMPACTION_THRESHOLD=float(_faiss.get("compaction_threshold", 0.2)),
        FAISS_SEARCH_FETCH_K_PADDING=int(_faiss.get("search_fetch_k_padding", 0)),
        FAISS_IVF_UPGRADE_THRESHOLD=int(_faiss.get("ivf_upgrade_threshold", 10_000)),
        FAISS_WRITE_LOCK_TIMEOUT_SECONDS=float(_faiss.get("write_lock_timeout_seconds", 30.0)),
        FAISS_PLATFORM_REVIEW_THRESHOLD=int(_faiss.get("platform_review_threshold", 100_000)),
        # LLM
        LLM_BACKEND=_llm.get("backend", "lmstudio"),
        LLM_API_BASE=_llm.get("api_base", "http://localhost:1234/v1"),
        LLM_MODEL=_llm.get("model", "qwen2.5-7b-instruct"),
        LLM_MAX_TOKENS=int(_llm.get("max_tokens", 2048)),
        LLM_TEMPERATURE=float(_llm.get("temperature", 0.3)),
        LLM_MIN_P=float(_llm.get("min_p", 0.05)),
        LLM_API_KEY=_llm.get("api_key", ""),
        LLM_CALL_TIMEOUT=float(_llm.get("call_timeout", 120)),
        LLM_CORRECTION_TIMEOUT=float(_llm.get("correction_timeout", 90)),
        LLM_VALIDATION_RETRY=_coerce_bool(_llm.get("validation_retry", True)),
        # Consolidation
        CONSOLIDATION_AUTO_RUN=_coerce_bool(_consol.get("auto_run", True)),
        CONSOLIDATION_INTERVAL_HOURS=float(_consol.get("interval_hours", 6)),
        CONSOLIDATION_CLUSTER_THRESHOLD=float(_consol.get("cluster_threshold", 0.78)),
        CONSOLIDATION_MIN_CLUSTER_SIZE=int(_consol.get("min_cluster_size", 2)),
        CONSOLIDATION_MAX_CLUSTER_SIZE=int(_consol.get("max_cluster_size", 20)),
        CONSOLIDATION_PRUNE_ENABLED=_coerce_bool(_consol.get("prune_enabled", False)),
        CONSOLIDATION_PRUNE_AFTER_DAYS=int(_consol.get("prune_after_days", 30)),
        DECAY_POLICIES={k: int(v) for k, v in cfg.get("decay_policies", {}).get("overrides", {}).items()},
        CONSOLIDATION_MAX_EPISODES_PER_RUN=int(_consol.get("max_episodes_per_run", 200)),
        CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD=float(_consol.get("topic_semantic_match_threshold", 0.84)),
        CONSOLIDATION_TOPIC_TITLE_OVERLAP_THRESHOLD=float(
            _consol.get("topic_title_overlap_threshold", 0.34)
        ),
        CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD=float(
            _consol.get("topic_force_semantic_threshold", 0.9)
        ),
        CONSOLIDATION_CONFIDENCE_COHERENCE_W=float(_consol.get("cluster_confidence_coherence_weight", 0.6)),
        CONSOLIDATION_CONFIDENCE_SURPRISE_W=float(_consol.get("cluster_confidence_surprise_weight", 0.4)),
        CONSOLIDATION_MAX_DURATION=float(_consol.get("max_duration", 1800)),
        CONSOLIDATION_MAX_ATTEMPTS=int(_consol.get("max_attempts", 5)),
        CONSOLIDATION_UTILITY_THRESHOLD=float(_consol.get("utility_threshold", 0.6)),
        CONSOLIDATION_UTILITY_WEIGHTS=_consol.get(
            "utility_weights",
            {
                "unconsolidated_backlog": 0.4,
                "recall_miss_fallback": 0.2,
                "contradiction_spike": 0.2,
                "challenged_claim_backlog": 0.2,
            },
        ),
        CONTRADICTION_SIMILARITY_THRESHOLD=float(_consol.get("contradiction_similarity_threshold", 0.7)),
        CONTRADICTION_LLM_ENABLED=_coerce_bool(_consol.get("contradiction_llm_enabled", True)),
        CONTRADICTION_MAX_CANDIDATE_PAIRS=int(_consol.get("contradiction_max_candidate_pairs", 24)),
        CONTRADICTION_LLM_BATCH_SIZE=int(_consol.get("contradiction_llm_batch_size", 6)),
        CONTRADICTION_LLM_MAX_RETRIES=int(_consol.get("contradiction_llm_max_retries", 1)),
        CONTRADICTION_PROMPT_RECORD_CHAR_LIMIT=int(
            _consol.get("contradiction_prompt_record_char_limit", 700)
        ),
        MERGE_DROP_DETECTION_ENABLED=_coerce_bool(_consol.get("merge_drop_detection_enabled", True)),
        MERGE_DROP_SIMILARITY_THRESHOLD=float(_consol.get("merge_drop_similarity_threshold", 0.5)),
        CONSOLIDATION_STOPWORDS=stopwords,
        CONSOLIDATION_PRIORITY_WEIGHTS=_consol.get(
            "priority_weights",
            {"surprise": 0.4, "recency": 0.35, "access_frequency": 0.25},
        ),
        KNOWLEDGE_MAX_VERSIONS=int(_consol.get("knowledge_max_versions", 5)),
        RENDER_MARKDOWN=_coerce_bool(_consol.get("render_markdown", True)),
        # Dedup
        DEDUP_SIMILARITY_THRESHOLD=float(_dedup.get("similarity_threshold", 0.95)),
        DEDUP_ENABLED=_coerce_bool(_dedup.get("enabled", True)),
        # Scoring
        SURPRISE_BOOST_PER_ACCESS=float(_scoring.get("surprise_boost_per_access", 0.02)),
        SURPRISE_DECAY_INACTIVE_DAYS=int(_scoring.get("surprise_decay_inactive_days", 7)),
        SURPRISE_DECAY_RATE=float(_scoring.get("surprise_decay_rate", 0.05)),
        SURPRISE_MIN=float(_scoring.get("surprise_min", 0.1)),
        SURPRISE_MAX=float(_scoring.get("surprise_max", 1.0)),
        # Storage
        MAX_BACKUPS=int(_storage.get("max_backups", 5)),
        # Recall
        RECALL_DEFAULT_N=int(_recall.get("default_n", 10)),
        RECALL_MAX_N=int(_recall.get("max_n", 50)),
        # Hybrid search
        HYBRID_SEARCH_ENABLED=_coerce_bool(_retrieval.get("hybrid_search_enabled", True)),
        HYBRID_SEMANTIC_WEIGHT=float(_retrieval.get("hybrid_semantic_weight", 0.7)),
        HYBRID_KEYWORD_WEIGHT=float(_retrieval.get("hybrid_keyword_weight", 0.3)),
        HYBRID_FTS_CANDIDATES=int(_retrieval.get("hybrid_fts_candidates", 50)),
        # Retrieval
        RECENCY_HALF_LIFE_DAYS=float(_retrieval.get("recency_half_life_days", 90.0)),
        KNOWLEDGE_SEMANTIC_WEIGHT=float(_retrieval.get("knowledge_semantic_weight", 0.8)),
        KNOWLEDGE_KEYWORD_WEIGHT=float(_retrieval.get("knowledge_keyword_weight", 0.2)),
        KNOWLEDGE_RELEVANCE_THRESHOLD=float(_retrieval.get("knowledge_relevance_threshold", 0.25)),
        KNOWLEDGE_MAX_RESULTS=int(_retrieval.get("knowledge_max_results", 5)),
        RECORDS_SEMANTIC_WEIGHT=float(_retrieval.get("records_semantic_weight", 0.9)),
        RECORDS_KEYWORD_WEIGHT=float(_retrieval.get("records_keyword_weight", 0.1)),
        RECORDS_RELEVANCE_THRESHOLD=float(_retrieval.get("records_relevance_threshold", 0.3)),
        RECORDS_MAX_RESULTS=int(_retrieval.get("records_max_results", 15)),
        EVOLVING_TOPIC_LOOKBACK_DAYS=int(_retrieval.get("evolving_topic_lookback_days", 30)),
        KNOWLEDGE_CONSISTENCY_THRESHOLD=float(
            _retrieval.get("knowledge_consistency_threshold", 0.995)
        ),
        RECALL_DEDUP_ENABLED=_coerce_bool(_retrieval.get("recall_dedup_enabled", True)),
        # Plugins
        PLUGINS_ENABLED=list(_plugins.get("enabled", [])),
        # Circuit breaker
        CIRCUIT_BREAKER_THRESHOLD=int(_cb.get("threshold", 3)),
        CIRCUIT_BREAKER_COOLDOWN=float(_cb.get("cooldown", 60.0)),
    )

    # Env vars override TOML (but not test overrides)
    if _load_env:
        _apply_env_overrides(c)

    # Apply any explicit overrides (used by reset_config)
    for k, v in overrides.items():
        if hasattr(c, k):
            object.__setattr__(c, k, v)

    c._recompute_paths()
    return c


# ── Singleton ────────────────────────────────────────────────────────────────

_config_instance: Config | None = None
_config_lock = _threading.Lock()


def get_config() -> Config:
    """Return the Config singleton, building from TOML on first access."""
    global _config_instance
    if _config_instance is not None:
        return _config_instance
    with _config_lock:
        if _config_instance is not None:
            return _config_instance
        toml = _load_toml()
        _config_instance = _build_config(toml, _load_env=True)
        _validate_config(_config_instance)

        try:
            maybe_migrate_to_projects(_config_instance._base_data_dir)
        except OSError as err:
            print(
                f"[consolidation-memory] Migration skipped (file locked?): {err}",
                file=sys.stderr,
            )

        logger = _logging.getLogger(__name__)
        config_path = _find_config_path()
        logger.info(
            "Config loaded: path=%s, embedding=%s/%s (dim=%d), llm=%s/%s",
            config_path or "(defaults)",
            _config_instance.EMBEDDING_BACKEND,
            _config_instance.EMBEDDING_MODEL_NAME,
            _config_instance.EMBEDDING_DIMENSION,
            _config_instance.LLM_BACKEND,
            _config_instance.LLM_MODEL,
        )
    return _config_instance


def reset_config(**overrides: object) -> Config:
    """Reset the singleton, optionally applying field overrides.

    Intended for tests::

        cfg = reset_config(
            _base_data_dir=tmp_path / "data",
            EMBEDDING_DIMENSION=384,
            EMBEDDING_BACKEND="fastembed",
        )

    After calling this, the next ``get_config()`` returns the fresh instance.
    """
    global _config_instance
    _config_instance = _build_config({}, _load_env=False, **overrides)
    return _config_instance


class override_config:
    """Context manager to temporarily override Config fields.

    Usage in tests::

        with override_config(FAISS_IVF_UPGRADE_THRESHOLD=100):
            ...  # get_config() returns instance with threshold=100

    Restores previous values on exit.
    """

    def __init__(self, **overrides: object) -> None:
        self._overrides = overrides
        self._originals: dict[str, object] = {}

    def __enter__(self) -> Config:
        cfg = get_config()
        for k, v in self._overrides.items():
            self._originals[k] = getattr(cfg, k)
            object.__setattr__(cfg, k, v)
        cfg._recompute_paths()
        return cfg

    def __exit__(self, *exc: object) -> None:
        cfg = get_config()
        for k, v in self._originals.items():
            object.__setattr__(cfg, k, v)
        cfg._recompute_paths()


# ── Project switching ────────────────────────────────────────────────────────


def get_active_project() -> str:
    """Return the currently active project name."""
    return get_config().active_project


def set_active_project(name: str | None = None) -> str:
    """Switch to a different project namespace.

    Validates the name, updates the active project, and recalculates
    every path so that consumer modules see the new values.

    Args:
        name: Project name to switch to.  If *None*, reads the
              ``CONSOLIDATION_MEMORY_PROJECT`` env var (default ``"default"``).

    Returns:
        The validated project name now active.

    Raises:
        ValueError: If *name* is invalid.
    """
    if name is None:
        name = os.environ.get("CONSOLIDATION_MEMORY_PROJECT", "default")

    cfg = get_config()
    cfg.active_project = validate_project_name(name)
    cfg._recompute_paths()
    # Drop stale thread-local DB connections so subsequent DB calls use
    # the new project's path immediately.
    try:
        from consolidation_memory.database import close_thread_local_connection
        close_thread_local_connection()
    except Exception:
        # Keep project switching resilient even if DB module isn't initialized.
        pass
    return cfg.active_project


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
        "faiss_tombstones.json", ".faiss_reload", ".faiss_write.lock",
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


# ── Validation ───────────────────────────────────────────────────────────────


def _validate_config(c: Config) -> None:
    """Validate config values. Raises ValueError on bad config."""
    errors = []

    if c.EMBEDDING_BACKEND not in _KNOWN_EMBEDDING_BACKENDS:
        errors.append(
            f"embedding.backend = {c.EMBEDDING_BACKEND!r}, "
            f"expected one of {_KNOWN_EMBEDDING_BACKENDS}"
        )
    if c.EMBEDDING_DIMENSION <= 0:
        errors.append(f"embedding.dimension = {c.EMBEDDING_DIMENSION}, must be > 0")
    if not c.EMBEDDING_MODEL_NAME:
        errors.append("embedding.model is empty")

    if c.LLM_BACKEND not in _KNOWN_LLM_BACKENDS:
        errors.append(
            f"llm.backend = {c.LLM_BACKEND!r}, "
            f"expected one of {_KNOWN_LLM_BACKENDS}"
        )
    if c.LLM_TEMPERATURE < 0:
        errors.append(f"llm.temperature = {c.LLM_TEMPERATURE}, must be >= 0")

    if not (0.0 < c.CONSOLIDATION_CLUSTER_THRESHOLD <= 1.0):
        errors.append(
            f"consolidation.cluster_threshold = {c.CONSOLIDATION_CLUSTER_THRESHOLD}, "
            f"must be in (0, 1]"
        )
    if not (0.0 < c.CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD <= 1.0):
        errors.append(
            "consolidation.topic_semantic_match_threshold = "
            f"{c.CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD}, must be in (0, 1]"
        )
    if not (0.0 <= c.CONSOLIDATION_TOPIC_TITLE_OVERLAP_THRESHOLD <= 1.0):
        errors.append(
            "consolidation.topic_title_overlap_threshold = "
            f"{c.CONSOLIDATION_TOPIC_TITLE_OVERLAP_THRESHOLD}, must be in [0, 1]"
        )
    if not (0.0 < c.CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD <= 1.0):
        errors.append(
            "consolidation.topic_force_semantic_threshold = "
            f"{c.CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD}, must be in (0, 1]"
        )
    if c.CONSOLIDATION_TOPIC_FORCE_SEMANTIC_THRESHOLD < c.CONSOLIDATION_TOPIC_SEMANTIC_THRESHOLD:
        errors.append(
            "consolidation.topic_force_semantic_threshold must be >= "
            "consolidation.topic_semantic_match_threshold"
        )
    if c.CONTRADICTION_MAX_CANDIDATE_PAIRS <= 0:
        errors.append(
            "consolidation.contradiction_max_candidate_pairs = "
            f"{c.CONTRADICTION_MAX_CANDIDATE_PAIRS}, must be > 0"
        )
    if c.CONTRADICTION_LLM_BATCH_SIZE <= 0:
        errors.append(
            "consolidation.contradiction_llm_batch_size = "
            f"{c.CONTRADICTION_LLM_BATCH_SIZE}, must be > 0"
        )
    if c.CONTRADICTION_LLM_MAX_RETRIES <= 0:
        errors.append(
            "consolidation.contradiction_llm_max_retries = "
            f"{c.CONTRADICTION_LLM_MAX_RETRIES}, must be > 0"
        )
    if c.CONTRADICTION_PROMPT_RECORD_CHAR_LIMIT < 128:
        errors.append(
            "consolidation.contradiction_prompt_record_char_limit = "
            f"{c.CONTRADICTION_PROMPT_RECORD_CHAR_LIMIT}, must be >= 128"
        )
    if not (0.0 < c.DEDUP_SIMILARITY_THRESHOLD <= 1.0):
        errors.append(
            f"dedup.similarity_threshold = {c.DEDUP_SIMILARITY_THRESHOLD}, "
            f"must be in (0, 1]"
        )
    if c.FAISS_WRITE_LOCK_TIMEOUT_SECONDS <= 0:
        errors.append(
            "faiss.write_lock_timeout_seconds = "
            f"{c.FAISS_WRITE_LOCK_TIMEOUT_SECONDS}, must be > 0"
        )
    if c.FAISS_PLATFORM_REVIEW_THRESHOLD <= 0:
        errors.append(
            "faiss.platform_review_threshold = "
            f"{c.FAISS_PLATFORM_REVIEW_THRESHOLD}, must be > 0"
        )

    if c.CONSOLIDATION_INTERVAL_HOURS <= 0:
        errors.append(f"consolidation.interval_hours = {c.CONSOLIDATION_INTERVAL_HOURS}, must be > 0")
    if c.CONSOLIDATION_MAX_DURATION <= 0:
        errors.append(f"consolidation.max_duration = {c.CONSOLIDATION_MAX_DURATION}, must be > 0")
    if not (0.0 <= c.CONSOLIDATION_UTILITY_THRESHOLD <= 1.0):
        errors.append(
            "consolidation.utility_threshold = "
            f"{c.CONSOLIDATION_UTILITY_THRESHOLD}, must be in [0, 1]"
        )
    if c.LLM_CALL_TIMEOUT <= 0:
        errors.append(f"llm.call_timeout = {c.LLM_CALL_TIMEOUT}, must be > 0")
    if c.CONSOLIDATION_MIN_CLUSTER_SIZE < 1:
        errors.append(f"consolidation.min_cluster_size = {c.CONSOLIDATION_MIN_CLUSTER_SIZE}, must be >= 1")
    if c.CONSOLIDATION_MAX_CLUSTER_SIZE < c.CONSOLIDATION_MIN_CLUSTER_SIZE:
        errors.append(
            f"consolidation.max_cluster_size = {c.CONSOLIDATION_MAX_CLUSTER_SIZE}, "
            f"must be >= min_cluster_size ({c.CONSOLIDATION_MIN_CLUSTER_SIZE})"
        )
    if c.SURPRISE_MIN >= c.SURPRISE_MAX:
        errors.append(
            f"surprise_min = {c.SURPRISE_MIN} must be < surprise_max = {c.SURPRISE_MAX}"
        )
    if c.CIRCUIT_BREAKER_THRESHOLD < 1:
        errors.append(f"circuit_breaker.threshold = {c.CIRCUIT_BREAKER_THRESHOLD}, must be >= 1")
    if c.RECENCY_HALF_LIFE_DAYS <= 0:
        errors.append(
            f"retrieval.recency_half_life_days = {c.RECENCY_HALF_LIFE_DAYS}, must be > 0"
        )
    if not (0.0 < c.KNOWLEDGE_CONSISTENCY_THRESHOLD <= 1.0):
        errors.append(
            "retrieval.knowledge_consistency_threshold = "
            f"{c.KNOWLEDGE_CONSISTENCY_THRESHOLD}, must be in (0, 1]"
        )

    # Weight sum validations
    weight_pairs = [
        ("hybrid_semantic_weight", c.HYBRID_SEMANTIC_WEIGHT,
         "hybrid_keyword_weight", c.HYBRID_KEYWORD_WEIGHT),
        ("knowledge_semantic_weight", c.KNOWLEDGE_SEMANTIC_WEIGHT,
         "knowledge_keyword_weight", c.KNOWLEDGE_KEYWORD_WEIGHT),
        ("records_semantic_weight", c.RECORDS_SEMANTIC_WEIGHT,
         "records_keyword_weight", c.RECORDS_KEYWORD_WEIGHT),
        ("consolidation_confidence_coherence_w", c.CONSOLIDATION_CONFIDENCE_COHERENCE_W,
         "consolidation_confidence_surprise_w", c.CONSOLIDATION_CONFIDENCE_SURPRISE_W),
    ]
    for name_a, val_a, name_b, val_b in weight_pairs:
        total = val_a + val_b
        if not (0.99 <= total <= 1.01):
            errors.append(
                f"{name_a} ({val_a}) + {name_b} ({val_b}) = {total}, should sum to 1.0"
            )

    _required_pw_keys = {"surprise", "recency", "access_frequency"}
    _actual_pw_keys = set(c.CONSOLIDATION_PRIORITY_WEIGHTS.keys())
    if _actual_pw_keys != _required_pw_keys:
        errors.append(
            f"consolidation.priority_weights must have keys {_required_pw_keys}, "
            f"got {_actual_pw_keys}"
        )
    pw_sum = sum(c.CONSOLIDATION_PRIORITY_WEIGHTS.values())
    if not (0.99 <= pw_sum <= 1.01):
        errors.append(
            f"consolidation.priority_weights values sum to {pw_sum}, should sum to 1.0"
        )

    required_utility_keys = _UTILITY_WEIGHT_KEYS
    actual_utility_keys = set(c.CONSOLIDATION_UTILITY_WEIGHTS.keys())
    if actual_utility_keys != required_utility_keys:
        errors.append(
            f"consolidation.utility_weights must have keys {required_utility_keys}, "
            f"got {actual_utility_keys}"
        )
    utility_sum = sum(c.CONSOLIDATION_UTILITY_WEIGHTS.values())
    if not (0.99 <= utility_sum <= 1.01):
        errors.append(
            f"consolidation.utility_weights values sum to {utility_sum}, should sum to 1.0"
        )
    for key, value in c.CONSOLIDATION_UTILITY_WEIGHTS.items():
        if value < 0:
            errors.append(
                f"consolidation.utility_weights[{key!r}] = {value}, must be >= 0"
            )

    if errors:
        raise ValueError(
            "Invalid consolidation_memory config:\n  " + "\n  ".join(errors)
        )

