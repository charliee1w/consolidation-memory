"""Result types for consolidation-memory operations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StoreResult:
    """Result of a memory store operation."""

    status: str  # "stored" | "duplicate_detected"
    id: str | None = None
    content_type: str | None = None
    tags: list[str] = field(default_factory=list)
    existing_id: str | None = None
    similarity: float | None = None
    message: str | None = None


@dataclass
class RecallResult:
    """Result of a memory recall operation."""

    episodes: list[dict] = field(default_factory=list)
    knowledge: list[dict] = field(default_factory=list)
    total_episodes: int = 0
    total_knowledge_topics: int = 0


@dataclass
class ForgetResult:
    """Result of a memory forget operation."""

    status: str  # "forgotten" | "not_found"
    id: str = ""


@dataclass
class StatusResult:
    """Result of a memory status query."""

    episodic_buffer: dict = field(default_factory=dict)
    knowledge_base: dict = field(default_factory=dict)
    last_consolidation: dict | None = None
    embedding_backend: str = ""
    embedding_model: str = ""
    faiss_index_size: int = 0
    faiss_tombstones: int = 0
    db_size_mb: float = 0.0
    version: str = ""


@dataclass
class ExportResult:
    """Result of a memory export operation."""

    status: str  # "exported"
    path: str = ""
    episodes: int = 0
    knowledge_topics: int = 0


@dataclass
class CorrectResult:
    """Result of a knowledge correction operation."""

    status: str  # "corrected" | "not_found" | "error"
    filename: str | None = None
    title: str | None = None
    message: str | None = None
