"""Episode embedding text and solution-shape helpers for retrieval alignment."""

from __future__ import annotations

import json
import re
from typing import Any

from consolidation_memory.anchors import extract_anchors
from consolidation_memory.consolidation.fast_path import (
    _PROBLEM_LINE_RE,
    _try_parse_structured_json,
)

_FIX_MARKERS = ("Fix:", "fix:", "Solution:", "SOLVED:", "Resolution:")
_PATH_IN_CONTENT_RE = re.compile(
    r"[A-Z]:\\[\w\\./\-]+|(?:^|[\s(])(?:src|tests|docs|benchmarks)/[\w./\-]+|~/[\w./\-]+",
    re.MULTILINE,
)
_ERROR_TOKEN_RE = re.compile(
    r"(?i)\b(?:error|failed|failure|timeout|exception|traceback|regression)\b"
)
_FILENAME_RE = re.compile(r"\b[\w.-]+\.(?:py|md|toml|yml|json|ps1)\b", re.IGNORECASE)
_FUNCTION_RE = re.compile(r"\b([a-z_][a-z0-9_]{5,})\(\)")
_COMMIT_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,12}\b", re.IGNORECASE)


def problem_query_from_content(content: str, *, max_len: int = 180) -> str:
    """Extract problem-shaped query text aligned with recall benchmarks."""
    text = str(content or "").strip()
    for marker in _FIX_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]
    return text


def _has_problem_marker(content: str) -> bool:
    text = str(content or "")
    if _PROBLEM_LINE_RE.search(text):
        return True
    lowered = text.lower()
    return "problem:" in lowered or '"problem"' in lowered


def _has_structured_solution(content: str) -> bool:
    record = _try_parse_structured_json(content)
    return (
        record is not None
        and str(record.get("type", "")).lower() == "solution"
        and bool(str(record.get("problem", "")).strip())
        and bool(str(record.get("fix", "")).strip())
    )


def _has_path_anchor(content: str) -> bool:
    for anchor in extract_anchors(content):
        if anchor.get("anchor_type") == "path":
            return True
    return bool(_PATH_IN_CONTENT_RE.search(content))


def solution_store_shape_warnings(content: str) -> list[str]:
    """Advisory warnings when a solution episode lacks retrieval-friendly shape."""
    text = str(content or "").strip()
    if not text:
        return ["Solution episode content is empty"]

    checks = (
        _has_problem_marker(text),
        _has_structured_solution(text),
        _has_path_anchor(text),
    )
    if any(checks):
        return []

    return [
        "Solution episode lacks Problem:/JSON problem+fix/path anchor; "
        "recall may rank poorly. Prefer: Problem: <symptom>\\nFix: <how>\\n"
        "Context: path:src/... or structured JSON with type=solution."
    ]


def _distinctive_tokens(content: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    for anchor in extract_anchors(content):
        anchor_type = str(anchor.get("anchor_type", ""))
        value = str(anchor.get("anchor_value", "")).strip()
        if not value or value in seen:
            continue
        if anchor_type in {"path", "commit", "tool"}:
            tokens.append(value)
            seen.add(value)

    for match in _PATH_IN_CONTENT_RE.finditer(content):
        value = match.group(0).strip()
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    if _ERROR_TOKEN_RE.search(content):
        for word in ("error", "failed", "failure", "timeout", "fix", "problem"):
            if re.search(rf"(?i)\b{word}\b", content) and word not in seen:
                tokens.append(word)
                seen.add(word)

    for match in _FILENAME_RE.finditer(content):
        value = match.group(0).strip().lower()
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    for match in _FUNCTION_RE.finditer(content):
        value = match.group(1).strip().lower()
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    for match in _COMMIT_RE.finditer(content):
        value = match.group(0).strip().lower()
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    return tokens[:20]


def distinctive_token_set(content: str) -> set[str]:
    """Normalized path/file/commit/tool tokens for recall overlap scoring."""
    normalized: set[str] = set()
    for token in _distinctive_tokens(content):
        value = str(token).strip().lower()
        if not value:
            continue
        normalized.add(value)
        if "/" in value or "\\" in value:
            leaf = value.replace("\\", "/").rsplit("/", 1)[-1]
            if leaf:
                normalized.add(leaf)
    return normalized


def embedding_text_for_episode(
    *,
    content: str,
    content_type: str,
    tags: list[str] | None = None,
) -> str:
    """Text used for FAISS embedding; full content remains stored in SQLite."""
    text = str(content or "").strip()
    if not text:
        return text

    if str(content_type).lower() != "solution":
        return text

    parts: list[str] = []

    structured = _try_parse_structured_json(text)
    if structured and str(structured.get("type", "")).lower() == "solution":
        problem = str(structured.get("problem", "")).strip()
        fix = str(structured.get("fix", "")).strip()
        if problem:
            parts.append(f"Problem: {problem}")
        if fix:
            parts.append(f"Fix: {fix}")
        context = str(structured.get("context", "")).strip()
        if context:
            parts.append(f"Context: {context}")
    else:
        problem = problem_query_from_content(text, max_len=220)
        if problem:
            if not problem.lower().startswith("problem:"):
                problem = f"Problem: {problem}"
            parts.append(problem)

    if tags:
        tag_text = " ".join(str(t).strip() for t in tags if str(t).strip())
        if tag_text:
            parts.append(f"tags: {tag_text}")

    parts.extend(_distinctive_tokens(text))

    merged = " ".join(part for part in parts if part).strip()
    return merged or text


def embedding_text_for_episode_row(episode: dict[str, Any]) -> str:
    """Build embedding text from a persisted episode row."""
    raw_tags = episode.get("tags")
    tags: list[str] | None
    if isinstance(raw_tags, list):
        tags = [str(t) for t in raw_tags]
    elif isinstance(raw_tags, str):
        try:
            parsed = json.loads(raw_tags)
            tags = [str(t) for t in parsed] if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            tags = None
    else:
        tags = None

    return embedding_text_for_episode(
        content=str(episode.get("content", "")),
        content_type=str(episode.get("content_type", "exchange")),
        tags=tags,
    )