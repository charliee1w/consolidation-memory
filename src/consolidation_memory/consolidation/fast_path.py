"""Deterministic consolidation fast path — no LLM required for structured episodes."""

from __future__ import annotations

import json
import logging
import re
from typing import TypedDict

from consolidation_memory.anchors import extract_anchors
from consolidation_memory.claim_graph import claim_from_record, normalize_record_type
from consolidation_memory.config import get_config
from consolidation_memory.database import claim_exists
from consolidation_memory.utils import parse_json_list

logger = logging.getLogger(__name__)

_PREFERENCE_USER_RE = re.compile(
    r"(?is)^user\s+prefers?\s+(?P<value>.+?)(?:\s+for\s+(?P<key>.+))?\s*\.?\s*$"
)
_PREFERENCE_KV_RE = re.compile(
    r"(?is)^preference\s*:\s*(?P<key>.+?)\s*[=:]\s*(?P<value>.+?)\s*\.?\s*$"
)
_PROBLEM_LINE_RE = re.compile(r"(?im)^(?:problem|issue|error)\s*:\s*(?P<text>.+?)\s*$")
_FIX_LINE_RE = re.compile(r"(?im)^(?:fix|solution|resolution)\s*:\s*(?P<text>.+?)\s*$")
_INLINE_FIX_RE = re.compile(r"(?is)(?P<prefix>.+?)\.\s*fix\s*:\s*(?P<fix>.+?)\s*\.?\s*$")
_TRIGGER_LINE_RE = re.compile(r"(?im)^trigger\s*:\s*(?P<text>.+?)\s*$")
_STEPS_LINE_RE = re.compile(r"(?im)^steps\s*:\s*(?P<text>.+?)\s*$")
_PROCEDURE_KV_RE = re.compile(
    r"(?is)^procedure\s*:\s*trigger\s*=\s*(?P<trigger>.+?)\s*,\s*steps\s*=\s*(?P<steps>.+?)\s*\.?\s*$"
)

_STRUCTURED_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "fact": ("subject", "info"),
    "solution": ("problem", "fix"),
    "preference": ("key", "value"),
    "procedure": ("trigger", "steps"),
    "strategy": ("problem_pattern", "strategy"),
}

_STRUCTURED_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "fact": (),
    "solution": ("context",),
    "preference": ("context",),
    "procedure": ("context",),
    "strategy": ("preconditions", "expected_signals", "failure_modes", "context"),
}


class FastPathResult(TypedDict):
    """Successful deterministic extraction for a cluster."""

    extraction_data: dict[str, object]
    path_kind: str


def _coerce_text_field(value: object) -> str:
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " | ".join(parts)
    return str(value or "").strip()


def _canonicalize_structured_record(parsed: dict[str, object]) -> dict[str, object] | None:
    record_type = parsed.get("type")
    if not record_type:
        return None
    try:
        normalized_type = normalize_record_type(str(record_type))
    except ValueError:
        return None

    record: dict[str, object] = {"type": normalized_type}
    for field_name in _STRUCTURED_REQUIRED_FIELDS[normalized_type]:
        value = _coerce_text_field(parsed.get(field_name))
        if not value:
            return None
        record[field_name] = value

    for field_name in _STRUCTURED_OPTIONAL_FIELDS[normalized_type]:
        value = _coerce_text_field(parsed.get(field_name))
        if value:
            record[field_name] = value
    return record


def _try_parse_structured_json(content: str) -> dict[str, object] | None:
    text = (content or "").strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return _canonicalize_structured_record(parsed)


def _try_parse_procedure(content: str, tags: list[str]) -> dict[str, object] | None:
    text = (content or "").strip()
    if not text:
        return None

    match = _PROCEDURE_KV_RE.match(text)
    if match:
        trigger = match.group("trigger").strip()
        steps = match.group("steps").strip()
        if trigger and steps:
            record: dict[str, object] = {
                "type": "procedure",
                "trigger": trigger,
                "steps": steps,
            }
            if tags:
                record["context"] = ", ".join(tags[:3])
            return record

    trigger_match = _TRIGGER_LINE_RE.search(text)
    steps_match = _STEPS_LINE_RE.search(text)
    if trigger_match and steps_match:
        trigger = trigger_match.group("text").strip()
        steps = steps_match.group("text").strip()
        if trigger and steps:
            record = {
                "type": "procedure",
                "trigger": trigger,
                "steps": steps,
            }
            if tags:
                record["context"] = ", ".join(tags[:3])
            return record

    return None


def _preference_key_from_tags(tags: list[str]) -> str:
    cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
    if not cleaned:
        return "general"
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:3])


def _try_parse_preference(content: str, tags: list[str]) -> dict[str, object] | None:
    text = (content or "").strip()
    if not text:
        return None

    match = _PREFERENCE_KV_RE.match(text)
    if match:
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key and value:
            return {"type": "preference", "key": key, "value": value}

    match = _PREFERENCE_USER_RE.match(text)
    if match:
        value = match.group("value").strip()
        key = (match.group("key") or "").strip() or _preference_key_from_tags(tags)
        if value:
            return {"type": "preference", "key": key, "value": value}

    return None


def _path_anchors(content: str) -> list[str]:
    return [
        anchor["anchor_value"]
        for anchor in extract_anchors(content)
        if anchor.get("anchor_type") == "path"
    ]


def _try_parse_solution(content: str) -> dict[str, object] | None:
    text = (content or "").strip()
    if not text:
        return None

    paths = _path_anchors(text)
    if not paths:
        return None

    problem = ""
    fix = ""

    problem_match = _PROBLEM_LINE_RE.search(text)
    if problem_match:
        problem = problem_match.group("text").strip()

    fix_match = _FIX_LINE_RE.search(text)
    if fix_match:
        fix = fix_match.group("text").strip()

    if not fix:
        inline_match = _INLINE_FIX_RE.match(text)
        if inline_match:
            problem = problem or inline_match.group("prefix").strip()
            fix = inline_match.group("fix").strip()

    if not problem:
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        problem = first_sentence or text

    if not fix:
        for line in text.splitlines():
            line_text = line.strip()
            if not line_text:
                continue
            if any(path in line_text for path in paths):
                fix = line_text
                break
        if not fix:
            fix = text

    if not problem or not fix:
        return None

    record: dict[str, object] = {
        "type": "solution",
        "problem": problem,
        "fix": fix,
    }
    if paths:
        record["context"] = ", ".join(paths[:5])
    return record


def _record_specificity_score(record: dict[str, object]) -> int:
    return len(json.dumps(record, sort_keys=True, default=str))


def _dedupe_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_claim: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for record in records:
        claim_id = claim_from_record(record)["id"]
        if claim_id not in best_by_claim:
            best_by_claim[claim_id] = record
            order.append(claim_id)
            continue
        if _record_specificity_score(record) >= _record_specificity_score(best_by_claim[claim_id]):
            best_by_claim[claim_id] = record
    return [best_by_claim[claim_id] for claim_id in order]


def _derive_title(records: list[dict[str, object]]) -> str:
    if not records:
        return "Consolidated knowledge"
    first = records[0]
    record_type = str(first.get("type", "fact"))
    if record_type == "preference":
        return f"Preference: {first.get('key', 'general')}"
    if record_type == "solution":
        problem = str(first.get("problem", "Solution"))
        return f"Solution: {problem[:72]}"
    if record_type == "procedure":
        return f"Procedure: {first.get('trigger', 'workflow')}"
    if record_type == "strategy":
        return f"Strategy: {first.get('problem_pattern', 'pattern')}"
    return f"Fact: {first.get('subject', 'topic')}"


def _derive_summary(records: list[dict[str, object]]) -> str:
    if not records:
        return ""
    snippets: list[str] = []
    for record in records[:3]:
        record_type = str(record.get("type", "fact"))
        if record_type == "preference":
            snippets.append(f"{record.get('key')}: {record.get('value')}")
        elif record_type == "solution":
            snippets.append(f"{record.get('problem')} -> {record.get('fix')}")
        elif record_type == "fact":
            snippets.append(f"{record.get('subject')}: {record.get('info')}")
        else:
            snippets.append(json.dumps(record, sort_keys=True, default=str)[:160])
    return " | ".join(snippets)


def _merge_tags(tags: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue
        norm = tag.lower()
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(tag)
    return merged[:8]


def _episode_tags(episode: dict[str, object]) -> list:
    raw_tags = episode.get("tags")
    if isinstance(raw_tags, (str, list)) or raw_tags is None:
        return parse_json_list(raw_tags)
    return parse_json_list(None)


def _extract_record_from_episode(episode: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    content = str(episode.get("content") or "")
    content_type = str(episode.get("content_type") or "exchange").strip().lower()
    tags = _episode_tags(episode)

    record = _try_parse_structured_json(content)
    kind = "structured" if record else None

    if record is None and content_type == "preference":
        record = _try_parse_preference(content, tags)
        kind = "preference" if record else None

    if record is None and content_type == "procedure":
        record = _try_parse_procedure(content, tags)
        kind = "procedure" if record else None

    if record is None and content_type in {"solution", "fact"}:
        record = _try_parse_solution(content)
        if record is not None:
            kind = "solution"

    if record is None:
        return None, None

    claim_id = claim_from_record(record)["id"]
    if claim_exists(claim_id):
        kind = "existing_claim"
    return record, kind or "structured"


def try_fast_path_extraction(cluster_episodes: list[dict[str, object]]) -> FastPathResult | None:
    """Build extraction payload without LLM when all episodes are structurally eligible."""
    if not get_config().CONSOLIDATION_FAST_PATH_ENABLED:
        return None
    if not cluster_episodes:
        return None

    records: list[dict[str, object]] = []
    kinds: list[str] = []
    all_tags: list[str] = []

    for episode in cluster_episodes:
        record, kind = _extract_record_from_episode(episode)
        if record is None or kind is None:
            logger.debug(
                "Fast-path skipped: episode %s not structurally eligible",
                episode.get("id"),
            )
            return None
        records.append(record)
        kinds.append(kind)
        all_tags.extend(_episode_tags(episode))

    deduped = _dedupe_records(records)
    if not deduped:
        return None

    if "existing_claim" in kinds:
        path_kind = "existing_claim"
    elif len(set(kinds)) == 1:
        path_kind = kinds[0]
    else:
        path_kind = "mixed"

    extraction_data: dict[str, object] = {
        "title": _derive_title(deduped),
        "summary": _derive_summary(deduped),
        "tags": _merge_tags(all_tags),
        "records": deduped,
    }
    return {
        "extraction_data": extraction_data,
        "path_kind": path_kind,
    }


__all__ = ["FastPathResult", "try_fast_path_extraction"]