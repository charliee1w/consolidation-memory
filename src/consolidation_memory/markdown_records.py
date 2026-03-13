"""Helpers for parsing rendered knowledge markdown back into structured records."""

from __future__ import annotations

import re

_MD_KV_BULLET_RE = re.compile(r"^[-*]\s+(?:\*\*(.+?)\*\*|([^:]+)):\s*(.+)$")
_MD_CONTEXT_RE = re.compile(r"^\*Context:\s*(.+?)\*\s*$")
_MD_VALUE_CONTEXT_RE = re.compile(r"^(.*?)\s+\(([^()]+)\)\s*$")
_MD_STRATEGY_FIELD_RE = re.compile(r"^-\s*(Preconditions|Expected signals|Failure modes):\s*(.+)$", re.IGNORECASE)


def _parse_markdown_kv_bullet(line: str) -> tuple[str, str] | None:
    """Parse markdown bullet lines like '- **Key**: Value'."""
    match = _MD_KV_BULLET_RE.match(line)
    if not match:
        return None
    key = (match.group(1) or match.group(2) or "").strip()
    value = match.group(3).strip()
    if not key or not value:
        return None
    return key, value


def parse_markdown_records(body: str) -> list[dict[str, str]]:
    """Extract structured records from rendered markdown sections."""
    records: list[dict[str, str]] = []
    lines = body.splitlines()
    section: str | None = None
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        lowered = stripped.lower()

        if lowered == "## facts":
            section = "facts"
            i += 1
            continue
        if lowered == "## solutions":
            section = "solutions"
            i += 1
            continue
        if lowered == "## preferences":
            section = "preferences"
            i += 1
            continue
        if lowered == "## procedures":
            section = "procedures"
            i += 1
            continue
        if lowered == "## strategies":
            section = "strategies"
            i += 1
            continue

        if section == "facts":
            parsed = _parse_markdown_kv_bullet(stripped)
            if parsed:
                subject, info = parsed
                records.append({"type": "fact", "subject": subject, "info": info})
        elif section == "preferences":
            parsed = _parse_markdown_kv_bullet(stripped)
            if parsed:
                key, value_text = parsed
                rec: dict[str, str] = {"type": "preference", "key": key, "value": value_text}
                ctx_match = _MD_VALUE_CONTEXT_RE.match(value_text)
                if ctx_match:
                    rec["value"] = ctx_match.group(1).strip()
                    rec["context"] = ctx_match.group(2).strip()
                records.append(rec)
        elif section in {"solutions", "procedures", "strategies"} and stripped.startswith("### "):
            header = stripped[4:].strip()
            j = i + 1
            body_lines: list[str] = []
            context = ""
            strategy_fields: dict[str, str] = {}
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt.startswith("## ") or nxt.startswith("### "):
                    break
                context_match = _MD_CONTEXT_RE.match(nxt)
                if context_match:
                    context = context_match.group(1).strip()
                elif section == "strategies":
                    strategy_field_match = _MD_STRATEGY_FIELD_RE.match(nxt)
                    if strategy_field_match:
                        label = strategy_field_match.group(1).strip().lower()
                        value = strategy_field_match.group(2).strip()
                        key_map = {
                            "preconditions": "preconditions",
                            "expected signals": "expected_signals",
                            "failure modes": "failure_modes",
                        }
                        field_key = key_map.get(label)
                        if field_key and value:
                            strategy_fields[field_key] = value
                    elif nxt:
                        body_lines.append(nxt)
                elif nxt:
                    body_lines.append(nxt)
                j += 1

            text_body = "\n".join(body_lines).strip()
            if header and text_body:
                if section == "solutions":
                    rec = {"type": "solution", "problem": header, "fix": text_body}
                elif section == "procedures":
                    rec = {"type": "procedure", "trigger": header, "steps": text_body}
                else:
                    rec = {
                        "type": "strategy",
                        "problem_pattern": header,
                        "strategy": text_body,
                    }
                    rec.update(strategy_fields)
                if context:
                    rec["context"] = context
                records.append(rec)
            i = j
            continue

        i += 1

    return records
