"""Tracked markdown must keep relative links and anchors resolvable.

Catches renamed sections and moved/renamed docs (e.g. a README heading
rename silently breaking guides that deep-link into it).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def _tracked_markdown() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return [ROOT / rel for rel in result.stdout.split("\0") if rel]


def _anchors(text: str) -> set[str]:
    """GitHub-style heading anchors (underscores kept, punctuation stripped)."""
    without_fences = _FENCE_RE.sub("", text)
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#+ (.+)$", without_fences, re.MULTILINE):
        anchor = re.sub(r"[^\w\- ]", "", heading.strip().lower())
        anchor = anchor.replace(" ", "-")
        seen = counts.get(anchor, 0)
        counts[anchor] = seen + 1
        anchors.add(anchor if seen == 0 else f"{anchor}-{seen}")
    return anchors


def test_tracked_markdown_links_resolve() -> None:
    failures: list[str] = []
    for md in _tracked_markdown():
        text = md.read_text(encoding="utf-8", errors="replace")
        local_anchors = _anchors(text)
        for link in _LINK_RE.findall(text):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            rel = md.relative_to(ROOT)
            if link.startswith("#"):
                if link[1:] not in local_anchors:
                    failures.append(f"{rel} -> {link} (missing in-file anchor)")
                continue
            path, _, anchor = link.partition("#")
            target = (md.parent / path).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                failures.append(f"{rel} -> {link} (escapes the repository)")
                continue
            if not target.exists():
                failures.append(f"{rel} -> {link} (missing file)")
                continue
            if anchor and target.suffix == ".md":
                target_text = target.read_text(encoding="utf-8", errors="replace")
                if anchor not in _anchors(target_text):
                    failures.append(f"{rel} -> {link} (missing anchor)")
    assert not failures, "broken markdown links:\n" + "\n".join(failures)
