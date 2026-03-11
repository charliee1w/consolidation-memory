"""Store and recall through the REST API."""

from __future__ import annotations

import os
from typing import Any

import httpx


BASE_URL = os.getenv("CONSOLIDATION_MEMORY_BASE_URL", "http://127.0.0.1:8080")
AUTH_TOKEN = os.getenv("CONSOLIDATION_MEMORY_REST_AUTH_TOKEN")


def _headers() -> dict[str, str]:
    if not AUTH_TOKEN:
        return {}
    return {"Authorization": f"Bearer {AUTH_TOKEN}"}


def _print_json(label: str, payload: Any) -> None:
    print(f"\n{label}:")
    print(payload)


def main() -> None:
    with httpx.Client(base_url=BASE_URL, headers=_headers(), timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        _print_json("health", health.json())

        store_payload = {
            "content": "The user wants release summaries to include validation commands.",
            "content_type": "preference",
            "tags": ["demo", "rest"],
        }
        stored = client.post("/memory/store", json=store_payload)
        stored.raise_for_status()
        _print_json("store", stored.json())

        recall_payload = {
            "query": "How should release summaries be formatted?",
            "n_results": 5,
            "include_knowledge": True,
        }
        recalled = client.post("/memory/recall", json=recall_payload)
        recalled.raise_for_status()
        recall_body = recalled.json()
        _print_json("recall", recall_body)

        status = client.get("/memory/status")
        status.raise_for_status()
        _print_json("status", status.json())


if __name__ == "__main__":
    main()
