from __future__ import annotations

from pathlib import Path


WORKFLOWS = {
    ".github/workflows/test.yml": {
        "expected_checkout_count": 4,
        "requires_contents_read": True,
    },
    ".github/workflows/novelty-full-nightly.yml": {
        "expected_checkout_count": 1,
        "requires_contents_read": True,
    },
    ".github/workflows/publish.yml": {
        "expected_checkout_count": 6,
        "requires_contents_read": False,
    },
}

CHECKOUT_LINE = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4"
TOKEN_LINE = (
    "token: ${{ secrets.CI_CHECKOUT_TOKEN != '' && "
    "secrets.CI_CHECKOUT_TOKEN || github.token }}"
)


def _count_occurrences(lines: list[str], needle: str) -> int:
    return sum(1 for line in lines if needle in line)


def test_checkout_steps_all_define_token_fallback() -> None:
    for workflow, expectations in WORKFLOWS.items():
        lines = Path(workflow).read_text(encoding="utf-8").splitlines()
        checkout_count = _count_occurrences(lines, CHECKOUT_LINE)
        token_count = _count_occurrences(lines, TOKEN_LINE)

        assert checkout_count == expectations["expected_checkout_count"], (
            f"{workflow} expected {expectations['expected_checkout_count']} checkout step(s), "
            f"found {checkout_count}"
        )
        assert token_count == checkout_count, (
            f"{workflow} has {checkout_count} checkout step(s) but only "
            f"{token_count} token fallback line(s)"
        )


def test_workflows_that_need_explicit_read_permissions_set_them() -> None:
    for workflow, expectations in WORKFLOWS.items():
        if not expectations["requires_contents_read"]:
            continue

        text = Path(workflow).read_text(encoding="utf-8")
        assert "permissions:" in text, f"{workflow} is missing a permissions block"
        assert "contents: read" in text, f"{workflow} is missing contents: read permission"
