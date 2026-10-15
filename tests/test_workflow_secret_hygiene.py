"""Release automation must not keep a push-capable token in the working tree.

`changelog-on-main.yml` installs the package and imports it, which executes code
resolved from PyPI. `actions/checkout` persists whatever token it is given into
`.git/config`, so handing it `RELEASE_AUTOMATION_PAT` gave those steps a
push-capable credential to read. The workflow now checks out with
`persist-credentials: false` and passes the PAT only to the push steps.

These tests read the workflow rather than the docs, because the claim that
matters is about the file CI executes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is not installed")

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"

PUSH_CAPABLE = "RELEASE_AUTOMATION_PAT"


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job]["steps"]


def _checkout_with(steps: list[dict]) -> list[dict]:
    return [step for step in steps if "actions/checkout" in str(step.get("uses", ""))]


@pytest.mark.parametrize("name", ["changelog-on-main.yml", "release-on-main.yml"])
def test_workflows_parse(name: str) -> None:
    assert _load(name)["jobs"]


def test_changelog_job_does_not_check_out_with_the_release_pat() -> None:
    steps = _steps(_load("changelog-on-main.yml"), "changelog")

    checkouts = _checkout_with(steps)
    assert len(checkouts) == 1
    checkout = checkouts[0]

    assert PUSH_CAPABLE not in str(checkout.get("with", {})), (
        "checkout persists its token into .git/config, and the later steps "
        "pip install and import the package"
    )


def test_changelog_job_disables_persisted_credentials() -> None:
    steps = _steps(_load("changelog-on-main.yml"), "changelog")

    checkout = _checkout_with(steps)[0]
    assert checkout["with"].get("persist-credentials") is False


def test_changelog_job_only_receives_the_pat_in_push_steps() -> None:
    """Every step that touches the PAT must be a step that pushes."""
    steps = _steps(_load("changelog-on-main.yml"), "changelog")

    for step in steps:
        env = step.get("env") or {}
        if PUSH_CAPABLE not in env:
            continue
        run = str(step.get("run", ""))
        assert "git push" in run, (
            f"{step.get('name', step)} receives {PUSH_CAPABLE} but does not push"
        )


def test_changelog_job_still_pushes_with_the_pat() -> None:
    """Least privilege must not quietly break the automation."""
    steps = _steps(_load("changelog-on-main.yml"), "changelog")

    push_steps = [
        step
        for step in steps
        if PUSH_CAPABLE in (step.get("env") or {}) and "git push" in str(step.get("run", ""))
    ]
    assert push_steps, "no step both receives the PAT and pushes"

    for step in push_steps:
        run = str(step["run"])
        assert PUSH_CAPABLE in run or "GIT_CONFIG" in run, (
            f"{step.get('name', step)} pushes without passing the PAT to git"
        )


def test_release_job_may_check_out_with_the_pat() -> None:
    """`release.py` pushes itself, so the credential has to survive into it.

    This is the one place the exemption is deliberate; if the release script ever
    stops pushing, this test is the reminder to narrow it too.
    """
    steps = _steps(_load("release-on-main.yml"), "release")

    checkout = _checkout_with(steps)[0]
    assert PUSH_CAPABLE in str(checkout.get("with", {}))

    release_script = ( _ROOT / "scripts" / "release.py").read_text(encoding="utf-8")
    assert '"push", "origin"' in release_script, (
        "release.py no longer pushes; the release job's PAT checkout is now "
        "unnecessarily broad"
    )


def test_no_workflow_uses_pull_request_target() -> None:
    """A `pull_request_target` checkout runs with repository secrets available."""
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        assert "pull_request_target" not in path.read_text(encoding="utf-8"), path.name