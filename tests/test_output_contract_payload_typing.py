"""Ratchet on the unvalidated share of the published output contracts.

A contract property typed ``dict[str, Any]`` or ``list[dict[str, Any]]`` publishes
as ``{"type": "object", "additionalProperties": true}``: the runtime validates
that the value is an object and nothing else, so any key upstream adds passes
silently and clients cannot rely on the shape. This file makes that blind spot
measurable instead of invisible.

Hand-writing models for the properties listed here is the fix; until then the
ceiling below must not move. Raising it is a deliberate decision, not a side
effect of adding a field — the test fails on growth so the number gets argued
about in review instead of drifting.

Current exposure: 32 of 234 published success-arm properties, including the
nested models hoisted into root ``$defs``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from consolidation_memory import server

# Ratchet, not a specification: bump it only alongside a commit that types the
# properties it replaces. Re-check with:
#   pytest tests/test_output_contract_payload_typing.py -q
MAX_UNVALIDATED_PROPERTIES = 32


def _is_opaque(spec: Any) -> bool:
    """True when a property's subtree is an object with no declared shape."""
    if not isinstance(spec, dict):
        return False
    kind = spec.get("type")
    if kind == "object":
        return not spec.get("properties") and spec.get("additionalProperties") is True
    if kind == "array":
        return _is_opaque(spec.get("items") or {})
    if "anyOf" in spec:
        return any(_is_opaque(arm) for arm in spec["anyOf"])
    return False


def _collect(
    node: Any,
    path: str,
    total: list[str],
    opaque: list[str],
    *,
    defs: dict[str, Any] | None = None,
) -> None:
    """Record every property under ``node``, resolving ``$ref`` into ``$defs``.

    ``_publish_output_schemas`` hoists ``$defs`` to the schema root beside
    ``anyOf``, not inside the success arm, so a walk of ``anyOf[0]`` alone never
    sees the nested contract models (``HealthOutput``, ``ScopeUsageEntry``,
    ``DriftClaimImpactOutput`` and the rest). They are properties like any other
    and a new ``dict[str, Any]`` inside one has to count.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and defs:
            name = ref.rsplit("/", 1)[-1]
            target = defs.get(name)
            if target is not None and name not in path.split("."):
                # One hop is enough: contract models reference each other
                # without cycles, and the guard stops a self-reference.
                _collect(target, f"{path}.{name}", total, opaque, defs=defs)
                return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, spec in properties.items():
                qualified = f"{path}.{name}" if path else name
                total.append(qualified)
                if _is_opaque(spec):
                    opaque.append(qualified)
                # Descend into the spec too: a property that is a `$ref` (under
                # `anyOf`, or bare) points at a model in `$defs`, and an inline
                # nested object publishes properties of its own. Both are
                # properties a caller depends on.
                _collect(spec, qualified, total, opaque, defs=defs)
        for key, value in node.items():
            if key not in ("properties", "$defs"):
                _collect(value, path, total, opaque, defs=defs)
    elif isinstance(node, list):
        for item in node:
            _collect(item, path, total, opaque, defs=defs)


def _exposure() -> tuple[list[str], list[str]]:
    """Every published property path, and the opaque subset."""
    total: list[str] = []
    opaque: list[str] = []
    for tool in asyncio.run(server.mcp.list_tools()):
        schema = tool.output_schema or {}
        arms = schema.get("anyOf")
        success = arms[0] if isinstance(arms, list) and arms else schema
        defs = schema.get("$defs") or success.pop("$defs", None) or {}
        _collect(success, tool.name, total, opaque, defs=defs)
    return total, opaque


TOTAL, OPAQUE = _exposure()


def test_exposure_is_reported() -> None:
    """The blind spot is measured, so review can see it move."""
    share = 100 * len(OPAQUE) / len(TOTAL)
    print(
        f"\nunvalidated payload exposure: {len(OPAQUE)}/{len(TOTAL)} published properties "
        f"({share:.1f}%) are dict[str, Any] / list[dict[str, Any]]",
    )
    assert TOTAL, "no published properties found; the walk is broken"
    assert OPAQUE, (
        "no opaque properties found; if the contracts were tightened, lower "
        "MAX_UNVALIDATED_PROPERTIES and delete this assertion"
    )


def test_unvalidated_surface_does_not_grow() -> None:
    """New dict[str, Any] properties are a decision, not a default."""
    offenders = sorted(OPAQUE)
    assert len(offenders) <= MAX_UNVALIDATED_PROPERTIES, (
        f"{len(offenders)} unvalidated properties exceeds the ceiling of "
        f"{MAX_UNVALIDATED_PROPERTIES} (added: {offenders}). Type the new "
        f"properties in tool_contracts.py, or raise the ceiling deliberately."
    )


def _descriptions_by_path(
    node: Any,
    path: str,
    found: dict[str, str],
    *,
    defs: dict[str, Any] | None = None,
) -> None:
    """Same walk as :func:`_collect`, but recording each property's description.

    Looking the description up by leaf name in the success arm missed every
    nested-model property: those live in ``$defs`` under their own name, so the
    lookup returned an empty spec and would have failed for the wrong reason.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and defs:
            name = ref.rsplit("/", 1)[-1]
            target = defs.get(name)
            if target is not None and name not in path.split("."):
                _descriptions_by_path(target, f"{path}.{name}", found, defs=defs)
                return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, spec in properties.items():
                qualified = f"{path}.{name}" if path else name
                if isinstance(spec, dict):
                    found[qualified] = str(spec.get("description") or "")
                _descriptions_by_path(spec, qualified, found, defs=defs)
        for key, value in node.items():
            if key not in ("properties", "$defs"):
                _descriptions_by_path(value, path, found, defs=defs)
    elif isinstance(node, list):
        for item in node:
            _descriptions_by_path(item, path, found, defs=defs)


def test_every_opaque_property_still_declares_a_description() -> None:
    """Opaque is acceptable; undocumented is not."""
    missing: list[str] = []
    for tool in asyncio.run(server.mcp.list_tools()):
        schema = tool.output_schema or {}
        arms = schema.get("anyOf")
        success = arms[0] if isinstance(arms, list) and arms else schema
        defs = schema.get("$defs") or success.pop("$defs", None) or {}
        found: dict[str, str] = {}
        _descriptions_by_path(success, tool.name, found, defs=defs)

        for path in OPAQUE:
            if not path.startswith(f"{tool.name}."):
                continue
            if not found.get(path, "").strip():
                missing.append(path)
    assert not missing, f"unvalidated properties without a description: {missing}"


def test_nested_model_properties_are_inside_the_ratchet() -> None:
    """The walk has to reach the models hoisted into root ``$defs``.

    Without this, adding a ``dict[str, Any]`` inside a nested output model moved
    no number and the ratchet stayed green.
    """
    nested = [path for path in TOTAL if path.count(".") > 1]
    assert nested, (
        "no nested-model property was counted; the walk is not following $defs, "
        "so nested properties escape the ratchet"
    )
    assert any("HealthOutput" in path or "ScopeUsageEntry" in path for path in nested)


@pytest.mark.parametrize("kind", ["object", "array"])
def test_opaque_detection_matches_the_published_shapes(kind: str) -> None:
    """The walk recognises both opaque shapes, and nothing else."""
    if kind == "object":
        assert _is_opaque({"type": "object", "additionalProperties": True})
        assert not _is_opaque({"type": "object", "properties": {"a": {"type": "string"}}})
    else:
        assert _is_opaque({"type": "array", "items": {"type": "object", "additionalProperties": True}})
        assert not _is_opaque({"type": "array", "items": {"type": "string"}})
    assert not _is_opaque({"type": "string"})
    assert not _is_opaque({"anyOf": [{"type": "string"}, {"type": "null"}]})
