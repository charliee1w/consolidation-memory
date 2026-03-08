# Schema Migration Plan

## Purpose

Define migration strategy from the current schema baseline to future universal-governance capabilities while preserving compatibility.

## Current Baseline

- Current schema version: `13`.
- Scope metadata persisted on episodes/topics/records.
- Trust tables already present: claims, sources, edges, events, anchors, contradiction log.

## Migration Principles

1. Additive first.
- Prefer new tables/columns and backfills over destructive rewrites.

2. Contract-preserving.
- Keep existing API behavior stable unless explicitly versioned.

3. Test-backed.
- Every migration requires upgrade-path tests and rollback strategy.

## Planned Migration Tracks

### Track A: Policy primitives

Add first-class policy tables (for example policy definitions and assignments) linked to existing scope metadata.

### Track B: Identity normalization

Optional normalization tables for namespace/app/agent/session identities while preserving current denormalized read paths.

### Track C: Governance audit extensions

Expand lifecycle/audit tables for richer shared-memory governance signals.

## Backfill Rules

- Existing rows inherit compatible defaults.
- Backfill scripts must be idempotent.
- New constraints should be enabled only after successful backfill verification.

## Required Migration Validation

1. Fresh install path initializes full schema.
2. Upgrade path from prior schema version succeeds.
3. Existing data remains queryable.
4. API responses remain contract-compatible.
5. Drift/claim workflows still pass regression tests.

## Operational Rollout

- Ship migration with feature flags when behavior changes are non-trivial.
- Provide one release cycle with compatibility defaults before tightening constraints.

## Out Of Scope

- Replacing SQLite/FAISS storage engines in this migration plan.
- Breaking current scope envelope contract.
