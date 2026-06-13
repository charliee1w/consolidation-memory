"""One-shot helper: split database.py into db/ submodules. Run from repo root."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "consolidation_memory"
DB_DIR = SRC / "db"
SOURCE = SRC / "database.py"


def _read_lines() -> list[str]:
    return SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)


def _extract(lines: list[str], start: int, end: int) -> str:
    """Extract 1-based inclusive line range."""
    return "".join(lines[start - 1 : end])


def _write(path: Path, header: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = header.rstrip() + "\n\n" + body.lstrip("\n")
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    lines = _read_lines()

    migrations_dict = _extract(lines, 42, 297)

    _write(
        DB_DIR / "_helpers.py",
        '''"""Shared database helpers and small utilities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Sequence

from consolidation_memory.utils import parse_datetime

OUTCOME_TYPES: tuple[str, ...] = (
    "success",
    "failure",
    "partial_success",
    "reverted",
    "superseded",
)
''',
        _extract(lines, 308, 333)
        + "\n"
        + _extract(lines, 4226, 4235),
    )

    scope_body = (
        _extract(lines, 44, 50).replace("_DEFAULT_CLAIM_PRECISION = 1.0\n", "")
        + _extract(lines, 335, 372)
        + _extract(lines, 405, 724)
    )
    _write(
        DB_DIR / "scope.py",
        '''"""Scope coercion, filters, and policy/ACL CRUD."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Mapping, Sequence

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection

_DEFAULT_NAMESPACE_SLUG = "default"
_DEFAULT_NAMESPACE_SHARING_MODE = "private"
_DEFAULT_APP_CLIENT_NAME = "legacy_client"
_DEFAULT_APP_CLIENT_TYPE = "python_sdk"

_POLICY_SELECTOR_KEYS: tuple[str, ...] = (
    "namespace_slug",
    "project_slug",
    "app_client_name",
    "app_client_type",
    "app_client_provider",
    "app_client_external_key",
    "agent_name",
    "agent_external_key",
    "session_external_key",
    "session_kind",
)

_EXACT_SCOPE_MATCH_KEYS: tuple[str, ...] = (
    "namespace_slug",
    "namespace_sharing_mode",
    "project_slug",
    "app_client_name",
    "app_client_type",
    "app_client_provider",
    "app_client_external_key",
    "agent_name",
    "agent_external_key",
    "session_external_key",
    "session_kind",
)
''',
        scope_body,
    )

    _write(
        DB_DIR / "connection.py",
        '''"""Thread-local SQLite connection pooling."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from consolidation_memory.config import get_config as _get_config

logger = logging.getLogger(__name__)

_local = threading.local()
_all_connections: list[sqlite3.Connection] = []
_conn_list_lock = threading.Lock()
''',
        _extract(lines, 726, 832),
    )

    _write(
        DB_DIR / "migrations.py",
        '''"""Schema versioning, migrations, and ensure_schema entry point."""

from __future__ import annotations

import logging
import sqlite3

from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import (
    _DEFAULT_APP_CLIENT_NAME,
    _DEFAULT_APP_CLIENT_TYPE,
    _DEFAULT_NAMESPACE_SHARING_MODE,
    _DEFAULT_NAMESPACE_SLUG,
    _default_project_slug,
)

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 20
'''
        + migrations_dict,
        _extract(lines, 834, 1317),
    )

    _write(
        DB_DIR / "episodes.py",
        '''"""Episode CRUD, FTS5, pruning/protection, and access counts."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import _normalize_id_tokens, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row
from consolidation_memory.utils import parse_json_list

logger = logging.getLogger(__name__)

_fts5_available: bool | None = None
_fts5_lock = threading.Lock()
_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
''',
        _extract(lines, 1338, 1911),
    )

    topics_body = _extract(lines, 373, 403) + _extract(lines, 1914, 2187)
    _write(
        DB_DIR / "topics.py",
        '''"""Knowledge topic CRUD and storage filename helpers."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import PurePosixPath
from typing import Any, Mapping

from consolidation_memory.db._helpers import _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import (
    _EXACT_SCOPE_MATCH_KEYS,
    _apply_exact_scope_filters,
    _apply_scope_filters,
    _coerce_scope_row,
)

_TOPIC_STORAGE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
''',
        topics_body,
    )

    _write(
        DB_DIR / "records.py",
        '''"""Knowledge record CRUD, temporal queries, contradictions, tag cooccurrence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from consolidation_memory.db._helpers import _normalize_utc_timestamp, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row
from consolidation_memory.utils import parse_json_list
''',
        _extract(lines, 2190, 2720),
    )

    _write(
        DB_DIR / "claims.py",
        '''"""Claims graph, precision, trust stats, anchors, and challenge flow."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import _normalize_utc_timestamp, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters
from consolidation_memory.utils import parse_datetime

logger = logging.getLogger(__name__)

_DEFAULT_CLAIM_PRECISION = 1.0
_PRECISION_REFRESH_EVENT_TYPES = frozenset({
    "contradiction",
    "challenged",
    "code_drift_detected",
})
''',
        _extract(lines, 2722, 3870),
    )

    consolidation_body = (
        _extract(lines, 1320, 1334)
        + _extract(lines, 3871, 4224)
        + _extract(lines, 5325, 5530)
    )
    _write(
        DB_DIR / "consolidation.py",
        '''"""Consolidation scheduler, runs, metrics, and attempt tracking."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence, cast

import sqlite3

from consolidation_memory.config import get_config as _get_config
from consolidation_memory.db._helpers import _normalize_id_tokens, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters
from consolidation_memory.types import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RunStatus,
)
from consolidation_memory.utils import parse_datetime, parse_json_list

logger = logging.getLogger(__name__)

_SCHEDULER_ROW_ID = "global"
''',
        consolidation_body,
    )

    outcomes_body = _extract(lines, 4238, 4686)
    _write(
        DB_DIR / "outcomes.py",
        '''"""Action outcomes, sources, refs, and claim outcome evidence."""

from __future__ import annotations

import json
import uuid
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import (
    OUTCOME_TYPES,
    _derive_action_key,
    _normalize_id_tokens,
    _normalize_outcome_type,
    _normalize_utc_timestamp,
    _now,
)
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row
''',
        outcomes_body,
    )

    export_body = _extract(lines, 4690, 5257)
    _write(
        DB_DIR / "export.py",
        '''"""Bulk export queries and claim graph snapshot import."""

from __future__ import annotations

import json
import uuid
from typing import Any, Mapping, Sequence

from consolidation_memory.db._helpers import OUTCOME_TYPES, _derive_action_key, _now
from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters, _coerce_scope_row
''',
        export_body,
    )

    _write(
        DB_DIR / "stats.py",
        '''"""Aggregate stats counters."""

from __future__ import annotations

from typing import Any, Mapping

from consolidation_memory.db.connection import get_connection
from consolidation_memory.db.scope import _apply_scope_filters
from consolidation_memory.types import StatsDict
''',
        _extract(lines, 5262, 5322),
    )

    _write(
        DB_DIR / "__init__.py",
        '"""SQLite persistence layer split by domain. Import via consolidation_memory.database facade."""',
        "",
    )

    # Patch connection.close_all_connections to lazy-import FTS cache reset.
    conn_path = DB_DIR / "connection.py"
    conn_text = conn_path.read_text(encoding="utf-8")
    conn_text = conn_text.replace(
        "    _reset_fts5_cache()",
        "    from consolidation_memory.db.episodes import _reset_fts5_cache\n\n    _reset_fts5_cache()",
    )
    conn_path.write_text(conn_text, encoding="utf-8")

    # Patch outcomes.record_action_outcome for claims circular import.
    outcomes_path = DB_DIR / "outcomes.py"
    outcomes_text = outcomes_path.read_text(encoding="utf-8")
    outcomes_text = outcomes_text.replace(
        "    _refresh_claim_precisions(claim_ids)",
        "    from consolidation_memory.db.claims import _refresh_claim_precisions\n\n    _refresh_claim_precisions(claim_ids)",
    )
    outcomes_path.write_text(outcomes_text, encoding="utf-8")

    # Patch claims.recompute_claim_precision for outcomes import.
    claims_path = DB_DIR / "claims.py"
    claims_text = claims_path.read_text(encoding="utf-8")
    claims_text = claims_text.replace(
        "    evidence = get_claim_outcome_evidence([token]).get(token, {})",
        "    from consolidation_memory.db.outcomes import get_claim_outcome_evidence\n\n    evidence = get_claim_outcome_evidence([token]).get(token, {})",
    )
    claims_text = claims_text.replace(
        "    scoped = filter_claims_for_scope(get_all_claims(), scope)",
        "    from consolidation_memory.db.export import get_all_claims\n\n    scoped = filter_claims_for_scope(get_all_claims(), scope)",
    )
    claims_path.write_text(claims_text, encoding="utf-8")

    # episodes insert_episode needs json import
    ep_path = DB_DIR / "episodes.py"
    ep_text = ep_path.read_text(encoding="utf-8")
    if "import json" not in ep_text:
        ep_text = ep_text.replace(
            "import logging",
            "import json\nimport logging",
        )
        ep_path.write_text(ep_text, encoding="utf-8")

    print(f"Wrote db package under {DB_DIR}")


if __name__ == "__main__":
    main()