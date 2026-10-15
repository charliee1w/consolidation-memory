# Graphical interfaces

Three graphical surfaces ship with the project. They are deliberately
**curated subsets** of the tool surface — ask/remember/browse style reading
and writing plus operational fix-it flows — while the full capability set
lives behind MCP, REST, the Python SDK and the OpenAI schemas (see
[CONTRIBUTING.md](../CONTRIBUTING.md) for the parity contract). The browser and
desktop GUIs reuse the same dispatch seam as those surfaces, so a `remember`
from either writes exactly the row an MCP call would write; the TUI is read-only
and never dispatches at all.

| Surface | Command | Extra | Backend path | Best for |
| --- | --- | --- | --- | --- |
| Browser UI | `consolidation-memory ui` | `[rest]` | REST API + web app on one port | Everyday use next to the browser; setup wizard; metrics |
| TUI dashboard | `consolidation-memory dashboard` | `[dashboard]` | Direct SQLite reads (`DashboardData`) | Fast read-only inspection in a terminal |
| Native desktop | `consolidation-memory app` | `[desktop]` | Direct SQLite + shared dispatch (`DesktopBackend`) | Tray-resident quick capture and lookup |

Install what you need:

```bash
pip install "consolidation-memory[rest]"        # browser UI (+ REST API)
pip install "consolidation-memory[dashboard]"   # Textual TUI
pip install "consolidation-memory[desktop]"     # PySide6 app
pip install "consolidation-memory[all]"         # everything
```

## Browser UI

```bash
consolidation-memory ui                    # http://127.0.0.1:8080/ui/
consolidation-memory ui --port 9000 --no-browser
consolidation-memory ui --host 0.0.0.0     # requires a REST auth token (see below)
```

`ui` starts the full REST application (FastAPI/uvicorn) and serves the
single-page app from `web/index.html` at `/ui/`. Because the REST API is
running, anything else that speaks to the REST surface works at the same
time.

### Tabs

| Tab | What it does |
| --- | --- |
| **Ask** | Plain-language question → `memory_ask` recall over episodes, topics, records and claims |
| **Remember** | Capture a note/fix/fact/preference → `memory_remember` (kind, tags; an empty tag list becomes `ui`) |
| **Browse** | Recent episodes and knowledge topics; delete an episode (→ `memory_forget`) |
| **Health** | Overview payload (version, project, DB/FAISS stats, health note) plus collected warnings, each with a fix action: consolidate, warmup, reindex, install the maintenance daemon |
| **Hygiene** | Corpus hygiene scan and cleanup (`memory_hygiene_scan` / `memory_hygiene_apply`) with dry-run awareness |
| **Metrics** | Charts from `real_world_eval` — the bundled published bundle plus a live report when present (see [REAL_WORLD_METRICS.md](REAL_WORLD_METRICS.md)) |

`POST /ui/api/ask` forwards the simple-surface arguments exactly as
`memory_ask` publishes them. `simple_api.build_ask_recall_arguments` is the
**only** translator to `memory_recall` arguments, and it always sets
`include_knowledge=True` — the page does not pre-translate, so a UI result and
an MCP `memory_ask` result are the same call.

### Setup wizard

When the config file is missing, the page opens an in-browser wizard:

- `GET /ui/api/setup/status` — is setup needed?
- `POST /ui/api/setup/quick` — runs the same one-command flow as
  `consolidation-memory init --quick` and reports the result.

The rest of the page is backed by small `/ui/api/*` routes. `ask`, `remember`,
`consolidate` and episode deletion go through the shared dispatch, so the UI
never has its own write path; `hygiene/*` calls the same `corpus_hygiene`
implementations the two hygiene tools call; `overview` and `recent` are read
helpers over `DashboardData`, while `metrics` serves the `real_world_eval` report
through `ui_ops.load_metrics_for_ui()` — the bundled
`web/published_metrics.json` when no live report is present, so it is not a
database read; and `warmup`, `reindex` and `daemon-install` reuse the same
helpers as the matching `/ops/*` routes.

### Security

The UI inherits the REST trust rules (see
[SECURITY.md](../SECURITY.md#rest-api)):

- bound to loopback by default — no token needed;
- any non-loopback bind is refused unless `CONSOLIDATION_MEMORY_REST_AUTH_TOKEN`
  is set (`ui` calls `validate_rest_bind` before starting);
- `CONSOLIDATION_MEMORY_REST_ALLOW_PUBLIC_BIND` overrides that refusal and
  waives the token requirement as well, so it exposes the API unauthenticated —
  it is not a supported configuration (see
  [SECURITY.md](../SECURITY.md#rest-api));
- the browser UI and the REST API share one process and one token.

## TUI dashboard

```bash
consolidation-memory dashboard
```

A Textual application for terminals: four tabs, switchable with keys `1`–`4`:

| Key | Tab | Contents |
| --- | --- | --- |
| `1` | **Episodes** | Recent episodic buffer rows |
| `2` | **Knowledge** | Topics and record counts |
| `3` | **Consolidation** | Recent runs, scheduler state |
| `4` | **Stats** | Database/FAISS/health counters |

Data comes from `dashboard_data.py`, which queries SQLite **directly** via
`database.py` — deliberately skipping engine, embedding and REST
initialization, so the dashboard starts instantly and stays read-only.

## Native desktop app

```bash
consolidation-memory app            # window + system tray icon
consolidation-memory app --no-tray  # close app together with the window
```

A PySide6 application with five tabs — **Ask · Remember · Browse · Health ·
Hygiene** — and a system-tray icon whose menu offers Open, Run consolidation and
Quit. Closing the window hides to tray with a notification; `--no-tray` makes
the window close for real.

`desktop_backend.py` backs it with two kinds of calls:

- **reads** — `DashboardData` (direct SQLite, same as the TUI);
- **writes, recall and maintenance** — the shared `execute_tool_call` dispatch
  (`memory_ask`, `memory_remember`, `memory_consolidate`, `memory_forget`,
  `memory_status`, `memory_hygiene_scan`, `memory_hygiene_apply`), so results and
  policies match every other surface; desktop remembers default to the `desktop`
  tag.

Heavy work runs on background `QThread` workers, so the window never blocks
on consolidation or recall.

## Which surface talks to what

```text
Browser UI  ── HTTP ──► REST API (same process) ──► tool_dispatch ──► MemoryClient ──► SQLite/FAISS
Desktop     ─────────────────────────────────────► tool_dispatch ──► MemoryClient ──► SQLite/FAISS
Desktop/TUI ── read-only ──► DashboardData ──► SQLite
```

Consequences worth knowing:

- **Policies apply everywhere**: scope/ACL denials surface in the UIs the
  same way they do over MCP (`status: "write_denied"`) — see
  [ACL.md](ACL.md).
- **The TUI and desktop reads bypass scopes** in the sense that they query
  the local database directly: they are trusted local operator tools, like
  the CLI. Do not expose them to untrusted users; the transport boundary is
  the protection (see [SECURITY.md](../SECURITY.md#trust-boundaries)).
- **No extra daemon is required**: `ui`/`dashboard`/`app` all work against
  the active project chosen by `CONSOLIDATION_MEMORY_PROJECT`.

## Related

- [README — extras table](../README.md#install-and-try-it)
- [MCP guide](MCP_GUIDE.md) — agent-facing surfaces
- [Real-world metrics](REAL_WORLD_METRICS.md) — data behind the Metrics tab
- [Roadmap](ROADMAP.md) — where the three surfaces sit
