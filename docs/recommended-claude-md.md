# Recommended CLAUDE.md Snippet

Add this to your `~/.claude/CLAUDE.md` (global) or project-level `CLAUDE.md` to ensure Claude Code proactively uses memory tools.

You can add this automatically with:

```bash
consolidation-memory setup-memory --path ~/.claude/CLAUDE.md
```

(`setup-claude` is still available as a legacy alias.)

Or copy the snippet below manually:

---

```markdown
## Memory

**Recall**: At the start of every new conversation, call `memory_recall`
with a query matching the user's opening message topic. This is your
persistent memory — always check it before responding.

**Store**: Proactively call `memory_store` whenever you:
- Learn something new about the user's setup, environment, or projects
- Solve a non-trivial problem (store both the problem AND the solution)
- Discover a user preference or workflow pattern
- Complete a significant task (summarize what was done and where)
- Encounter something surprising or noteworthy

Write each memory as a self-contained note that future-you can understand
without context. Use appropriate `content_type` (fact, solution, preference,
exchange) and add `tags` for organization. Do NOT store trivial exchanges
like greetings or simple Q&A.
```

---

## Recommended Hook Config

For automatic memory recall on every session start, add a Claude Code hook:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "echo 'Memory system active — recall will be triggered by CLAUDE.md instructions'"
      }
    ]
  }
}
```

The CLAUDE.md instructions above are the primary mechanism for ensuring proactive memory use. The hook serves as a reminder that the memory system is available.
