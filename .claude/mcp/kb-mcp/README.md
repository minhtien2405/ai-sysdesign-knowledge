# kb-mcp

Local MCP server cho repo `ai-sysdesign-knowledge`. Expose 6 tools để Claude Code agents query state của knowledge base trong context.

## Tools

| Tool | Mục đích |
|---|---|
| `kb_get_terminology(en_term)` | Look up VI mapping cho 1 EN technical term từ `docs/terminology.md` |
| `kb_list_case_studies(scope?, status?)` | List case studies với filter scope (1-4) / status (done/draft/planned) |
| `kb_get_case_study(case_id)` | Fetch full markdown + frontmatter của 1 case study by id |
| `kb_find_cross_refs(topic, limit?)` | Keyword search across title/tags/summary — auto-cross-link |
| `kb_propose_next_topic(scope?)` | List 📋 planned items chưa viết — gap analysis |
| `kb_validate_refs(case_id, full?)` | Run `scripts/validate_refs.py` cho 1 case study, return verdict |

## Install + run

Server được register tự động qua `.mcp.json` ở repo root. Yêu cầu:

1. **`uv`** đã install (xem [root README](../../../README.md#yêu-cầu-để-run-agentic-infra)).
2. Restart Claude Code session sau khi clone repo — `uv` sẽ tự cài deps lần đầu chạy.

Manual run cho debugging:

```bash
cd /path/to/ai-sysdesign-knowledge
uv run --directory .claude/mcp/kb-mcp python server.py
```

Server chạy stdio transport, sẽ block đợi MCP client connect.

## Testing

```bash
cd .claude/mcp/kb-mcp
uv run pytest -v
```

Tests cover:
- `kb_get_terminology` — known term lookup
- `kb_list_case_studies` — filter by scope/status
- (smoke-tested) các tool khác qua Claude Code session

## Design notes

- Repo root được infer từ `__file__` (3 levels up). Không dùng env var.
- Terminology cache loaded lazily lần gọi đầu, persistent trong process lifetime → invalidate bằng cách restart server (Claude Code restart).
- Tất cả tool đều **read-only** — không bao giờ mutate filesystem. Mutation work do `/promote-draft` skill + scripts handle.

## Adding a new tool

Edit `server.py`:

```python
@mcp.tool()
def kb_my_new_tool(arg: str) -> dict:
    """Docstring — Claude reads this để biết khi nào dùng tool."""
    return {"result": ...}
```

Restart Claude Code → tool xuất hiện với prefix `mcp__kb-mcp__kb_my_new_tool`.
