# `.claude/` — Agentic Infrastructure

> Đây là **operator layer** của knowledge base. Reader chỉ cần đọc `case-studies/` + `README.md`. Operator (người maintain + extend wiki) đọc folder này.

Lấy cảm hứng từ Karpathy ethos: **show your tools** — repo công khai cả nội dung lẫn pipeline tạo ra nó.

## Mục đích

Biến `agentic-aisys-wiki` thành **self-extending wiki** — clone repo + có Claude Code là chạy được toàn bộ workflow:

- Scaffold case study mới qua slash command
- Review chất lượng tự động trước khi promote
- Regenerate INDEX/progress từ filesystem (không drift)
- Hook lint trên mỗi save
- (Phase 2) MCP server expose state cho agents
- (Phase 3) Cron auto-fill backlog từ arxiv/blogs

## Layout

```text
.claude/
├── README.md                              # File này
├── settings.json                          # Permissions + hooks (project-scope)
├── settings.local.json                    # User-local overrides (gitignored)
├── agents/                                # 5 subagent definitions
│   ├── ai-sysdesign-knowledge-writer.md   # Writer chính (fill 1500–2000 dòng)
│   ├── case-study-reviewer.md             # Quality gate (8 dimensions PASS/FAIL)
│   ├── topic-researcher.md                # Arxiv + blog mining → proposals/
│   ├── cross-ref-finder.md                # Auto-backfill wiki graph links
│   └── reference-validator.md             # URL liveness + stale-number flag
├── skills/                                # 6 slash commands
│   ├── new-case-study/SKILL.md            # /new-case-study <id>  (scaffold + dup prompt)
│   ├── update-index/SKILL.md              # /update-index
│   ├── promote-draft/SKILL.md             # /promote-draft <id>   (review gate)
│   ├── lint-bilingual/SKILL.md            # /lint-bilingual <file>
│   ├── propose-topic/SKILL.md             # /propose-topic <scope>
│   └── validate-refs-full/SKILL.md        # /validate-refs-full <id|all>
└── mcp/
    ├── kb-mcp/                            # Local Python MCP server (uv-managed)
    │   ├── server.py                      # 6 tools (see Phase 2 below)
    │   ├── pyproject.toml + uv.lock
    │   └── tests/                         # 11 pytest cases
    └── arxiv-storage/                     # Runtime cache for arxiv MCP (gitignored)
```

Bổ trợ ở ngoài `.claude/`:

```text
.mcp.json                                  # Repo-root MCP server registry (kb-mcp + arxiv)
scripts/
├── build_index.py                         # Regen INDEX.md + docs/progress.md, --check-duplicates
├── validate_refs.py                       # Frontmatter + structure + URL checks
└── install_user_agents.sh                 # (Optional) mirror agents sang ~/.claude/agents/
templates/
└── case_study.md                          # Skeleton có YAML frontmatter
docs/
├── planned.yaml                           # Source of truth cho 📋 planned entries
├── style-guide.md                         # Bilingual VI-EN writing rules
├── terminology.md                         # VI-EN technical term mappings
└── progress.md                            # Auto-generated snapshot
proposals/                                 # Topic-researcher output (pending review)
drafts/                                    # WIP case studies (status: draft)
case-studies/                              # Published (status: done)
```

## The Wiki Loop

```text
                ┌───── topic-researcher (Phase 3: weekly cron) ─────┐
                │                                                    │
                ▼                                                    │
   proposals/PROPOSAL_*.md  ──(user approve)─▶  docs/planned.yaml    │
                                                      │              │
                                                      │              │
                                              /new-case-study        │
                                              ├─ pre-check planned.yaml overlap
                                              ├─ if collide: ASK user
                                              │  ("ghi đè cái cũ" | "bỏ cái mới")
                                              └─ on "ghi đè": prune planned entry
                                                      │
                                                      ▼
                                              drafts/SX-YY_*.md  ──▶ @ai-sysdesign-knowledge-writer
                                                      │                       │
                                                      │                       ▼
                                                      │              fill 7 sections,
                                                      │              1500–2000 dòng
                                                      │                       │
                                                      │     ┌─────────────────┘
                                                      ▼     ▼
                                              /promote-draft
                                              ├─ @case-study-reviewer (8-dim PASS/FAIL)
                                              ├─ FAIL → STOP, in report
                                              └─ PASS → mv drafts/ → case-studies/<scope>/
                                                                    status: draft → done
                                                                    build_index.py regen
                                                                            │
                                                                            ▼
                                                              case-studies/<scope>/SX-YY_*.md
                                                                            │
                                                                            ▼
                                                              @cross-ref-finder
                                                              ├─ scan ≤9 cũ
                                                              ├─ edit tối đa 3 file
                                                              └─ inline link densify graph
                                                                            │
                                                                            ▼
                                                              (Phase 3: quarterly)
                                                              /validate-refs-full all
                                                              @reference-validator URL liveness
```

**3-layer integrity guard** (đảm bảo invariant: 1 id ở 1 source duy nhất):

1. `build_index.py` abort exit-1 nếu detect duplicate (any normal run aborts).
2. `/new-case-study` skill: AskUserQuestion khi id đã có ở `planned.yaml` + tự prune entry sau scaffold.
3. PostToolUse hook: chạy `build_index.py --check-duplicates` sau mỗi Write/Edit lên `drafts/`, `case-studies/`, hoặc `docs/planned.yaml`.

## Quy trình thường gặp

### 1. Bắt đầu case study mới

```text
/new-case-study S3-03
```

→ Scaffold `drafts/S3-03_vector_db_internals.md` từ `templates/case_study.md`, pre-fill frontmatter từ `docs/planned.yaml`.

Sau đó invoke writer agent để fill 7 sections (depth target 1500-2000 dòng):

```text
@ai-sysdesign-knowledge-writer
Viết drafts/S3-03_vector_db_internals.md, deep dive HNSW vs IVF-PQ,
compare Pinecone/Weaviate/Milvus/pgvector.
```

### 2. Review draft

```text
@case-study-reviewer
Review drafts/S3-03_vector_db_internals.md
```

→ Báo PASS/FAIL trên 8 dimensions (frontmatter, structure, depth, bilingual style, diagrams, refs, cross-refs, hallucination smell-test).

### 3. Promote draft → published

```text
/promote-draft S3-03
```

→ Auto chạy reviewer; nếu PASS thì move sang `case-studies/03-modern-stack/`, đổi `status: done`, xóa khỏi `planned.yaml`, regen INDEX.

### 4. Sau khi sửa file, update INDEX

```text
/update-index
```

→ Run `python scripts/build_index.py`. Cũng tự chạy ngầm qua PostToolUse hook trên mỗi Write/Edit case study.

### 5. Lint quick

```text
/lint-bilingual case-studies/03-modern-stack/S3-03_vector_db_internals.md
```

→ Grep over-translation, missing acronym expansion, structural issues.

## Frontmatter schema (canonical)

Xem [`templates/case_study.md`](../templates/case_study.md). Tóm tắt:

| Field | Type | Auto-managed? | Mục đích |
|---|---|---|---|
| `id` | `SX-YY` | no | Immutable identifier |
| `title` | str | no | Display name (quote nếu có `:`) |
| `slug` | snake_case | no | Filename suffix |
| `scope` | 1-4 | no | Maps to folder |
| `scope_name` | str | no | Display name |
| `difficulty` | enum | no | foundational \| intermediate \| intermediate-advanced \| advanced |
| `status` | enum | no | planned \| draft \| done |
| `summary` | str | no | 1-sentence for INDEX table |
| `tags` | list | no | Search/filter facet |
| `cross_refs` | list[id] | no | Knowledge graph edges |
| `created` | ISO date | no | First commit date |
| `last_validated` | ISO date | yes (validate_refs) | Last URL check |
| `line_count` | int | yes (build_index) | Filesystem-derived |

## Permissions (settings.json)

Project-scope allowlist scoped tightly:

- ✅ `Bash(python scripts/*)`, `Bash(git status/diff/log)`, `Bash(ls/grep/wc/find ...)`, `Read`, `Edit`, `Write(drafts/* | proposals/* | case-studies/* | docs/* | INDEX.md)`
- ❌ `Bash(rm -rf*)`, `Bash(git push*)`, `Bash(git reset --hard*)`, `Write(.git/*)`

User vẫn được prompt cho commit/push — đây là cố ý (Karpathy: human-in-loop ở các thao tác có blast radius).

## Hooks (settings.json)

Hai PostToolUse hook trên `Write|Edit`, cả hai non-blocking (chỉ inform):

1. **Validate refs on save**: nếu file path match `(case-studies|drafts)/*.md` thì auto-run `validate_refs.py` và in các dòng `FAIL|missing|invalid`. Catch structural issues sớm.
2. **Duplicate-id check**: nếu file path match `(case-studies|drafts|docs/planned\.yaml)` thì auto-run `python scripts/build_index.py --check-duplicates` và in `ERROR`/id collision. Catch state drift giữa 3 source.

## Roadmap phases

### ✅ Phase 1 — Foundation (đã xong)

- Frontmatter schema + template
- `build_index.py` + `validate_refs.py`
- `case-study-reviewer` agent
- 4 skills (new-case-study, update-index, promote-draft, lint-bilingual)
- Settings.json với permissions + 1 hook
- Backfill 9 case studies hiện có

### ✅ Phase 2 — MCP + 3 agents (đã xong 2026-05-20)

- `.claude/mcp/kb-mcp/` Python MCP server (uv-managed), expose 6 tools:
  - `kb_get_terminology(en_term)`
  - `kb_list_case_studies(scope?, status?)`
  - `kb_get_case_study(case_id)`
  - `kb_find_cross_refs(topic, limit?)`
  - `kb_propose_next_topic(scope?)`
  - `kb_validate_refs(case_id, full?)`
- `.mcp.json` ở repo root register `kb-mcp` (local) + `arxiv` (external `arxiv-mcp-server`)
- `topic-researcher` agent (WebSearch + arxiv MCP + kb-mcp)
- `cross-ref-finder` agent (auto-backfill links sau promote)
- `reference-validator` agent (URL liveness + stale-number flag)
- `/propose-topic <scope>` skill + `/validate-refs-full <id|all>` skill
- 11/11 pytest pass trên kb-mcp core tools

### 📋 Phase 3 — Autonomy

- Weekly cron: `topic-researcher` fill `proposals/`
- Quarterly cron: `reference-validator --full` open PR cho dead URLs
- Post-merge hook: trigger `cross-ref-finder` qua Claude API

## Installation cho user khác clone repo

1. Clone repo
2. Cài Claude Code (CLI hoặc IDE) — Opus model
3. Cài `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (cho MCP servers + scripts)
4. `cd .claude/mcp/kb-mcp && uv sync --group dev` (cài deps cho local MCP)
5. (Optional) `pip install pyyaml` system-wide nếu muốn chạy `scripts/build_index.py` trực tiếp
6. **Mở Claude Code AT repo root** — agents + skills + hooks + MCP auto-detect

> ⚠ **Workspace root quan trọng**: Claude Code chỉ scan `.claude/` ở workspace root, KHÔNG recurse subdirs. Mở workspace ở parent folder (vd `/projects/` thay vì `/projects/agentic-aisys-wiki/`) sẽ làm tất cả project-scope agents/skills/MCP invisible. Symptom: `@case-study-reviewer` báo "Agent not found".

Nếu muốn xài `ai-sysdesign-knowledge-writer` ở mọi project khác (cross-project user-scope):

```bash
./scripts/install_user_agents.sh writer
# Restart Claude Code
```

## Philosophy

**Karpathy-style operating principles**:

1. **Sparse but deep** — không bloat. Mỗi file `.claude/` phải earn its place.
2. **Single source of truth** — filesystem + frontmatter. Không có state ngoài đây.
3. **Human-in-loop at high-blast-radius** — promote/commit/push cần xác nhận. Researcher đề xuất nhưng không tự merge.
4. **Show your tools** — repo public cả nội dung lẫn pipeline. Không có hidden CI magic.
5. **Quality > quantity** — reviewer agent từ chối draft kém. Wiki không thêm nội dung mỏng.
