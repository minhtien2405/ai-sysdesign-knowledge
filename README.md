# AI System Design Knowledge Base

> Self-extending bilingual VI-EN wiki về AI system design tại big tech, dành cho AI/ML engineer Việt Nam. Lấy cảm hứng từ Karpathy ethos: **sparse-but-deep, show your tools, single source of truth**.

## Mục tiêu

Knowledge base này tập trung trả lời 4 câu hỏi lớn cho mỗi hệ thống AI/ML production:

1. **Hệ thống được thiết kế thế nào?** — architecture, data flow, components.
2. **Tại sao chọn approach đó?** — trade-offs vs alternatives, R&D evolution.
3. **Mechanism hoạt động ra sao?** — algorithm step-by-step, intuition, math khi cần.
4. **Đã được cải tiến qua thời gian như thế nào?** — failure modes, successor systems, current best practices.

Mỗi case study target **1500–2000 dòng markdown** (từ S1-03 trở đi), có ASCII diagrams, concrete numbers (QPS, latency, model size), pseudo-code Python, và references đến papers/engineering blogs thật.

## Bắt đầu từ đâu

- **Đọc nội dung**: [INDEX.md](INDEX.md) — roadmap 18 case studies + reading order.
- **Hiểu workflow agentic**: [`.claude/README.md`](.claude/README.md) — agents, skills, hooks, scripts.
- **Style rules**: [docs/style-guide.md](docs/style-guide.md) + [docs/terminology.md](docs/terminology.md).

## Cấu trúc repo

```text
agentic-aisys-wiki/
├── README.md                          # File này
├── INDEX.md                           # Auto-generated roadmap (do NOT edit manually)
├── LICENSE
│
├── case-studies/                      # ◀── Reader đọc folder này
│   ├── 01-foundations/                # Scope 1: ML system foundations
│   ├── 02-model-development/          # Scope 2: Model architecture & training
│   ├── 03-modern-stack/               # Scope 3: LLM / RAG / Agent / CV
│   └── 04-production/                 # Scope 4: Production ML systems
│
├── drafts/                            # WIP case studies (status: draft)
├── proposals/                         # Topic proposals từ @topic-researcher (pre-approval)
├── templates/
│   └── case_study.md                  # Skeleton với YAML frontmatter
│
├── docs/
│   ├── style-guide.md                 # Bilingual VI-EN writing rules
│   ├── terminology.md                 # VI-EN technical term mappings
│   ├── progress.md                    # Auto-generated progress snapshot
│   └── planned.yaml                   # Source of truth cho 📋 planned entries
│
├── scripts/                           # Build pipeline + install helpers
│   ├── build_index.py                 # Regen INDEX + progress từ frontmatter
│   ├── validate_refs.py               # Frontmatter + structure + URL checks
│   └── install_user_agents.sh         # (Optional) mirror agents to ~/.claude/agents/
│
├── .mcp.json                          # MCP servers config (kb-mcp + arxiv)
│
└── .claude/                           # ◀── Operator đọc folder này (agentic infra)
    ├── README.md                      # Workflow + roadmap chi tiết
    ├── settings.json                  # Permissions + hooks (project-scope)
    ├── agents/                        # 5 subagent definitions
    │   ├── ai-sysdesign-knowledge-writer.md
    │   ├── case-study-reviewer.md
    │   ├── topic-researcher.md
    │   ├── cross-ref-finder.md
    │   └── reference-validator.md
    ├── skills/                        # 6 slash commands
    │   ├── new-case-study/SKILL.md
    │   ├── update-index/SKILL.md
    │   ├── promote-draft/SKILL.md
    │   ├── lint-bilingual/SKILL.md
    │   ├── propose-topic/SKILL.md
    │   └── validate-refs-full/SKILL.md
    └── mcp/
        ├── kb-mcp/                    # Local Python MCP server (uv-managed)
        └── arxiv-storage/             # Runtime cache for arxiv-mcp (gitignored)
```

## Self-extending workflow

Knowledge base **self-maintains** qua Claude Code agents + skills. Một id case-study đi qua 3 stage có boundary rõ ràng — `planned.yaml` → `drafts/` → `case-studies/<scope>/`:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  RESEARCH                                                                │
│    /propose-topic <scope>                                                │
│      └─ @topic-researcher  mines arxiv + blogs                           │
│         └─ writes proposals/PROPOSAL_*.md  (status: pending-review)      │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           ▼
                              (manual) review + approve
                                           ▼
                                   docs/planned.yaml
                                           │
┌──────────────────────────────────────────┴───────────────────────────────┐
│  SCAFFOLD                                                                │
│    /new-case-study S3-XX                                                 │
│      ├─ if id ∈ planned.yaml: ASK user — "ghi đè cái cũ" | "bỏ cái mới"  │
│      └─ on "ghi đè": scaffold drafts/SX-XX_*.md                          │
│                      AND remove planned.yaml entry (invariant)           │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           ▼
┌──────────────────────────────────────────┴───────────────────────────────┐
│  AUTHOR                                                                  │
│    @ai-sysdesign-knowledge-writer  fill drafts/SX-XX                     │
│      └─ 7 sections, 1500–2000 dòng, bilingual VI-EN, real references     │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           ▼
┌──────────────────────────────────────────┴───────────────────────────────┐
│  REVIEW + PUBLISH                                                        │
│    /promote-draft S3-XX                                                  │
│      ├─ @case-study-reviewer  8-dimension PASS/FAIL gate                 │
│      ├─ FAIL → STOP, in report                                           │
│      └─ PASS → mv drafts/ → case-studies/<scope>/                        │
│                status: draft → done                                      │
│                build_index.py regen INDEX + progress                     │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           ▼
┌──────────────────────────────────────────┴───────────────────────────────┐
│  GRAPH BACKFILL                                                          │
│    @cross-ref-finder                                                     │
│      └─ scan existing studies, edit ≤3 với inline link tới new study     │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           ▼
                          (quarterly) /validate-refs-full all
                          @reference-validator URL liveness audit
```

**3 layer defense chống duplicate id**:

1. `scripts/build_index.py` abort exit-1 nếu id xuất hiện ở > 1 source (`case-studies/`, `drafts/`, `planned.yaml`).
2. `/new-case-study` skill: pre-check overlap với `planned.yaml`, prompt user ("ghi đè cái cũ" vs "bỏ cái mới") rồi auto-prune planned entry.
3. PostToolUse hook: chạy `build_index.py --check-duplicates` ngầm sau mỗi Write/Edit lên `drafts/`, `case-studies/`, hoặc `docs/planned.yaml`.

**Invariant**: mỗi id case-study tồn tại ở **EXACTLY MỘT** trong 3 nguồn. Bất kỳ vi phạm nào đều bị flag immediately.

**3 phases roadmap**:

| Phase | Status | Components |
|---|---|---|
| 1 — Foundation | ✅ done | Frontmatter schema, `build_index.py`, `validate_refs.py`, writer + reviewer agents, 4 skills, validate-on-save hook |
| 2 — MCP + 3 agents + integrity | ✅ done (2026-05-20) | `kb-mcp` server (6 tools), arxiv MCP integration, topic-researcher / cross-ref-finder / reference-validator agents, `/propose-topic` + `/validate-refs-full` skills, duplicate-prevention mechanism |
| 3 — Autonomy | 🚧 scaffolded | 3 GitHub Actions trong `.github/workflows/`: weekly `@topic-researcher` cron, quarterly `@reference-validator --full` cron, post-merge `@cross-ref-finder`. Activate bằng cách add `ANTHROPIC_API_KEY` secret — xem [`.github/README.md`](.github/README.md). |

Chi tiết operator-side: [`.claude/README.md`](.claude/README.md).

## Phong cách viết — Bilingual VI-EN

**Giữ nguyên tiếng Anh**: technical terms (embedding, feature store, two-tower, KV cache, PagedAttention), tên hệ thống/công ty (Michelangelo, vLLM, DLRM), acronyms (CTR, QPS, P99, NDCG), framework names.

**Dùng tiếng Việt**: diễn giải, phân tích, trade-off analysis, intuition, transition.

**Ví dụ câu chuẩn**:

> "YouTube sử dụng kiến trúc two-tower model để xử lý candidate generation, trong đó user tower và item tower được train riêng biệt nhưng share chung embedding space."

Xem [docs/style-guide.md](docs/style-guide.md) đầy đủ.

## Conventions

- **No hallucination**: nếu không có public source → ghi "based on public information, internal details may differ".
- **Cite sources**: paper (arXiv link + năm), engineering blog (URL + năm), conference talk.
- **Concrete numbers**: ưu tiên số cụ thể, ghi nguồn nếu approximation.
- **Cross-reference**: link giữa các case studies qua `cross_refs` trong frontmatter + relative path trong body.

## Yêu cầu để run agentic infra

- [Claude Code](https://claude.com/claude-code) CLI hoặc IDE extension (Opus model)
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) — Python package + venv manager

## Setup (clone → chạy được)

```bash
# 1. Install uv (1 lần per máy)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc để persist

# 2. Clone + sync kb-mcp deps
git clone <this-repo>
cd agentic-aisys-wiki
cd .claude/mcp/kb-mcp && uv sync --group dev && cd ../../..

# 3. (Optional) install pyyaml system-wide để chạy scripts/ trực tiếp
pip install pyyaml

# 4. Set up .claude/setting.local.json, example:
{
  "permissions": {
    "allow": [
      "WebSearch",
      "Bash(uv tool *)",
      "Read(//home/tienpham/.cache/uv/**)",
      "Bash(gh api *)",
      "Bash(python3 *)",
      "Bash(git check-ignore *)",
      "Bash(cp INDEX.md /tmp/INDEX.before.md)",
      "Bash(cp docs/progress.md /tmp/progress.before.md)"
    ]
  },
  "enabledMcpjsonServers": [
    "kb-mcp",
    "arxiv"
  ]
}


# 4. Mở Claude Code AT REPO ROOT (workspace phải là agentic-aisys-wiki/,
#    KHÔNG phải parent folder. Nếu mở parent thì project-scope .claude/ không được scan.)
cd agentic-aisys-wiki
claude                           # CLI
# hoặc VS Code: File → Open Folder → chọn agentic-aisys-wiki/
```

Sau khi mở repo, Claude Code tự load:

- **6 skills**: `/new-case-study`, `/update-index`, `/promote-draft`, `/lint-bilingual`, `/propose-topic`, `/validate-refs-full`
- **5 agents**: `@ai-sysdesign-knowledge-writer`, `@case-study-reviewer`, `@topic-researcher`, `@cross-ref-finder`, `@reference-validator`
- **2 MCP servers**: `kb-mcp` (6 tools, local), `arxiv` (search/download papers)
- **Hooks**: validate-on-save + duplicate-check trên mỗi Write/Edit

(Optional) `./scripts/install_user_agents.sh writer` — mirror writer agent sang `~/.claude/agents/` để dùng cross-project.

Chi tiết operator-side: [`.claude/README.md`](.claude/README.md).

## Đóng góp / viết thêm case study

Full pipeline (mỗi step là 1 command / agent invocation):

```text
1. /propose-topic 3                                  # mines arxiv + blogs → proposals/PROPOSAL_*.md
   ▼
2. (manual) review proposals/, approve nào → thêm vào docs/planned.yaml
   ▼
3. /new-case-study S3-XX                             # scaffold drafts/, prompt nếu id collide planned.yaml
   ▼
4. @ai-sysdesign-knowledge-writer fill drafts/S3-XX  # 1500–2000 dòng, bilingual, references thật
   ▼
5. /promote-draft S3-XX                              # @case-study-reviewer gate; if PASS → case-studies/
   ▼
6. @cross-ref-finder                                 # backfill inline links từ ≤3 case studies cũ
   ▼
7. (Optional) /validate-refs-full S3-XX              # quarterly URL liveness check
```

Mỗi bước có quality gate riêng — pipeline tự stop nếu draft chưa đủ depth, references thiếu URL, hoặc id collide.

## License

MIT — xem [LICENSE](LICENSE).

## Disclaimer

Đây là knowledge base **học tập cá nhân**. Nội dung dựa trên public engineering blogs, papers, và conference talks. Internal architecture của các công ty có thể khác. Numbers/scale là approximation tại thời điểm reference, không phản ánh state hiện tại.
