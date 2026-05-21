# `.github/` — Phase 3 Autonomy

GitHub Actions workflows tự động hoá knowledge-base maintenance. Đây là **Phase 3** của roadmap (xem [`../.claude/README.md`](../.claude/README.md)) — repo-wide autonomy chạy độc lập trên GitHub infra, không cần local Claude Code session.

## 3 workflows

| File | Trigger | Producer | Output |
|---|---|---|---|
| [`workflows/weekly-topic-research.yml`](workflows/weekly-topic-research.yml) | `cron: 0 0 * * 1` (Monday 00:00 UTC) + manual | `@topic-researcher` | New `proposals/PROPOSAL_*.md` + PR |
| [`workflows/quarterly-ref-validation.yml`](workflows/quarterly-ref-validation.yml) | `cron: 0 0 1 */3 *` (1st of every 3rd month) + manual | `@reference-validator` | URL liveness report; PR fix dead links nếu có |
| [`workflows/post-merge-cross-ref.yml`](workflows/post-merge-cross-ref.yml) | `pull_request: types: [closed]` to `main` | `@cross-ref-finder` | Edit ≤3 case studies cũ với inline link tới new study + PR |

## Enable Phase 3 (1-time setup)

Phase 3 workflows scaffolded sẵn nhưng **chưa active** cho tới khi bạn:

### 1. Add `ANTHROPIC_API_KEY` secret

Repo settings → **Secrets and variables** → **Actions** → **New repository secret**:

| Field | Value |
|---|---|
| Name | `ANTHROPIC_API_KEY` |
| Secret | Anthropic API key (từ console.anthropic.com → API Keys) |

> Cost note: Mỗi run dùng Claude Opus 4.7 với prompt + agent tool use. Estimate cost: weekly research ~$0.50–2, quarterly validation ~$5–15 (vì check tất cả URL), post-merge ~$0.20–1 per merge.

### 2. Grant workflow permissions

Repo settings → **Actions** → **General** → **Workflow permissions**:

- ✅ Read and write permissions (cho phép workflow commit + open PR)
- ✅ Allow GitHub Actions to create and approve pull requests

### 3. (Optional) Adjust cron timing

Mặc định weekly = Monday 00:00 UTC, quarterly = 1st of Jan/Apr/Jul/Oct. Sửa `on.schedule.cron` trong file YAML nếu muốn khác.

### 4. Test manual trigger trước

Actions tab → chọn workflow → **Run workflow** → kiểm tra log + PR output. Sau khi confirm chạy đúng, cron tự fire theo schedule.

## Workflow dependencies

Mỗi workflow cần:

- **Repo state**: `.claude/agents/`, `.claude/mcp/kb-mcp/`, `.mcp.json`, `scripts/build_index.py`, `scripts/validate_refs.py`, `docs/planned.yaml`, `templates/case_study.md` — tất cả đã có trong repo.
- **GitHub Action**: [`anthropics/claude-code-action@v1`](https://github.com/anthropics/claude-code-action) — runs Claude với full agent + skill access.
- **Runner**: `ubuntu-latest` (GitHub-hosted). Cần Python 3.10 + `uv`.
- **Secret**: `ANTHROPIC_API_KEY`.

## Local test trước khi push

Nếu muốn test workflow local trước khi commit:

```bash
# Install act (local GitHub Actions runner): https://github.com/nektos/act
act -W .github/workflows/weekly-topic-research.yml \
    --secret ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    workflow_dispatch
```

## Disable Phase 3

Nếu không muốn cron tự fire:

- **Disable specific workflow**: Actions tab → workflow → "..." → Disable workflow.
- **Remove all**: `rm -rf .github/workflows/` (giữ `.github/README.md` để document).
- **Pause temporarily**: Comment out `on.schedule` block trong file YAML (giữ `workflow_dispatch:` cho manual trigger).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Workflow chạy nhưng không tạo PR | Workflow permissions chưa enable "Read and write" + "create PRs" |
| `ANTHROPIC_API_KEY` not found | Secret chưa add hoặc tên sai (case-sensitive) |
| Agent failed mid-run | Agent timeout (mặc định 30 min weekly, 60 min quarterly) — adjust `timeout-minutes` |
| Duplicate detection fails | `/new-case-study` skill chưa được agent dùng, agent ghi file thẳng vào drafts/. Cần review prompt. |
| Cron không fire | GitHub disables cron sau 60 ngày không activity. Push commit hoặc manual trigger để re-activate. |

## Karpathy ethos compliance

Phase 3 workflows giữ nguyên 5 nguyên tắc (xem `.claude/README.md` § Philosophy):

1. **Sparse but deep** — 3 workflows, mỗi cái 1 mục đích rõ. Không spawn random automation.
2. **Single source of truth** — agents đọc filesystem + frontmatter, không có hidden state ngoài repo.
3. **Human-in-loop at high-blast-radius** — output là PR (không direct push to main), reviewer-gate vẫn áp dụng nếu workflow tạo case-study mới.
4. **Show your tools** — workflow YAML public, full prompt visible, không có hidden secret logic.
5. **Quality > quantity** — researcher chỉ propose, không tự promote; ref-validator open PR fix, không tự override.
