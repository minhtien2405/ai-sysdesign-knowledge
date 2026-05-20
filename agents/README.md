# ai-sysdesign-knowledge-writer — Claude Code Agent

Specialized **Claude Code subagent** chuyên viết knowledge documents về AI system design tại big tech companies theo phong cách bilingual Việt-Anh. Đây là agent đã được dùng để tạo toàn bộ case studies trong repo này.

## Agent capabilities

- **Domain expertise**: recommendation systems, ads ranking, search ranking, feature stores, model serving, ML training platforms, LLM serving infrastructure, MLOps, drift detection.
- **Bilingual writing style**: giữ nguyên technical terms tiếng Anh, diễn giải tiếng Việt — phù hợp với AI/ML engineer Việt Nam.
- **Structured output**: mỗi case study tuân thủ 7-section template (Overview → Requirements → Architecture → Deep dive → Trade-offs → Lessons learned → References).
- **Depth-first**: target 1500–2000 dòng/case study với mechanism deep-dive, R&D evolution, improvements over time, implementation depth.
- **No hallucination**: cite engineering blogs, arXiv papers, conference talks — nếu không chắc thì ghi rõ.
- **Persistent memory**: agent có file-based memory ở `~/.claude/agent-memory/ai-sysdesign-knowledge-writer/` để giữ consistency (terminology, progress tracking) qua nhiều sessions.

## Cài đặt

### Yêu cầu

- [Claude Code](https://claude.com/claude-code) đã cài đặt (CLI hoặc IDE extension).
- Subscription Claude (Opus model recommended — agent dùng `model: opus` trong frontmatter).

### Bước cài đặt

```bash
# Tạo thư mục agents nếu chưa có
mkdir -p ~/.claude/agents

# Copy agent definition vào user-scope
cp ai-sysdesign-knowledge-writer.md ~/.claude/agents/

# Restart Claude Code session (close và mở lại CLI/IDE extension)
```

Sau khi restart, agent sẽ xuất hiện trong danh sách:

```bash
claude /agents
# Hoặc trong session:
# Type /agent → autocomplete sẽ hiện ai-sysdesign-knowledge-writer
```

### Verify cài đặt

Mở Claude Code session mới và check:

```
/agent ai-sysdesign-knowledge-writer
```

Agent sẽ load và sẵn sàng nhận yêu cầu.

## Cách dùng

### Pattern 1 — Viết case study mới

```
Viết case study về [system name], deep dive vào mechanism + R&D evolution + improvements.
Output: case-studies/0X-scope-name/SX-YY_topic_slug.md
```

Ví dụ:
```
Viết case study về Alibaba DIN/DIEN — attention-based user interest modeling.
```

Agent sẽ:
1. Đọc agent memory (project status, terminology) để giữ consistency.
2. Plan structure 7 sections.
3. Viết case study target 1500-2000 dòng.
4. Update INDEX.md (chuyển 📋 → ✅).
5. Update agent memory (progress, terminology mới nếu có).

### Pattern 2 — Refine case study đã có

```
Refine case study S1-01 — thêm deep dive về candidate generation network architecture.
```

### Pattern 3 — Tạo comparison document

```
Tạo comparison document so sánh các vector databases: Pinecone, Weaviate, Milvus, pgvector, Qdrant.
Format: comparison table với columns System | Approach | Pros | Cons | Use case.
```

## Configuration trong agent file

Agent definition (`ai-sysdesign-knowledge-writer.md`) có 4 fields quan trọng trong frontmatter:

| Field | Value | Mục đích |
|---|---|---|
| `name` | `ai-sysdesign-knowledge-writer` | Identifier dùng khi gọi `/agent` |
| `description` | (xem file) | Claude main agent dùng để auto-route requests |
| `model` | `opus` | Force dùng Claude Opus (best for deep technical writing) |
| `memory` | `user` | Memory persist ở user-scope, share across projects |

## Memory system

Agent có 4 loại memory tại `~/.claude/agent-memory/ai-sysdesign-knowledge-writer/`:

| File | Type | Mục đích |
|---|---|---|
| `user_profile.md` | user | Profile + learning goals của bạn |
| `project_knowledge_base.md` | project | Progress tracker, file naming convention, structure rules |
| `terminology_consistency.md` | reference | VI-EN technical term mappings — giữ consistency cross-sessions |
| `feedback_depth_first.md` | feedback | Rule depth-first (1500-2000 dòng/case study) |

Snapshot của các memory files này được copy vào [`docs/`](../docs/) folder của repo để public reader có thể tham khảo style guide và terminology.

## Quality standards

Agent enforce 6 quality rules:

1. **Accuracy first** — chỉ cite source xác thực
2. **No hallucination** — không bịa names/numbers
3. **Concrete numbers** — QPS, latency, model size phải có
4. **Depth over breadth** — sâu 1 component hơn lướt qua nhiều
5. **ASCII/Mermaid diagrams** — cho mọi architectural concept
6. **Cite sources** — paper + URL + năm

## Troubleshooting

**Agent không xuất hiện sau cài đặt?**
- Confirm file ở đúng path: `~/.claude/agents/ai-sysdesign-knowledge-writer.md`.
- Restart hoàn toàn Claude Code (đóng terminal, mở lại).
- Check frontmatter YAML hợp lệ (không thiếu `---` mở/đóng).

**Agent viết case study quá ngắn?**
- Check feedback memory `feedback_depth_first.md` có tồn tại không.
- Yêu cầu rõ trong prompt: "target 1500-2000 dòng, deep dive mechanism + R&D evolution".

**Agent dùng sai terminology (vd dịch "embedding" thành "vector nhúng")?**
- Check `terminology_consistency.md` đã được agent đọc chưa.
- Update file nếu thấy term mới chưa có mapping.

## License

Agent definition share theo MIT (cùng với repo).

## Credits

Agent được tạo và refine qua các sessions với Claude Code (Opus 4.7). Persona và quality standards được customize cho mục tiêu học AI system design của AI/ML engineer Việt Nam.
