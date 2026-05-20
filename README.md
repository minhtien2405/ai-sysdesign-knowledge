# AI System Design Knowledge Base

> Bộ case studies chuyên sâu về AI system design tại big tech companies — viết theo phong cách **bilingual Việt-Anh** dành cho AI/ML engineer Việt Nam ôn luyện system design, hiểu sâu cơ chế, và bắt kịp tech stack hiện đại.

## Mục tiêu

Knowledge base này tập trung trả lời 4 câu hỏi lớn cho mỗi hệ thống AI/ML production:

1. **Hệ thống được thiết kế thế nào?** — architecture, data flow, components.
2. **Tại sao chọn approach đó?** — trade-offs vs alternatives, R&D evolution.
3. **Mechanism hoạt động ra sao?** — algorithm step-by-step, intuition, math khi cần.
4. **Đã được cải tiến qua thời gian như thế nào?** — failure modes, successor systems, current best practices.

Mỗi case study dài **1500–2000 dòng markdown** (từ S1-03 trở đi), có ASCII diagrams, concrete numbers (QPS, latency, model size), pseudo-code Python, và references đến papers/engineering blogs thật.

## Bắt đầu từ đâu

- **Mới hoàn toàn**: đọc [INDEX.md](INDEX.md) → bắt đầu với S1-01 (YouTube reco) theo suggested reading order.
- **Đã có background recommendation/ads**: skip foundations, đi thẳng vào S1-03 (PinSage GNN) → S2-02 (CTR ranking lineage) → S2-01 (DLRM).
- **Tập trung LLM stack**: S3-01 (vLLM serving) → S3-02 (Production RAG).
- **Tập trung MLOps/production**: S4-01 (Michelangelo) → S4-02 (A/B testing).

## Cấu trúc repo

```text
ai-sysdesign-knowledge/
├── README.md                       # File này
├── INDEX.md                        # Roadmap toàn bộ 18 case studies + progress tracker
├── case-studies/
│   ├── 01-foundations/             # Scope 1: ML system foundations
│   ├── 02-model-development/       # Scope 2: Model architecture & training
│   ├── 03-modern-stack/            # Scope 3: LLM / RAG / Agent / CV
│   └── 04-production/              # Scope 4: Production ML systems
├── agents/                         # Claude Code agent dùng để viết case studies
│   ├── ai-sysdesign-knowledge-writer.md
│   └── README.md                   # Hướng dẫn install + use
└── docs/
    ├── style-guide.md              # Bilingual VI-EN writing rules
    ├── terminology.md              # VI-EN technical term mappings (consistency)
    └── progress.md                 # Snapshot tiến độ chi tiết
```

## Scope coverage

| Scope | Focus | Case studies done | Planned |
|---|---|---|---|
| 1 — Foundations | End-to-end ML pipeline, recsys two-stage, online learning, GNN retrieval | 3 (S1-01, S1-02, S1-03) | 1 |
| 2 — Model dev | Architecture lineage, CTR ranking models, embedding tables | 2 (S2-01, S2-02) | 3 |
| 3 — Modern stack | LLM serving (vLLM), RAG, vector DB, agents, OCR | 2 (S3-01, S3-02) | 3 |
| 4 — Production | Feature store, A/B testing, drift, GPU mgmt | 2 (S4-01, S4-02) | 2 |
| **Total** | | **9 / 18** | **9** |

Xem [INDEX.md](INDEX.md) cho danh sách đầy đủ + status.

## Phong cách viết — Bilingual VI-EN

**Giữ nguyên tiếng Anh**: technical terms (embedding, feature store, two-tower, KV cache, PagedAttention), tên hệ thống/công ty (Michelangelo, vLLM, DLRM), acronyms (CTR, QPS, P99, NDCG), framework names, paper names.

**Dùng tiếng Việt**: diễn giải, phân tích, trade-off analysis, intuition, transition.

**Ví dụ câu chuẩn**:

> "YouTube sử dụng kiến trúc two-tower model để xử lý candidate generation, trong đó user tower và item tower được train riêng biệt nhưng share chung embedding space."

Xem [docs/style-guide.md](docs/style-guide.md) và [docs/terminology.md](docs/terminology.md) cho rules đầy đủ.

## Đóng góp / viết thêm case study

Knowledge base này được duy trì bởi một **Claude Code agent** chuyên biệt (`ai-sysdesign-knowledge-writer`). Nếu bạn dùng Claude Code và muốn tự viết thêm case studies theo cùng phong cách:

1. Copy [`agents/ai-sysdesign-knowledge-writer.md`](agents/ai-sysdesign-knowledge-writer.md) vào `~/.claude/agents/`.
2. Restart Claude Code session.
3. Gọi `/agent ai-sysdesign-knowledge-writer` và yêu cầu case study mới.

Chi tiết xem [agents/README.md](agents/README.md).

## Conventions quan trọng

- **No hallucination**: nếu không có public source → ghi rõ "based on public information, internal details may differ".
- **Cite sources**: paper (arXiv link + năm), engineering blog (URL + năm), conference talk.
- **Concrete numbers**: ưu tiên số cụ thể, ghi nguồn nếu approximation.
- **Cross-reference**: link giữa các case studies khi relevant (vd S1-03 PinSage → S3-03 Vector DB).

## License

MIT — xem [LICENSE](LICENSE).

## Disclaimer

Đây là knowledge base **học tập cá nhân**. Nội dung dựa trên public engineering blogs, papers, và conference talks. Internal architecture của các công ty có thể khác. Numbers/scale là approximation tại thời điểm reference, không phản ánh state hiện tại.
