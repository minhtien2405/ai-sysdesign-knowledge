# Progress Snapshot — AI System Design Knowledge Base

> Snapshot trạng thái tại thời điểm restructure (2026-05-20). Để xem trạng thái real-time, đọc [`../INDEX.md`](../INDEX.md).

## Scope (4 mảng)

- **Scope 1** — AI/ML System Design Foundations (data pipeline, training, serving, monitoring, MLOps)
- **Scope 2** — Model Development & Training
- **Scope 3** — Modern Tech Stack (LLM/RAG/Agent/CV)
- **Scope 4** — Production AI Systems

## File naming convention

`Sx-yy_<topic_slug>.md`

- `x` = scope number (1-4)
- `yy` = sequence number within scope (01, 02, …)
- `topic_slug` = snake_case, descriptive

Files được organize theo folder structure:

```text
case-studies/
├── 01-foundations/        # Sx = 1
├── 02-model-development/  # Sx = 2
├── 03-modern-stack/       # Sx = 3
└── 04-production/         # Sx = 4
```

## Completed (✅) — 9 case studies

| File | Scope | Topic | Lines |
|---|---|---|---|
| `01-foundations/S1-01_youtube_recommendation_end_to_end.md` | 1 | YouTube two-stage reco (candidate gen + ranking) | 496 |
| `01-foundations/S1-02_tiktok_monolith_realtime_recommendation.md` | 1 | TikTok Monolith — online learning, collisionless embedding | 601 |
| `01-foundations/S1-03_pinterest_pinsage_graph_retrieval.md` | 1 | Pinterest PinSage — GNN 3B pins, MapReduce inference, PinnerSage/TransAct | 1514 |
| `02-model-development/S2-01_meta_dlrm_architecture.md` | 2 | Meta DLRM — sparse embeddings, hybrid parallelism | 450 |
| `02-model-development/S2-02_wide_deep_deepfm_dcn_evolution.md` | 2 | Wide & Deep → DeepFM → DCN-V2 evolution cho CTR | 654 |
| `03-modern-stack/S3-01_vllm_paged_attention_continuous_batching.md` | 3 | vLLM — PagedAttention + continuous batching | 511 |
| `03-modern-stack/S3-02_production_rag_system_architecture.md` | 3 | Production RAG — hybrid retrieval, reranker, contextual, eval | 796 |
| `04-production/S4-01_uber_michelangelo_feature_store.md` | 4 | Uber Michelangelo — feature store + ML platform | 562 |
| `04-production/S4-02_ab_testing_experimentation_platforms.md` | 4 | A/B testing — CUPED, mSPRT, interleaving, MAB, SRM | 763 |

**Total**: ~6,347 dòng markdown across 9 files.

## Planned (📋) — 9 more case studies

- **Scope 1**: S1-04 Netflix personalization stack
- **Scope 2**: S2-03 Alibaba DIN/DIEN, S2-04 LLM pretraining at scale, S2-05 Finetune vs RAG vs Prompt framework
- **Scope 3**: S3-03 Vector DB (HNSW vs IVF-PQ), S3-04 Agent frameworks, S3-05 OCR/Document AI
- **Scope 4**: S4-03 Drift detection, S4-04 GPU cluster mgmt & cost

**Total planned**: 18 case studies.

## Depth target (từ S1-03 trở đi)

Target line count = **1500-2000 dòng/case study** với:

1. **Mechanism deep-dive** — algorithm step-by-step, intuition, concrete examples với numbers
2. **R&D evolution** — predecessor systems, paper lineage, alternatives đã thử và bị loại
3. **Improvements over time** — failure modes, fixes, successor systems
4. **Implementation depth** — đủ chi tiết để reader có thể tự build

## Structure cố định mỗi case study

1. **Overview** — bối cảnh, business problem, tại sao quan trọng
2. **System Requirements** — functional, non-functional với concrete numbers, constraints
3. **High-level Architecture** — ASCII diagram + data flow
4. **Deep dive các components chính**
5. **Trade-offs & Design decisions** — comparison tables
6. **Lessons learned & Best practices**
7. **References** — papers, engineering blogs, talks (đánh dấu độ tin cậy)

## Style rules

Xem [`style-guide.md`](style-guide.md) cho rules đầy đủ. Tóm tắt:

- **Bilingual**: technical terms tiếng Anh, diễn giải/phân tích tiếng Việt
- ASCII diagrams (preferred) hoặc Mermaid
- Pseudo-code Python-flavored, comment tiếng Việt
- Concrete numbers — luôn ghi source/year, hoặc note "approximation from public info"
- KHÔNG hallucinate names/components/numbers — không chắc thì ghi rõ

## Suggested reading orders

**Người mới (9 files hiện có)**:

1. Foundations: S1-01 → S1-02 → S1-03
2. Model dev: S2-02 → S2-01
3. Production loop: S4-01 → S4-02
4. Modern LLM stack: S3-01 → S3-02

**Đã có background reco/ads** (skip foundations):

- S1-03 → S2-02 → S2-01 → S4-01 → S4-02 → S3-01 → S3-02

## Notes cho contributors

- Khi add/complete case study mới, **luôn update INDEX.md** (chuyển 📋 → ✅, update line count).
- Cho mỗi case study mới, cross-reference đến case studies hiện có (vd S1-01 ↔ S2-01 ↔ S4-01).
- Nếu deep dive một technique cụ thể (vd AWQ quantization), tạo sub-file dạng `Sx-yy-a_<sub_topic>.md` hoặc append vào case study cha.
- File `terminology.md` phải được update khi có technical terms mới xuất hiện để giữ consistency.
