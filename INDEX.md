# AI System Design Knowledge Base — Case Study Roadmap

> Knowledge base cho AI/ML engineer ôn luyện system design qua case studies thực tế của big tech.
> Phong cách viết: **bilingual VI-EN** (giữ nguyên technical terms tiếng Anh, diễn giải tiếng Việt).
> Depth target: **1500–2000 dòng/case study** (từ S1-03 trở đi) với ASCII diagrams, concrete numbers, pseudo-code, references thật.

---

## Cách dùng knowledge base này

1. **Đọc theo scope** nếu bạn muốn tập trung một mảng (foundations / model dev / modern stack / production).
2. **Đọc theo difficulty** (foundational → intermediate → advanced) nếu bạn muốn build kiến thức tuần tự.
3. **Mỗi file** đều có structure 7 sections cố định: Overview → Requirements → Architecture → Deep dive → Trade-offs → Lessons learned → References. Bạn có thể skip phần Architecture ở lần đọc đầu, quay lại sau khi đã hiểu trade-offs.
4. **Status icons**:
   - ✅ — đã viết, đã review
   - 📋 — planned, chưa viết
   - 🚧 — đang viết (work in progress)

---

## Scope 1 — AI/ML System Design Foundations

Mục tiêu: hiểu cấu trúc end-to-end của một ML system production (data → training → serving → monitoring), nắm vững các thuật ngữ và kiến trúc cơ bản trước khi đào sâu vào component cụ thể.

| # | Case study | Difficulty | Mô tả |
|---|---|---|---|
| ✅ S1-01 | [YouTube Recommendation System End-to-End](case-studies/01-foundations/S1-01_youtube_recommendation_end_to_end.md) | Foundational | Two-stage architecture (candidate generation + ranking), DNN models, serving infra ở scale tỷ user. |
| ✅ S1-02 | [TikTok Monolith / Real-time Recommendation](case-studies/01-foundations/S1-02_tiktok_monolith_realtime_recommendation.md) | Intermediate–Advanced | Collisionless embedding table, online learning, multi-channel retrieval, cold-start strategies. |
| ✅ S1-03 | [Pinterest PinSage — Graph-based Retrieval](case-studies/01-foundations/S1-03_pinterest_pinsage_graph_retrieval.md) | Intermediate–Advanced | GNN ở scale 3 tỷ pins: random-walk sampling, importance pooling, MapReduce inference, PinnerSage/TransAct evolution. |
| 📋 S1-04 | Netflix Personalization Stack | Foundational | Multi-armed bandits, contextual ranking, offline/online evaluation, A/B test culture. |

## Scope 2 — Model Development & Training

Mục tiêu: hiểu cách big tech chọn architecture, train mô hình ở scale, debug overfit/underfit, evaluation methodology. Focus vào model design trade-offs hơn là infra.

| # | Case study | Difficulty | Mô tả |
|---|---|---|---|
| ✅ S2-01 | [Meta DLRM — Deep Learning Recommendation Model](case-studies/02-model-development/S2-01_meta_dlrm_architecture.md) | Intermediate | Dense + sparse features, embedding tables 100GB+, model + data parallelism, FBGEMM. |
| ✅ S2-02 | [Wide & Deep / DeepFM / DCN evolution](case-studies/02-model-development/S2-02_wide_deep_deepfm_dcn_evolution.md) | Intermediate | Tiến hoá architecture cho CTR prediction từ Google Play → Huawei → DCN-V2, foundation cho mọi ranking architecture sau. |
| 📋 S2-03 | Alibaba DIN / DIEN — User Interest Modeling | Advanced | Attention over user behavior sequence, GRU-based interest evolution. |
| 📋 S2-04 | LLM Pretraining at Scale (GPT/LLaMA-style) | Advanced | Data curation, tokenizer choices, 3D parallelism, training stability (loss spikes, gradient clipping). |
| 📋 S2-05 | Finetune vs RAG vs Prompt Engineering — Decision Framework | Foundational | Khi nào nên finetune, khi nào RAG, khi nào chỉ cần prompt — qua case study thực tế. |

## Scope 3 — Modern Tech Stack (LLM / RAG / Agent / CV)

Mục tiêu: làm chủ tech stack hiện đại — LLM serving, retrieval, agent frameworks, vector DB, OCR/CV pipelines.

| # | Case study | Difficulty | Mô tả |
|---|---|---|---|
| ✅ S3-01 | [vLLM Deep Dive — PagedAttention & Continuous Batching](case-studies/03-modern-stack/S3-01_vllm_paged_attention_continuous_batching.md) | Advanced | KV cache management, block table, prefill/decode scheduling, throughput vs latency. |
| ✅ S3-02 | [Production RAG System Architecture](case-studies/03-modern-stack/S3-02_production_rag_system_architecture.md) | Intermediate | Chunking strategies, hybrid retrieval (BM25 + dense + RRF), reranker, contextual retrieval, RAGAS eval. |
| 📋 S3-03 | Vector Database Internals — HNSW vs IVF-PQ | Intermediate | Index structures, recall/QPS trade-off, Pinecone/Weaviate/Milvus/pgvector comparison. |
| 📋 S3-04 | Agent Framework Architecture (ReAct, tool use, multi-agent) | Advanced | Planning, tool calling, memory, error recovery, observability. |
| 📋 S3-05 | OCR + Document AI Pipeline (LayoutLM / Donut / multimodal LLM) | Intermediate | End-to-end document understanding, table extraction, layout-aware models. |

## Scope 4 — Production AI Systems

Mục tiêu: vận hành ML system ở production — scaling, latency, A/B test, drift, cost, GPU management.

| # | Case study | Difficulty | Mô tả |
|---|---|---|---|
| ✅ S4-01 | [Uber Michelangelo — End-to-End ML Platform & Feature Store](case-studies/04-production/S4-01_uber_michelangelo_feature_store.md) | Advanced | Feature store (online + offline), training pipeline, model registry, monitoring. |
| ✅ S4-02 | [A/B Testing & Experimentation Platforms](case-studies/04-production/S4-02_ab_testing_experimentation_platforms.md) | Intermediate | Sample size, CUPED, sequential testing (mSPRT), interleaving, MAB, SRM detection, guardrails. |
| 📋 S4-03 | Data & Model Drift Detection in Production | Intermediate | PSI, KS test, embedding drift, prediction drift, alerting strategies. |
| 📋 S4-04 | GPU Cluster Management & Cost Optimization for LLM Inference | Advanced | Multi-tenant serving, GPU pooling, spot vs reserved, quantization (INT8/FP8/AWQ/GPTQ). |

---

## Progress tracker

- **Total planned**: 18 case studies
- **Completed**: 9
- **In progress**: 0
- **Planned**: 9

**Đã viết (sorted theo file path):**
1. ✅ S1-01 YouTube recommendation system end-to-end — foundations
2. ✅ S1-02 TikTok Monolith real-time recommendation — online learning paradigm
3. ✅ S1-03 Pinterest PinSage — graph-based retrieval at 3B pins scale
4. ✅ S2-01 Meta DLRM architecture — model dev trade-off
5. ✅ S2-02 Wide & Deep / DeepFM / DCN evolution — ranking architecture lineage
6. ✅ S3-01 vLLM deep dive — modern LLM serving stack
7. ✅ S3-02 Production RAG system architecture — hybrid retrieval + reranker + eval
8. ✅ S4-01 Uber Michelangelo feature store — production ML platform
9. ✅ S4-02 A/B testing & experimentation platforms — closure cho ML loop

**Suggested reading order cho người mới (9 files hiện có):**
1. **Foundations**: S1-01 (YouTube reco) → S1-02 (TikTok real-time online learning paradigm) → S1-03 (PinSage graph-based retrieval).
2. **Model dev**: S2-02 (Wide&Deep/DeepFM/DCN evolution — ranking foundations) → S2-01 (DLRM — Meta's scaled-up version).
3. **Production loop**: S4-01 (Michelangelo platform) → S4-02 (A/B test để verify wins).
4. **Modern LLM stack**: S3-01 (vLLM serving) → S3-02 (Production RAG architecture).

**Reading order cho người đã có background reco/ads** (skip foundations):
- S1-03 (PinSage — GNN approach) → S2-02 → S2-01 → S4-01 → S4-02 → S3-01 → S3-02.

---

## Conventions sử dụng trong knowledge base

- **Numbers / scale**: trích từ engineering blog hoặc paper, ghi rõ năm. Nếu không chắc → ghi "based on public information, internal numbers may differ".
- **Diagram**: ASCII art (preferred) hoặc Mermaid. Diagram phải show data flow, không chỉ static boxes.
- **Pseudo-code**: Python-flavored, có comment tiếng Việt giải thích intuition.
- **Comparison tables**: System | Approach | Pros | Cons | Use case.
- **References**: prioritize engineering blogs > arXiv papers > conference talks > textbook. Ghi rõ URL và năm publication.

## Project structure

```text
ai-sysdesign-knowledge/
├── README.md                       # Project overview (start here)
├── INDEX.md                        # This file — roadmap + progress
├── case-studies/
│   ├── 01-foundations/             # Scope 1
│   ├── 02-model-development/       # Scope 2
│   ├── 03-modern-stack/            # Scope 3
│   └── 04-production/              # Scope 4
├── agents/
│   ├── ai-sysdesign-knowledge-writer.md  # Agent definition (Claude Code)
│   └── README.md                   # How to install/use the agent
└── docs/
    ├── style-guide.md              # Bilingual VI-EN writing rules
    ├── terminology.md              # VI-EN technical term mappings
    └── progress.md                 # Detailed progress snapshot
```

## Glossary nhanh (chi tiết hơn ở [docs/terminology.md](docs/terminology.md))

- **Candidate generation / retrieval**: pha đầu tiên trong reco system, giảm corpus từ triệu/tỷ items xuống vài trăm/nghìn candidates.
- **Ranking**: pha thứ hai, score chính xác các candidates và sắp xếp.
- **Two-tower model**: kiến trúc user tower + item tower, train chung embedding space, dùng cho retrieval bằng ANN.
- **Online inference vs batch inference**: realtime (latency-sensitive) vs precomputed offline.
- **Feature store**: kho lưu features dùng chung cho training và serving, đảm bảo training/serving skew = 0.
- **KV cache**: trong LLM serving, cache key/value tensor của các token đã sinh để không phải recompute attention.
- **Continuous batching**: dynamic batching cho LLM serving, không đợi full batch — flush request ngay khi done.
- **GNN (Graph Neural Network)**: neural net hoạt động trên graph data, mỗi node update representation từ neighbors qua aggregator function.
- **Inductive vs transductive GNN**: inductive (GraphSAGE, PinSage) compute embedding cho unseen nodes; transductive (GCN) chỉ học cho fixed node set.
- **Random walk sampling**: lấy mẫu neighbors qua random walk trên graph, visit count là importance signal.
- **Importance pooling aggregator**: weighted sum của neighbor representations với weights = visit counts.
