# Terminology Consistency — VI-EN Bilingual Mappings

Mục đích: giữ consistency thuật ngữ giữa các case studies. **Luôn giữ tiếng Anh** cho technical terms, dùng tiếng Việt cho diễn giải.

## Recommendation systems

| EN (giữ nguyên) | VI (diễn giải nếu cần) | Notes |
|---|---|---|
| candidate generation / retrieval | pha sinh candidates / pha truy hồi | Stage 1 of two-stage reco |
| ranking | pha xếp hạng / pha ranking | Stage 2 |
| two-tower model | mô hình hai tháp / two-tower | User tower + item tower |
| embedding | embedding (giữ EN) | KHÔNG dịch "vector nhúng" |
| sparse features / dense features | sparse/dense features (giữ EN) | |
| feature store | feature store (giữ EN) | KHÔNG dịch |
| online/offline store | online/offline store | |
| training/serving skew | training/serving skew | KHÔNG dịch — keep idiomatic |
| feedback loop | feedback loop / vòng phản hồi | |
| cold-start | cold-start | |
| position bias | position bias | |
| sampled softmax | sampled softmax | |
| ANN (approximate nearest neighbor) | ANN | |
| HNSW, IVF, IVF-PQ, ScaNN, FAISS | (tên thư viện/algorithm — giữ EN) | |

## Graph Neural Networks (GNN) — added S1-03 PinSage

| EN (giữ nguyên) | VI (diễn giải nếu cần) | Notes |
|---|---|---|
| GNN (Graph Neural Network) | GNN | KHÔNG dịch "mạng nơ-ron đồ thị" — over-translate |
| GCN (Graph Convolutional Network) | GCN | Kipf-Welling 2017 |
| GraphSAGE | GraphSAGE | Hamilton et al. NeurIPS 2017 — keep name |
| PinSage | PinSage | Pinterest's production GNN |
| PinnerSage | PinnerSage | Multi-embedding user version |
| TransAct | TransAct | Pinterest 2023, transformer on PinSage tokens |
| inductive vs transductive | inductive / transductive | KHÔNG dịch |
| random walk | random walk | KHÔNG dịch "bước đi ngẫu nhiên" |
| neighborhood sampling | neighborhood sampling | |
| k-hop sampling / k-hop neighborhood | k-hop sampling | |
| aggregator function | aggregator function | KHÔNG dịch |
| importance pooling | importance pooling | Pinterest's aggregator innovation |
| mean / max / LSTM aggregator | (giữ EN — aggregator names) | |
| visit count | visit count | Random walk frequency |
| bipartite graph | bipartite graph / đồ thị hai phía | "bipartite" giữ EN |
| edge weight / edge importance | edge weight | |
| message passing | message passing | KHÔNG dịch |
| receptive field (in GNN) | receptive field | |
| over-smoothing | over-smoothing | GNN deep-stacking problem |
| PageRank / Personalized PageRank (PPR) | PageRank / PPR | |
| Pixie (Pinterest's random walk engine) | Pixie | |
| DeepWalk, node2vec | (giữ EN — algorithm names) | |
| hard negative mining | hard negative mining | KHÔNG dịch |
| curriculum learning | curriculum learning | |
| max-margin loss / triplet loss | max-margin / triplet loss | |
| in-batch negative | in-batch negative | |
| MapReduce inference | MapReduce inference | |
| memoization (in inference) | memoization | KHÔNG dịch |
| producer-consumer pipeline | producer-consumer pipeline | |
| graph snapshot | graph snapshot | |
| temporal GNN / dynamic GNN | temporal GNN / dynamic GNN | |

## CTR / Ads / Ranking

| EN | VI |
|---|---|
| CTR (Click-Through Rate) | CTR |
| CVR (Conversion Rate) | CVR |
| watch time | watch time / thời gian xem |
| dwell time | dwell time |
| MMoE (Multi-gate Mixture of Experts) | MMoE |
| multi-task ranking | multi-task ranking |
| factorization machine (FM) | FM |
| DLRM, DeepFM, Wide & Deep, DCN | (giữ EN) |

## LLM / Modern stack

| EN | VI |
|---|---|
| KV cache | KV cache | KHÔNG dịch "bộ nhớ đệm KV" |
| PagedAttention | PagedAttention |
| continuous batching | continuous batching / batch liên tục | |
| static batching | static batching |
| prefill / decode | prefill / decode | KHÔNG dịch |
| TTFT (time to first token) | TTFT |
| TPOT (time per output token) | TPOT |
| tensor parallelism (TP) | tensor parallelism |
| pipeline parallelism (PP) | pipeline parallelism |
| speculative decoding | speculative decoding |
| prefix caching | prefix caching |
| chunked prefill | chunked prefill |
| quantization (FP16/FP8/INT8/INT4) | quantization |
| AWQ, GPTQ | (giữ EN — tên algorithm) |
| vLLM, TGI, TensorRT-LLM, SGLang | (tên framework — giữ EN) |
| RAG (Retrieval-Augmented Generation) | RAG |
| retriever / generator | retriever / generator |
| chunking strategy | chunking strategy / chiến lược chunking |
| reranker | reranker |
| BM25 / dense retrieval / hybrid | (giữ EN) |
| vector database | vector database / vector DB |
| bi-encoder / cross-encoder | bi-encoder / cross-encoder |
| RRF (Reciprocal Rank Fusion) | RRF |
| HyDE (Hypothetical Document Embedding) | HyDE |
| MMR (Maximal Marginal Relevance) | MMR |
| contextual retrieval | contextual retrieval |
| semantic chunking | semantic chunking |
| recursive chunking | recursive chunking |
| hierarchical chunking / auto-merging | hierarchical chunking / auto-merging |
| Matryoshka embedding | Matryoshka embedding |
| RAGAS (eval framework) | RAGAS |
| faithfulness / context precision / answer relevance | (giữ EN — RAGAS metrics) |
| recall@k / MRR / NDCG@k | (giữ EN — IR metrics) |
| HNSW (Hierarchical Navigable Small World) | HNSW |
| IVF-PQ (Inverted File - Product Quantization) | IVF-PQ |
| prompt caching | prompt caching |
| grounding / citation | grounding / citation |

## Training / model dev

| EN | VI |
|---|---|
| overfit / underfit | overfit / underfit |
| pretraining / finetuning | pretraining / finetuning | KHÔNG dịch "tiền huấn luyện" |
| transfer learning | transfer learning |
| backpropagation | backpropagation |
| gradient clipping | gradient clipping |
| learning rate schedule | learning rate schedule |
| data parallel / model parallel | data parallel / model parallel |
| Horovod, Megatron, DeepSpeed | (tên framework — giữ EN) |
| checkpoint | checkpoint |
| inference vs training | inference vs training |

## Production / MLOps

| EN | VI |
|---|---|
| online inference / batch inference | online inference / batch inference |
| shadow mode / canary deployment | shadow mode / canary deployment |
| A/B test | A/B test |
| drift (data/concept/model) | drift |
| PSI (Population Stability Index) | PSI |
| KS test (Kolmogorov-Smirnov) | KS test |
| model registry | model registry |
| lineage | lineage |
| SLA / SLO | SLA / SLO |
| QPS (queries per second) | QPS |
| P50 / P99 latency | P50 / P99 latency |
| autoscaling | autoscaling |
| GPU pooling / multi-tenant serving | GPU pooling / multi-tenant serving |
| spot instance / reserved instance | spot / reserved instance |
| CUPED (Controlled-experiment Using Pre-Experiment Data) | CUPED |
| CUPAC (CUPED with Any Covariate) | CUPAC |
| variance reduction | variance reduction / giảm phương sai |
| SRM (Sample Ratio Mismatch) | SRM |
| AA test | AA test |
| MDE (Minimum Detectable Effect) | MDE |
| mSPRT (mixture Sequential Probability Ratio Test) | mSPRT |
| sequential testing / always-valid inference | sequential testing |
| interleaving / Team Draft Interleaving (TDI) | interleaving |
| MAB (Multi-Armed Bandit) | MAB |
| Thompson sampling | Thompson sampling |
| HTE (Heterogeneous Treatment Effect) | HTE |
| ATE (Average Treatment Effect) | ATE |
| SUTVA (Stable Unit Treatment Value Assumption) | SUTVA |
| guardrail metrics | guardrail metrics |
| holdback / holdout group | holdback / holdout group |
| novelty effect / primacy effect | novelty effect / primacy effect |
| FDR (False Discovery Rate) / Bonferroni | FDR / Bonferroni |
| Welch's t-test | Welch's t-test |
| confidence interval (CI) | CI |

## Phong cách câu mẫu

Đúng (bilingual tự nhiên):
- "YouTube sử dụng kiến trúc two-tower model để xử lý candidate generation, trong đó user tower và item tower được train riêng biệt nhưng share chung embedding space."
- "Khi QPS tăng đột biến, hệ thống trigger autoscaling dựa trên P99 latency thay vì CPU utilization."
- "vLLM giải quyết memory fragmentation của KV cache bằng kỹ thuật paging, giảm waste từ 60-80% xuống dưới 5%."

Sai (over-translation):
- "YouTube sử dụng kiến trúc hai-tháp để xử lý quá trình sinh ứng viên..." → over-translate, mất technical specificity.
- "Bộ nhớ đệm khóa-giá trị" thay vì "KV cache" → không idiomatic.

Sai (không đủ tiếng Việt):
- "vLLM solves KV cache fragmentation through paging, reducing waste from 60-80% to under 5%." → toàn EN, mất bilingual flavor.

## Quy ước viết số liệu

- Năm phải ghi rõ khi cite: "paper 2019", "based on 2023 blog".
- Đơn vị: ms, s, GB, TB, QPS — giữ chuẩn quốc tế.
- Nếu approximation: "approximately", "based on public information", "internal numbers may differ".
- Range: "60-80%" (dùng dash), "vài chục GB" (tiếng Việt cho approximation mềm).

## Acronyms danh sách cần expand lần đầu (mỗi document)

Một số acronym mà người đọc có thể chưa biết — expand lần đầu xuất hiện:
- PSI = Population Stability Index
- KS = Kolmogorov-Smirnov
- MMR = Maximal Marginal Relevance
- MMoE = Multi-gate Mixture of Experts
- HNSW = Hierarchical Navigable Small World
- AWQ = Activation-aware Weight Quantization
- TGI = Text Generation Inference
- TTFT, TPOT — expand lần đầu trong LLM context
