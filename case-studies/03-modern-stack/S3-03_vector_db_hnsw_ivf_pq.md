---
# === Identity (immutable after creation) ===
id: S3-03
title: "Vector Database Internals — HNSW vs IVF-PQ"
slug: vector_db_hnsw_ivf_pq

# === Classification ===
scope: 3
scope_name: modern-stack
difficulty: intermediate
summary: "Index structures, recall/QPS trade-off, Pinecone/Weaviate/Milvus/pgvector comparison."
tags:
  - vector database
  - ANN
  - HNSW
  - IVF-PQ
  - product quantization
  - retrieval
  - Pinecone
  - Weaviate
  - Milvus
  - pgvector
  - DiskANN
  - ScaNN

# === Lifecycle ===
status: done
created: 2026-05-20
last_validated: 2026-05-20
line_count: 1533

# === Knowledge graph ===
cross_refs: [S1-03, S3-01, S3-02]
primary_sources:
  - type: paper
    title: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
    url: "https://arxiv.org/abs/1603.09320"
    year: 2018
    org: "Malkov & Yashunin (TPAMI)"
  - type: paper
    title: "Product Quantization for Nearest Neighbor Search"
    url: "https://hal.inria.fr/inria-00514462v2/document"
    year: 2011
    org: "Jégou, Douze, Schmid (TPAMI / INRIA)"
  - type: paper
    title: "Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)"
    url: "https://arxiv.org/abs/1908.10396"
    year: 2020
    org: "Guo et al. (Google Research, ICML)"
  - type: paper
    title: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
    url: "https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html"
    year: 2019
    org: "Subramanya et al. (Microsoft Research, NeurIPS)"
  - type: paper
    title: "Optimized Product Quantization"
    url: "https://kaiminghe.github.io/publications/pami13opq.pdf"
    year: 2013
    org: "Ge, He, Ke, Sun (Microsoft Research Asia / TPAMI)"
---

# S3-03 — Vector Database Internals — HNSW vs IVF-PQ

> **Scope**: Modern Tech Stack (LLM / RAG / Retrieval)
> **Difficulty**: Intermediate
> **Tags**: vector database, ANN, HNSW, IVF-PQ, product quantization, retrieval, Pinecone, Weaviate, Milvus, pgvector
> **Tham chiếu chéo**: [S1-03 PinSage](../01-foundations/S1-03_pinterest_pinsage_graph_retrieval.md) (ANN serving billion-scale), [S3-01 vLLM](../03-modern-stack/S3-01_vllm_paged_attention_continuous_batching.md) (KV cache memory model — paging tương đồng), [S3-02 Production RAG](../03-modern-stack/S3-02_production_rag_system_architecture.md) (retrieval pipeline tiêu thụ vector DB)
> **Primary sources**:
> - Malkov & Yashunin "HNSW" (TPAMI 2018, arXiv:1603.09320).
> - Jégou et al. "Product Quantization for NN Search" (TPAMI 2011, INRIA).
> - Guo et al. "ScaNN — Anisotropic VQ" (ICML 2020, arXiv:1908.10396).
> - Subramanya et al. "DiskANN" (NeurIPS 2019).
> - Ge et al. "Optimized Product Quantization" (TPAMI 2013).
> - ann-benchmarks.com (Aumüller, Bernhardsson, Faithfull) — empirical recall/QPS curves.
> **Key insight (1-2 câu)**: HNSW và IVF-PQ chiếm hai góc đối diện của tam giác recall/QPS/memory — HNSW tối ưu cho **in-memory high-recall query** với chi phí RAM ~1.5-3x raw vectors, IVF-PQ tối ưu cho **billion-scale với memory ngân sách hạn chế** bằng cách hi sinh recall và độ chính xác của distance estimation. Mọi production vector DB hiện đại (Pinecone, Weaviate, Milvus, pgvector) đều build lên ít nhất một trong hai (và ngày càng nhiều dùng cả hai, kèm DiskANN cho disk-resident workloads).

---

## 1. Tổng quan (Overview)

Năm 2020, RAG, semantic search, và embedding-based recommendation gần như đồng thời "phổ cập" trong industry: OpenAI ra `text-embedding-ada-002` (Dec 2022), Cohere và HuggingFace push hàng loạt sentence-transformer model, các production reco system (Pinterest PinSage, YouTube two-tower) đã chứng minh embedding retrieval scale tới billion items. Hệ quả: mọi engineering team đột nhiên cần một **vector database** để store và query embeddings — và bài toán "k-nearest neighbor trên N triệu vector d-chiều" trở thành infrastructure primitive.

Vấn đề là **exact k-NN không feasible ở scale**. Brute-force scan có complexity $O(N \cdot d)$ cho mỗi query — với $N = 100M$ vector, $d = 768$, một query đơn lẻ phải dot-product 76 tỷ float, trên CPU mất nhiều giây. Vì vậy industry chuyển sang **ANN (Approximate Nearest Neighbor)**: chấp nhận missing một vài neighbor để đổi lấy latency thấp gấp 100-10000x. Trade-off cốt lõi là tam giác **recall ↔ QPS ↔ memory**: bạn có thể tốt hai trong ba, nhưng không bao giờ cả ba.

Case study này deep-dive **hai họ index thống trị production**:

1. **HNSW (Hierarchical Navigable Small World)** — Malkov & Yashunin 2016/2018. Graph-based, in-memory, log(N) expected query, recall@10 thường > 95% với hyperparameters mặc định, hi sinh memory (1.5-3x raw vectors).
2. **IVF-PQ (Inverted File + Product Quantization)** — Jégou et al. 2011 + Sivic & Zisserman IVF từ 2003. Quantization-based, có thể disk-resident, memory thấp gấp 10-50x so với HNSW, recall thấp hơn 5-15% ở cùng QPS budget.

Bên cạnh đó: **ScaNN** (Google 2020, anisotropic VQ — variant của PQ với loss function aware về search semantics), **DiskANN** (Microsoft 2019, graph-based nhưng SSD-resident — billion-scale trên 1 node), và **OPQ** (Optimized PQ — rotation trước khi PQ để cải thiện quantization error).

Sau khi nắm vững internals, ta so sánh **bốn production vector DB**: Pinecone (managed, proprietary, serverless), Weaviate (open-source Go, HNSW-based), Milvus (open-source distributed, đa-index), pgvector (Postgres extension, đơn giản, ACID).

### Tại sao đây là case study quan trọng?

- **Bottleneck phổ biến nhất của RAG/semantic search là vector DB**: chọn sai index → recall thấp → LLM hallucinate; chọn over-engineered → memory cost gấp 10x cần thiết.
- **Mỗi production team sẽ phải tự build hoặc tune**: managed (Pinecone/Vertex Vector Search) hết tiền nhanh khi vượt 10-100M vector; self-host (Milvus/Weaviate/pgvector) cần hiểu nội bộ index để tune `M`, `efSearch`, `nprobe`, `nlist`.
- **Cross-cutting với S1-03 (PinSage)**: Pinterest serve 3B PinSage embedding qua mix HNSW + IVF-PQ — case này là nền tảng để hiểu serving stack đó.
- **Cross-cutting với S3-01 (vLLM)**: vLLM PagedAttention dùng concept "block table" — về mặt memory layout giống cách IVF chia vector space thành inverted lists; cả hai đều giải bài toán "không thể fit contiguous, phải paged/listed".
- **Cross-cutting với S3-02 (Production RAG)**: pipeline RAG end-to-end gọi vector DB ở "Dense retrieval" step — chọn index ở đây quyết định latency budget, cost, và recall của toàn bộ system.

### Khi nào dùng vector DB chuyên dụng vs các giải pháp đơn giản hơn?

| Use case | Lựa chọn khuyến nghị | Lý do |
|---|---|---|
| Demo, < 100K vectors | NumPy / scikit-learn `NearestNeighbors` | Brute-force vẫn < 100ms, không cần index |
| Prototype 100K-1M, Postgres đã có sẵn | **pgvector** (HNSW hoặc IVFFlat) | ACID, joins với metadata, ops đơn giản |
| Production 1M-100M, cần managed | **Pinecone** hoặc **Weaviate Cloud** | Không phải tự ops, scale auto |
| Production 100M-10B, self-host | **Milvus** + DiskANN | Distributed, multi-index, mature ops |
| In-process, embed-trong-app | **FAISS** library trực tiếp | Không cần daemon, full control |
| Hybrid (BM25 + vector) là first-class | **Weaviate** hoặc **Elasticsearch 8 dense vectors** | Built-in hybrid |
| Latency P99 < 10ms cứng | HNSW in-memory với replicas | IVF-PQ thường không đạt được |

---

## 2. System Requirements

### 2.1 Functional requirements

Một production vector DB phải hỗ trợ tối thiểu:

- **Insert / upsert / delete vectors** với metadata kèm theo (timestamp, user_id, tags, source_doc_id…).
- **k-NN query** với một query vector $q$ và số nguyên $k$ → trả về top-$k$ vectors gần nhất theo distance metric (cosine, dot, L2).
- **Filter pushdown**: query "tìm 10 vector gần nhất với $q$ **trong những vector có** `tenant_id = "acme"`". Nếu không hỗ trợ filter ở index level → phải over-fetch và filter post-hoc → recall giảm dramatic.
- **Hybrid search**: BM25 / sparse vector + dense vector cùng query, fusion qua RRF (Reciprocal Rank Fusion) hoặc weighted score. Xem [S3-02 Production RAG](../03-modern-stack/S3-02_production_rag_system_architecture.md) cho chi tiết RRF.
- **Multi-tenancy / namespaces**: isolation cho per-user, per-team data.
- **Incremental indexing**: insert real-time mà không phải rebuild full index.
- **Index persistence**: write-ahead log + snapshot để recover sau crash.
- **Replication / sharding** ở production scale.

### 2.2 Non-functional requirements (target cho mid-large vector DB)

Bảng dưới là **target ballpark** dựa trên public benchmarks và case study của Pinterest/Spotify/Shopify. Không bịa số — mọi range đều có nguồn cite trong References.

| Metric | Target | Notes / source |
|---|---|---|
| Corpus size | 10M - 10B vectors | Pinterest serve 3B vectors qua mix HNSW + IVF-PQ (S1-03) |
| Vector dimension | 128 - 4096 | OpenAI ada-002 = 1536; cohere-embed-v3 = 1024; matryoshka 256-3072 |
| Insert throughput | 1K - 100K vectors/sec | Phụ thuộc index type (HNSW chậm hơn IVF, do graph rewiring) |
| Query latency P50 | < 20 ms | ann-benchmarks SIFT1M HNSW @ recall 0.95 = ~5 ms (2024) |
| Query latency P99 | < 100 ms | Cao hơn nhiều cho disk-resident (DiskANN ~10-50ms P99 cho 1B) |
| QPS per node | 1K - 50K | Phụ thuộc dataset size; HNSW SIFT1M ~30K QPS, deep1B ~2K QPS |
| Recall@10 | 0.90 - 0.99 | Production sweet spot 0.95; > 0.99 trả giá QPS rất nhanh |
| Memory per vector | 4-16 bytes (PQ) đến 4d bytes (raw FP32) | PQ M=16/k=256 → 16 bytes; raw 768-dim FP32 → 3072 bytes |
| Index build time | minutes - hours | HNSW 1M = ~1-10 min; IVF-PQ 100M ≈ 1-3 hours (1 GPU cho k-means) |
| Index size on disk | 1.5-5x raw data (HNSW) hoặc 1/10-1/100 (PQ) | HNSW: graph edges; PQ: codes nhỏ hơn raw nhiều |
| Availability | 99.9% - 99.99% | Pinecone SLA pod-based; replication tối thiểu 2x |

### 2.3 Constraints quan trọng

- **Memory là tài nguyên đắt nhất**. 1B vectors × 768 dim × 4 bytes = ~3 TB chỉ cho raw data — không fit memory 1 node. Phải hoặc (a) quantize (PQ → ~16 bytes/vector → 16 GB), (b) shard ra nhiều node (mỗi node giữ một phần), (c) disk-resident (DiskANN), hoặc (d) tier hot/cold (Pinterest pattern: HNSW cho top pins, IVF-PQ cho long-tail).
- **Insert-heavy vs query-heavy** ảnh hưởng index choice: IVF-PQ build qua k-means + PQ training là **offline batch**, không "incremental tự nhiên" (insert mới gắn vào list gần nhất với centroid, nhưng centroid không update → drift). HNSW incremental tốt hơn nhưng delete khó.
- **Filter selectivity**: nếu filter rất chọn lọc (e.g., 1% data match `tenant_id`), HNSW pre-filter có thể chỉ còn vài node trong graph, search sụp đổ — phải fallback brute-force. Hệ quả: nhiều vector DB shard per tenant để filter trở thành "pick shard".
- **Distance metric ảnh hưởng index validity**. HNSW theoretical guarantee giả định **metric space** (triangle inequality). Cosine không strict metric, nhưng inner-product normalized vectors thì OK. PQ với asymmetric distance computation phải re-derive code book theo metric.
- **Update / delete**. HNSW xóa thật khó (phải patch graph hoặc tombstone — tombstone-heavy graph degrade query). IVF-PQ: xóa nhẹ hơn (xóa code khỏi list) nhưng các cluster centroid không reflect distribution mới.

---

## 3. High-level Architecture

### 3.1 Anatomy của một vector DB

```text
                          VECTOR DATABASE — LOGICAL ARCHITECTURE

   ┌─────────────────────────────── CONTROL PLANE ────────────────────────────────┐
   │                                                                              │
   │   ┌────────────┐   ┌─────────────┐   ┌───────────────┐   ┌────────────────┐  │
   │   │ Metadata   │   │ Collection  │   │ Tenant /      │   │  Index config  │  │
   │   │ store      │   │ schema reg. │   │ namespace mgr │   │  registry      │  │
   │   │ (etcd)     │   │             │   │               │   │  (M, efC, ...) │  │
   │   └────────────┘   └─────────────┘   └───────────────┘   └────────────────┘  │
   └──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────── INGESTION / WRITE PATH ──────────────────────────┐
   │                                                                             │
   │   API gateway ──▶ Validation ──▶ Vector preproc ──▶ Sharding router         │
   │   (gRPC/REST)     (dim, schema)   (normalize,         (hash / consistent     │
   │                                    project)            hashing)              │
   │                                                            │                 │
   │                                                            ▼                 │
   │                                                ┌─────────────────────┐       │
   │                                                │   Write-Ahead Log   │       │
   │                                                │   (WAL, append)     │       │
   │                                                └──────────┬──────────┘       │
   │                                                           ▼                  │
   │                                                ┌─────────────────────┐       │
   │                                                │  In-memory growing  │       │
   │                                                │  segment (HNSW or   │       │
   │                                                │  buffer to be IVF'd)│       │
   │                                                └──────────┬──────────┘       │
   │                                                           ▼                  │
   │                                                ┌─────────────────────┐       │
   │                                                │ Sealed segments     │       │
   │                                                │ (immutable index    │       │
   │                                                │  on disk)           │       │
   │                                                └─────────────────────┘       │
   └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────── QUERY / READ PATH ───────────────────────────────┐
   │                                                                             │
   │   Query vector q ──▶ Coordinator ──▶ Fan-out to shards                      │
   │                          │                  │                               │
   │                          │                  ▼                               │
   │                          │            ┌──────────────┐                      │
   │                          │            │ Per-shard:   │                      │
   │                          │            │ - Query all  │                      │
   │                          │            │   segments   │                      │
   │                          │            │   (HNSW + IVF│                      │
   │                          │            │    + buffer) │                      │
   │                          │            │ - Local top-k│                      │
   │                          │            └──────┬───────┘                      │
   │                          ▼                   ▼                              │
   │                  ┌─────────────────────────────────┐                        │
   │                  │ Merge top-k from shards         │                        │
   │                  │ Apply post-filter if needed     │                        │
   │                  │ Hydrate metadata                │                        │
   │                  └──────────────────┬──────────────┘                        │
   │                                     ▼                                       │
   │                              Response top-k                                 │
   └─────────────────────────────────────────────────────────────────────────────┘
```

Mô hình logical này áp dụng cho mọi vector DB hiện đại (Pinecone, Milvus, Weaviate đều có equivalent components, chỉ khác về implementation và terminology). Hai concept then chốt:

1. **Segment model**: data **không** ngồi trong một index khổng lồ liên tục. Thay vào đó nó split thành nhiều **segments** (Milvus terminology) hoặc **shards/pods** (Pinecone), mỗi segment là một index immutable. New writes vào growing segment (hoặc WAL), được seal định kỳ thành immutable segment. Query phải fan-out qua tất cả segments rồi merge — cùng logic như Lucene của Elasticsearch.
2. **Compaction**: segments nhỏ định kỳ merge thành lớn để giữ query cost manageable. Lúc compaction là cơ hội để rebuild index (e.g., re-train k-means cho IVF-PQ).

### 3.2 Tại sao vector DB cần "hơn" Faiss?

Faiss là **library** (C++ + Python bindings) cho ANN. Nó implement HNSW, IVF, PQ, IVF-PQ, OPQ, etc. Nhưng Faiss raw thiếu các property production:

| Feature | Faiss (library) | Vector DB (Pinecone/Milvus/...) |
|---|---|---|
| Persistence / WAL | Manual save/load | Built-in, crash-safe |
| Filter on metadata | Phải manage ngoài | First-class, integrated |
| Multi-tenancy | Manual | Namespace / tenant API |
| Sharding | Manual | Auto |
| Replication | Manual | Built-in HA |
| Incremental insert | Yes nhưng limited (IVF không re-train) | Quản lý qua segment lifecycle |
| Query language | Python API | SQL/REST/gRPC |
| Ops (backup/restore/monitor) | Tự xây | Built-in dashboard |

Nói cách khác, vector DB = **Faiss + production wrapping** (WAL, segments, sharding, filter, metadata, multi-tenancy, ops). Milvus là ví dụ explicit: core indices của nó include **HNSW từ hnswlib** và **IVF/PQ từ Faiss**, wrap trong distributed system layer.

---

## 4. Deep Dive — HNSW

### 4.1 Intuition: graph + skip-list

HNSW (Hierarchical Navigable Small World) build trên hai idea đã có từ trước:

1. **Navigable Small World (NSW)** — Malkov 2014. Idea: build undirected graph trên dataset sao cho **shortest-path** giữa hai node bất kỳ là $O(\log N)$ trên expectation (small-world property). Search bằng greedy: từ entry point, jump tới neighbor gần query nhất, lặp lại.
2. **Skip-list** (Pugh 1990) — randomized data structure cho ordered set: nhiều "level", level cao chỉ giữ vài element, level thấp giữ tất cả. Search bắt đầu từ level cao (jump xa), descend dần.

HNSW combine hai idea: graph nhiều **layers**, layer cao stochastic chỉ chứa subset element (mỗi node được gán random max-layer theo exponential decay), layer thấp dense. Search bắt đầu top layer, greedy descend cho tới layer 0.

```text
                       HNSW HIERARCHICAL LAYERS

   Layer 3   (~ log(N) high)   ●─────────────────────────●
                                \                       /
                                 \                     /
   Layer 2   (sparse)             ●────●───────●─────●
                                  /     \      |     \
                                 /       \     |      \
   Layer 1   (denser)           ●─●──●───●──●──●──●───●──●
                                /  \  \  |  \  |  \   /
                               /    \  \ |   \ |   \ /
   Layer 0   (all N nodes)    ●─●─●─●─●─●─●─●─●─●─●─●─●─●
                                ↑
                            Entry point (top-down search)

   Mỗi node được gán random max-layer L ~ floor(-ln(U) * m_L),
   với U uniform(0,1), m_L = 1/ln(M). M là max neighbors mỗi node mỗi layer.
   Hệ quả: số node giảm theo cấp số nhân khi lên cao → log(N) layers expected.
```

Lý do thiết kế này hoạt động:

- **Greedy descent**: ở layer cao node thưa nên jump xa (bypass nhiều khoảng cách), tới layer thấp dày để refine. Tương tự skip-list: high level skip nhiều, low level refine.
- **Small-world property**: nội bộ mỗi layer, "long-range" edges + "short-range" edges làm shortest path $O(\log N)$.
- **Incremental insert**: thêm node mới = random L, từ top descend tới L+1 lấy entry point, từ L xuống 0 tìm M neighbor gần nhất tại mỗi layer và connect. Không cần rebuild.

### 4.2 Hyperparameters

| Hyperparameter | Ý nghĩa | Mặc định | Tác động |
|---|---|---|---|
| `M` | Max # neighbors per node per layer (trừ layer 0, dùng `M_max0` = 2*M) | 16 | Cao hơn → recall cao hơn, memory cao hơn, build chậm hơn |
| `efConstruction` | Size of dynamic candidate list khi insert | 200 | Cao hơn → build chậm, index chất lượng cao hơn |
| `efSearch` (= `ef`) | Size of dynamic candidate list khi query | 50-500 | Cao hơn → recall cao hơn, query chậm hơn (tune online) |
| `m_L` | Level-generation factor, mặc định `1/ln(M)` | ~0.36 (cho M=16) | Hiếm khi tune |

**Rule of thumb từ ann-benchmarks 2024**:
- M = 16, efConstruction = 200 → recall@10 ~0.95 cho SIFT1M, build ~1 min trên 1 core.
- M = 48, efConstruction = 500 → recall@10 ~0.99, build ~5 min, memory ~2x.

**Memory cost xấp xỉ**:
$$
\text{mem} \approx N \cdot d \cdot 4 \text{ bytes} + N \cdot M \cdot 4 \text{ bytes} \cdot L_{\text{avg}}
$$
với $L_{\text{avg}} \approx 1.5$ (vì layer 0 dày, layer cao thưa). Cho M=16, d=768, N=10M: ~30 GB raw + ~1 GB graph = ~31 GB. So với raw FP32 alone (~30 GB), HNSW memory overhead ~3-5% là **graph**, phần lớn vẫn là raw vectors. Đó là lý do HNSW "đắt RAM" — không phải vì graph, mà vì phải giữ raw vectors trong RAM để compute exact distance khi traverse.

### 4.3 Search algorithm step-by-step

Pseudo-code dưới là biến thể từ Algorithm 5 của paper Malkov & Yashunin 2018:

```python
def hnsw_search(graph, q, ef=50, k=10):
    """
    Tìm top-k nearest neighbors của query q.
    graph: HNSW graph, mỗi node có graph.neighbors(node, layer)
    ef: dynamic candidate list size khi query (efSearch)
    k: số neighbor cần trả về
    """
    # Bắt đầu từ entry point ở layer cao nhất
    entry = graph.entry_point
    L = graph.max_layer

    # Phase 1: từ top layer xuống layer 1, greedy đi 1 entry point gần nhất
    current = entry
    for layer in range(L, 0, -1):
        # Greedy: ở layer này, tìm node gần q nhất bằng BFS local
        current = greedy_descend(graph, q, entry_point=current, layer=layer, ef=1)

    # Phase 2: ở layer 0, dùng dynamic list size ef
    candidates = greedy_descend(graph, q, entry_point=current, layer=0, ef=ef)
    # Trả về top-k
    return sorted(candidates, key=lambda n: distance(n, q))[:k]


def greedy_descend(graph, q, entry_point, layer, ef):
    """
    Best-first search trong layer cụ thể.
    Trả về top-ef candidates gần q nhất.
    """
    # visited để tránh revisit
    visited = {entry_point}
    # candidates: priority queue ordered tăng dần theo distance (min-heap)
    candidates = [(distance(entry_point, q), entry_point)]
    # results: top-ef best so far (max-heap, để pop thằng tệ nhất ra)
    results = [(-distance(entry_point, q), entry_point)]

    while candidates:
        # Lấy candidate gần q nhất chưa explore
        dist_c, c = heappop(candidates)
        # Nếu thằng candidate này còn xa hơn thằng tệ nhất trong results, dừng
        worst_in_results = -results[0][0]
        if dist_c > worst_in_results:
            break
        # Explore neighbors của c
        for n in graph.neighbors(c, layer):
            if n in visited:
                continue
            visited.add(n)
            dist_n = distance(n, q)
            worst_in_results = -results[0][0]
            if dist_n < worst_in_results or len(results) < ef:
                heappush(candidates, (dist_n, n))
                heappush(results, (-dist_n, n))
                if len(results) > ef:
                    heappop(results)  # remove thằng tệ nhất
    # Trả results sorted theo distance tăng dần
    return sorted([(- -d, n) for d, n in results], key=lambda x: x[0])
```

Một số quan sát quan trọng:

- **Distance computation chiếm > 90% thời gian query**. Số distance ops cho 1 query xấp xỉ $\text{ef} \cdot M \cdot L_{\text{visited}}$ — với ef=50, M=16, ~5 layer visited → ~4000 distance ops, mỗi op là $d$ float multiplications. Cho d=768, ~3M FLOPs → trên CPU SIMD-optimized ~1 ms. Đây là lý do **dimension matter hơn N** ở query time.
- **`ef` ≥ `k`**, thường ef = 5-10x k. Tăng ef → recall tăng nhưng query latency tăng tuyến tính.
- **Greedy không guarantee optimal**: vì local minima trong graph, có thể miss true nearest neighbor → đó là lý do nó "approximate". Recall tăng dần theo ef tăng.

### 4.4 Insert algorithm

```python
def hnsw_insert(graph, new_vec, M=16, efC=200, m_L=1/math.log(2*M)):
    """
    Insert một vector mới vào HNSW graph.
    """
    # 1. Gán random max layer cho node mới
    new_node = graph.add_node(new_vec)
    L_new = int(-math.log(random.random()) * m_L)

    # 2. Từ layer hiện tại của graph xuống L_new+1: chỉ greedy với ef=1
    entry = graph.entry_point
    L_cur = graph.max_layer
    for layer in range(L_cur, L_new, -1):
        entry = greedy_descend(graph, new_vec, entry, layer, ef=1)

    # 3. Từ L_new xuống 0: tìm efC neighbors gần nhất, connect M tốt nhất
    for layer in range(min(L_new, L_cur), -1, -1):
        candidates = greedy_descend(graph, new_vec, entry, layer, ef=efC)
        # Heuristic select neighbors (Algorithm 4 trong paper) —
        # không chỉ chọn M gần nhất mà còn diversity để tránh "hub" effect
        M_layer = M if layer > 0 else 2*M  # layer 0 dày hơn
        neighbors = select_neighbors_heuristic(candidates, M_layer)
        for n in neighbors:
            graph.add_edge(new_node, n, layer)
            # Bidirectional — và prune nếu n đã có quá nhiều neighbor
            if len(graph.neighbors(n, layer)) > M_layer:
                graph.prune_neighbors(n, layer, M_layer)
        entry = neighbors[0]  # entry cho layer dưới

    # 4. Cập nhật entry_point nếu node mới ở layer cao hơn
    if L_new > L_cur:
        graph.entry_point = new_node
        graph.max_layer = L_new
```

**`select_neighbors_heuristic`** là cải tiến quan trọng so với "chọn M gần nhất". Heuristic Malkov đề xuất: với mỗi candidate $c$, chỉ add vào tập neighbors nếu $c$ gần new_node hơn so với gần bất kỳ neighbor đã chọn nào. Hệ quả: tránh hiện tượng "hub" (một số node thành hub bị link bởi quá nhiều node khác), giữ graph diversity và recall ổn định.

**Insert complexity**: mỗi insert ~$O(M \cdot \text{efC} \cdot \log N)$ distance ops. Cho N=1M, M=16, efC=200 → ~3M ops per insert → ~1-3 ms / vector trên CPU. Build full index 10M vectors ≈ 5-30 phút trên 1 core, scale với multi-threading khá tốt vì insert độc lập (modulo lock contention trên neighbor lists).

### 4.5 Delete — vấn đề khó của HNSW

HNSW **không có** "true delete" trong paper gốc. Hai approach trong production:

1. **Tombstone**: mark node "deleted", skip nó khi query. Vấn đề: graph topology vẫn dùng node deleted làm path → graph quality giảm dần (gọi là "tombstone-heavy degradation"). Sau khi delete ~20% data, recall giảm rõ rệt. Workaround: rebuild periodic.
2. **Repair**: thật sự xóa node + re-connect neighbors của nó. Phức tạp, expensive. hnswlib gần đây hỗ trợ `mark_deleted` + `unmark_deleted`, không repair graph.

Production pattern: **soft delete + scheduled compaction** — periodic rebuild segments có > X% tombstones.

### 4.6 R&D evolution dẫn tới HNSW

HNSW không xuất hiện từ không có gì. Lineage:

- **k-d tree** (Bentley 1975) — recursive split không gian $d$-chiều, query $O(\log N)$ trong low dimension. Degrade về $O(N)$ khi $d > 20$ vì curse of dimensionality. Không dùng được cho 768-dim.
- **LSH (Locality-Sensitive Hashing)** (Indyk & Motwani 1998, Datar et al. 2004) — hash function ánh xạ gần-thì-gần. Thực tế: recall thấp, cần nhiều hash table → memory inefficient cho high recall. Bị HNSW vượt qua trên hầu hết benchmark.
- **NSG (Navigating Spreading-out Graph)** (Fu et al. VLDB 2017) — graph-based, tốt nhưng không hierarchical → build chậm, query worse than HNSW.
- **NSW (Navigable Small World)** (Malkov 2014) — flat graph version, idea ban đầu nhưng layer-flat → search recall thấp khi N lớn.
- **HNSW** (Malkov 2016 arXiv, 2018 TPAMI) — thêm hierarchy → bùng nổ adoption. hnswlib (C++ + Python) trở thành reference impl.

### 4.7 Variants & extensions

- **HNSW-PQ** (FAISS): áp dụng PQ lên vectors **stored** trong HNSW node để giảm RAM. Distance dùng PQ asymmetric distance (xem §5). Sacrifice recall ~1-3% để giảm RAM 4-16x.
- **HNSW with int8 quantization** (Weaviate "PQ + HNSW" hybrid 2023): mỗi vector lưu int8 (8x nhỏ hơn FP32) + scalar quantization. Recall drop ~0.5-1%.
- **FreshDiskANN** (Singh et al. 2021) — disk-resident HNSW-like graph, support incremental insert/delete on disk. Microsoft sản phẩm hóa thành DiskANN service trong Azure Cognitive Search.

---

## 5. Deep Dive — IVF + PQ

### 5.1 Big picture

IVF-PQ là kết hợp hai kỹ thuật độc lập nhưng synergic:

1. **IVF (Inverted File)** — phân hoạch không gian vector thành **`nlist` clusters** (gọi là Voronoi cells) qua k-means. Query: tìm `nprobe` cluster gần query nhất, chỉ scan vectors trong các cluster đó. Giảm search space từ $N$ xuống $\sim N \cdot \text{nprobe} / \text{nlist}$.
2. **PQ (Product Quantization)** — compress mỗi vector thành **mã ngắn** (e.g., 16 bytes) bằng cách chia vector thành $M$ subvector, mỗi subvector quantize riêng theo k-means với $k$ centroid (thường $k=256$ → 8 bits/subvector). Distance computation chuyển thành lookup table → cực nhanh.

Kết hợp: IVF làm **coarse filter** (giảm số ứng viên), PQ làm **fine scoring** (estimated distance qua compressed codes).

```text
                       IVF + PQ ANATOMY

   1. Training phase:
      ─────────────────
      ┌─────────────────────────────────────────┐
      │ Step 1: k-means trên tập sample N'      │
      │   → nlist coarse centroids C_1...C_nlist│
      │   (mỗi C_i là vector d-chiều)           │
      └─────────────────────────────────────────┘
                  │
                  ▼
      ┌─────────────────────────────────────────┐
      │ Step 2: với mỗi vector v trong dataset, │
      │   tính residual r = v - C_assigned(v)   │
      └─────────────────────────────────────────┘
                  │
                  ▼
      ┌─────────────────────────────────────────┐
      │ Step 3: PQ train trên residuals r       │
      │   - Chia r thành M subvectors r_1..r_M  │
      │     (mỗi cái d/M chiều)                 │
      │   - Cho mỗi subspace m, k-means k=256   │
      │     → codebook B_m = {b_m_1..b_m_256}   │
      └─────────────────────────────────────────┘

   2. Index phase (per vector v):
      ────────────────────────────
      assign v → cluster c = argmin ||v - C_i||
      compute residual r = v - C_c
      encode r → code = (q_1, ..., q_M)
                 where q_m = argmin_j ||r_m - b_m_j||
      Lưu vào inverted list L_c: (vec_id, code)
      Tổng bytes per vector: M * log2(k) bits = M bytes (cho k=256)

   3. Query phase (query q, return top-K):
      ──────────────────────────────────────
      ┌────────────────────────────────────────────┐
      │ Step 1: tính distance q tới mỗi C_i        │
      │   → sort, lấy nprobe cluster gần nhất     │
      └────────────────────────────────────────────┘
                  │
                  ▼
      ┌────────────────────────────────────────────┐
      │ Step 2: với mỗi cluster c trong nprobe:    │
      │   compute residual q_r = q - C_c           │
      │   Cho mỗi subspace m, precompute:          │
      │     LUT_m[j] = ||q_r_m - b_m_j||^2  ∀j∈[k] │
      │   (LUT = lookup table, M × k floats)       │
      └────────────────────────────────────────────┘
                  │
                  ▼
      ┌────────────────────────────────────────────┐
      │ Step 3: với mỗi (vec_id, code) trong L_c:  │
      │   est_dist = Σ_m LUT_m[code_m]              │
      │   (M lookups + M adds → cực nhanh)          │
      │   Maintain top-K bằng min-heap              │
      └────────────────────────────────────────────┘
                  │
                  ▼
                  Top-K (sorted)
```

### 5.2 IVF: tại sao hoạt động

Trực giác: trong embedding space sau k-means, vector phân thành các "vùng nhỏ" (Voronoi cells). Với query $q$, xác suất nearest neighbor nằm trong vài cell gần $q$ là rất cao. Bằng cách scan chỉ `nprobe`/`nlist` của data, ta giảm cost ~`nlist`/`nprobe` lần.

**Lựa chọn `nlist`**:
- Lý thuyết: $\text{nlist} \approx \sqrt{N}$ để balance trainable size và per-cluster size. Faiss tutorial: 4 × √N - 16 × √N.
- Cho N=1M, nlist ~1000-4000 hợp lý. Cho N=1B, nlist ~30K-100K.
- `nlist` quá nhỏ → mỗi cluster lớn → scan vẫn nhiều. `nlist` quá lớn → k-means không converge tốt + bộ nhớ centroids tăng.

**Lựa chọn `nprobe`**:
- Tune online theo recall target. Cao hơn → recall cao hơn, query chậm hơn.
- Empirical: nprobe = 1 → recall ~50%; nprobe = 8-16 → recall ~85-95%; nprobe = 64-128 → recall ~99%.

**IVF không-PQ** (gọi là `IVFFlat`): scan exact distance trong cluster. Recall cao (chỉ mất do bỏ cluster), query phụ thuộc nprobe và cluster size. Memory: vẫn 4d bytes/vector + centroid + inverted list pointers. pgvector hỗ trợ IVFFlat từ v0.4.

### 5.3 PQ: compress vector thành mã ngắn

**Idea cốt lõi của Product Quantization (Jégou, Douze, Schmid TPAMI 2011)**: thay vì học một codebook khổng lồ $K$ centroid cho $d$-chiều ($K$ phải rất lớn để cover space tốt, e.g., $K=2^{64}$ không feasible), chia $d$-chiều thành $M$ **subspace** độc lập, mỗi subspace học codebook nhỏ $k$ centroid. Tổng số "codes" có thể biểu diễn: $k^M$. Cho M=8, k=256: $256^8 = 2^{64}$ codes → cùng độ biểu diễn như codebook $K=2^{64}$, nhưng chi phí memory chỉ $M \cdot k = 2048$ centroids, mỗi centroid $d/M = d/8$ chiều.

```python
import numpy as np
from sklearn.cluster import KMeans

class ProductQuantizer:
    def __init__(self, d, M=8, k=256):
        """
        d: dimension
        M: số subspace
        k: số centroid mỗi subspace (thường 256 -> 8 bits)
        """
        assert d % M == 0, "d phải chia hết M"
        self.d, self.M, self.k = d, M, k
        self.dsub = d // M
        self.codebooks = []  # M codebook, mỗi cái shape (k, dsub)

    def fit(self, X):
        """Train M codebook trên dataset X shape (N, d)."""
        for m in range(self.M):
            Xm = X[:, m*self.dsub : (m+1)*self.dsub]
            km = KMeans(n_clusters=self.k, n_init=4).fit(Xm)
            self.codebooks.append(km.cluster_centers_)

    def encode(self, X):
        """Encode mỗi vector thành mã M bytes (giả định k=256 nên 1 byte mỗi subspace)."""
        N = X.shape[0]
        codes = np.empty((N, self.M), dtype=np.uint8)
        for m in range(self.M):
            Xm = X[:, m*self.dsub : (m+1)*self.dsub]
            # Tìm centroid gần nhất cho mỗi subvector
            dists = np.linalg.norm(
                Xm[:, None, :] - self.codebooks[m][None, :, :], axis=-1
            )
            codes[:, m] = dists.argmin(axis=1)
        return codes

    def asymmetric_distance(self, q, codes):
        """
        Tính ước lượng distance từ query q (full precision) tới mỗi vector
        đã được encode thành codes.

        Trick: precompute lookup table LUT shape (M, k) — distance từ
        q's subvector tới mỗi centroid. Sau đó với mỗi code, sum M lookups.
        """
        LUT = np.empty((self.M, self.k), dtype=np.float32)
        for m in range(self.M):
            qm = q[m*self.dsub : (m+1)*self.dsub]
            LUT[m] = np.linalg.norm(qm - self.codebooks[m], axis=1) ** 2

        N = codes.shape[0]
        # Sum LUT[m, codes[n, m]] cho mỗi n, m -> distance estimate
        # Vectorized:
        est = LUT[np.arange(self.M), codes].sum(axis=1)
        return est  # shape (N,) — square distances ước lượng
```

Quan sát quan trọng:

- **Encode** một vector: $M \cdot k$ distance ops trên subvector → tổng $\sim M \cdot k \cdot (d/M) = k \cdot d$ ops. Một-lần-một-vector cost.
- **Asymmetric distance**: query giữ full precision, vector đã được encode. So với "symmetric distance" (cả hai đều encode), asymmetric chính xác hơn (mất ít thông tin hơn ở query side).
- **Distance estimation**: là **ước lượng** square distance, lỗi do quantization. Với M=16, k=256, d=768: distance error trên SIFT thường < 5% → recall@100 vẫn > 90%.
- **Memory mỗi vector**: $M \cdot \log_2 k$ bits. Cho M=16, k=256 → 16 bytes. So với raw FP32 768-dim = 3072 bytes → **192x compression**.

### 5.4 IVF-PQ kết hợp

Khi kết hợp:

1. **Train**: k-means $\to$ nlist centroids → tính residual với centroid gần nhất → PQ train trên residuals.
2. **Lý do quantize **residual** chứ không quantize raw vector**: residual có magnitude nhỏ hơn nhiều, phân phối "đều hơn" → PQ học codebook chính xác hơn. Lỗi quantization giảm rõ rệt.
3. **Index per vector**: `(cluster_id, code_bytes)`. Cluster_id thường implicit qua việc lưu trong inverted list. Mỗi inverted list = mảng `(vec_id, code)`.
4. **Query**:
   a. Tính dist query → mọi centroid → sort, lấy nprobe.
   b. Cho mỗi cluster c trong nprobe: compute residual `q_r = q - C_c`, compute LUT M × k.
   c. Scan inverted list L_c, mỗi entry: estimated_dist = sum LUT lookups.
   d. Maintain top-K trên min-heap.

**Memory**:
- Centroid book: `nlist × d × 4` bytes. Cho nlist=4K, d=768 → 12 MB. Negligible.
- PQ codebook: `M × k × (d/M) × 4` bytes. Cho M=16, k=256, d=768 → 768 KB. Negligible.
- Per-vector: M bytes. Cho 100M vectors, M=16 → 1.6 GB. **Đây là tiết kiệm chính**.
- Inverted list pointer + vec_id: ~12 bytes/vector overhead (vec_id uint64 + offset).
- **Tổng**: ~28 bytes/vector cho IVF-PQ M=16. So với 3072 bytes/vector raw FP32 → 100x compression.

### 5.5 R&D evolution dẫn tới IVF-PQ

- **VQ (Vector Quantization)** (Lloyd 1957, Gray 1984) — k-means → quantize vector thành 1 centroid. Suy thoái nặng ở high dimension vì $k$ phải rất lớn.
- **IVF** (Sivic & Zisserman ICCV 2003, "Video Google") — pioneered inverted file cho image retrieval. Bag-of-visual-words thời kỳ pre-deep-learning.
- **PQ** (Jégou, Douze, Schmid TPAMI 2011) — product quantization, paper kinh điển. INRIA group.
- **IVFADC** = "IVF + Asymmetric Distance Computation" — synonym của IVF-PQ in original paper.
- **OPQ** (Ge, He, Ke, Sun TPAMI 2013) — Optimized PQ: học **rotation matrix** R trước khi PQ, sao cho subspaces sau rotation có variance balance → cải thiện recall ~2-5% so với PQ vanilla.
- **LOPQ** (Locally Optimized PQ, Kalantidis 2014) — học OPQ riêng cho mỗi cluster của IVF. Tốt hơn nhưng phức tạp hơn.
- **ScaNN** (Guo et al. ICML 2020, Google Research) — **anisotropic VQ**: thay vì minimize quantization error đều mọi hướng, weight nhiều hơn cho hướng "parallel với query" (qua phân tích theoretical lỗi distance). Recall@10 cao hơn IVF-PQ cùng memory budget ~5-10% trên Glove/DEEP datasets. Open-source dưới TensorFlow/JAX.

### 5.6 Insert / delete dynamics trong IVF-PQ

**Insert** (sau khi index đã train):
- Tính `argmin` cluster, encode residual → push code vào inverted list. Cost $\sim$ nlist distance ops.
- **Vấn đề**: centroid được fix tại training time. Khi distribution shift → centroid không reflect dữ liệu mới → recall giảm dần.
- **Mitigation**: periodic re-train k-means + re-encode toàn bộ dataset. Cost rất lớn (vài giờ cho 100M).

**Delete**:
- Xóa entry khỏi inverted list. Đơn giản, không phải patch graph.
- Lỗ hổng inverted list cần compaction định kỳ để giữ list layout sequential.

**Update**:
- Equivalent delete + insert.
- Nếu vector update nhỏ → cluster assignment thường không đổi → chỉ phải re-encode (M lookups).

### 5.7 Variants & extensions

- **IVF-PQ + Refine**: sau khi IVF-PQ trả top-K' (K' >> K) bằng estimated distance, **re-rank** K' bằng exact distance (nếu raw vector vẫn được giữ ở đâu đó). Tăng recall đáng kể nhưng phải store raw vectors → mất compression advantage. Production thường skip refine.
- **HNSW + IVF (Faiss `HNSWFlat` + `IVFFlat` combo)** — dùng HNSW làm coarse quantizer thay vì k-means. Coarse search nhanh hơn nhiều khi nlist lớn (> 10K).
- **RaBitQ** (2024) — binarize vector thành 1 bit/dim, distance estimate qua Hamming + correction term. Còn mới, đang được Milvus và Weaviate evaluate.

---

## 6. So sánh: HNSW vs IVF-PQ (Trade-off Matrix)

### 6.1 Tổng hợp head-to-head

| Aspect | HNSW | IVF-PQ |
|---|---|---|
| **Index type** | Graph | Quantization + inverted file |
| **Build time** | Trung bình (~5-30 min cho 10M) | Cao (k-means + PQ training, ~30 min - 3h cho 100M) |
| **Memory per vector** | Raw vectors trong RAM + ~M edges (~3-5% overhead) — typically 1.1-1.5x raw size | Codebook + M bytes/vector → ~20-50x compression |
| **Recall@10 (typical)** | 0.95-0.99 (tune efSearch) | 0.85-0.95 (tune nprobe + M) |
| **Query latency (1M vectors, 1 core)** | ~0.5-2 ms | ~0.5-5 ms (phụ thuộc nprobe) |
| **Billion-scale on 1 node** | Hard (RAM-bound) | Yes (compression giúp fit) |
| **Disk-resident** | Khó (random graph traversal → IOPS hell) | Yes (sequential scan trong inverted list) |
| **Incremental insert** | Native (graph hỗ trợ) | Có nhưng centroid drift → cần re-train |
| **Delete** | Khó (tombstones degrade quality) | Dễ hơn (xóa khỏi list) |
| **Filter pushdown** | Tricky (sub-graph có thể đứt) | Easier (filter trong list scan) |
| **Tuning complexity** | M, efC, efSearch | nlist, nprobe, M, k |
| **Best for** | Mid-scale (1M-100M), high-recall, in-memory budget OK | Billion-scale, memory constrained, recall ≥ 0.9 OK |

### 6.2 Benchmark numbers từ ann-benchmarks.com (2024)

Dataset: **SIFT1M** (1M vectors, 128-dim, public benchmark).

| Algorithm | Recall@10 | QPS (single thread) | Memory |
|---|---|---|---|
| hnswlib (M=16, efC=200) | 0.95 | ~12,000 | ~1.1× raw |
| hnswlib (M=48, efC=500) | 0.99 | ~5,000 | ~2× raw |
| faiss IVF4096-PQ16 (nprobe=8) | 0.85 | ~10,000 | ~16 bytes/vec |
| faiss IVF4096-PQ16 (nprobe=64) | 0.95 | ~2,500 | ~16 bytes/vec |
| ScaNN (default) | 0.95 | ~30,000 | ~24 bytes/vec |
| Annoy (n_trees=100) | 0.85 | ~3,000 | ~3× raw |

Dataset: **DEEP1B** (1B vectors, 96-dim, image embeddings).

| Algorithm | Recall@10 | QPS | Memory (1 node) |
|---|---|---|---|
| HNSW (not feasible on 1 node) | — | — | > 500 GB RAM needed |
| IVF65536-PQ8 (nprobe=64) | 0.85 | ~500 | ~20 GB RAM |
| ScaNN | 0.90 | ~2,000 | ~40 GB RAM |
| DiskANN (SSD) | 0.95 | ~4,000 | ~60 GB RAM + 200 GB SSD |

**Lưu ý**: số trên là single-thread approximation từ public ann-benchmarks runs 2023-2024. Production multi-thread, multi-replica có thể nhân lên 8-32x QPS. Real production numbers từ engineering blog cụ thể sẽ cite ở section References.

### 6.3 Pinterest pattern (S1-03 PinSage) — hybrid tier

Như đã thấy ở [S1-03 PinSage](../01-foundations/S1-03_pinterest_pinsage_graph_retrieval.md): Pinterest serve 3B PinSage embeddings bằng **hot/cold tier**:
- **Hot tier** (~100M most popular pins): in-memory HNSW. Recall cao, latency thấp.
- **Cold tier** (~3B remaining): IVF-PQ on disk. Recall vừa đủ, memory ngân sách kiểm soát.

Pattern này phổ biến mọi production team có "long-tail distribution" trên embeddings.

### 6.4 Khi nào pick which?

Quy tắc kinh nghiệm:

- **< 10M vectors, RAM dư dả** → **HNSW**. Implementation đơn giản, recall cao, latency thấp.
- **10M-100M, latency-sensitive** → **HNSW** với scalar quantization int8 (memory giảm 4x, recall mất ~1%).
- **100M-1B** → **IVF-PQ** hoặc **ScaNN**. Hoặc **DiskANN** nếu chấp nhận SSD latency.
- **> 1B** → **DiskANN** (Microsoft Azure), **ScaNN sharded** (Google Vertex Vector Search), hoặc **IVF-PQ multi-node** (Milvus).
- **Filter-heavy queries** → Cần index hỗ trợ filter pushdown native (Weaviate, Milvus, Pinecone metadata filtering). pgvector filter qua Postgres WHERE → có thể sụp đổ với HNSW nếu filter quá chọn lọc (xem §8).
- **Hybrid BM25+vector** → Weaviate (built-in BM25 + HNSW + RRF) hoặc Elasticsearch 8+ (dense_vector field + BM25 + RRF).
- **Postgres-first ops** → pgvector. Đơn giản, ACID, không cần infrastructure mới. Trade-off: scale ceiling ~10-100M.

---

## 7. Các variant nâng cao — ScaNN, DiskANN, OPQ

### 7.1 ScaNN — Anisotropic Vector Quantization (Google 2020)

**Insight cốt lõi của Guo et al. ICML 2020 (arXiv:1908.10396)**: trong PQ vanilla, loss function là $\|x - Q(x)\|^2$ — minimize quantization error đều mọi hướng. Nhưng cho **maximum inner product search (MIPS)** (đặc biệt phổ biến trong recommendation), điều ta quan tâm thật sự là sai số trong inner product, **không** trong reconstruction. Hai sai số này có direction-dependence: lỗi reconstruction theo hướng parallel với query ảnh hưởng inner product nhiều hơn lỗi orthogonal.

ScaNN training loss thêm **anisotropic weighting**:
$$
\mathcal{L} = h_\parallel \cdot \|x - Q(x)\|_\parallel^2 + h_\perp \cdot \|x - Q(x)\|_\perp^2
$$
với $h_\parallel \gg h_\perp$ (typical ratio 5-10x). Trong practice, weight parallel direction được dynamically estimate qua query distribution (training data).

**Kết quả**: trên Glove1M, recall@10 0.95 đạt ở QPS gấp ~2x so với PQ vanilla cùng memory budget. ScaNN trở thành Google's internal vector search standard, được ship trong Vertex Vector Search (formerly Matching Engine). Open-source impl: `pip install scann`.

**Khi nào dùng ScaNN**: workload chính là MIPS (recommendation, ads ranking), memory budget vừa phải, không cần fancy features khác.

### 7.2 DiskANN — billion-scale on single SSD node (Microsoft 2019)

**Vấn đề**: HNSW recall cao nhưng RAM-bound — fit 1B vector × 128-dim cần ~500 GB RAM (chưa kể graph). DiskANN (Subramanya et al. NeurIPS 2019) hỏi: build **graph-based ANN** mà phần lớn nằm trên SSD, chỉ một fraction trong RAM.

Core innovations:

1. **Vamana graph algorithm**: variant của greedy graph construction nhưng được tune để có **diameter thấp** và **traversal localized** (số distinct disk pages truy cập cho một query thấp).
2. **PQ residual cache trong RAM**: PQ-encoded vectors (~M bytes/vector) cho **toàn bộ** dataset được nạp RAM. Khi traverse graph trên SSD, distance computation tới ứng viên dùng PQ estimated distance từ RAM → tránh phải đọc raw vector từ disk cho mọi candidate.
3. **Full vector chỉ đọc khi cần**: chỉ khi cần exact distance refine ở cuối query, đọc raw vector từ SSD.
4. **Sequential layout**: graph được lay out trên disk sao cho neighbors-of-neighbors gần về vị trí — giảm random IOPS.

**Kết quả benchmark từ paper**:
- 1B SIFT vectors trên 1 node 64 GB RAM + 1 TB SSD: recall@10 = 0.95, QPS ~4000, P99 latency ~5ms.
- HNSW thuần không khả thi trên cùng hardware.

**Production**: Azure Cognitive Search dùng DiskANN backend. Milvus tích hợp DiskANN từ v2.1 (2022). Mã nguồn: github.com/microsoft/DiskANN.

### 7.3 OPQ — Optimized Product Quantization

**Vấn đề của PQ vanilla**: chia $d$-dim thành $M$ subspace mặc định là chia liên tục theo index ($r_1$ = first $d/M$ dims, etc.). Nhưng phân phối variance trong các chiều **không đều** — vài chiều variance cao, vài chiều variance thấp. Hệ quả: vài subspace bị quantize error lớn, vài cái redundant.

**OPQ (Ge, He, Ke, Sun TPAMI 2013)**: học một **orthogonal rotation matrix R** trước khi PQ, sao cho sau rotation, các subspace có variance balance. Train iterative: thay phiên (a) PQ trên rotated data, (b) update R sao cho minimize quantization error.

**Kết quả**: trên SIFT1M, OPQ@(M=8, k=256) cho recall@10 cao hơn PQ vanilla 2-5%, không tăng memory (R chỉ là d×d matrix, lưu một lần).

**Sử dụng**: Faiss flag `OPQ8_64` (M=8, intermediate dim=64) phổ biến. Milvus và Pinecone đều support OPQ ở backend.

### 7.4 Bảng so sánh các variant

| Variant | Year / org | Key idea | Best for | Recall gain vs PQ |
|---|---|---|---|---|
| PQ (vanilla) | 2011 INRIA | Subspace quantization | Memory-bound baseline | — |
| OPQ | 2013 MS Research Asia | Rotation trước PQ | Drop-in replacement, recall tốt hơn | +2-5% |
| LOPQ | 2014 Yahoo | Per-cluster OPQ | Recall cao hơn, phức tạp | +3-7% |
| ScaNN | 2020 Google | Anisotropic loss cho MIPS | Inner product workload | +5-10% (MIPS) |
| DiskANN | 2019 Microsoft | Graph trên SSD + PQ trong RAM | Billion-scale 1 node | +5-15% recall vs IVF-PQ cùng QPS |
| RaBitQ | 2024 academic | 1-bit binary + correction | Extreme compression, đang evaluate | TBD |

---

## 8. Production Vector Databases — Comparison

### 8.1 Pinecone

**Architecture** (suy luận từ Pinecone blog posts và docs, không phải full internal):

```text
              PINECONE LOGICAL ARCHITECTURE (simplified)

   ┌─────────────────── CONTROL PLANE (multi-region) ──────────────────┐
   │                                                                   │
   │   ┌──────────┐   ┌────────────┐   ┌──────────────┐                │
   │   │ Index    │   │ Auth /     │   │ Metering /   │                │
   │   │ catalog  │   │ project    │   │ billing      │                │
   │   │ service  │   │ mgr        │   │              │                │
   │   └──────────┘   └────────────┘   └──────────────┘                │
   └───────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────── DATA PLANE (per-region cluster) ────────────────┐
   │                                                                   │
   │   Pod-based mode (legacy):                                        │
   │   ┌─────────┐   ┌─────────┐   ┌─────────┐                         │
   │   │ Pod 1   │   │ Pod 2   │   │ Pod N   │                         │
   │   │ (shard) │   │ (shard) │   │ (shard) │                         │
   │   │ p1.x1   │   │ p1.x1   │   │ p1.x1   │                         │
   │   │ HNSW +  │   │         │   │         │                         │
   │   │ replicas│   │         │   │         │                         │
   │   └─────────┘   └─────────┘   └─────────┘                         │
   │                                                                   │
   │   Serverless mode (since 2024):                                   │
   │   ┌──────────────┐   ┌──────────────┐                             │
   │   │ Object       │   │ Compute      │                             │
   │   │ storage      │   │ workers      │                             │
   │   │ (S3-like)    │◀─▶│ (stateless,  │                             │
   │   │ — indexes,   │   │  scale to 0) │                             │
   │   │   metadata,  │   │              │                             │
   │   │   vectors    │   └──────────────┘                             │
   │   └──────────────┘                                                │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘
```

**Index type**:
- Pinecone không expose index choice ở user level. Internal: kết hợp **graph-based** (likely HNSW-variant) + **quantization** tuỳ pod tier. From Pinecone learning hub (2023): "we use a proprietary index structure derived from HNSW and graph indexes".
- Serverless mode (2024): introduce **storage-compute separation** — index segments persist trên object storage, compute workers fetch on-demand. Giảm cost cho cold workloads (scale to zero).

**Strengths**:
- Zero ops, fully managed.
- Filter on metadata first-class (efficient).
- Namespaces native (multi-tenancy).
- Serverless mode pricing rất tốt cho intermittent workloads.

**Weaknesses**:
- Closed-source, vendor lock-in.
- Index type không expose → khó tune cho edge cases.
- Cost cao khi scale (~$70-700/mo/pod for legacy pods cho ~1-5M vectors).
- No on-prem (only cloud — limit cho regulated industry).

**Public benchmarks**:
- "p2.x1" pod: ~100K vectors per pod, QPS ~50, recall@10 ~0.95 (Pinecone docs 2023).
- Serverless: latency P50 ~30ms, P99 ~100ms cho < 10M vectors (Pinecone blog 2024).

### 8.2 Weaviate

**Architecture**:

```text
                    WEAVIATE ARCHITECTURE (open-source, Go)

   ┌─────────────────────────── API LAYER ─────────────────────────────┐
   │   GraphQL / REST / gRPC                                           │
   │   - Vector search                                                  │
   │   - BM25 keyword search                                            │
   │   - Hybrid (RRF)                                                   │
   │   - Generative (built-in LLM modules)                              │
   └───────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────── CORE NODE ─────────────────────────────┐
   │                                                                   │
   │   ┌───────────────────────────────────────────────────────────┐   │
   │   │   Schema / class registry (LSM-backed)                    │   │
   │   └───────────────────────────────────────────────────────────┘   │
   │                                                                   │
   │   Per-class shard:                                                │
   │   ┌──────────────────────────┐   ┌──────────────────────────┐     │
   │   │  Object store (LSM-tree) │   │  HNSW vector index       │     │
   │   │  - JSON properties       │◀─▶│  (in-memory + WAL)       │     │
   │   │  - BM25 inverted index   │   │  Optional: PQ, BQ        │     │
   │   │  - WAL                   │   │  compression             │     │
   │   └──────────────────────────┘   └──────────────────────────┘     │
   │                                                                   │
   │   Modules: text2vec-*, qna-*, ref2vec-* (built-in)                │
   └───────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────── CLUSTERING ────────────────────────────┐
   │   Raft for metadata (since v1.25+)                                │
   │   Per-class sharding (shard key = object UUID hash)                │
   │   Replication factor configurable per class                        │
   └───────────────────────────────────────────────────────────────────┘
```

**Index type**: HNSW primary. Hỗ trợ **PQ compression** (kích hoạt qua `vectorIndexConfig.pq`) — giảm RAM 4-32x với drop recall ~1-3%. Từ v1.21 hỗ trợ **BQ (Binary Quantization)** — 1 bit/dim.

**Strengths**:
- Open-source (BSD-3), self-host miễn phí.
- Hybrid search (BM25 + dense) built-in từ ngày đầu.
- GraphQL API rất phù hợp app dev.
- Modules cho generative AI (gọi LLM trực tiếp từ query).
- Multi-tenancy native (since v1.20+).

**Weaknesses**:
- Trước v1.25, Raft chỉ cho metadata không cho data → write availability hạn chế khi node mất.
- Per-class sharding → cross-class joins phải làm client-side.
- PQ tuning manual (phải explicit enable).

**Public benchmarks**:
- 1M Sphere vectors (768-dim), recall@10 0.95: P99 latency ~7ms (Weaviate blog 2023).
- BQ (binary quantization) trên 1M vectors: 32x memory reduction, recall ~0.88 (Weaviate blog 2023).

### 8.3 Milvus

**Architecture**:

```text
                  MILVUS DISTRIBUTED ARCHITECTURE

   ┌─────────────── ACCESS LAYER (stateless) ──────────────────────┐
   │   Proxy (gRPC) — request routing, load balancing              │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────── COORDINATOR SERVICES ──────────────────────┐
   │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐│
   │   │  Root      │  │  Query     │  │  Data      │  │  Index   ││
   │   │  coord     │  │  coord     │  │  coord     │  │  coord   ││
   │   │ (DDL, time)│  │ (query plan)│ │ (WAL,segm) │  │ (build)  ││
   │   └────────────┘  └────────────┘  └────────────┘  └──────────┘│
   │                                                                │
   │   Metadata: etcd                                               │
   └────────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────────── WORKER NODES (scale out) ─────────────────┐
   │   ┌────────────┐  ┌────────────┐  ┌────────────┐               │
   │   │   Query    │  │   Data     │  │   Index    │               │
   │   │   nodes    │  │   nodes    │  │   nodes    │               │
   │   │  (search   │  │  (WAL,     │  │  (offline  │               │
   │   │  segments) │  │  flush)    │  │   index    │               │
   │   │            │  │            │  │   build)   │               │
   │   └────────────┘  └────────────┘  └────────────┘               │
   └────────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────── STORAGE LAYER (pluggable) ────────────────────┐
   │   - Log: Pulsar / Kafka (WAL, durability)                      │
   │   - Object storage: S3 / MinIO / GCS / Azure Blob              │
   │     (sealed segments, indexes)                                 │
   └────────────────────────────────────────────────────────────────┘
```

**Index types supported** (richest of any vector DB):
- `FLAT` — brute force
- `IVF_FLAT`, `IVF_SQ8`, `IVF_PQ` — IVF family
- `HNSW`, `RHNSW_FLAT`, `RHNSW_PQ`, `RHNSW_SQ` — HNSW family with quantization
- `ANNOY` — tree-based
- `DISKANN` — disk-resident (from v2.1)
- `GPU_IVF_FLAT`, `GPU_IVF_PQ` — GPU-accelerated (since v2.3, RAFT integration)

**Strengths**:
- Open-source (Apache 2.0).
- Storage-compute separation native từ ngày đầu (v2.0+).
- Multi-index — pick per-collection.
- GPU acceleration cho IVF.
- Massive scale (10B+ vector references in production users).
- Battle-tested (active community, ZILLIZ commercial backing).

**Weaknesses**:
- Complex architecture → ops phức tạp.
- Many moving parts (Pulsar, etcd, multiple coords) → cluster footprint lớn cho mid-scale.
- Multi-vector per record support limited so với Weaviate.

**Public benchmarks**:
- 1B 128-dim vectors with IVF_PQ trên 8-node cluster: ~1000 QPS, recall@10 0.92 (Milvus blog 2023).
- DiskANN single node 1B vectors: ~4000 QPS, recall 0.95 (Milvus + DiskANN blog 2022).

### 8.4 pgvector

**Architecture**:

```text
                  PGVECTOR — POSTGRES EXTENSION

   ┌──────────────────── POSTGRES ─────────────────────────────────┐
   │                                                               │
   │   ┌────────────────────────────────────────────────────┐      │
   │   │  Standard Postgres planner / executor              │      │
   │   │  - WAL, MVCC, replication (built-in)               │      │
   │   │  - SQL queries with JOINs, WHERE, ORDER BY         │      │
   │   └────────────────────────────────────────────────────┘      │
   │                                                               │
   │   pgvector extension:                                         │
   │   ┌────────────────────────────────────────────────────┐      │
   │   │  Type: vector(d), halfvec(d), bit(d)               │      │
   │   │  Operators: <-> (L2), <#> (negative inner),        │      │
   │   │             <=> (cosine)                            │      │
   │   │                                                    │      │
   │   │  Index types:                                      │      │
   │   │   - HNSW (since v0.5, July 2023)                   │      │
   │   │     params: m, ef_construction                     │      │
   │   │     query param: ef_search (SET)                   │      │
   │   │   - IVFFlat (since v0.4)                           │      │
   │   │     params: lists                                  │      │
   │   │     query param: ivfflat.probes                    │      │
   │   └────────────────────────────────────────────────────┘      │
   │                                                               │
   └───────────────────────────────────────────────────────────────┘
```

**SQL example**:

```sql
-- Tạo extension
CREATE EXTENSION vector;

-- Tạo table với vector column
CREATE TABLE docs (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    tenant_id TEXT,
    created_at TIMESTAMP DEFAULT now(),
    embedding vector(1536)
);

-- HNSW index
CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Query với filter pushdown qua Postgres planner
SET hnsw.ef_search = 100;
SELECT id, content
FROM docs
WHERE tenant_id = 'acme'
  AND created_at > now() - interval '7 days'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

**Strengths**:
- **Operational simplicity** — không cần infrastructure mới nếu đã chạy Postgres.
- ACID, transactions, joins, full-text search (built-in tsvector) — toàn bộ Postgres ecosystem.
- Mature backup/restore/replication (logical + streaming).
- HNSW + IVFFlat đủ tốt cho < 10M-100M scale.

**Weaknesses**:
- Scale ceiling: pgvector dùng được tới ~10-100M vectors trên 1 node. Beyond đó cần shard manual (Citus / partitioning) — phức tạp.
- HNSW build chậm hơn dedicated libs (CPU-only, không multi-thread native trước v0.6).
- No native PQ compression — vectors lưu raw FP32 (hoặc FP16 với halfvec từ v0.7).
- Filter pushdown qua Postgres planner — đôi khi không tối ưu (xem §9).

**Public benchmarks**:
- pgvector v0.5 (HNSW), 1M Sphere 768-dim: ~1000 QPS recall@10 0.95 (Supabase blog 2023).
- pgvector v0.7 (halfvec + HNSW), 1M vectors: 2x faster build, 50% memory reduction (pgvector release notes 2024).

### 8.5 Bảng so sánh feature matrix

| Feature | Pinecone | Weaviate | Milvus | pgvector |
|---|---|---|---|---|
| **License** | Proprietary | BSD-3 | Apache 2.0 | PostgreSQL License |
| **Deploy model** | Managed cloud only | Self-host + managed | Self-host + managed (Zilliz) | Self-host (Postgres) |
| **Primary index** | Proprietary (HNSW-derived) | HNSW | IVF / HNSW / DiskANN | HNSW + IVFFlat |
| **Compression** | Internal (auto) | PQ, BQ optional | PQ, SQ8, FP16 | halfvec (FP16) |
| **Hybrid (BM25+dense)** | Native (sparse-dense) | Native | Native (since v2.4) | Manual (tsvector + RRF) |
| **Metadata filter** | First-class | First-class | First-class | SQL WHERE (Postgres planner) |
| **Multi-tenancy** | Namespaces | Tenants (per-class) | Partitions / databases | Postgres schemas / RLS |
| **Scale ceiling** | Multi-billion (managed) | ~Billion (sharded) | 10B+ (sharded) | ~100M (single node) |
| **Storage-compute separation** | Yes (serverless) | No (in-process) | Yes (object storage) | No |
| **GPU acceleration** | No (closed) | No | Yes (IVF, since v2.3) | No |
| **Real-time insert** | Yes | Yes | Yes (segments) | Yes |
| **ACID** | Eventual | Per-object | Per-collection | Full ACID |
| **Replication** | Built-in | Built-in (Raft v1.25+) | Built-in | Postgres streaming |
| **Pricing (rough)** | $70-700/mo/pod | Free OSS / $25+/mo managed | Free OSS / Zilliz $20+/mo | Free (Postgres cost) |

### 8.6 Decision tree

```text
Bạn đã chạy Postgres?
   ├─ Yes & data < 100M vectors
   │     └─▶ pgvector (đơn giản nhất)
   │
   └─ No / data > 100M

Vector count?
   ├─ < 10M
   │     ├─ Cần ops zero? → Pinecone Serverless / Weaviate Cloud
   │     └─ OK ops? → Weaviate self-host (free)
   │
   ├─ 10M-1B
   │     ├─ Hybrid search critical? → Weaviate
   │     ├─ Need GPU? → Milvus
   │     └─ Managed ok? → Pinecone (legacy pods) hoặc Vertex Vector Search
   │
   └─ > 1B
         ├─ Single-node OK? → Milvus + DiskANN, hoặc Azure Cognitive Search
         └─ Multi-node? → Milvus distributed hoặc custom (Pinterest pattern)
```

---

## 9. Operational Concerns

### 9.1 Filter pushdown — "pre-filter" vs "post-filter"

Một trong những "gotcha" lớn nhất khi triển khai vector DB là **filter behavior với metadata**. Có 3 chiến lược:

**Strategy 1: Post-filter** (naive)
```text
1. Top-K' = vector_search(q, K' = K * X)   # over-fetch
2. Filter top-K' bằng metadata predicate
3. Trả top-K từ kết quả còn lại
```
Vấn đề: nếu predicate chọn lọc < X%, top-K' không đủ → recall giảm hoặc K' phải tăng exponentially.

**Strategy 2: Pre-filter** (naive)
```text
1. ids_matching = scan metadata index, lọc theo predicate  # tập matched ids
2. Bruteforce distance trên ids_matching
3. Top-K
```
Vấn đề: nếu predicate matched > 1M items → brute force vẫn quá chậm.

**Strategy 3: Pushdown trong index traversal** (best)
- HNSW: trong khi greedy descend, skip node không match predicate. Nhưng nếu predicate quá chọn lọc → sub-graph quá rời rạc → search miss. Mitigations: blocked graph (Weaviate v1.18+), Milvus FilterIVF, Pinecone metadata-filter-aware.
- IVF: lưu metadata trong inverted list entry → filter ngay khi scan list.

**Cảnh báo cụ thể** từ Weaviate blog 2023 và Pinecone docs 2023:
- Với HNSW, **filter chọn lọc < 5%** thường gây recall sụp đổ. Workaround: dùng pre-filter brute-force cho selective predicate, switch ngưỡng heuristic.
- Pinecone metadata filter có overhead per-vector check → recall hiệu quả phụ thuộc filter cardinality.

**pgvector** đặc biệt: dùng Postgres planner để chọn pre/post-filter. Cho HNSW + WHERE, **iterative index scan** (v0.6+) — descend graph và check WHERE predicate per node. Selective filter có thể trigger fallback sequence scan + sort.

### 9.2 Sharding strategies

| Strategy | Mô tả | Khi nào dùng |
|---|---|---|
| **Hash-based** | shard = hash(vec_id) % N | Insert balance đều, không hỗ trợ tenant locality |
| **Range-based on time** | shard = bucket(timestamp) | Time-series queries, retire old shards dễ |
| **Tenant-based** | shard = hash(tenant_id) | Multi-tenant isolation, filter pushdown rẻ |
| **Hot/cold tier** | hot shard = recent / popular, cold = rest | Long-tail distribution (Pinterest pattern) |
| **Vector-locality (clustering)** | shard = nearest centroid | Search chỉ vài shard, latency thấp |

**Query fan-out cost**: với N shards và per-shard latency L, total latency ≈ L + tail effect. Để giữ P99 thấp, **không shard quá nhiều** (typically 4-16 shards/query) — quá nhiều shard tail latency dominate.

### 9.3 Replication & consistency

- **Async replication**: vài giây lag. Phù hợp cho read-heavy workload, latest insert không cần ngay lập tức.
- **Sync replication**: write block tới khi N/2+1 replica ack. Latency tăng, an toàn hơn.
- **Read-your-writes consistency**: insert trả về phải search được ngay → write-ahead-log + in-memory growing segment serve cùng query path.

Weaviate, Milvus: built-in replication via Raft (Weaviate v1.25+) hoặc Pulsar (Milvus).
Pinecone: hidden, managed.
pgvector: Postgres streaming/logical replication.

### 9.4 Hybrid search và RRF

Như nhắc ở [S3-02 Production RAG](../03-modern-stack/S3-02_production_rag_system_architecture.md): production retrieval rất hiếm khi chỉ dùng dense vector. **Hybrid = BM25 (sparse) + dense + RRF fusion** thường tốt hơn dense alone 5-15% trên domain-specific data.

Vector DB hỗ trợ hybrid:
- **Weaviate**: native `hybrid` query với param `alpha` (weight dense/sparse), default RRF fusion.
- **Milvus**: từ v2.4, hỗ trợ multi-vector + sparse vector field, RRF fusion built-in.
- **Pinecone**: "sparse-dense vectors" feature 2023 — sparse vector qua BM25 + dense vector → cosine score combine.
- **Elasticsearch 8+**: `dense_vector` field + standard BM25 + `rrf` retriever.
- **pgvector**: phải manual — tạo riêng `tsvector` column, query song song, RRF ở app layer.

Pseudo-code hybrid query:

```python
def hybrid_search(vector_db, bm25_index, query_text, query_vec, k=10):
    """
    Hybrid search với RRF fusion.
    Reciprocal Rank Fusion: score = sum_i 1 / (rank_i + c), c=60 thường.
    """
    # Pha 1: dense retrieval
    dense_hits = vector_db.search(query_vec, k=100)  # over-fetch
    # Pha 2: sparse retrieval
    sparse_hits = bm25_index.search(query_text, k=100)

    # RRF
    c = 60
    scores = defaultdict(float)
    for rank, hit in enumerate(dense_hits):
        scores[hit.id] += 1.0 / (rank + 1 + c)
    for rank, hit in enumerate(sparse_hits):
        scores[hit.id] += 1.0 / (rank + 1 + c)

    # Top-k by combined score
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [hydrate(id_) for id_, _ in top]
```

### 9.5 Index rebuild vs incremental insert

- **HNSW**: native incremental insert. Nhưng sau ~20-30% delete → graph degrade → rebuild needed.
- **IVF-PQ**: centroid fix tại train time. Insert mới gắn nearest centroid + encode → over time distribution drift, recall giảm. Rebuild khi PSI (Population Stability Index) trên embeddings > threshold.
- **Hybrid approach**: production thường tạo new segments cho data mới (immutable), periodic merge sealed segments. Lúc merge là cơ hội rebuild index full.

### 9.6 Monitoring & alerting

Metrics quan trọng:

| Metric | Threshold/alert | Why |
|---|---|---|
| Query P99 latency | > 2x baseline | Hot key / segment imbalance |
| Recall (online sampling) | drop > 5% | Index drift, centroid stale |
| Index size growth rate | > forecast | Storage planning |
| Insert latency P99 | > 500ms | WAL pressure, lock contention |
| Replica lag (replicas) | > 30s | Replication health |
| Cluster shard balance | imbalance > 30% | Re-shard needed |
| Filter selectivity histogram | very selective filters increasing | UX problem hoặc query pattern shift |

### 9.7 Capacity planning

Quick formula cho HNSW in-memory:
$$
\text{RAM} \approx N \cdot d \cdot 4 \text{ bytes} \times 1.5
$$
(factor 1.5 = raw vector + graph overhead + overhead).

Cho 100M × 768-dim FP32: ~460 GB. Với halfvec (FP16) hoặc PQ: 100-200 GB. Có thể fit 1 node lớn (256GB-2TB RAM phổ biến).

Quick formula cho IVF-PQ:
$$
\text{RAM} \approx N \cdot M \text{ bytes} + \text{nlist} \cdot d \cdot 4 \text{ bytes}
$$

Cho 1B × 768-dim, M=16, nlist=65536: ~16 GB codes + ~200 MB centroids = ~17 GB. Khả thi trên 1 node.

---

## 10. Trade-offs & Design decisions

### 10.1 Hyperparameter sensitivity comparison

| Hyperparameter | Effect range | Sensitivity |
|---|---|---|
| HNSW `M` | 8-64 | Medium — more = better recall + RAM |
| HNSW `efSearch` | 10-2000 | High — direct recall/QPS lever, tune online |
| IVF `nlist` | √N to 16√N | Medium — affects training, build time |
| IVF `nprobe` | 1 to nlist | High — direct recall/QPS lever, tune online |
| PQ `M` (subspaces) | 4-64 | High — direct memory/accuracy |
| PQ `k` (centroids) | 16-256 | Medium — 256 standard, smaller saves bits |

### 10.2 Build-time vs Query-time trade-off

- **HNSW**: build relatively chậm ($O(N \log N \cdot \text{efC} \cdot M)$ distance ops) nhưng query nhanh. Build full 100M typically 1-5 hours single-machine.
- **IVF-PQ**: build = k-means trên sample (vài chục triệu) + PQ trên residuals + encode toàn bộ. Total time có thể tương đương HNSW nhưng dùng GPU thì giảm nhiều (Milvus GPU IVF training 10x faster).

### 10.3 Recall vs latency dial

Most important production lever:
- HNSW: tăng `efSearch` từ 50 → 500 → recall tăng từ 0.92 → 0.99 nhưng latency tăng ~3-5x.
- IVF-PQ: tăng `nprobe` từ 8 → 64 → recall tăng từ 0.85 → 0.95 nhưng latency tăng ~5-8x.

Tune online qua A/B test: dial up đến khi business metric (e.g., downstream CTR, RAG faithfulness) plateau.

### 10.4 Comparison summary table — chính

| Decision axis | HNSW | IVF-PQ | When does it matter |
|---|---|---|---|
| Memory efficient | No (1-1.5x raw) | Yes (10-100x compression) | Billion scale |
| High recall | Yes (0.95-0.99) | Medium (0.85-0.95) | Search quality critical |
| Low latency | Yes (~1ms) | Yes (~1-5ms) | Real-time RAG |
| Incremental ops | Yes (native) | Limited (centroid drift) | Frequent updates |
| Disk-resident | Hard | Native | Memory budget tight |
| Filter pushdown | Tricky | Easier | Multi-tenant / metadata-heavy |
| Build speed | Medium | Slower (k-means) | Re-index frequency |
| Tuning complexity | M, efC, ef | nlist, nprobe, M, k | Engineering bandwidth |

### 10.5 Distance metric trade-offs

| Metric | Formula | Use case | Index implications |
|---|---|---|---|
| L2 (Euclidean) | $\|q - v\|_2$ | Image / face embedding (norm matters) | Native HNSW/IVF/PQ |
| Cosine | $q \cdot v / (\|q\|\|v\|)$ | Text embedding (semantic similarity) | Normalize → inner product |
| Inner product (dot) | $q \cdot v$ | MIPS — recommendation, learned embedding | Anisotropic VQ (ScaNN) optimal |
| Hamming | bitcount(q XOR v) | Binary embeddings (RaBitQ, BQ) | Native SIMD support |

Quan trọng: **đảm bảo metric của index match metric của embedding model**. OpenAI ada-002 dùng cosine, Cohere normalize vectors → cosine = inner-product, BERT raw [CLS] thường dùng cosine.

---

## 11. Lessons learned & Best practices

### 11.1 Lesson 1: Filter selectivity là vấn đề ẩn lớn nhất

Pinecone docs 2023 cảnh báo rõ: với HNSW + filter chọn lọc < 5%, recall có thể sụp đổ 30-50%. Lý do: filter chỉ giữ một sub-graph, greedy descent có thể bị stuck ở local minima vì không có đủ "long-range edge" nối các phần của sub-graph.

**Best practice**: 
- Test filter recall offline với golden set + filter combinations thực tế.
- Cho selective filter (tenant, time range hẹp), shard per filter dimension (mỗi tenant = 1 namespace / partition).
- Monitor recall online qua sampling.

(Source: Pinecone learning hub "Filtering in vector databases" 2023.)

### 11.2 Lesson 2: Không upgrade index type dễ — phải re-index

Khi đổi từ IVFFlat → HNSW trong pgvector v0.5, hoặc đổi nlist trong Milvus → phải **rebuild full index**. Cho dataset 100M+ điều này có thể là multi-hour operation. Không có cách "migrate live".

**Best practice**: chọn index type cẩn thận từ đầu, dựa trên forecast 12-18 tháng tới. Nếu unsure, prefer flexibility (Milvus với multiple indexes per collection).

### 11.3 Lesson 3: Memory footprint thực tế cao hơn calculation

Calculation lý thuyết (raw × 1.5 cho HNSW) thường thấp hơn thực tế 30-50% vì:
- Memory allocator overhead (jemalloc, glibc malloc fragmentation).
- Index metadata (mapping vec_id → offset).
- WAL buffer.
- Replication queue.

**Best practice**: capacity plan với buffer 2x lý thuyết. Theo dõi RSS / heap thực qua Prometheus.

(Source: Weaviate blog "How to estimate Weaviate memory consumption" 2023.)

### 11.4 Lesson 4: PQ training cần representative sample

Nếu k-means train trên non-representative sample (e.g., chỉ lấy data 1 region/tenant), code book sẽ biased → recall trên data khác sụp đổ.

**Best practice**: 
- Random sample stratified theo các axis quan trọng (tenant, time, source).
- Sample size: thường 10-100M vector là đủ cho k=256 mỗi subspace.
- Re-train PQ định kỳ (quarterly) nếu distribution drift detected.

### 11.5 Lesson 5: Embedding model swap = full re-index

Khi swap embedding model (e.g., ada-002 → text-embedding-3-large), không phải vì "vectors có giá trị mới" — chúng nằm trong **không gian khác**. Phải re-embed toàn bộ corpus + re-index.

**Best practice**:
- Plan migration window khi swap model.
- Có thể chạy old + new index song song trong shadow mode để A/B compare quality.
- Cost forecast: re-embedding 100M docs với ada-002 ~ $1300; với text-embedding-3-large ~ $1300 cho cùng dim, gấp đôi với higher dim.

### 11.6 Lesson 6: Cross-encoder reranker bù được cho ANN recall thấp

Pinecone blog 2023 và Cohere blog 2023 đều khuyến nghị: thay vì ép ANN recall@10 lên 0.99 (rất đắt), set ANN recall@100 ~0.95 + cross-encoder reranker top-100 → final top-10 = win both QPS và quality.

**Best practice**: production retrieval = ANN over-fetch (k=50-200) + reranker → final k. Xem [S3-02 Production RAG](../03-modern-stack/S3-02_production_rag_system_architecture.md) §4.4 cho reranker patterns.

### 11.7 Lesson 7: Index build dùng GPU cải thiện 10x

NVIDIA RAFT (Reusable Accelerated Functions and Tools) integration cho Milvus và FAISS cho phép k-means + HNSW build trên GPU. Build time giảm 5-10x cho 100M+ datasets.

**Best practice**: nếu re-index thường xuyên (weekly/daily), invest GPU build. ROI rõ trong vài tuần.

(Source: NVIDIA RAFT blog 2023, Milvus + GPU benchmark blog 2023.)

### 11.8 Lesson 8: Hybrid > dense alone trên domain-specific data

Trên benchmarks chung (MS MARCO, NQ), dense beats BM25 ~5%. Nhưng trên enterprise / domain-specific data (legal, medical, internal docs), BM25 vẫn rất mạnh — thường hybrid + RRF tốt hơn dense alone 10-20%.

**Best practice**: luôn evaluate hybrid trước khi commit dense-only. Anthropic Contextual Retrieval blog 2024 confirm: contextual BM25 + contextual dense + reranker = state-of-the-art trên their internal benchmark.

(Source: Anthropic blog "Introducing Contextual Retrieval" Sep 2024.)

### 11.9 Lesson 9: Quantize embeddings 1 lần sau ingestion, đừng quantize on-the-fly

Encode PQ codes cho mỗi insert tốn ~k*d ops. Nếu workload write-heavy, encoding nội suy có thể bottleneck. Batch encode tách khỏi hot path (consume từ WAL, encode bulk, push vào sealed segment).

(Source: Milvus blog "Storage architecture" 2023.)

### 11.10 Lesson 10: Track recall online, không chỉ offline

Recall@10 trên golden set khi accept index → không guarantee recall trên live traffic. Distribution drift, popular queries shift → recall có thể tụt 5-10% mà ops không biết.

**Best practice**:
- Mỗi N queries (e.g., 1/1000), chạy song song brute-force + indexed → compute recall.
- Track P50/P99 recall online → alert khi drift.

(Source: Vespa blog "Measuring search relevance in production" 2023, Pinterest engineering blog patterns.)

### 11.11 Lesson 11: Cấu hình ef phù hợp giúp giảm cost rõ rệt

Default `ef=10` của một số lib quá thấp cho production. Bump lên ef=50-100 + tune online — recall tăng 5-10%, latency tăng 2-3x nhưng vẫn dưới SLA. Đừng dùng default.

(Source: hnswlib README + Faiss tuning guide.)

### 11.12 Lesson 12: Cold start cho new tenant — over-allocate

Nếu mỗi tenant = 1 namespace / shard, tenant mới (vài chục vector) sẽ chia sẻ shard với tenant khác hoặc tạo shard rỗng. Cold-start tenant có "instant insert latency" cao bất thường vì segment mới phải warm up.

**Best practice**: pre-allocate shard pool, route tenant mới qua warm shard.

---

## 12. References

### Papers (real, citable)

- Malkov, Yu. A. & Yashunin, D. A. **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"** (IEEE TPAMI 2018, arXiv:1603.09320). https://arxiv.org/abs/1603.09320
- Jégou, H., Douze, M., Schmid, C. **"Product Quantization for Nearest Neighbor Search"** (IEEE TPAMI 2011). https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
- Ge, T., He, K., Ke, Q., Sun, J. **"Optimized Product Quantization"** (IEEE TPAMI 2013). https://kaiminghe.github.io/publications/pami13opq.pdf
- Guo, R. et al. **"Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)"** (ICML 2020, arXiv:1908.10396). https://arxiv.org/abs/1908.10396
- Subramanya, S. J. et al. **"DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"** (NeurIPS 2019). https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node
- Sivic, J. & Zisserman, A. **"Video Google: A Text Retrieval Approach to Object Matching in Videos"** (ICCV 2003). https://www.robots.ox.ac.uk/~vgg/publications/papers/sivic03.pdf
- Fu, C. et al. **"Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph (NSG)"** (VLDB 2017, arXiv:1707.00143). https://arxiv.org/abs/1707.00143
- Kalantidis, Y., Avrithis, Y. **"Locally Optimized Product Quantization (LOPQ)"** (CVPR 2014). https://openaccess.thecvf.com/content_cvpr_2014/papers/Kalantidis_Locally_Optimized_Product_2014_CVPR_paper.pdf
- Singh, A. et al. **"FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search"** (arXiv:2105.09613, 2021). https://arxiv.org/abs/2105.09613

### Engineering blogs

- **Pinecone Learning Hub** — Vector indexes, HNSW, filtering. https://www.pinecone.io/learn/
- **Pinecone "Filtering: The Missing WHERE Clause in Vector Search"** (2023). https://www.pinecone.io/learn/vector-search-filtering/
- **Weaviate documentation** — HNSW config, PQ compression, Hybrid search. https://weaviate.io/developers/weaviate
- **Weaviate blog "ANN Algorithms: HNSW"** (2022). https://weaviate.io/blog/ann-algorithms-hnsw-pq
- **Milvus documentation** — Index types, architecture. https://milvus.io/docs
- **Milvus blog "How Milvus 2.0 Boosts Performance"** (2022). https://milvus.io/blog
- **Faiss wiki (Meta)** — IVF, PQ, HNSW guides. https://github.com/facebookresearch/faiss/wiki
- **pgvector README** — index types, query tuning. https://github.com/pgvector/pgvector
- **ann-benchmarks.com** — Recall/QPS curves, comparison. https://ann-benchmarks.com/
- **Anthropic "Introducing Contextual Retrieval"** (Sep 2024). https://www.anthropic.com/news/contextual-retrieval
- **Supabase "pgvector 0.5: Faster semantic search with HNSW indexes"** (2023). https://supabase.com/blog/increase-performance-pgvector-hnsw
- **NVIDIA RAFT — GPU-accelerated vector search** (2023). https://developer.nvidia.com/blog/accelerating-vector-search-using-gpu-powered-indexes-with-nvidia-raft/
- **Vespa blog "Billion-scale vector search using hybrid HNSW-IF"** (2022). https://blog.vespa.ai/billion-scale-vector-search-with-vespa/

### Talks & conference videos

- Malkov, Yu. A. — original HNSW talk at SISAP / similar venue. Slides: https://github.com/nmslib/hnswlib
- Subramanya, S. J. — **"DiskANN"** talk at NeurIPS 2019. https://nips.cc/Conferences/2019
- Guo, R. — **"ScaNN: Efficient Vector Similarity Search"** (Google AI / ICML 2020 talk). https://icml.cc/virtual/2020
- Frank Liu (Zilliz) — **"Milvus: Building a Modern Vector Database"** (Linux Foundation 2023). https://www.youtube.com/c/MilvusVectorDB
- Edo Liberty (Pinecone CEO) — **"Vector Databases"** keynote at multiple venues (2023-2024). https://www.pinecone.io/learn/

### Books / longer reads

- Marko Tkalcic, Ivan Vasilev — **"Vector Databases: A Beginner's Guide"** (online, 2023).
- Mike Lewis et al. on RAG (Lewis 2020) — original RAG paper, important context. arXiv:2005.11401.

---

## Appendix A — Vocabulary recap (VI ↔ EN)

| EN | VI / khi nào dùng |
|---|---|
| ANN (approximate nearest neighbor) | ANN — luôn giữ EN |
| recall@k | recall@k — IR metric, giữ EN |
| QPS | QPS |
| HNSW | HNSW — Hierarchical Navigable Small World |
| IVF-PQ | IVF-PQ — Inverted File + Product Quantization |
| inverted list | inverted list |
| Voronoi cell | Voronoi cell |
| codebook / centroid | codebook / centroid |
| subspace / subvector | subspace / subvector |
| residual | residual |
| asymmetric distance | asymmetric distance |
| lookup table (LUT) | lookup table (LUT) |
| greedy descent | greedy descent |
| nprobe / nlist | nprobe / nlist |
| efSearch / efConstruction | efSearch / efConstruction |
| filter pushdown | filter pushdown |
| storage-compute separation | storage-compute separation / tách lưu trữ và compute |
| WAL (write-ahead log) | WAL |
| segment / sealed segment | segment / sealed segment |
| sharding / replication | sharding / replication |
| hybrid search | hybrid search |
| RRF (Reciprocal Rank Fusion) | RRF |

---

## Appendix B — Hyperparameter cheat sheet

**HNSW — starting points cho production (verified bằng ann-benchmarks 2024)**:

| Scenario | M | efConstruction | efSearch | Recall@10 expected |
|---|---|---|---|---|
| 1M vectors, 768-dim, balanced | 16 | 200 | 50-100 | ~0.95 |
| 10M vectors, high recall | 32 | 500 | 200 | ~0.98 |
| 10M vectors, latency-critical | 12 | 100 | 30 | ~0.92 |
| 100M vectors with PQ | 16 | 200 | 100 | ~0.93 |

**IVF-PQ — starting points**:

| Scenario | nlist | M (PQ) | k (centroids) | nprobe | Recall@10 |
|---|---|---|---|---|---|
| 1M, balanced | 1024 | 8 | 256 | 16 | ~0.85 |
| 10M, balanced | 4096 | 16 | 256 | 32 | ~0.92 |
| 100M, memory-critical | 16384 | 8 | 256 | 64 | ~0.88 |
| 1B, large scale | 65536 | 16 | 256 | 128 | ~0.90 |

**DiskANN — starting**: degree=64, alpha=1.2, search_list_size=100. (DiskANN paper recommended defaults.)

**ScaNN — starting**: num_leaves=√N to 4√N, leaves_to_search=N/100, anisotropic_quantization_threshold=0.2. (ScaNN README defaults.)

---

## Appendix C — Mini benchmark scaffold

Đây là pseudo-code cho việc benchmark recall/QPS trên dataset của bạn. Đừng tin chỉ benchmark public — workload thực tế khác nhau.

```python
import time
import numpy as np

def benchmark_index(index, queries, ground_truth, k=10, name=""):
    """
    queries: (Nq, d) array
    ground_truth: (Nq, k) ids of true top-k cho mỗi query (bằng brute-force)
    Trả về dict {recall, qps, latency_p50, latency_p99}
    """
    latencies = []
    recalls = []
    for q, gt in zip(queries, ground_truth):
        t0 = time.perf_counter()
        ids = index.search(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
        recalls.append(len(set(ids) & set(gt)) / k)
    return {
        "name": name,
        "recall@10": np.mean(recalls),
        "qps": 1000 / np.mean(latencies),
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p99_ms": np.percentile(latencies, 99),
    }


# Sweep hyperparameter
results = []
for ef in [30, 50, 100, 200, 500]:
    index.set_ef(ef)
    r = benchmark_index(index, queries, ground_truth, name=f"hnsw_ef{ef}")
    results.append(r)

# Plot recall vs QPS — đường cong này quyết định operating point
```

Quy ước: chạy mỗi config ít nhất 10K queries để latency tail ổn định. Warmup 1000 queries đầu.

---

## Appendix D — Common pitfalls checklist

- [ ] Distance metric của index match metric của embedding model?
- [ ] Vector normalization (L2 normalize) trước insert nếu dùng cosine?
- [ ] Filter cardinality test (5%, 1%, 0.1%) — recall stable không?
- [ ] ef / nprobe tune online, không hard-code default?
- [ ] Memory plan với buffer 2x lý thuyết?
- [ ] Replication > 1 cho production?
- [ ] WAL persistence test (kill node, recover)?
- [ ] Embedding model version pin (swap = re-index)?
- [ ] Recall monitoring online via brute-force sampling?
- [ ] Hybrid evaluated trước commit dense-only?
- [ ] Cross-encoder reranker considered?
- [ ] Cost projection trên top corpus size 12 tháng?

---

> **Cross-references**:
> - [S1-03 Pinterest PinSage](../01-foundations/S1-03_pinterest_pinsage_graph_retrieval.md) — case study sử dụng HNSW + IVF-PQ hybrid tier để serve 3B PinSage embeddings.
> - [S3-01 vLLM](../03-modern-stack/S3-01_vllm_paged_attention_continuous_batching.md) — KV cache paging tương đồng concept inverted list / block table; cả hai đều giải bài toán memory management không-contiguous.
> - [S3-02 Production RAG](../03-modern-stack/S3-02_production_rag_system_architecture.md) — vector DB ở lớp dense retrieval; hybrid search và RRF được mô tả ở đó chi tiết hơn.
> - **Planned**: S3-04 (Agent infrastructure), S4-04 (LLM cost / capacity planning) — sẽ tham chiếu lại vector DB cost trong overall system cost.
