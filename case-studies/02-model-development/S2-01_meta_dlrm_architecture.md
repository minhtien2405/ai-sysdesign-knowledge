# S2-01 — Meta DLRM (Deep Learning Recommendation Model)

> **Scope**: Model Development & Training
> **Difficulty**: Intermediate
> **Tags**: recommendation systems, embedding tables, sparse features, model parallelism, FBGEMM, CTR prediction
> **Primary sources**: Naumov et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems" (arXiv 1906.00091, 2019), Mudigere et al. "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models" (ISCA 2022), Meta Engineering blogs.

---

## 1. Tổng quan (Overview)

DLRM (Deep Learning Recommendation Model) là **architecture template** mà Meta (Facebook) công bố năm 2019 để mô hình hoá personalization/recommendation workloads. Đây không chỉ là một model — đó là một **family of models** thống nhất cách Meta xử lý các tasks như:

- **News Feed ranking** trên Facebook
- **Instagram Feed/Reels recommendation**
- **Ads ranking** (CTR/CVR prediction) — workload tiêu thụ phần lớn ML compute của Meta
- **Marketplace, Groups, Pages recommendations**

DLRM nổi tiếng vì hai đặc điểm:
1. **Massive sparse features** — embedding tables có thể chiếm **hàng terabyte memory** (tổng hợp), với hàng tỷ entries.
2. **Hybrid parallelism** trong training — **model parallel** cho embedding tables (vì quá lớn để fit một GPU), **data parallel** cho dense MLP layers.

### Tại sao DLRM là case study tốt cho model development?

- Minh hoạ **trade-off classical**: dense features vs sparse features, factorization machines vs deep models, interaction order, embedding dimension.
- Là **open-source benchmark** chuẩn cho industry (MLPerf Recommendation track dùng DLRM).
- Cho thấy **production architecture không chỉ là model code** — phải hiểu memory layout, partitioning strategy, kernel optimization (FBGEMM).
- Ads/recsys consume ~50%+ ML compute ở big tech như Meta/Google → hiểu DLRM là hiểu một mảng huge của industry.

### Lịch sử ngắn

- **2016**: Wide & Deep (Google) — combine memorization (wide linear) + generalization (deep MLP) cho Google Play recommendations.
- **2017**: DeepFM (Huawei) — combine factorization machines (FM) với deep network.
- **2019**: Meta release DLRM as a generalization + simplification → unified framework với explicit feature interaction via dot products.
- **2020+**: Variants — DLRM-v2, DHEN (Deep & Hierarchical Ensemble Network, Meta 2022), embeddings-only sparse architectures.

---

## 2. System Requirements

### 2.1 Functional requirements

- Predict **probability of user engagement** (click, like, conversion) cho một (user, item, context) tuple.
- Hỗ trợ **hundreds of sparse features** (user_id, item_id, age_bucket, country, device_type, page_category, ...) với cardinality từ vài chục đến hàng tỷ.
- Hỗ trợ **dense features** (numerical) — counts, ratios, time features, image/text embeddings.
- Score **hundreds to thousands candidates per request** trong ads/feed ranking.

### 2.2 Non-functional requirements

| Metric | Approximation (public) | Notes |
|---|---|---|
| Total embedding table memory | **TB-scale** (single model) | Tổng hợp tất cả sparse features |
| Largest single table | **vài chục GB** (e.g. user_id hash with billions of entries) | |
| Dense MLP size | **~MB** | Bottom + top MLP, không lớn so với embeddings |
| Training data | **petabyte** per training run | Multi-day logs |
| Training time | **hours to days** trên cluster với hundreds of GPUs/TPUs | Mudigere et al. 2022 report ZionEX hệ thống của Meta |
| Inference latency budget | **vài ms per inference** | Batch 100-1000 candidates trong 10-100 ms |
| Training throughput | **millions of samples/sec** trên cluster | |

### 2.3 Constraints quan trọng

- **Memory constraint là dominant**: embedding tables quá lớn cho một GPU (GPU memory ~80 GB max trên A100, table có thể 100GB+).
- **Embedding lookup là sparse + irregular** — pattern access không đều → khó GPU-friendly.
- **Communication overhead** trong distributed training là bottleneck — embedding gradients phải all-to-all giữa các workers.

---

## 3. High-level Architecture

DLRM model structure (Naumov et al. 2019):

```
   Dense features                              Sparse features
       (numerical)                  (categorical, multi-hot, sequences)
            │                                       │
            ▼                                       ▼
    ┌───────────────┐                  ┌──────────────────────────┐
    │  Bottom MLP   │                  │  Embedding tables        │
    │  (Dense tower)│                  │  - 1 table per sparse    │
    │               │                  │    feature               │
    │  e.g. 3 FC    │                  │  - Lookup + (optional)   │
    │  layers       │                  │    sum/avg pooling for   │
    └───────┬───────┘                  │    multi-hot features    │
            │                          └─────────────┬────────────┘
            │ dense vec d                            │  m embeddings, each dim d
            │                                        │
            └─────────────┬──────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────────┐
            │  Feature interaction layer   │
            │  (pairwise dot products      │
            │   between all m+1 vectors)   │
            └──────────────┬───────────────┘
                           │  (m+1) choose 2 scalars + dense vec
                           ▼
                  ┌────────────────┐
                  │   Top MLP      │
                  │   (Predictor)  │
                  │   sigmoid out  │
                  └────────┬───────┘
                           ▼
                   P(click | x)
```

**Key insight**: feature interaction layer là **dot product giữa tất cả các cặp embeddings + dense vector**. Đây là cách DLRM **explicitly model second-order interactions** (giống FM), nhưng feature input của interaction là embedding (giống DeepFM), và top MLP học thêm higher-order interactions trên top.

---

## 4. Deep dive các components chính

### 4.1 Sparse features và embedding tables

Đây là **trái tim** của DLRM. Mỗi sparse feature có một embedding table:

```
Sparse feature i → categorical ID (e.g. user_id=12345, item_id=678) 
                → table lookup → embedding vector v_i ∈ R^d
```

#### Cardinality và memory math

Ví dụ một production model có 100 sparse features. Một số table tiêu biểu:

| Feature | Cardinality | Embedding dim | Table memory |
|---|---|---|---|
| user_id | 3 × 10^9 | 64 | ~768 GB (FP32) |
| item_id | 10^8 | 128 | ~51 GB |
| ad_creative_id | 10^7 | 128 | ~5 GB |
| country | 200 | 16 | ~12 KB |
| device_type | 50 | 8 | ~1.6 KB |
| ... | | | |

**Tổng** dễ vượt **1 TB** memory. Không GPU đơn lẻ nào fit được. → **Phải shard tables across nhiều GPUs (model parallelism)**.

#### Hash trick cho high-cardinality features

Với features như user_id 3 × 10^9 entries, full table là impractical. Solution: **hash trick** — hash ID xuống một bucket nhỏ hơn (e.g. 10^7 buckets):

```
user_id_hash = hash(user_id) % 10_000_000
embedding = table[user_id_hash]
```

Trade-off: hash collisions → multiple users share embedding. Nhưng:
- Với 10^7 buckets, collision rate cho 3 × 10^9 users là ~300x (mỗi bucket trung bình 300 users).
- Model học **average representation** của những users này — vẫn tốt nếu collision random.
- Memory giảm từ 768GB → 2.5GB (300x savings).

Trick này được gọi là **feature hashing** hay **hashing trick**. Đôi khi dùng **multi-hash** (mehrhash, mỗi feature có k hashes, embedding = sum) để giảm collision impact.

#### Multi-hot và sequence features

Một số features là **multi-hot** (e.g. user đã xem 50 videos gần đây — 50 IDs). Lookup tất cả 50 embeddings rồi **sum/mean pooling**:

```python
def multi_hot_lookup(table, ids):  # ids: list of categorical IDs
    embeddings = [table[i] for i in ids]
    return sum(embeddings) / len(embeddings)  # mean pooling
```

Pooling có thể là sum, mean, max, hoặc weighted (attention-based — như DIN từ Alibaba, S2-03).

### 4.2 Dense MLP (bottom + top)

Hai dense MLP trong DLRM:

#### Bottom MLP (Dense tower)

Process **dense features** (numerical: counts, ratios, time, image/text embeddings nếu có):

```
Bottom MLP: dense_features → FC(512) → ReLU → FC(256) → ReLU → FC(d) 
                              (output dim = embedding dim d, để align với embeddings)
```

Bottom MLP transform dense features thành một vector có cùng dim với embeddings → đưa vào interaction layer.

#### Top MLP (Predictor)

Sau khi có feature interactions, top MLP học higher-order combinations và predict probability:

```
Top MLP: interaction_output → FC(1024) → ReLU → FC(1024) → ReLU → FC(512) → ReLU → FC(1) → sigmoid
```

Top MLP thường lớn hơn bottom MLP, nhưng vẫn nhỏ so với embedding tables.

### 4.3 Feature interaction layer

Đây là phần **đặc trưng của DLRM**, khác với pure-MLP architecture.

Có m sparse features → m embeddings v_1, ..., v_m, mỗi dim d.
Bottom MLP output dense vector v_0 (cũng dim d).

**Interaction**: dot products giữa **tất cả các cặp** trong {v_0, v_1, ..., v_m}:

```
interactions = [v_i · v_j  for i in range(m+1) for j in range(i+1, m+1)]
             # số phần tử = C(m+1, 2) = (m+1)*m/2
```

Sau đó concat với dense vector v_0:

```
top_mlp_input = concat([v_0, interactions])
```

#### Tại sao dot products?

- **Second-order interactions explicit** — giống Factorization Machines (Rendle 2010). FM cho thấy second-order interactions là quan trọng nhất cho CTR prediction.
- **Efficient** — dot product là O(d), tổng m^2 dot products = O(m^2 * d). Cho m=100, d=64 → 100^2 * 64 = 640K ops, very cheap.
- **Inductive bias** — buộc model phải học rằng interactions là pairwise; tránh top MLP phải tự figure ra.

#### Alternative interaction designs

| Design | Source | Note |
|---|---|---|
| Pairwise dot product (DLRM) | Naumov et al. 2019 | Simple, explicit |
| Concat embeddings + deep MLP (no explicit interaction) | Cheng et al. 2016 (Wide & Deep deep side) | MLP tự học interaction nhưng implicit |
| Factorization Machine (FM) | Rendle 2010 | Linear + pairwise FM, no deep |
| DeepFM | Guo et al. 2017 | FM + DNN parallel |
| Cross Network (DCN) | Wang et al. 2017 | Multi-order interactions via cross layers |
| Outer product (NCF / xDeepFM) | He 2017 / Lian 2018 | Heavier but more expressive |
| DHEN (Meta 2022) | Meta | Hierarchical ensemble of interactions |

DLRM chọn dot product vì đơn giản + đủ tốt + efficient.

### 4.4 Training infrastructure — model parallelism

Đây là phần **engineering-heavy** của DLRM, thường được coverage trong Mudigere et al. (Meta ZionEX paper, ISCA 2022).

#### Vấn đề

- Embedding tables tổng TB → không fit 1 GPU.
- Dense MLP nhỏ → có thể replicate trên mọi GPU (data parallel).

#### Solution: Hybrid parallelism

```
                    ┌───────────────────────────────────────────┐
                    │            Training cluster                │
                    │     N GPUs (e.g., N=128 trên ZionEX)      │
                    └───────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  ┌──────────┐                  ┌──────────┐                  ┌──────────┐
  │  GPU 0   │                  │  GPU 1   │      ...         │  GPU N-1 │
  │          │                  │          │                  │          │
  │ Embed    │                  │ Embed    │                  │ Embed    │
  │ table    │                  │ table    │                  │ table    │
  │ shard 0  │                  │ shard 1  │                  │ shard N-1│
  │ (model   │                  │ (model   │                  │ (model   │
  │  parallel)│                 │  parallel)│                 │  parallel)│
  │          │                  │          │                  │          │
  │ Dense MLP│                  │ Dense MLP│                  │ Dense MLP│
  │ (replica)│                  │ (replica)│                  │ (replica)│
  │ (data    │                  │ (data    │                  │ (data    │
  │  parallel)│                 │  parallel)│                 │  parallel)│
  └──────────┘                  └──────────┘                  └──────────┘
```

**Mỗi training step**:

1. **Sample batch B** — chia thành N micro-batches, mỗi GPU một micro-batch (data parallel cho dense).
2. **Embedding lookup**: mỗi GPU có sparse IDs từ micro-batch của mình. Vì IDs có thể nằm ở table shard trên GPU khác → cần **all-to-all communication**:
   - Mỗi GPU gửi requests cho IDs nó cần đến GPU chứa shard tương ứng.
   - GPU đó lookup và gửi embedding vectors về.
   - Đây là **expensive collective communication step**.
3. **Forward pass**: Bottom MLP + interaction + Top MLP (data parallel trên mỗi GPU).
4. **Backward pass**: gradients flow ngược.
5. **Embedding gradient update**: cần all-to-all một lần nữa để route gradients về đúng shard.
6. **Dense gradient update**: standard all-reduce (data parallel).

#### Sharding strategy

Sharding sao cho **balanced** là vấn đề lớn. Strategies:

| Strategy | Idea | Pros | Cons |
|---|---|---|---|
| **Table-wise** | Mỗi table fit trọn 1 GPU | Đơn giản | Imbalance nếu table sizes khác nhau lớn |
| **Row-wise** | Shard mỗi table by row | Balance tốt hơn | Cần access pattern phù hợp |
| **Column-wise** | Shard embedding dim | Tốt cho large embedding dim | Phức tạp |
| **Hybrid** | Mix các strategies | Tối ưu nhất | Complex tooling |

Meta dùng **TorchRec** (open-source) cho automatic sharding với optimization-based partitioner.

### 4.5 FBGEMM — kernel optimization

Embedding lookup trông đơn giản (chỉ là gather), nhưng:
- Lookup pattern không regular → cache-unfriendly.
- Pooling (sum/mean over multiple IDs) cần aggregation.
- Multi-table có thể fuse để giảm kernel launch overhead.

**FBGEMM** (Facebook General Matrix Multiplication) là Meta's library tối ưu cho:
- Quantized embedding lookups (INT8/INT4 embeddings để giảm memory).
- Fused multi-table batched lookups.
- Optimized pooling.

FBGEMM_GPU mở rộng cho GPU. Performance impact: **2-10x speedup** so với naive PyTorch implementation (per Meta engineering blog 2020-2021).

### 4.6 Inference / serving

Inference DLRM khác training:
- **Latency budget rất chặt** (vài ms cho ranking).
- **Batch size nhỏ hơn** training (vì cần score realtime).
- Có thể **quantize embeddings** (INT8) để giảm memory + latency.
- Embedding tables thường **không full replicate** trên inference machines — vì quá lớn. Thay vào đó, có thể dùng:
  - Dedicated embedding servers (parameter server style).
  - Replicate frequently-accessed embeddings, sharding rare ones.

```python
# Pseudo-code inference
def dlrm_inference(dense_features, sparse_ids):
    # 1. Embedding lookup (có thể từ remote embedding server)
    embeddings = []
    for table_id, ids in enumerate(sparse_ids):
        emb = embedding_server.lookup(table_id, ids)  # có thể remote RPC
        embeddings.append(emb)
    
    # 2. Bottom MLP
    dense_vec = bottom_mlp(dense_features)
    
    # 3. Interaction (all pairwise dot products)
    all_vecs = [dense_vec] + embeddings
    interactions = []
    for i in range(len(all_vecs)):
        for j in range(i+1, len(all_vecs)):
            interactions.append(dot(all_vecs[i], all_vecs[j]))
    
    # 4. Top MLP
    top_input = concat([dense_vec] + interactions)
    return sigmoid(top_mlp(top_input))
```

---

## 5. Trade-offs & Design decisions

### 5.1 Embedding dim — bigger is better?

Không hẳn. Trade-off:

| Embedding dim | Pros | Cons |
|---|---|---|
| Small (8-16) | Memory tiết kiệm, fast lookup | Capacity thấp cho high-cardinality features |
| Medium (32-64) | Balance | |
| Large (128-256) | Capacity cao | Memory blow up, có thể overfit |

**Insight thực tế**: high-cardinality features (user_id, item_id) thường cần dim lớn hơn. Low-cardinality features (country, device) chỉ cần dim nhỏ. → **Variable embedding dim per feature** (Meta dùng technique này — paper "Mixed Dimension Embeddings" 2019).

### 5.2 Hash trick vs full table

| Approach | Pros | Cons |
|---|---|---|
| **Full table** | Mỗi ID có embedding riêng, capacity max | Memory huge cho high-cardinality |
| **Hash trick** | Memory bounded | Collisions, slight accuracy loss |
| **Frequency-based** | Top-K frequent IDs full, rest hash | Best of both | Implementation phức tạp |

Production thường dùng **frequency-based** — embeddings cho IDs xuất hiện > threshold lần được học riêng, rest dùng hash bucket.

### 5.3 Pairwise dot product vs deeper interactions

DLRM chỉ dùng pairwise. Có thể model higher-order interactions (3-way, 4-way) nhưng:

| Order | Cost | Benefit | Verdict |
|---|---|---|---|
| 2nd order (DLRM) | O(m^2 d) | Capture most useful interactions | Default |
| 3rd order | O(m^3 d) | Marginal gain | Thường không worth |
| Implicit via deep MLP | Cheaper but implicit | Top MLP đã học implicit higher-order |  |

DLRM rely on **Top MLP để học higher-order interactions implicitly** sau khi có explicit 2nd-order signal. Đây là design trade-off chính.

### 5.4 Training: hybrid parallelism vs pure data parallel

| Approach | Pros | Cons |
|---|---|---|
| **Pure data parallel** | Đơn giản, framework support tốt | Không khả thi với TB embeddings |
| **Pure model parallel** | Memory scale | Slow communication, low utilization |
| **Hybrid (DLRM chọn)** | Best memory + compute trade-off | Implementation complex (TorchRec/FBGEMM help) |

### 5.5 Online learning vs batch retraining

Production reco models cần fresh — user behavior changes nhanh. Options:

- **Daily batch retrain**: simple, stable. Models hơi stale (1 ngày).
- **Online learning (incremental updates)**: fresh ngay, nhưng instable, hard to debug.
- **Streaming features + batch model**: middle ground — model batch, features (counters) cập nhật realtime. Đây là choice phổ biến.

DLRM paper không cover online learning chi tiết — TikTok Monolith (S1-02) là case study tốt cho online learning.

---

## 6. Lessons learned & Best practices

1. **Trong recsys, embedding tables thường > 99% parameters của model**. Tối ưu memory + lookup cost của embeddings impact lớn hơn tối ưu MLP.

2. **Bắt đầu với DLRM baseline trước khi thử fancy architectures**. DLRM đơn giản, well-understood, performance baseline tốt. Chỉ deviate khi có lý do rõ ràng (e.g. sequence modeling cho user history → cần Transformer).

3. **Hash trick + frequency-based embeddings là default cho high-cardinality**. Đừng cố giữ full table cho features có >10^7 unique values.

4. **Variable embedding dim per feature** — không phải mọi feature đều cần dim 64. Country dim 8 đủ; user_id cần dim 128.

5. **Đầu tư vào FBGEMM-equivalent kernels** nếu serving custom — naive PyTorch embedding lookup slow nhiều lần so với optimized kernels.

6. **Communication là bottleneck của distributed training DLRM** — embedding all-to-all chiếm 30-50% step time. → Tối ưu network (NVLink, InfiniBand), gradient compression, embedding caching.

7. **Quantize embeddings cho serving (INT8)** — accuracy loss thường < 0.1% AUC, memory giảm 4x, latency cải thiện đáng kể.

8. **Check training/serving skew** — bottom MLP có thể nhạy với feature scale; nếu pipeline training và serving compute features khác nhau (dù chỉ slightly), model degrade.

9. **Multi-task heads phía top MLP** — thường share embeddings + bottom MLP + interactions, chỉ split ở top heads (CTR, CVR, dwell time...). Xem S1-01 cho YouTube multi-task ranking.

10. **A/B test offline metrics có thể lie**. AUC tăng 0.01 offline có thể chỉ là noise. Production decision phải dựa vào A/B test với business metrics (revenue per user, retention).

---

## 7. References

### Papers

1. **Naumov et al. (Meta).** "Deep Learning Recommendation Model for Personalization and Recommendation Systems." arXiv:1906.00091, 2019. [Link](https://arxiv.org/abs/1906.00091) — paper gốc DLRM.
2. **Mudigere et al. (Meta).** "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models" (ZionEX). ISCA 2022. [arXiv:2104.05158](https://arxiv.org/abs/2104.05158) — infrastructure deep dive.
3. **Cheng et al. (Google).** "Wide & Deep Learning for Recommender Systems." DLRS 2016. [arXiv:1606.07792](https://arxiv.org/abs/1606.07792).
4. **Guo et al. (Huawei).** "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI 2017. [arXiv:1703.04247](https://arxiv.org/abs/1703.04247).
5. **Rendle.** "Factorization Machines." ICDM 2010 — paper foundational FM.
6. **Wang et al. (Google).** "Deep & Cross Network for Ad Click Predictions." ADKDD 2017. [arXiv:1708.05123](https://arxiv.org/abs/1708.05123).
7. **Ginart et al. (Stanford/Meta).** "Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems." 2019. [arXiv:1909.11810](https://arxiv.org/abs/1909.11810).
8. **DHEN (Meta).** "DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction." 2022.

### Engineering blogs / open source

9. **Meta AI Blog.** "Open-sourcing FBGEMM for state-of-the-art server-side inference." 2018-2020 series.
10. **TorchRec (PyTorch).** [github.com/pytorch/torchrec](https://github.com/pytorch/torchrec) — Meta's open-source library cho DLRM-style models.
11. **MLPerf Recommendation Benchmark** — DLRM là reference model. [mlcommons.org](https://mlcommons.org/).
12. **PyTorch DLRM repo.** [github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm) — official reference implementation.

### Related case studies (đọc tiếp)

- **S2-02 Wide & Deep / DeepFM / DCN evolution** — chi tiết các architecture predecessor.
- **S2-03 Alibaba DIN/DIEN** — user behavior sequence modeling, complement DLRM.
- **S1-01 YouTube reco** — multi-stage architecture mà DLRM-style ranking sẽ nằm ở stage 2.
- **S4-01 Uber Michelangelo** — feature store + platform để serve DLRM-style models ở production.

### Độ tin cậy

- DLRM paper 2019 + ZionEX paper 2022 là **chính thức từ Meta**, high confidence.
- Số liệu cụ thể (cardinality, latency, GPU count cho production models) **không được Meta public** — các con số trên là **approximations từ industry**.
- Mention về variable embedding dim, frequency-based hash là từ papers + blogs, có thể không phải technique chính xác Meta dùng trong production hiện tại.
