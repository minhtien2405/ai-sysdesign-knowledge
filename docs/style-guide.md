# Bilingual VI-EN Writing Style Guide

> Rules viết case study cho knowledge base này — đảm bảo consistency, tự nhiên, và đúng technical specificity.

## Nguyên tắc cốt lõi

Phong cách viết là **bilingual VI-EN tự nhiên**, KHÔNG phải dịch word-by-word từ English. Mục tiêu: AI/ML engineer Việt Nam đọc thấy dễ hiểu mà vẫn giữ được technical specificity.

## Khi nào giữ tiếng Anh

**LUÔN giữ nguyên tiếng Anh** cho 5 nhóm sau:

### 1. Tên hệ thống, sản phẩm, công ty
- Michelangelo, TFX, Borg, Spanner, Aurora
- vLLM, TGI, TensorRT-LLM, SGLang
- Google, Meta, Uber, ByteDance, Pinterest

### 2. Technical terms chuyên ngành
- embedding, feature store, two-tower model
- candidate generation, retrieval, ranking
- KV cache, PagedAttention, continuous batching
- model parallelism, data parallelism, tensor parallelism
- training/serving skew, online/offline inference

**Tuyệt đối KHÔNG dịch** kiểu:
- ❌ "vector nhúng" → ✅ "embedding"
- ❌ "bộ nhớ đệm khoá-giá trị" → ✅ "KV cache"
- ❌ "mô hình hai tháp" → ✅ "two-tower model" (chỉ dùng "hai tháp" khi đã expand lần đầu)
- ❌ "huấn luyện trước" → ✅ "pretraining"

### 3. Acronyms
CTR, CVR, AUC, NDCG, QPS, SLA, P99 latency, MAU, DAU, TTFT, TPOT, MMoE, FM, GNN, GCN, RAG, HNSW, IVF-PQ, BM25, RRF, MMR, CUPED, SRM, MDE, MAB, HTE, ATE, FDR, PSI.

Khi acronym xuất hiện lần đầu trong document → expand: "CTR (Click-Through Rate)".

### 4. Code, API names, framework names
- PyTorch, TensorFlow, JAX, Triton, FAISS, ScaNN
- Spark, Flink, Kafka, Cassandra, Redis
- Kubernetes, Docker, gRPC, Protobuf

### 5. Tên paper, algorithm, technique
- DLRM, Wide & Deep, DeepFM, DCN, DIN, DIEN
- Transformer, BERT, GPT, LLaMA
- DeepWalk, node2vec, GraphSAGE, PinSage
- AWQ, GPTQ, SmoothQuant

## Khi nào dùng tiếng Việt

**Dùng tiếng Việt** cho 4 nhóm:

### 1. Câu giải thích, phân tích, so sánh

✅ Đúng: "YouTube sử dụng kiến trúc two-tower model để xử lý candidate generation, trong đó user tower và item tower được train riêng biệt nhưng share chung embedding space."

❌ Sai (over-translate): "YouTube sử dụng kiến trúc hai-tháp để xử lý quá trình sinh ứng viên..."

### 2. Connecting words, transitions

- "Tuy nhiên", "Mặt khác", "Trong khi đó", "Để giải quyết bài toán này"
- "Lý do đằng sau", "Hệ quả là", "Cụ thể"
- "Đáng chú ý", "Quan trọng", "Cần lưu ý"

### 3. Intuition và lý do thiết kế

✅ "Lý do TikTok chọn online learning thay vì batch training là vì user interest evolution rất nhanh — chỉ vài phút có thể trend mới xuất hiện."

### 4. Đánh giá ưu/nhược điểm

✅ "Approach này có ưu điểm là implementation đơn giản, nhưng nhược điểm là không scale được khi corpus vượt 1M items."

## Ví dụ câu mẫu chuẩn

### Recommendation systems
> "YouTube sử dụng kiến trúc two-tower model để xử lý candidate generation, trong đó user tower và item tower được train riêng biệt nhưng share chung embedding space."

### LLM serving
> "vLLM giải quyết memory fragmentation của KV cache bằng kỹ thuật PagedAttention, giảm waste từ 60-80% xuống dưới 5% — đây là innovation chính cho phép throughput cao gấp 24x so với HuggingFace Transformers."

### Production
> "Khi QPS tăng đột biến, hệ thống trigger autoscaling dựa trên P99 latency thay vì CPU utilization, vì latency phản ánh chính xác hơn user experience."

### A/B testing
> "CUPED (Controlled-experiment Using Pre-Experiment Data) giảm variance của metric bằng cách regress trên pre-experiment covariate, hiệu quả tăng statistical power tương đương 50-80% sample size."

## Câu chưa đạt — và cách fix

### Vấn đề 1: Over-translation

❌ "Khi truy vấn-mỗi-giây tăng đột biến, hệ thống kích hoạt tự-mở-rộng dựa trên độ-trễ-phân-vị-99 thay vì sử dụng CPU."

✅ "Khi QPS tăng đột biến, hệ thống trigger autoscaling dựa trên P99 latency thay vì CPU utilization."

### Vấn đề 2: Toàn tiếng Anh (mất bilingual flavor)

❌ "vLLM solves KV cache fragmentation through paging, reducing waste from 60-80% to under 5%."

✅ "vLLM giải quyết KV cache fragmentation bằng paging, giảm waste từ 60-80% xuống dưới 5%."

### Vấn đề 3: Code-switching không tự nhiên

❌ "The system uses feature store để store features cho training."

✅ "Hệ thống dùng feature store để lưu features cho training."

## Quy ước viết số liệu

- **Năm**: phải ghi rõ khi cite paper/blog: "DLRM paper 2019", "based on Pinterest 2023 blog post".
- **Đơn vị quốc tế**: ms, μs, s, GB, TB, PB, QPS, FLOPs — không dịch.
- **Approximation**:
  - "approximately 200ms" hoặc "khoảng 200ms"
  - "based on public information, internal numbers may differ"
  - "vài chục GB" (tiếng Việt OK cho approximation mềm)
- **Range**: dùng dash — "60-80%", "100-200ms", "1-10M users".

## Format markdown

### Headings hierarchy

- `#` — Document title (1 lần duy nhất)
- `##` — Section chính (Overview, Requirements, Architecture, …)
- `###` — Subsection trong deep dive
- `####` — Sub-subsection (hiếm khi cần)

### Tables

**Comparison table** chuẩn:

```markdown
| System | Approach | Pros | Cons | Use case |
|---|---|---|---|---|
| Option A | … | … | … | … |
| Option B | … | … | … | … |
```

### Code blocks

- Luôn specify language: ```python, ```sql, ```bash, ```yaml, ```text (cho ASCII diagrams).
- Pseudo-code: Python-flavored, **comment bằng tiếng Việt** cho intuition.

```python
def random_walk_sample(graph, source, T=50):
    """Random walk từ source, trả về top-T neighbors theo visit count."""
    visit_count = defaultdict(int)
    for _ in range(N_WALKS):
        # Mỗi walk dài tối đa MAX_LEN bước
        current = source
        for _ in range(MAX_LEN):
            current = random.choice(graph.neighbors(current))
            visit_count[current] += 1
    return top_k(visit_count, T)
```

### ASCII diagrams

Dùng cho architecture/data flow. Ví dụ:

```text
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ User request │ ───▶ │  Retrieval   │ ───▶ │   Ranking    │
└──────────────┘      └──────────────┘      └──────────────┘
                              │                     │
                              ▼                     ▼
                       ┌──────────────┐      ┌──────────────┐
                       │  ANN index   │      │  Feature     │
                       │  (HNSW)      │      │  store       │
                       └──────────────┘      └──────────────┘
```

### Math formulas

Inline: `$f(x) = Wx + b$`
Block:
```text
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$
```

## Citation format

### Papers
> "PinSage paper (Ying et al. KDD 2018, arXiv:1806.01973)"

### Engineering blogs
> "Source: Pinterest Engineering Blog (2023), URL: https://medium.com/pinterest-engineering/..."

### Conference talks
> "vLLM SOSP 2023 talk: https://www.youtube.com/watch?v=..."

## Cross-reference giữa case studies

Khi đề cập đến case study khác trong repo, dùng relative link:

```markdown
Xem thêm chi tiết về feature store trong [S4-01 Michelangelo](../04-production/S4-01_uber_michelangelo_feature_store.md).
```

Hoặc dùng tham chiếu inline:
> "Architecture này tương tự two-tower model trong S1-01 (YouTube reco)."

## Checklist trước khi finalize case study

- [ ] Technical terms giữ nguyên tiếng Anh? (check vs `terminology.md`)
- [ ] Giải thích bằng tiếng Việt tự nhiên, không dịch word-by-word?
- [ ] Concrete numbers + năm cite?
- [ ] ASCII diagram cho mỗi architectural concept?
- [ ] Pseudo-code cho key algorithms?
- [ ] Comparison tables ở phần Trade-offs?
- [ ] References có URL đầy đủ?
- [ ] Cross-references đến case studies liên quan?
- [ ] No hallucinated names/numbers?
- [ ] 7 sections structure đầy đủ?
- [ ] Length 1500-2000 dòng (từ S1-03 trở đi)?
