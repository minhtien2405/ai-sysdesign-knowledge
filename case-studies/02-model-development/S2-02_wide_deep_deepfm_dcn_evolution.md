---
id: S2-02
title: "Wide & Deep / DeepFM / DCN: Tiến hoá Architecture cho CTR Prediction"
summary: "Tiến hoá architecture cho CTR prediction từ Google Play → Huawei → DCN-V2, foundation cho mọi ranking architecture sau."
slug: wide_deep_deepfm_dcn_evolution
scope: 2
scope_name: model-development
difficulty: intermediate
status: done
tags:
  - CTR prediction
  - feature interaction
  - factorization machines
  - deep learning
  - recommendation
  - ads ranking
cross_refs: [S1-01, S2-01]
created: 2026-05-20
last_validated: 2026-05-20
line_count: 676
---

# S2-02 — Wide & Deep / DeepFM / DCN: Tiến hoá Architecture cho CTR Prediction

> **Scope**: Model Development & Training
> **Difficulty**: Intermediate
> **Tags**: CTR prediction, feature interaction, factorization machines, deep learning, recommendation, ads ranking
> **Primary sources**:
> - Cheng et al. "Wide & Deep Learning for Recommender Systems" (DLRS 2016, arXiv 1606.07792) — Google Play.
> - Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (IJCAI 2017, arXiv 1703.04247) — Huawei.
> - Wang et al. "Deep & Cross Network for Ad Click Predictions" (ADKDD 2017, arXiv 1708.05123) — Google.
> - Wang et al. "DCN V2: Improved Deep & Cross Network..." (WWW 2021, arXiv 2008.13535) — Google.

---

## 1. Tổng quan (Overview)

Trước khi industry tiến đến các kiến trúc phức tạp như DLRM (xem [S2-01](S2-01_meta_dlrm_architecture.md)), DIN/DIEN, transformer-based ranking, có **3 generation models nền tảng** mà mọi AI engineer làm reco/ads nên hiểu rõ:

1. **Wide & Deep (Google, 2016)** — kết hợp memorization (wide linear part) và generalization (deep MLP part).
2. **DeepFM (Huawei, 2017)** — thay wide part bằng Factorization Machine (FM) để tự động học second-order interactions, không cần feature engineering thủ công.
3. **DCN / DCN-V2 (Google, 2017 / 2021)** — Cross Network học higher-order feature interactions một cách explicit và có thể stack nhiều layers.

Đây không phải các model "lỗi thời" — chúng vẫn là **production baseline** ở nhiều công ty, và là **building blocks** cho các architecture phức tạp hơn (DLRM thực chất có thể xem như generalization của FM + DNN, DCN-V2 vẫn được Google dùng trong YouTube ranking đến 2024 theo các talks gần đây).

### Tại sao cần học evolution này?

- **Foundation cho mọi architecture sau**: hiểu Wide & Deep → hiểu DLRM → hiểu interaction networks → hiểu transformer-based ranker.
- **Trade-off framework**: mỗi generation giải quyết một limitation cụ thể của generation trước → là case study mẫu về **incremental improvement** trong ML research.
- **Production reality**: ở các công ty không có infra như Meta/Google, Wide & Deep hoặc DeepFM vẫn là baseline rất mạnh, ROI cao trên effort.
- **Interview material**: câu hỏi "compare Wide & Deep vs DeepFM vs DCN" xuất hiện rất nhiều trong ML system design interviews ở big tech.

### Bối cảnh business chung — bài toán CTR prediction

Cho một tuple `(user, item, context)`:
- **Input**: tens to hundreds of features (sparse categorical + dense numerical + cross features).
- **Output**: `P(click | user, item, context)` ∈ [0, 1].
- **Usage**: dùng để rank candidates trong ads/feed/search. Predicted CTR (pCTR) thường multiply với bid (cho ads) hoặc dùng làm input cho objective function tổng hợp (cho feed/search).

Trước Wide & Deep, các approach phổ biến là:
- **Logistic Regression với manual cross features**: scalable, interpretable, nhưng feature engineering thủ công cực kỳ tốn effort.
- **GBDT (Gradient Boosting Decision Trees)** như XGBoost: tốt cho dense features, không scale tốt cho sparse high-cardinality features.
- **Factorization Machines (Rendle 2010)**: học second-order interactions automatic nhưng limited capacity, không capture được higher-order non-linear patterns.

3 models trong case study này về cơ bản là câu trả lời cho câu hỏi: **làm sao kết hợp ưu điểm của linear/FM (memorization, low-order interactions) với deep networks (generalization, higher-order non-linear interactions)?**

---

## 2. System Requirements

### 2.1 Functional requirements

- Predict pCTR (hoặc pCVR — conversion rate) cho một (user, item, context) instance.
- Hỗ trợ **mixed feature types**: sparse categorical (user_id, item_id, age_bucket, country, ...), dense numerical (counts, ratios), multi-hot (list of user historical items).
- Hỗ trợ **online inference** với latency budget vài ms per request, batch hundreds-thousands candidates.
- Có thể **incrementally update** với fresh training data (Wide & Deep paper mention hourly retraining cho Google Play).

### 2.2 Non-functional requirements (typical numbers, Google/Huawei era 2016-2017)

| Metric | Wide & Deep (Google Play 2016) | DeepFM (Huawei 2017) | DCN-V2 (Google 2021) |
|---|---|---|---|
| Training data volume | ~500B examples | ~hundreds of millions | TB-scale |
| Number of sparse features | hundreds | tens to hundreds | hundreds |
| Vocabulary size (largest) | ~100M (app IDs, user IDs) | millions | billions (with hashing) |
| Embedding dimension | 32 | 10 | 32-128 |
| Model size | ~MB-GB (excluding embeddings) | ~MB | ~MB |
| Embedding table total | ~GB | hundreds MB-GB | up to TB (full prod) |
| Serving latency budget | ~10 ms p99 | ~10 ms | ~10 ms |
| Retrain frequency | hourly / daily | daily | hourly |

### 2.3 Constraints quan trọng

- **Generalization vs memorization trade-off**: linear models nhớ tốt các pattern đã thấy (memorization), nhưng kém với combinations chưa thấy trong training set. Deep models generalize tốt nhưng over-generalize → recommend irrelevant items.
- **Sparse + high-cardinality features**: làm sao học interactions giữa hai features có vocab 100M mỗi cái? Không thể có bảng cross product 100M × 100M.
- **Cold-start**: items mới chưa có history → embeddings random → interactions chưa meaningful.
- **Feature engineering effort**: tradition tốn rất nhiều engineer effort để hand-craft cross features (e.g. `query × app_category`). Mục tiêu là giảm thiểu effort này.

---

## 3. High-level Architecture — So sánh 3 generations

```
   ┌──────────────────────────────────────────────────────────────┐
   │                       Wide & Deep (2016)                     │
   │                                                              │
   │   Sparse features       Sparse features       Dense features │
   │   (raw + crosses)       (embedded)            (numerical)    │
   │         │                     │                   │          │
   │         ▼                     ▼                   ▼          │
   │   ┌──────────┐         ┌─────────────────────────────────┐   │
   │   │  WIDE    │         │           DEEP                  │   │
   │   │ (linear) │         │   Embedding lookup + concat     │   │
   │   │          │         │   → MLP (3 hidden layers)       │   │
   │   └────┬─────┘         └────────────────┬────────────────┘   │
   │        │                                │                    │
   │        └───────────► σ(W·x + ŷ_deep) ◄──┘                    │
   │                       sigmoid output                          │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │                        DeepFM (2017)                         │
   │                                                              │
   │   All sparse features → shared embeddings (key insight!)     │
   │                              │                               │
   │             ┌────────────────┼─────────────────┐             │
   │             ▼                ▼                 ▼             │
   │   ┌──────────────┐    ┌──────────────┐  ┌────────────┐       │
   │   │  FM 1st-     │    │  FM 2nd-     │  │   DEEP     │       │
   │   │  order       │    │  order       │  │   (MLP)    │       │
   │   │  (linear)    │    │  (pairwise   │  │            │       │
   │   │              │    │   dot prod)  │  │            │       │
   │   └──────┬───────┘    └──────┬───────┘  └─────┬──────┘       │
   │          │                   │                │              │
   │          └───────────► σ(sum) ◄───────────────┘              │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │                    DCN / DCN-V2 (2017/2021)                  │
   │                                                              │
   │   All features → embeddings + dense → concat → x_0           │
   │                                          │                   │
   │                    ┌─────────────────────┴─────┐             │
   │                    ▼                           ▼             │
   │           ┌─────────────────┐           ┌────────────┐       │
   │           │  CROSS NETWORK  │           │   DEEP     │       │
   │           │  x_{l+1} =      │           │   (MLP)    │       │
   │           │   x_0·(x_l·w_l) │           │            │       │
   │           │     + b_l + x_l │           │            │       │
   │           │  L cross layers │           │            │       │
   │           └────────┬────────┘           └─────┬──────┘       │
   │                    │                          │              │
   │                    └──────► concat ◄──────────┘              │
   │                                │                             │
   │                                ▼                             │
   │                        σ(W·[xc; xd] + b)                     │
   └──────────────────────────────────────────────────────────────┘
```

### Điểm khác nhau cốt lõi (high-level)

| Aspect | Wide & Deep | DeepFM | DCN / DCN-V2 |
|---|---|---|---|
| Wide part | Linear LR với **manual cross features** | **FM** (automatic 2nd-order, no manual) | **Cross Network** (higher-order, learnable) |
| Embedding sharing | Wide và Deep **không share** | Wide (FM) và Deep **share embeddings** | Cross và Deep **share embeddings** |
| Interaction order | 1st-order + manual crosses + deep | 1st + automatic 2nd + deep | Up to L+1 order (controllable) + deep |
| Feature engineering | **Cần manual crosses** | **Không cần** | **Không cần** |
| Parameter count cho crosses | O(N²) cross features × cardinality | O(N · k) với k = embed dim | O(L · d²) trong DCN-V2 hoặc O(L · d) trong DCN-V1 |

---

## 4. Deep dive từng generation

### 4.1 Wide & Deep (Google, 2016)

#### Motivation

Google Play Store cần model rank hàng triệu apps cho mỗi user query. Bài toán có hai requirements xung đột:
- **Memorization**: nhớ rằng "user query 'fried chicken' → click 'chicken nuggets app'" — pattern explicit, observed nhiều lần trong training.
- **Generalization**: gợi ý các apps user chưa thấy nhưng tương tự — cần học representation chung.

Google đề xuất kiến trúc **joint training** hai branch:
- **Wide part**: linear model on raw sparse features + **manually crafted cross-product features**. Cross-product giúp memorize specific co-occurrence patterns.
- **Deep part**: feed-forward NN trên embeddings của sparse features + concatenated dense features. Học representation continuous, generalize tốt.

#### Mathematical formulation

```
ŷ = σ(W_wide · [x; φ(x)] + W_deep · a^(L) + b)
```

Trong đó:
- `x` = raw sparse features (one-hot).
- `φ(x)` = **cross-product transformation** — tích Cartesian của một tập feature subsets. Ví dụ:
  - `AND(user_installed_app=netflix, impression_app=hulu)` = 1 nếu cả hai điều đúng.
- `a^(L)` = activation của hidden layer cuối của deep part.
- `σ` = sigmoid.

#### Cross-product feature — phần "manual" của Wide & Deep

Đây là điểm yếu chính (và sau này được DeepFM/DCN giải quyết). Engineer phải **chọn tay** các cross combinations meaningful. Ví dụ Wide & Deep paper:

```python
# Crossed columns trong Google Play
crossed_features = [
    ("user_installed_app_ids", "impression_app_id"),
    ("user_installed_app_categories", "impression_app_category"),
]
```

Effort tăng quadratic với số features → không scale khi có hundreds of features.

#### Deep part details (Google Play production)

- 3 hidden layers, sizes [1024, 512, 256], ReLU activation.
- Embedding dim 32 cho hầu hết sparse features.
- Total params (excluding embeddings): ~vài MB.

#### Training

- **Joint training** (không phải ensemble) — backprop qua cả hai branches cùng lúc, optimize chung một loss.
- Wide part: **FTRL (Follow-The-Regularized-Leader)** optimizer với L1 regularization → sparse weights.
- Deep part: **AdaGrad** optimizer.
- Loss: binary log-loss (cross entropy).

#### Pseudo-code (PyTorch-flavored)

```python
class WideAndDeep(nn.Module):
    def __init__(self, sparse_vocab_sizes, cross_vocab_size,
                 dense_dim, embed_dim=32, hidden=[1024, 512, 256]):
        super().__init__()
        # Wide part — linear trên cross features (one-hot encoded)
        self.wide_linear = nn.Linear(cross_vocab_size, 1, bias=False)

        # Deep part — embeddings cho mỗi sparse feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(v, embed_dim) for v in sparse_vocab_sizes
        ])
        deep_in = embed_dim * len(sparse_vocab_sizes) + dense_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(deep_in, h), nn.ReLU()]
            deep_in = h
        layers += [nn.Linear(deep_in, 1)]
        self.deep_mlp = nn.Sequential(*layers)

    def forward(self, sparse_ids, dense_feats, cross_onehot):
        # Wide branch — direct linear trên cross features
        wide_logit = self.wide_linear(cross_onehot)

        # Deep branch — embed + concat + MLP
        embeds = [emb(ids) for emb, ids in zip(self.embeddings, sparse_ids)]
        deep_in = torch.cat(embeds + [dense_feats], dim=-1)
        deep_logit = self.deep_mlp(deep_in)

        # Joint: sum logits, then sigmoid
        return torch.sigmoid(wide_logit + deep_logit)
```

#### Kết quả Google Play (paper 2016)

- **+3.9% online app acquisition rate** so với baseline (deep-only model).
- **+1% so với wide-only** baseline.
- Joint training quan trọng hơn từng phần riêng lẻ.

### 4.2 DeepFM (Huawei, 2017)

#### Motivation — fix limitation của Wide & Deep

Wide & Deep có 3 vấn đề:
1. **Manual cross features tốn effort** — engineers phải hand-craft.
2. **Cross features không generalize** sang combinations chưa thấy trong training.
3. **Wide và Deep dùng features khác nhau** → có thể bị skew, không share information optimally.

DeepFM giải quyết bằng cách:
- Thay Wide part bằng **Factorization Machine (FM)** — model nổi tiếng của Rendle (2010), tự động học second-order interactions từ embeddings.
- **Share embeddings** giữa FM và Deep — cùng một embedding lookup feeds vào cả FM và MLP. Đây là key insight.

#### Mathematical formulation

FM 1st-order:
```
y_FM_1 = Σ_i w_i · x_i           (linear part)
```

FM 2nd-order:
```
y_FM_2 = Σ_{i<j} <v_i, v_j> · x_i · x_j
```
Trong đó `v_i` là embedding vector của feature `i`. Đây là **dot product giữa các embeddings** của các features active.

Trick toán học (Rendle 2010): có thể compute O(N·k) thay vì O(N²·k):
```
y_FM_2 = 0.5 · Σ_f [(Σ_i v_{i,f}·x_i)² - Σ_i (v_{i,f}·x_i)²]
```

Deep part:
```
y_deep = MLP(concat([v_1, v_2, ..., v_n]))
```
**Note**: dùng cùng embeddings `v_i` như FM part.

Output cuối:
```
ŷ = σ(y_FM_1 + y_FM_2 + y_deep)
```

#### Tại sao shared embeddings là key insight?

- **Efficiency**: chỉ cần 1 embedding lookup per feature, không phải 2 (như Wide & Deep nếu deep part dùng embedding).
- **Regularization**: FM signal cung cấp inductive bias cho embeddings — buộc các embedding capture meaningful pairwise interactions, giúp deep part learn faster.
- **Cold-start**: embedding được train từ cả hai signals (low-order + high-order) → robust hơn.

#### Pseudo-code

```python
class DeepFM(nn.Module):
    def __init__(self, sparse_vocab_sizes, dense_dim,
                 embed_dim=10, hidden=[400, 400, 400]):
        super().__init__()
        # Shared embeddings — dùng cho cả FM và Deep
        self.embeddings = nn.ModuleList([
            nn.Embedding(v, embed_dim) for v in sparse_vocab_sizes
        ])
        # FM 1st-order weights — embedding dim 1
        self.fm_linear = nn.ModuleList([
            nn.Embedding(v, 1) for v in sparse_vocab_sizes
        ])
        # Deep MLP
        deep_in = embed_dim * len(sparse_vocab_sizes) + dense_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(deep_in, h), nn.ReLU()]
            deep_in = h
        layers += [nn.Linear(deep_in, 1)]
        self.deep_mlp = nn.Sequential(*layers)

    def forward(self, sparse_ids, dense_feats):
        # Lookup embeddings (shared!)
        embeds = [emb(ids) for emb, ids in zip(self.embeddings, sparse_ids)]
        embeds_stack = torch.stack(embeds, dim=1)   # [B, F, k]

        # FM 1st-order: Σ w_i
        linear_logit = sum(lin(ids).squeeze(-1)
                           for lin, ids in zip(self.fm_linear, sparse_ids))

        # FM 2nd-order: trick O(F·k)
        sum_square = embeds_stack.sum(dim=1) ** 2      # [B, k]
        square_sum = (embeds_stack ** 2).sum(dim=1)    # [B, k]
        fm_2nd = 0.5 * (sum_square - square_sum).sum(dim=-1)  # [B]

        # Deep
        deep_in = torch.cat([e for e in embeds] + [dense_feats], dim=-1)
        deep_logit = self.deep_mlp(deep_in).squeeze(-1)

        return torch.sigmoid(linear_logit + fm_2nd + deep_logit)
```

#### Kết quả paper (Huawei dataset + Criteo benchmark)

- DeepFM > Wide & Deep ~0.3-0.5% AUC trên Criteo.
- Không cần feature engineering thủ công.
- Training cũng nhanh hơn vì share embeddings → fewer parameters.

### 4.3 DCN — Deep & Cross Network (Google, 2017)

#### Motivation

DeepFM giới hạn ở **second-order interactions** (pairwise). Nhưng nhiều real-world patterns là **higher-order**: ví dụ `(country, device_type, hour_of_day, app_category)` — 4-way interaction.

DCN đề xuất một **Cross Network** có thể tự động học interactions bậc cao hơn, một cách **explicit** và **parameter-efficient**.

#### Cross layer (DCN-V1)

Mỗi cross layer:
```
x_{l+1} = x_0 · (x_l^T · w_l) + b_l + x_l
```

Trong đó:
- `x_0` = input embedding concat (fixed reference).
- `x_l` = output của cross layer thứ l.
- `w_l, b_l` = learnable parameters của layer l.
- Sau L layers, output capture interactions up to order L+1.

**Key property**: parameter count = `O(L · d)` (linear với feature dim d). Compare với feed-forward NN: O(L · d²).

#### DCN-V2 (Google 2021) — improvement

DCN-V1 có limitation: weight `w_l` là vector (d,), expressiveness hạn chế. DCN-V2 thay bằng **weight matrix** (d, d):

```
x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
```

Trong đó `⊙` là element-wise product, `W_l ∈ R^{d×d}`.

Parameter count: O(L · d²). Để giảm cost, DCN-V2 propose **low-rank decomposition**:
```
W_l ≈ U_l · V_l^T   với U_l, V_l ∈ R^{d×r}, r << d
```
Hoặc **mixture-of-experts variant**:
```
x_{l+1} = Σ_k g_k(x_l) · [x_0 ⊙ (U_k · V_k^T · x_l + b_k)]
```

#### Kiến trúc tổng thể DCN (parallel structure)

```
   embeddings + dense  →  x_0
        │
        ├──► Cross Network (L layers) ─→ x_c
        │
        └──► Deep Network (MLP) ─────────→ x_d
        
   ŷ = σ(W · [x_c; x_d] + b)
```

DCN-V2 paper cũng đề xuất **stacked structure** (cross trước, deep sau hoặc ngược lại), và experiment shows parallel thường tốt hơn trên ad click datasets.

#### Pseudo-code DCN-V2 cross layer

```python
class CrossLayerV2(nn.Module):
    def __init__(self, dim, rank=None):
        super().__init__()
        if rank is None:
            # Full matrix W ∈ R^{d×d}
            self.W = nn.Linear(dim, dim, bias=True)
        else:
            # Low-rank: W ≈ U @ V^T
            self.U = nn.Linear(dim, rank, bias=False)
            self.V = nn.Linear(rank, dim, bias=True)
        self.full = (rank is None)

    def forward(self, x0, xl):
        if self.full:
            wx = self.W(xl)
        else:
            wx = self.V(self.U(xl))
        return x0 * wx + xl   # element-wise product + residual


class DCNv2(nn.Module):
    def __init__(self, embed_dim_total, n_cross=3, hidden=[256, 256],
                 cross_rank=None):
        super().__init__()
        self.crosses = nn.ModuleList([
            CrossLayerV2(embed_dim_total, rank=cross_rank)
            for _ in range(n_cross)
        ])
        layers = []
        in_d = embed_dim_total
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        self.deep = nn.Sequential(*layers)
        self.head = nn.Linear(embed_dim_total + hidden[-1], 1)

    def forward(self, x0):
        # Cross branch
        xc = x0
        for cross in self.crosses:
            xc = cross(x0, xc)
        # Deep branch
        xd = self.deep(x0)
        # Concat and predict
        return torch.sigmoid(self.head(torch.cat([xc, xd], dim=-1)))
```

#### Kết quả DCN-V2 (Google 2021)

- Trên Criteo Display Ads: +0.1% AUC so với DCN-V1, +0.3% so với DeepFM.
- **Quan trọng hơn**: production deployment ở Google đem lại significant business metrics. Theo paper, Cross Network capture interactions mà MLP-only không học được — explainability tốt hơn (có thể visualize feature interaction weights).
- Được dùng trong YouTube ranking (theo các Google talks về ranking architecture, post-2021).

---

## 5. Trade-offs & Design decisions

### 5.1 So sánh tổng quan

| Aspect | Wide & Deep | DeepFM | DCN-V2 |
|---|---|---|---|
| Cần manual feature crosses | **Yes** (drawback chính) | No | No |
| Embedding sharing | Wide ↔ Deep: No | FM ↔ Deep: **Yes** | Cross ↔ Deep: **Yes** |
| Max interaction order | 2 (qua manual crosses) + deep | 2 (FM) + deep | L+1 (Cross) + deep |
| Parameter complexity (interactions) | O(N²) for crosses | O(N·k) | O(L·d²) (DCN-V2) hoặc O(L·d) (V1) |
| Interpretability | High (linear part) | Medium (FM weights) | High (Cross weights → feature interaction matrix) |
| Production proven (big tech) | Google Play, many others | Alibaba, Huawei | Google, YouTube |
| Training stability | Cần 2 optimizers (FTRL + AdaGrad) | Single optimizer | Single optimizer |
| Cold-start (new features) | Wide phần fail (no learned cross) | OK (FM còn meaningful với 1 feature seen) | OK |

### 5.2 Khi nào dùng cái nào?

**Dùng Wide & Deep khi:**
- Đã có hệ thống feature engineering tốt sẵn (hand-crafted crosses).
- Cần **interpretability cao** trên linear part (debug ads spend, regulatory).
- Có domain knowledge cụ thể về cross interactions quan trọng.

**Dùng DeepFM khi:**
- Không có resource cho manual feature engineering.
- Sparse features dominant, second-order interactions là enough.
- Want fastest path to production với baseline mạnh.
- **Recommendation**: DeepFM là default choice cho team mới làm reco/ads.

**Dùng DCN-V2 khi:**
- Cần capture higher-order interactions (>2-way).
- Có compute budget cho training/serving phức tạp hơn.
- Muốn explainability của cross weights (Google paper provide visualization).
- **Recommendation**: production ranker ở scale lớn, post-DeepFM stage of maturity.

### 5.3 Common pitfalls

**1. Embedding dimension trade-off**
- Quá nhỏ (k < 8): không đủ capacity, FM/Cross signal yếu.
- Quá lớn (k > 64): overfit, memory explosion với sparse features high-cardinality.
- **Rule of thumb**: k = 8-32 cho production. Có thể dùng **mixed-dimension embedding** (như Meta DLRM): features cao-cardinality dùng dim lớn, low-cardinality dùng dim nhỏ.

**2. Dense vs sparse handling**
- Wide & Deep paper: dense features concat trực tiếp vào deep branch.
- DCN-V2: dense features cũng được "embedded" qua linear layer để cùng dim với sparse embeddings.
- DeepFM: dense features thường được bucketize → categorical → embedded. (Tránh dùng raw dense vì FM interaction giữa raw dense và categorical embedding không meaningful.)

**3. Embedding initialization**
- Init quá lớn → loss diverge sớm.
- Init quá nhỏ → gradient vanish.
- **Standard**: Xavier/Glorot init hoặc N(0, 0.01).

**4. Batch normalization vs layer norm**
- BN trên sparse embeddings = bad idea (batch statistics không stable với sparse features).
- LayerNorm trong deep MLP = OK, thường +0.1-0.2% AUC.

**5. Negative sampling**
- Tất cả 3 models giả sử có positive + negative examples balanced.
- Trong ads/feed, negative rate thường <5% → cần **negative downsampling** + **calibration** sau prediction để recover true pCTR.
  ```
  p_true = p_sampled / (p_sampled + (1 - p_sampled) / w)
  ```
  với `w` là downsampling rate của negatives.

### 5.4 Cross-references đến knowledge base

- Architecture sau DCN-V2 trong evolution: [S2-01 DLRM](S2-01_meta_dlrm_architecture.md) — Meta's take với explicit dot-product interaction + massive embedding tables.
- Sequential user modeling (next step): DIN/DIEN (Alibaba) — sẽ cover ở S2-03, attention over user behavior sequence.
- Khi deploy production: serving infrastructure → [S4-01 Michelangelo](S4-01_uber_michelangelo_feature_store.md).
- Recommendation system context: [S1-01 YouTube](S1-01_youtube_recommendation_end_to_end.md) — DCN-V2 được dùng trong YouTube ranking stage.

---

## 6. Lessons learned & Best practices

### 6.1 Lesson 1: Joint training > ensemble

Wide & Deep paper highlight rằng **joint training là khác với ensemble**:
- Ensemble: train 2 models độc lập, combine predictions cuối.
- Joint: train cùng lúc, gradient flow giữa các branches.

Trong joint training:
- Wide part chỉ cần ít features (focus on memorization).
- Deep part nhận signal từ wide để focus on generalization gap.

Production wisdom: **luôn jointly train các branches** trong các architecture hybrid (kể cả MMoE, DLRM, DCN). Đừng ensemble unless có lý do strong (e.g. model diversity).

### 6.2 Lesson 2: Shared embeddings là powerful regularization

DeepFM's insight (FM ↔ Deep share embeddings) đã propagate sang nhiều architectures sau:
- DCN: Cross và Deep share embeddings.
- DLRM: Bottom MLP và Top MLP share embeddings qua dot-product interaction.
- Two-tower retrieval: query và candidate towers share **vocabulary** (không nhất thiết weights).

**Heuristic**: nếu hai branches operate trên cùng features, **luôn share embeddings**, đặt downstream branches là "specialized heads" trên top.

### 6.3 Lesson 3: Embedding lookup là bottleneck thực sự, không phải MLP

Ở production scale:
- MLP forward/backward: vài hundred FLOPs, GPU compute-bound (fast).
- Embedding lookup: irregular memory access pattern, memory-bound.
- **80%+ training time** trong các production CTR models là embedding lookup (theo Meta DLRM paper, ISCA 2022).

→ Tối ưu phải focus vào: embedding sharding, row-wise table compression, FBGEMM-style optimized kernels. Đừng waste time optimize MLP.

### 6.4 Lesson 4: Production iteration là incremental

Đây là một meta-lesson về cách big tech evolve architectures:
- Mỗi generation chỉ improve **0.3-0.5% AUC** so với previous.
- Nhưng 0.5% AUC trên revenue tỷ đô = hundreds of millions $ impact.
- Don't expect "10x improvement" — production ML là grinding incremental wins, not paradigm shifts.

### 6.5 Lesson 5: Always have a strong baseline

- Trước khi build DCN-V2, đảm bảo DeepFM baseline đã được tune tốt.
- Trước khi DeepFM, có LR + manual crosses baseline.
- **Skipping baselines = không hiểu nguồn gốc gain đến từ đâu**. Có thể gain đến từ feature engineering tốt hơn, không phải model.

### 6.6 Lesson 6: Feature engineering chưa chết

Dù DeepFM/DCN tự động học interactions, vẫn có những features cần engineering:
- **Time features**: hour_of_day, day_of_week, time_since_last_action — model không tự derive được từ raw timestamp.
- **Behavioral aggregates**: count of impressions last 7 days, mean watch time of category X. Có thể compute trong feature store.
- **Cross-domain features**: user's behavior trên Facebook → feature cho Instagram recommendation (Meta-internal).

**Production reality**: 60-70% AUC gain trong production ranker đến từ **features**, không phải model architecture.

### 6.7 Lesson 7: Calibration matters trong ads

Cho ads, pCTR phải **calibrated** — predicted CTR = actual CTR observed. Reasons:
- Auction mechanism: bid = pCTR × CPC. Mis-calibrated pCTR → wrong allocation.
- Budget pacing: predicted spend based on pCTR.

Sau khi train DeepFM/DCN, **luôn check calibration plot**:
```
predicted_bucket_mean vs observed_CTR_in_bucket
```
Nếu lệch → áp dụng **Platt scaling** hoặc **isotonic regression** post-hoc.

### 6.8 Best practice — production checklist

- [ ] Negative downsampling + calibration recovery formula.
- [ ] Embedding init Xavier hoặc N(0, 0.01).
- [ ] Shared embeddings nếu có hybrid architecture.
- [ ] Joint training, không ensemble.
- [ ] Validation split theo **time** (chronological), không random — tránh data leakage.
- [ ] Track AUC + LogLoss + Calibration Error đồng thời.
- [ ] Profile training: nếu >50% thời gian là embedding ops → cần FBGEMM / fused kernels.
- [ ] Monitor embedding norm trong training để detect divergence.

---

## 7. References

### Foundational papers (highly reliable)

1. **Cheng et al. "Wide & Deep Learning for Recommender Systems"** — DLRS workshop, RecSys 2016. arXiv: [1606.07792](https://arxiv.org/abs/1606.07792). Google Play Store case study, joint training framework.

2. **Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"** — IJCAI 2017. arXiv: [1703.04247](https://arxiv.org/abs/1703.04247). Huawei research, shared-embedding insight.

3. **Wang et al. "Deep & Cross Network for Ad Click Predictions"** — ADKDD 2017. arXiv: [1708.05123](https://arxiv.org/abs/1708.05123). Google, original DCN with vector weights.

4. **Wang et al. "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems"** — WWW 2021. arXiv: [2008.13535](https://arxiv.org/abs/2008.13535). Google, matrix weights + low-rank + production lessons.

5. **Rendle "Factorization Machines"** — ICDM 2010. The FM paper underlying DeepFM. Crucial to understand DeepFM intuition.

### Engineering blog posts

6. **Google AI Blog — "Wide & Deep Learning: Better Together with TensorFlow"** (2016): [link](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html). Production context cho Google Play.

7. **TensorFlow tutorial on Wide & Deep** — official TF guide, có code template.

### Surveys & comparative studies

8. **Zhang et al. "Deep Learning based Recommender System: A Survey and New Perspectives"** — ACM Computing Surveys 2019. arXiv: [1707.07435](https://arxiv.org/abs/1707.07435). Bao quát evolution của reco architectures.

9. **Zhao et al. "AutoFIS: Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction"** — KDD 2020. Follow-up direction sau DeepFM/DCN: auto-select important interactions.

### Critical evaluations / industry feedback

10. **Rendle et al. "Neural Collaborative Filtering vs. Matrix Factorization Revisited"** — RecSys 2020. arXiv: [2005.09683](https://arxiv.org/abs/2005.09683). Counter-paper cho thấy gain của neural models trên CF benchmark có thể nhỏ hơn báo cáo. Bài học: luôn cẩn thận với benchmarks.

11. **Anelli et al. "How Neural Are Neural Recommenders?"** — RecSys 2022. Phân tích critical về NN gains trong reco.

### Talks (video, có thể xem)

12. **Heng-Tze Cheng (Google) — Wide & Deep talk at TF Dev Summit 2017**: YouTube search "Wide and Deep TensorFlow". Production motivation explained clearly.

13. **DCN-V2 talk at WWW 2021** — Ruoxi Wang. Search "DCN V2 talk" trên YouTube.

### Code & reproduction

14. **DeepCTR library** (PyTorch & TF implementations): [github.com/shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR). Reference implementation của tất cả 3 models + benchmarks trên Criteo, Avazu, MovieLens.

15. **TorchRec** (Meta) — production-grade library với DLRM, DeepFM examples, FBGEMM kernels.

---

> **Tóm tắt 1 dòng**: Wide & Deep → DeepFM → DCN-V2 là evolution **giảm dần manual feature engineering** đồng thời **tăng dần explicit modeling của feature interactions**. Hiểu rõ 3 generations này là nền cho DLRM, DIN, và các transformer-based ranker hiện đại.
