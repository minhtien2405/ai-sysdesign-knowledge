---
id: S1-03
title: "Pinterest PinSage: Graph-based Retrieval at 3 Billion Pins Scale"
summary: "GNN ở scale 3 tỷ pins: random-walk sampling, importance pooling, MapReduce inference, PinnerSage/TransAct evolution."
slug: pinterest_pinsage_graph_retrieval
scope: 1
scope_name: foundations
difficulty: intermediate-advanced
status: done
tags:
  - GNN
  - graph neural network
  - PinSage
  - random walk sampling
  - importance pooling
  - MapReduce inference
cross_refs: [S1-01, S2-01, S3-03]
created: 2026-05-20
last_validated: 2026-05-20
line_count: 1536
---

# S1-03 — Pinterest PinSage: Graph-based Retrieval at 3 Billion Pins Scale

> **Difficulty**: Intermediate–Advanced
> **Scope**: Foundations (Scope 1) — graph neural network applied to web-scale recommendation
> **Key insight**: PinSage là production GNN đầu tiên ở scale tỷ items, đã chứng minh rằng graph-based retrieval không chỉ là academic toy mà có thể serve 3 tỷ pins thật. Bí quyết nằm ở 3 thứ: (1) **random-walk neighborhood sampling** thay vì k-hop, (2) **importance pooling aggregator** thay vì mean/LSTM, (3) **MapReduce-based offline inference** thay vì online forward-pass.

---

## Mục lục

1. [Overview — Pinterest trước và sau PinSage](#1-overview)
2. [System Requirements — scale, latency, freshness](#2-system-requirements)
3. [Foundational Research — GCN, GraphSAGE và lineage dẫn đến PinSage](#3-foundational-research)
4. [PinSage Architecture Deep Dive](#4-pinsage-architecture-deep-dive)
5. [Training Infrastructure Innovation](#5-training-infrastructure)
6. [Production Deployment & Serving](#6-production-deployment)
7. [Improvements Over Time — PinnerSage, TransAct, multi-modal fusion](#7-improvements-over-time)
8. [Trade-offs & Design Decisions](#8-trade-offs)
9. [Common Pitfalls When Implementing GNN-based Reco](#9-common-pitfalls)
10. [Lessons Learned & Best Practices](#10-lessons-learned)
11. [References](#11-references)

---

## 1. Overview

### 1.1 Pinterest — bối cảnh 2018 và bài toán recommendation

Pinterest là một visual discovery platform. Khác với feed-based social networks (Facebook, TikTok) nơi user follow accounts, Pinterest có cấu trúc dữ liệu đặc biệt mà mỗi user **collect "pins"** (image bookmarks) vào các **"boards"** (themed collections). Một pin có thể được "re-pin" sang nhiều board khác nhau, tạo ra một mạng lưới quan hệ pin–board cực kỳ phong phú.

Đặc điểm dữ liệu Pinterest 2018 (theo paper Ying et al. KDD 2018):
- **~3 billion pins** (pin = image + metadata + URL).
- **~18 billion board–pin edges** (tức là 18 tỷ pin-placement events).
- **~200M+ monthly active users**.
- **~1B+ user-pin interactions per day** (click, save, hide, comment).

Bài toán: với mỗi pin query (hoặc user query), recommend top-K relevant pins. Đây là cốt lõi của nhiều surface:
- **Related Pins** — module "more like this" dưới mỗi pin.
- **Home Feed** — feed cá nhân hoá ở trang chủ.
- **Search** — return pins liên quan đến text query.
- **Email digest** — chọn pins cho weekly digest.

### 1.2 Pre-PinSage era — Pinterest đã làm reco thế nào trước GNN?

Trước PinSage (deploy 2018), Pinterest đã trải qua nhiều generation của retrieval:

**Generation 1 (~2013–2015): Collaborative Filtering + Visual Embeddings**

Pinterest dùng matrix factorization classic (Alternating Least Squares, ALS) trên ma trận user × pin với implicit feedback (save, click). Vấn đề:
- **Cold-start nặng** — pin mới (rất nhiều mỗi ngày) không có lịch sử tương tác → MF không score được.
- **Sparse signal** — average user chỉ tương tác với vài trăm pins / tổng 3B pins → ma trận cực sparse (sparsity > 99.99%).

Pinterest bổ sung **visual embeddings** từ một CNN (ResNet finetuned trên Pinterest data) để giải quyết cold-start cho pin mới. Nhưng MF vẫn là backbone cho user side.

**Generation 2 (~2015–2017): Pixie — Random Walk on Pin–Board Graph**

Đây là tiền thân tinh thần của PinSage. Pinterest engineering blog (2017) mô tả **Pixie**: một real-time random walk engine chạy trên pin–board bipartite graph.

Ý tưởng Pixie rất intuitive:
- Khi user click pin A, start random walk từ A trên graph.
- Mỗi step: từ pin → board (random chọn 1 trong các boards chứa pin đó), từ board → pin (random chọn 1 pin khác trong board đó).
- Đếm visit count cho mỗi pin được "thăm" qua walk.
- Pin nào có visit count cao → recommend.

Pixie là một dạng **personalized PageRank** chạy real-time. Ưu điểm:
- **Personalized**: walk start từ pin/user-specific seed.
- **Real-time**: không cần precompute toàn bộ score.
- **Explainable**: có thể trace lại path random walk.

Nhược điểm:
- **Không có learned embedding** — Pixie chỉ count visits, không generalize được pattern.
- **Cold-start pin** vẫn khó — pin mới ít edges → random walk hiếm khi reach được.
- **No content signal** — Pixie không dùng được pin image / text features.

Pixie chạy production từ 2016 và phục vụ tốt, nhưng Pinterest team nhận ra cần **learned representation** để (1) combine graph signal với content signal, (2) generalize cho pin mới (inductive). Đó là motivation cho PinSage.

**Generation 3 (2018): PinSage — GNN học representation từ pin–board graph**

PinSage thừa kế ý tưởng random walk từ Pixie nhưng dùng nó như **sampling strategy** để train một Graph Neural Network. Output của PinSage là **256-dim embedding cho mỗi pin**, có thể dùng cho ANN retrieval, downstream ranking, hoặc các task khác.

### 1.3 Tại sao GNN — graph-based intuition cho Pinterest data

Tại sao graph-based learning lại phù hợp với Pinterest?

1. **Cấu trúc dữ liệu vốn là graph**: pin–board bipartite graph là first-class citizen, không phải feature engineering bịa ra.

2. **Multi-hop signal đáng giá**: Nếu pin A và pin B chia sẻ nhiều boards chung, A và B có quan hệ trực tiếp. Nhưng nếu A share boards với C, và C share boards với B, thì A–B có quan hệ 2-hop — vẫn đáng exploit. GNN với K layers captures K-hop neighborhood.

3. **Co-occurrence pattern mạnh hơn content**: Hai pins có image rất khác (e.g. công thức nấu ăn và bức tranh phòng bếp) nhưng cùng xuất hiện trong nhiều boards "kitchen aesthetics" → user thực sự coi chúng liên quan. Content-only model (visual embedding) không capture được pattern này; graph capture được.

4. **Inductive setting**: Pin mới được tạo liên tục. GraphSAGE-style inductive GNN có thể compute embedding cho pin mới bằng cách aggregate từ neighbors, không cần retrain.

### 1.4 Business impact của PinSage

Pinterest công bố (KDD 2018 paper + engineering blogs):
- **Related Pins** A/B test: PinSage so với production baseline (Pixie-based + visual embedding) cho **150%+ improvement** trong head-to-head human evaluation (raters chọn PinSage suggestion thường xuyên hơn).
- **Hit-rate@10** (recall metric): PinSage vượt baseline khoảng **+30%**.
- **MRR (Mean Reciprocal Rank)**: cải thiện ~22% so với best content-based baseline.

Sau PinSage, gần như toàn bộ retrieval surface ở Pinterest chuyển sang dùng PinSage embedding hoặc các successor (PinnerSage, TransAct). PinSage trở thành **case study kinh điển** cho production GNN — được cite hàng nghìn lần và inspire hàng loạt graph-based reco systems ở Alibaba (GraphRec, EGES), Uber (Graph Learning), LinkedIn, Snap.

---

## 2. System Requirements

### 2.1 Scale (numbers từ paper KDD 2018, internal numbers hiện tại lớn hơn nhiều)

| Dimension | Value (2018) | Estimated 2024+ |
|---|---|---|
| Number of pins | ~3 billion | ~10 billion+ |
| Number of boards | ~1 billion | ~3 billion+ |
| Pin–board edges | ~18 billion | ~60 billion+ |
| Active users / month | ~200M | ~500M+ |
| New pins per day | ~hundreds of millions | ~billions |
| Embedding dimension | 256 (float32) → 1024 byte/pin | typically 128-256 |
| Total embedding storage | ~3B × 1KB ≈ 3 TB (float32) | ~10TB+ (compress) |

### 2.2 Functional requirements

- **Retrieval API**: given a pin id (or user id), return top-K similar pins (K = 100-1000) using PinSage embedding + ANN.
- **Cold-start support**: pin mới (created < 1 hour) phải có embedding ngay, không chờ retrain.
- **Quality metrics**: hit-rate@K, MRR, recall@K trên human-curated ground truth.
- **Multi-surface serving**: cùng embedding dùng cho Related Pins, Home Feed candidate gen, Search candidate gen.

### 2.3 Non-functional requirements

- **Latency budget**: retrieval phải xong trong **~50ms P99** (để leave budget cho ranking downstream, total budget ~200ms cho cả pipeline).
  - Concretely: ANN search top-1000 trong embedding space 256-dim qua 3B pins phải < 30ms; embedding lookup < 5ms; network overhead < 15ms.
- **Throughput**: ước tính ~hundred thousand QPS cho retrieval ở peak (Home Feed + Related Pins combined).
- **Training time**: PinSage paper báo cáo train trên 7.5B pins sample được ~1 ngày trên cluster GPU. Full retrain end-to-end < 1 tuần.
- **Freshness**:
  - Embedding cho pin **mới tạo** (< 1 hour): tính toán bằng forward pass single-pin (latency ~tens of ms).
  - Toàn bộ embedding table: refresh **daily** (rebuild qua MapReduce pipeline).
  - Graph structure: update daily (board–pin edges).
- **Hardware** (2018 paper):
  - Training: ~32 NVIDIA Tesla K80 GPUs (paper era). Hiện tại nâng cấp lên A100/H100.
  - Inference (offline): ~378 Hadoop nodes cho MapReduce embedding pipeline.
  - Serving: standard CPU clusters + ANN index (HNSW or custom Pinterest "Manas" search infra).

### 2.4 Constraints

- **Không thể train transductive GNN** (như GCN nguyên bản): graph quá lớn, full-batch training infeasible. Phải inductive.
- **Không thể online inference** cho mọi pin trong realtime: 3B pins × 50ms = 150M giây CPU time mỗi lần "refresh all". Phải batch offline.
- **Không thể đơn giản dùng visual embedding**: pin sharing same visual có khi rất khác nhau về intent (e.g. cùng một ảnh chiếc ghế xuất hiện trong board "industrial design" vs "rustic farmhouse" sẽ co-occur với hai cluster khác nhau).

---

## 3. Foundational Research — GCN, GraphSAGE và lineage dẫn đến PinSage

Để hiểu PinSage, phải hiểu hai paper foundational: **GCN (Kipf & Welling, ICLR 2017)** và **GraphSAGE (Hamilton et al., NeurIPS 2017)**.

### 3.1 GCN — Graph Convolutional Network (Kipf & Welling 2017)

**Setup**: graph $G = (V, E)$ với $|V| = N$ nodes, adjacency matrix $A \in \{0,1\}^{N \times N}$, node feature matrix $X \in \mathbb{R}^{N \times d}$.

**GCN layer formula**:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

Trong đó:
- $\tilde{A} = A + I$ — adjacency với self-loops.
- $\tilde{D}$ — degree matrix của $\tilde{A}$.
- $H^{(0)} = X$ — initial node features.
- $W^{(l)}$ — learnable weight matrix tại layer $l$.
- $\sigma$ — non-linearity (typically ReLU).

**Intuition**: mỗi node update representation bằng **weighted average của neighbors' representations** (weight = inverse degree để normalize). Stacking $K$ layers → mỗi node thấy được $K$-hop neighborhood.

**Tại sao GCN KHÔNG dùng được ở Pinterest scale?**

1. **Full-batch training**: GCN tính toàn bộ $H^{(l+1)}$ cho tất cả nodes mỗi forward pass. Với 3B nodes, $H \in \mathbb{R}^{3B \times 256}$ ≈ 3 TB float32 — không fit GPU memory.

2. **Sparse matrix multiplication trên adjacency**: $\tilde{A}$ có 18B non-zero entries. Mỗi forward pass cần SpMM (Sparse Matrix Multiplication) trên matrix khổng lồ — even với distributed sparse compute, latency / memory không khả thi.

3. **Transductive**: GCN học embedding cho **fixed set of nodes** đã thấy trong training. Pin mới (created after training) không có embedding — phải retrain lại. Với pin volume Pinterest, retrain mỗi giờ là không khả thi.

4. **Receptive field explosion**: nếu degree trung bình của graph là $d$, sau $K$ layers mỗi node có thể "see" $d^K$ neighbors. Pinterest graph có nodes high-degree (popular pins on hundreds of boards) → $d^K$ có thể là millions sau $K=2$ → mỗi minibatch implicit cần load nhiều node features.

GCN là breakthrough conceptual nhưng **không production-ready cho web scale**. Đó là motivation cho GraphSAGE.

### 3.2 GraphSAGE — Inductive node embeddings via neighborhood sampling (Hamilton et al. 2017)

GraphSAGE đưa ra 2 đổi mới chính giải quyết các vấn đề của GCN:

1. **Neighborhood sampling**: thay vì dùng toàn bộ neighbors, mỗi layer **sample fixed-size** neighbors (e.g. 25 ở layer 1, 10 ở layer 2). Điều này (a) giới hạn computation, (b) khiến receptive field constant — không explode theo degree.

2. **Inductive aggregation**: GraphSAGE học **aggregator function** (mean, LSTM, pooling) tham số hoá bởi neural net. Aggregator này có thể apply cho node mới (chỉ cần neighbors của nó) → inductive.

**GraphSAGE forward pass (algorithm 1 từ paper)**:

```
Input: graph G(V, E), input features {x_v ∀ v ∈ V},
       depth K, weight matrices W^k, non-linearity σ,
       neighbor sampling function N: v → 2^V

Output: vector representations z_v cho mọi v ∈ V

1. h_v^0 ← x_v cho mọi v ∈ V
2. for k = 1 to K do:
3.   for v ∈ V do:
4.     h_{N(v)}^k ← AGGREGATE_k({h_u^{k-1}, ∀ u ∈ N(v)})
5.     h_v^k ← σ(W^k · CONCAT(h_v^{k-1}, h_{N(v)}^k))
6.   h_v^k ← h_v^k / ||h_v^k||_2   # L2 normalize
7. z_v ← h_v^K
```

**Step-by-step ví dụ concrete** (graph 5 nodes A, B, C, D, E):

Giả sử graph:
```
        A
       / \
      B   C
     /|   |\
    D E   E F
```
- Neighbors: N(A) = {B, C}, N(B) = {A, D, E}, N(C) = {A, E, F}, N(D) = {B}, N(E) = {B, C}, N(F) = {C}
- Initial features (giả sử 2-dim cho dễ minh hoạ):
  - $x_A = [1, 0]$, $x_B = [0, 1]$, $x_C = [1, 1]$, $x_D = [0.5, 0.5]$, $x_E = [1, 0.5]$, $x_F = [0.2, 0.8]$

**Layer 1, mean aggregator, sample 2 neighbors**:

Tính $h_A^1$:
- Sample 2 neighbors của A từ {B, C} → giả sử sample được {B, C} (vì chỉ có 2).
- $h_{N(A)}^1 = \text{mean}([x_B, x_C]) = \text{mean}([0, 1], [1, 1]) = [0.5, 1.0]$
- Concat: $\text{CONCAT}(x_A, h_{N(A)}^1) = [1, 0, 0.5, 1.0]$ (4-dim)
- Multiply $W^1 \in \mathbb{R}^{4 \times h}$ (giả sử $h = 2$, $W^1 = \text{some random init}$).
- $h_A^1 = \sigma(W^1 \cdot [1, 0, 0.5, 1.0])$
- L2 normalize.

Tương tự cho $h_B^1, h_C^1, ...$.

**Layer 2** dùng $h^1$ làm input, lặp lại process → $h^2$ captures 2-hop neighborhood.

**Aggregator function trade-offs**:

| Aggregator | Formula | Pros | Cons |
|---|---|---|---|
| Mean | $\frac{1}{|N(v)|}\sum_{u} h_u$ | Simple, permutation-invariant, cheap | Mất discriminative power, identical pairs of nodes có same neighbors → same embedding |
| LSTM | Sequential LSTM over random permutation của neighbors | Expressive, can learn order patterns | Not permutation-invariant về mặt lý thuyết (paper hack bằng random permute); slow on GPU |
| Pooling | $\max(\{\sigma(W \cdot h_u + b)\})$ | Good discriminative power, permutation-invariant | Element-wise max loses information, harder to train |

Paper GraphSAGE tìm thấy **pooling** thường tốt nhất, LSTM thứ hai, mean baseline.

### 3.3 Tại sao Pinterest cần modify GraphSAGE thay vì dùng nguyên bản

Pinterest team đã thử implement GraphSAGE thẳng vào pin–board graph. Phát hiện 3 vấn đề:

1. **Random uniform neighbor sampling kém quality**: trong pin–board graph, một pin có thể trong hàng nghìn boards (popular pins on viral topics), và board có thể chứa hàng nghìn pins. Uniform sampling K neighbors → khả năng cao sample được **noise neighbors** (boards có pin nhưng không thực sự relevant với pin source). Cần importance-weighted sampling.

2. **Mean/LSTM aggregator không exploit edge importance**: tất cả neighbors được treat bằng nhau (mean), nhưng intuitively pin A liên quan đến pin B mạnh hơn nếu họ co-occur trong nhiều boards. Mean throws away this signal.

3. **Naive forward pass cho 3B pins quá đắt**: even với neighbor sampling, mỗi pin cần $O(\text{sampled neighbors per layer}^K)$ leaf node lookups. 3B × cost-per-pin = nhiều CPU-years. Cần inference architecture mới (MapReduce-based).

PinSage giải quyết cả 3 vấn đề trên qua **random-walk based sampling**, **importance pooling**, và **producer-consumer MapReduce inference**. Đó là 3 đóng góp chính của paper.

### 3.4 Lineage tóm tắt

```
2013-2015: Pinterest CF + ALS + Visual Embeddings
   |
   ↓ (cần personalized, cần real-time)
2016-2017: Pixie (random walk on bipartite graph, real-time PPR)
   |
   ↓ (cần learned representation, cần generalize)
2016: DeepWalk, Node2Vec (skip-gram on random walks)  ← idea: random walk + word2vec-style learning
   |
   ↓
2017: GCN (Kipf-Welling) ← idea: convolution on graph
   |
   ↓ (transductive, không scale)
2017: GraphSAGE (Hamilton et al.) ← idea: inductive, neighborhood sampling, aggregator
   |
   ↓ (uniform sampling kém quality, mean aggregator weak)
2018: PinSage (Ying et al. KDD 2018) ← random-walk sampling + importance pooling + MapReduce inference
   |
   ↓ (single embedding bottleneck cho diverse user interest)
2020: PinnerSage (Pal et al. KDD 2020) ← multi-embedding per user via hierarchical clustering
   |
   ↓ (need sequence + multi-modal fusion)
2023: TransAct (Pinterest KDD 2023) ← transformer-based sequence model fusing real-time actions + PinSage embeds
```

Lineage này quan trọng: **mỗi generation giải quyết bottleneck của generation trước**. Khi học, hỏi câu "predecessor đã fail ở đâu?" giúp hiểu motivation của design choice.

---

## 4. PinSage Architecture Deep Dive

### 4.1 Bipartite graph construction

**Pin–board bipartite graph** $G = (V_P \cup V_B, E)$:
- $V_P$ — set of pins (~3B nodes).
- $V_B$ — set of boards (~1B nodes).
- $E$ — edges: $(p, b) \in E$ nếu pin $p$ được "saved to" board $b$.

**Tại sao bipartite thay vì pin-pin graph trực tiếp?**

Pinterest có thể tạo pin-pin graph bằng nhiều cách: (a) co-view (user A xem pin X rồi Y), (b) co-save trong cùng board, (c) co-click trong cùng session. Tuy nhiên team chọn **bipartite pin-board** vì:

1. **Board carries semantic signal**: board là **human-curated grouping** với theme rõ ràng ("Bohemian living room ideas", "Vegan dinner recipes"). Hai pins trong cùng board không phải ngẫu nhiên — user actively chọn put them together → strong relevance signal.

2. **Reduces noise**: co-view signal rất noisy (user scroll quickly past many pins). Co-save trong board là **explicit, deliberate action** → much stronger.

3. **Sparser graph → tractable**: pin-pin direct co-occurrence graph sẽ dày hơn nhiều (mỗi pin link với nhiều pins qua nhiều paths). Bipartite filter through "board bottleneck" reduces edge count significantly.

4. **Reflects user mental model**: Pinterest UX is built around boards. Users think "this pin belongs to my kitchen-ideas board" — model nên align với cognitive structure này.

**Edge formation policy**:
- Edge $(p, b)$ added khi user save pin $p$ vào board $b$. Multiple users saving cùng pin vào cùng board → still one edge (paper version) hoặc edge weight = save count (extended versions).
- Edge removal: nếu user delete pin from board, edge removed. Stale edges (boards inactive > 1 year) có thể được decay-weighted.
- Self-loops: không có (pin chỉ liên kết với boards, board chỉ liên kết với pins → strict bipartite).

**Edge weight**: paper không emphasize edge weight (binary edges trong original PinSage); một số follow-up dùng weight = số user đã re-pin (popularity within board).

**Pin features** $x_p \in \mathbb{R}^{d_x}$:
- **Visual features**: 4096-dim từ VGG-16 pretrained, sau đó project xuống 2048-dim (lúc paper được viết). Sau này upgrade ResNet, EfficientNet, CLIP.
- **Text features**: 256-dim từ word2vec embedding của pin title + description, average pooled.
- **Concat**: $x_p = [\text{visual}; \text{text}]$ → 2048 + 256 = 2304-dim initial feature.

**Board features**: paper xử lý board như "virtual node" với feature aggregated từ member pins, hoặc dùng zero vector + learned bias. (Note: PinSage paper thực ra **chỉ học embedding cho pins**, board chỉ là intermediate node để traverse — không output board embedding.)

**Graph storage**: 18B edges cần distributed storage.
- Pinterest dùng **Hadoop HDFS** lưu graph dưới dạng adjacency lists.
- Per-pin adjacency list: list of board IDs (varint encoded).
- Per-board adjacency list: list of pin IDs.
- Lookup cost: O(log N) qua sorted lookup hoặc O(1) qua distributed key-value store như Pinterest's RocksDB-backed services.

**Storage size calculation**:
- Mỗi edge ~ 16 bytes raw (2× int64 IDs) → 18B × 16 = ~288 GB raw.
- Với varint encoding (giảm avg byte per ID xuống ~5-6 bytes do skewed distribution), tổng còn ~180 GB.
- Sharded qua HDFS: shard theo pin_id mod N (N = số shards, e.g. 1000) → mỗi shard ~180 MB.
- Replication factor 3 cho durability → effective storage ~540 GB.

**Graph access patterns**:
- **Sequential scan**: trong MapReduce inference, scan toàn bộ pins → optimal layout là sort by pin_id, scan sequentially → ~minutes per shard.
- **Random lookup**: trong online inference cho new pins, lookup neighbors của 1 pin → key-value lookup ~ms.
- **Adjacency expansion**: trong random walk, từ pin → boards → pins → cần 2-hop lookup hiệu quả → typically batch lookups qua batch RPC.

### 4.2 Random walk-based neighborhood sampling

Đây là **đổi mới đầu tiên** của PinSage. Thay vì uniform sample K neighbors, PinSage chạy short random walks từ pin source và đếm visit count.

**Algorithm (pseudo-code Python)**:

```python
def sample_pinsage_neighbors(p: PinID, T: int = 50,
                              num_walks: int = 1000,
                              walk_length: int = 2) -> List[Tuple[PinID, float]]:
    """
    Sample top-T neighbors for pin p using random walks.
    Returns list of (neighbor_pin_id, importance_score).
    """
    visit_counts = defaultdict(int)

    for _ in range(num_walks):
        current = p
        for step in range(walk_length):
            # Bước 1: pin -> board (random chọn 1 board chứa pin hiện tại)
            boards = get_boards_containing_pin(current)
            if not boards:
                break
            chosen_board = random.choice(boards)

            # Bước 2: board -> pin (random chọn 1 pin trong board, không phải pin nguồn)
            pins_in_board = get_pins_in_board(chosen_board)
            pins_in_board = [pin for pin in pins_in_board if pin != p]
            if not pins_in_board:
                break
            current = random.choice(pins_in_board)

            visit_counts[current] += 1

    # Top-T neighbors by visit count
    sorted_neighbors = sorted(visit_counts.items(),
                              key=lambda x: -x[1])[:T]

    # Normalize visit counts để dùng làm importance weight
    total = sum(c for _, c in sorted_neighbors)
    weighted = [(pin, count / total) for pin, count in sorted_neighbors]
    return weighted
```

**Tại sao random walk thay vì k-hop sampling như GraphSAGE original?**

1. **Importance signal "free"**: visit count tự nhiên cao cho neighbors gần & well-connected, thấp cho noise neighbors. Trong k-hop uniform sampling, không có signal để rank neighbors.

2. **Constant receptive field**: walk length fixed (e.g. 2) → mỗi pin có **fixed budget T=50 neighbors**, bất kể degree của pin. Tránh receptive field explosion ở high-degree pins (viral pins on hundreds of boards).

3. **Personalization-friendly**: random walks là personalized — start từ pin source, neighbors là pins "related" theo path. Đây là idea trực tiếp từ Pixie / Personalized PageRank.

4. **Visit count = soft importance**: pin gần (1-hop) sẽ có visit count cao hơn pin xa (2-hop), tự nhiên reflect importance.

**Tại sao T = 50?**

Paper đề cập T=50 là sweet spot empirical:
- T < 20: information loss, embedding quality giảm.
- T > 200: diminishing returns, training cost tăng linearly.
- T = 50 cân bằng quality và compute (mỗi forward pass cost ~ 50 neighbor lookups per layer).

**Walk length / num walks**: typical setting walks length 2 (paper), num walks ~ vài nghìn để estimate stable visit count. Length 2 đủ để hit boards rồi pins khác, không quá xa để noise.

**Visit count distribution intuition**:

Khi chạy 1000 random walks từ pin $p$, distribution của visit counts tuân theo power-law:
- Top 5-10 pins: visit count 30-100 (very close neighbors, co-occur with $p$ trong nhiều boards).
- Pins thứ 10-50: visit count 5-30 (moderate co-occurrence).
- Pins thứ 50+: visit count 1-5 (tail, mostly noise).

T=50 chính là chỗ "elbow" của distribution — beyond T=50, visit counts decay nhanh và signal-to-noise ratio giảm mạnh.

**Comparison concrete với uniform sampling**:

Giả sử pin $p$ là một pin về "Italian pasta recipes" trong 200 boards:
- **Uniform K-hop sample** (K=2, sample 50 neighbors uniformly): có thể sample được pins về pasta, nhưng cũng có thể sample pins về politics, sports, beauty (boards có pasta pin có thể có pins random khác). Noise ratio ~30-40%.
- **Random walk sample** (visit count weighted): top-50 pins gần như toàn là pasta/Italian-food pins (visit count concentrates trên neighbors thực sự semantically related). Noise ratio ~5-10%.

Đây là lý do PinSage achieve cao hơn ~15-20% recall@K so với baseline GraphSAGE với uniform sampling.

**Precomputation strategy**:

Vì random walk relatively expensive (1000 walks × walk_length × graph lookups), PinSage **precompute** neighborhood sample sets cho tất cả pins **offline, daily**, store kết quả trong HDFS. Training/inference không lặp lại random walks — chỉ load precomputed neighbors.

```python
# Daily preprocessing job (MapReduce):
# Input: full pin-board graph
# Output: {pin_id: [(neighbor_id_1, importance_1), ..., (neighbor_id_50, importance_50)]}

# Mapper: for each pin, run random walks
def preprocess_neighbors_mapper(pin_id, graph_partition):
    neighbors_with_importance = sample_pinsage_neighbors(
        pin_id, T=50, num_walks=1000, walk_length=2
    )
    yield (pin_id, neighbors_with_importance)

# Output sharded by pin_id mod N, stored as Parquet/Avro on HDFS.
# Total storage: 3B pins × (50 × (8 byte ID + 4 byte float)) = ~1.8 TB.
```

Trade-off: storage ~1.8TB but query time amortized → net win cho training/inference throughput.

### 4.3 Importance pooling aggregator

**Đổi mới thứ hai** của PinSage. Thay vì mean/LSTM/pooling như GraphSAGE, PinSage dùng **weighted aggregation với visit counts làm weights**.

**Formula**:

$$h_{N(v)}^{(k)} = \gamma\left(\{ \text{ReLU}(Q^{(k)} h_u^{(k-1)} + q^{(k)}) \cdot \alpha_u : u \in N(v) \}\right)$$

Trong đó:
- $Q^{(k)}, q^{(k)}$ — learnable weight matrix + bias tại layer $k$ (transform neighbor features).
- $\alpha_u$ — importance weight = visit count của $u$ trong random walk từ $v$, normalized.
- $\gamma$ — pooling function: **weighted sum** (paper) hoặc weighted-mean.

Sau khi có $h_{N(v)}^{(k)}$, kết hợp với node feature hiện tại:

$$h_v^{(k)} = \text{ReLU}\left(W^{(k)} \cdot \text{CONCAT}(h_v^{(k-1)}, h_{N(v)}^{(k)}) + w^{(k)}\right)$$

$$h_v^{(k)} \leftarrow h_v^{(k)} / \|h_v^{(k)}\|_2$$

**ASCII diagram forward pass cho 1 node 1 layer**:

```
                                                                      
   Source pin v                                                       
    ┌────────────┐                                                    
    │ h_v^{k-1}  │ ──────────────────┐                                
    └────────────┘                   │                                
                                     │                                
   Random walk sampling              │                                
   from v: top-T neighbors {u_1..u_T}│ CONCAT                         
   với importance α_1, .., α_T       ▼                                
                                ┌─────────┐                           
    ┌────────────┐ × α_1     →  │ neighbor│      ┌─────────┐          
    │ h_{u_1}    │ →  Q^k  →    │ aggregat│  →   │  W^k    │  ReLU →  
    └────────────┘ + ReLU       │ : weight│  →   │ + bias  │  → L2    
                                │ ed sum  │      └─────────┘  norm    
    ┌────────────┐ × α_2     →  │ over    │           │               
    │ h_{u_2}    │ →  Q^k  →    │ T neigh-│           │               
    └────────────┘ + ReLU       │ bors    │           ▼               
                                └─────────┘    ┌───────────┐          
    ...                              ▲         │  h_v^{k}  │          
                                     │         └───────────┘          
    ┌────────────┐ × α_T     →       │                                
    │ h_{u_T}    │ →  Q^k  → ────────┘                                
    └────────────┘ + ReLU                                             
```

**Tại sao importance pooling tốt hơn mean/LSTM của GraphSAGE?**

1. **Exploit signal sẵn có**: random walk đã produce importance score $\alpha$. Mean throws it away; importance pooling sử dụng nó. Đây là "free lunch" — info đã tính rồi, không dùng thì phí.

2. **Discriminative power cao hơn mean**: hai nodes có cùng neighbor set nhưng khác importance distribution sẽ produce different embeddings — mean không phân biệt được.

3. **Compute rẻ hơn LSTM**: LSTM phải xử lý neighbors sequentially (hoặc giả-sequentially); weighted sum là vectorized matrix op trên GPU, parallel hoàn toàn.

4. **Permutation-invariant**: weighted sum không phụ thuộc thứ tự neighbor (vì mỗi neighbor có importance riêng) — tốt cho graph data.

**Concrete example** (3 neighbors):
- $u_1$: visit count 30, normalized $\alpha_1 = 0.6$
- $u_2$: visit count 15, normalized $\alpha_2 = 0.3$
- $u_3$: visit count 5, normalized $\alpha_3 = 0.1$

Sau khi transform:
- $\text{ReLU}(Q h_{u_1} + q) = [1.2, 0.0, 0.5]$
- $\text{ReLU}(Q h_{u_2} + q) = [0.0, 0.8, 0.3]$
- $\text{ReLU}(Q h_{u_3} + q) = [0.5, 0.5, 0.0]$

Weighted sum: $h_{N(v)} = 0.6 \times [1.2, 0.0, 0.5] + 0.3 \times [0.0, 0.8, 0.3] + 0.1 \times [0.5, 0.5, 0.0]$
$= [0.72 + 0 + 0.05, 0 + 0.24 + 0.05, 0.3 + 0.09 + 0] = [0.77, 0.29, 0.39]$

So với mean (equal weights 1/3 each): $[\text{avg}] = [0.567, 0.433, 0.267]$ — distribution khác hẳn → embedding khác.

### 4.4 Multi-layer architecture (K=2)

PinSage dùng **K = 2 layers** (theo paper). Tại sao chỉ 2?

**Receptive field analysis**:

Với $T = 50$ neighbors per node per layer:
- Layer 1: mỗi pin "thấy" 50 neighbor pins.
- Layer 2: mỗi pin "thấy" 50 × 50 = 2500 pins (qua chain neighbor-of-neighbor).
- Layer 3: 50³ = 125,000 pins — explosion.

K=2 đã đủ cover 2-hop neighborhood ~ 2500 pins, vừa đủ context cho mỗi pin. K=3 explosion compute, và thực tế **over-smoothing** (xem section 9.1) làm quality giảm.

**Layer structure**:

```
Input pin features (visual + text) ∈ R^{2304}
        │
        ▼
   ┌──────────────────────────────┐
   │ Layer 1: importance pooling  │   Aggregate over 50 random-walk
   │ Q^1, q^1, W^1                │   neighbors. Each neighbor uses
   │ Output: h^1 ∈ R^{1024}       │   raw input features x_u.
   └──────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────┐
   │ Layer 2: importance pooling  │   Aggregate over 50 layer-1
   │ Q^2, q^2, W^2                │   embeddings of neighbors.
   │ Output: h^2 ∈ R^{512}        │
   └──────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────┐
   │ Dense projection layer        │   2 fully-connected layers
   │ Output: z ∈ R^{256}          │   to project to final embed.
   └──────────────────────────────┘
        │
        ▼
   L2 normalize → unit vector ∈ R^{256}
```

Final embedding $z_v \in \mathbb{R}^{256}$, L2-normalized → similarity = dot product = cosine similarity.

**Parameter count analysis**:

Cho 2-layer PinSage với input 2304-dim, hidden 1024 → 512, output 256:
- $Q^1 \in \mathbb{R}^{1024 \times 2304}$: ~2.4M params.
- $W^1 \in \mathbb{R}^{1024 \times (2304 + 1024)}$: ~3.4M params.
- $Q^2 \in \mathbb{R}^{512 \times 1024}$: ~0.5M params.
- $W^2 \in \mathbb{R}^{512 \times (1024 + 512)}$: ~0.8M params.
- Dense projection (512 → 256, 256 → 256): ~0.2M params.
- **Total**: ~7-10M params.

So với DLRM/Wide&Deep ranking models (100M+ params), PinSage relatively **lightweight** — vì work chính nằm ở graph aggregation, không phải parameters.

**Receptive field math chi tiết**:

Với T=50 neighbors per node, K=2 layers:
- Mỗi pin embedding final $z_v$ depends on features của ~$T^K = 2500$ unique pins (assuming no overlap, real count slightly less do graph clustering).
- Compute cost per pin forward pass: $O(T \cdot d_{hidden})$ per layer × $K$ layers = $O(K \cdot T \cdot d_{hidden})$ FLOPs.
- Numerically: 2 × 50 × 1024 = ~100K FLOPs per pin for aggregation, plus matrix mults ~10M FLOPs total per pin (~10ms on GPU, ~100ms on CPU).

### 4.5 Hard negative mining curriculum

**Đổi mới thứ ba** quan trọng. PinSage train với **max-margin ranking loss**:

$$\mathcal{L}(z_q, z_p, z_n) = \max(0, z_q \cdot z_n - z_q \cdot z_p + \Delta)$$

Trong đó:
- $z_q$ — query pin embedding.
- $z_p$ — positive pin (pin actually relevant — co-occur trong board của query, hoặc click sequence).
- $z_n$ — negative pin.
- $\Delta$ — margin (e.g. 0.1).

Loss này push positive closer hơn negative ít nhất margin.

**Vấn đề với random negative**: nếu sample negative uniformly từ 3B pins, khả năng cực cao pick được pin **hoàn toàn irrelevant** (e.g. query là pin về recipes, negative là pin về cars). Loss này quá easy — model học được embedding rộng nhưng không phân biệt fine-grained.

**Hard negative**: pin **gần với query trong embedding space** nhưng **không thực sự relevant**. Example: query là "rustic farmhouse kitchen design", hard negative có thể là "industrial loft kitchen design" — cùng đều là kitchen design nhưng style hoàn toàn khác.

**Hard negative mining curriculum** trong PinSage:

```python
def hard_negative_mining(epoch: int, query_pin: PinID,
                          all_pins: List[PinID],
                          current_embeddings: Dict[PinID, ndarray]) -> PinID:
    """
    Curriculum: epoch sớm dùng easy negatives, epoch muộn dùng hard.
    """
    if epoch < 3:
        # Easy negative: random sample
        return random.choice(all_pins)
    else:
        # Hard negative: rank candidates by PageRank score (excluding
        # actual neighbors of query), then pick from rank 1000-2000.
        # Rank 1-100: too similar, possibly true positive.
        # Rank > 5000: too dissimilar, easy.
        # Rank 1000-2000: "hard" — similar enough to confuse model.

        query_neighbors = set(get_random_walk_neighbors(query_pin, T=500))
        candidates = [p for p in all_pins if p not in query_neighbors]

        # Compute personalized PageRank scores từ query_pin
        ppr_scores = compute_personalized_pagerank(query_pin, candidates)
        sorted_by_ppr = sorted(candidates, key=lambda p: -ppr_scores[p])

        # Pick from rank 1000-2000
        hard_pool = sorted_by_ppr[1000:2000]

        return random.choice(hard_pool)
```

**Tại sao curriculum quan trọng?**

Nếu start training ngay với hard negatives, model chưa biết embedding cấu trúc → loss noisy, không converge. Bắt đầu easy → model học coarse structure → sau đó hard negative finetune fine-grained boundaries. Đây là idea cổ điển trong curriculum learning (Bengio et al. 2009).

Paper PinSage reports: dùng hard negative curriculum **cải thiện hit-rate@10 thêm ~12%** so với chỉ easy negatives.

**Number of hard negatives per batch**: paper recommend ~6 hard negatives + reuse các positive trong batch làm "in-batch negatives" (như sampled softmax trick).

**Math chi tiết loss function** với hard negatives:

$$\mathcal{L}_{batch} = \sum_{i=1}^{B} \left[ \sum_{j \in H_i} \max(0, z_{q_i} \cdot z_{n_j} - z_{q_i} \cdot z_{p_i} + \Delta) + \sum_{k \neq i} \max(0, z_{q_i} \cdot z_{p_k} - z_{q_i} \cdot z_{p_i} + \Delta_{soft}) \right]$$

Trong đó:
- $B$ — batch size (512 trong paper).
- $H_i$ — set hard negatives cho query $i$ (~6 items).
- $\Delta$ — hard margin (e.g. 0.1).
- $\Delta_{soft}$ — soft margin cho in-batch negatives (e.g. 0.05, smaller vì in-batch có thể là false negatives — pins khác trong batch có thể thực ra relevant).

**Gradient flow intuition**:
- Khi $z_q \cdot z_n + \Delta > z_q \cdot z_p$ (negative quá gần), gradient kéo $z_q$ và $z_p$ closer, đẩy $z_n$ xa ra. Margin loss → linear gradient (not exploding như cross-entropy).
- Khi margin được satisfied (positive far enough), gradient = 0, no update — efficient compute.

**Why max-margin thay vì cross-entropy / softmax?**
- Max-margin only updates "violated" examples → less wasted compute on already-correct.
- More robust to noise: hard examples không bị over-weighted như trong softmax với temperature scaling.
- Embedding norm controlled implicitly by L2 normalization step.

---

## 5. Training Infrastructure Innovation

Đây là phần khiến PinSage **production-ready** — không chỉ có model design tốt mà còn có training architecture chịu được scale.

### 5.1 Producer-consumer training pipeline

**Vấn đề**: GPU forward/backward pass cho mini-batch PinSage rất nhanh (~tens of ms), nhưng **prepare mini-batch** (random walk sampling, feature lookup từ HDFS) cực kỳ chậm nếu làm sequential. Nếu GPU phải đợi data → GPU utilization < 20% → waste hardware.

**Giải pháp**: separate **producer** (CPU) và **consumer** (GPU) qua queue.

```
┌───────────────────────────────────────────────────────────────────┐
│ HDFS / Distributed graph storage                                  │
│ - Pin features (visual + text)                                    │
│ - Pin-board adjacency lists                                       │
│ - Board-pin adjacency lists                                       │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ MapReduce feature precomputation (offline, daily)                 │
│ - Random walk sampler (~100 worker processes)                     │
│ - Per pin: precompute T=50 neighbors + importance weights         │
│ - Output: HDFS shards with                                        │
│     {pin_id: [(neighbor_id, importance), ..., feature_vector]}    │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ CPU Producer Pool (many threads)                                  │
│ - Sample batch of query pins                                      │
│ - For each query: lookup positive (from board co-occurrence)      │
│ - Lookup hard negatives (curriculum-based)                        │
│ - Lookup feature tensors for query + positives + negatives        │
│ - Lookup 2-hop neighbor features (50×50 = 2500 features per pin)  │
│ - Pack into tensors, send to GPU queue                            │
└─────────────────────────────┬─────────────────────────────────────┘
                              │ batches queued
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ GPU Consumer (1 or more GPUs, data parallel)                      │
│ - Pop batch from queue                                            │
│ - Forward pass through 2-layer PinSage                            │
│ - Compute max-margin loss                                         │
│ - Backward + Adam update                                          │
│ - Sync weights across GPUs (if data parallel)                     │
└───────────────────────────────────────────────────────────────────┘
```

**Producer-consumer ratio**: paper sử dụng ~32 CPU producer threads per GPU consumer để keep GPU fed. Modern setup (GPUs nhanh hơn) có thể cần 64+ producers.

### 5.2 Constructing mini-batches

**Batch composition** (paper):
- **512 positive pairs** per batch.
- Mỗi positive pair: (query pin, related pin). Source data: pins co-occurring in same board within short time window, hoặc click sequences (user clicked A then B).
- **6 hard negatives** per query pin (sampled per epoch curriculum).
- **Implicit in-batch negatives**: cho mỗi positive pair (q_i, p_i), các p_j với $j \neq i$ làm negatives cho q_i (free, không cần extra lookup).

**Memory layout**:
- Query features: $[512, 2304]$ float32.
- Positive features: $[512, 2304]$.
- Negatives: $[512, 6, 2304]$.
- For each pin, 50 neighbor features at layer 1: $[512, 50, 2304]$ — đây là chunk lớn nhất.
- For each pin, 50 layer-1 neighbors's 50 layer-2 neighbors: $[512, 50, 50, 2304]$ — chunk khổng lồ! Đây là lý do batch size cap ở 512.

**Total batch memory**: ~few GB GPU memory chỉ riêng cho data, chưa kể activations. K80 (24GB) per paper era đã căng — modern A100 (80GB) thoải mái hơn.

### 5.3 Multi-GPU training

PinSage paper báo cáo train với **32 GPUs data parallel**:
- Mỗi GPU process 1 batch (512 pairs) → effective batch size = 32 × 512 = 16,384 pairs per step.
- Gradient synchronization: AllReduce qua MPI / NCCL.
- Sync mode: synchronous SGD (paper era — async ít dùng cho production).

**Hardware paper era vs modern**:
| Component | 2018 paper | Modern (2024+) |
|---|---|---|
| GPU | NVIDIA K80 (24GB) × 32 | A100 80GB × 8-16, hoặc H100 |
| GPU memory | 24GB | 80GB |
| Interconnect | PCIe (no NVLink at K80) | NVLink + InfiniBand |
| Framework | Custom MXNet-based | PyTorch + PyG hoặc DGL |
| Throughput | ~hours per epoch on 7.5B pin subgraph | Estimate 3-5× faster |

### 5.4 MapReduce inference pipeline (KEY innovation)

**Vấn đề**: sau khi train xong, cần compute embedding cho **toàn bộ 3B pins**. Naive approach (mỗi pin chạy 1 forward pass online): 3B × ~10ms = 30M giây ≈ 1 năm CPU time / chỉ tính theo single thread. Không khả thi.

**Insight**: nhiều pin **share neighbors**. Pin A và pin B nếu cùng được link từ board "cooking", có thể share 30/50 neighbors. Naive inference compute layer-1 embedding cho từng neighbor riêng cho A và riêng cho B → duplicate work.

**Giải pháp**: **Two-pass MapReduce inference** với memoization.

**Pass 1**: Compute layer-1 embedding cho **tất cả pins** trước.

```
MapReduce Pass 1:
  Input: {pin_id: (raw_features, [list of 50 random-walk neighbor IDs])}
  
  Map phase:
    For each pin p:
      Emit (p, "layer1_input", raw_features)
      For each neighbor u of p:
        Emit (u, "needed_for", p)   # mark that u's raw features needed
  
  Reduce phase (per pin u):
    Collect raw_features of u
    For each p that "needs" u, just provide u's raw features in next map.
  
  Then a "compute" map:
    For each pin p:
      Lookup raw_features of p's 50 neighbors
      Run layer 1 forward pass: h^1_p = importance_pool(Q^1, neighbors)
      Emit (p, h^1_p)
  
  Output: {pin_id: h^1 ∈ R^{1024}} — 3B records.
```

**Pass 2**: Compute layer-2 embedding using layer-1 outputs.

```
MapReduce Pass 2:
  Input: {pin_id: h^1} từ Pass 1
  
  Map phase:
    For each pin p:
      For each neighbor u (2-hop neighbor of p actually = 1-hop của pass 2):
        Emit (p, "needs_h1_of", u)
  
  Reduce: collect h^1 vectors needed.
  
  Compute map:
    For each pin p:
      Lookup h^1 of p's 50 neighbors
      Run layer 2 forward pass: h^2_p = importance_pool(Q^2, h^1 neighbors)
      Project to 256-dim, L2 normalize
      Emit (p, z_p)
  
  Output: {pin_id: z ∈ R^{256}} — final embeddings.
```

**Memoization win**: trong Pass 1, mỗi pin compute $h^1$ **đúng một lần**, dùng lại bởi tất cả pins cần nó (qua data redistribution của MapReduce reduce phase). Naive approach: nếu pin A có 50 neighbors, mỗi neighbor có 50 neighbors → A cần compute 50×50 = 2500 layer-1 forward passes (duplicated). MapReduce: compute once per pin → 3B passes total, not 3B × 50.

**Speedup empirical**: paper báo cáo MapReduce inference cho toàn bộ corpus xong **trong 1 ngày trên ~378 Hadoop nodes**. Naive would take orders of magnitude longer.

**Pseudo-code Python flavor cho Pass 1**:

```python
# Pass 1 - compute layer-1 embeddings for all pins
def map_phase_pass1(pin_record):
    """Map: emit raw features keyed by pin_id."""
    pin_id, features, neighbors, importance = pin_record
    yield (pin_id, ("self", features, neighbors, importance))
    # Also emit neighbor "needs" to redistribute features:
    for u in neighbors:
        yield (u, ("needed_by", pin_id))

def reduce_phase_pass1(pin_id, records):
    """Reduce: collect features + tags."""
    self_rec = None
    needed_by = []
    for rec in records:
        if rec[0] == "self":
            self_rec = rec
        elif rec[0] == "needed_by":
            needed_by.append(rec[1])
    yield (pin_id, self_rec, needed_by)

# Now a second pass: each pin already has its raw features broadcasted.
def compute_layer1(pin_id, self_rec, neighbor_features):
    """Compute h^1 for this pin using its neighbors' raw features."""
    _, features, neighbors, importance = self_rec
    
    # Transform neighbors via Q^1 + ReLU
    transformed = []
    for u, alpha in zip(neighbors, importance):
        x_u = neighbor_features[u]  # raw features
        t_u = relu(Q1 @ x_u + q1)
        transformed.append((t_u, alpha))
    
    # Importance-weighted sum
    h_neighbor = sum(alpha * t for t, alpha in transformed)
    
    # Combine with self
    h_self = relu(W1 @ concat(features, h_neighbor) + w1)
    h_self = h_self / norm(h_self)
    return h_self
```

Cấu trúc Pass 2 tương tự, chỉ thay raw features bằng layer-1 embeddings.

### 5.5 Detailed MapReduce execution analysis

**Resource accounting cho 1 full inference run (3B pins)**:

| Stage | Time | Compute | I/O |
|---|---|---|---|
| Load graph + precomputed neighbors | ~30 min | 378 nodes × 8 cores | Read ~2 TB từ HDFS |
| Pass 1 map (compute h^1 cho all pins) | ~4-6 hours | 378 nodes × 8 cores × GPU optional | Read raw features ~3 TB, write h^1 ~3 TB |
| Shuffle phase (broadcast h^1 to neighbors needing it) | ~2 hours | Network-bound | Network shuffle ~15 TB total |
| Pass 2 map (compute z = h^2 cho all pins) | ~4-6 hours | 378 nodes | Read h^1 ~3 TB, write z ~750 GB (after fp16) |
| Index build (HNSW / IVF-PQ) | ~3 hours | 100+ nodes | Write index ~200 GB |
| Verification / sanity checks | ~30 min | Single node | Recall@K on golden set |
| **Total** | **~18-24 hours** | | |

**Key insight**: shuffle phase là **bottleneck**. Mỗi pin layer-1 embedding cần được broadcast tới tất cả pins có nó làm neighbor (avg 50 per pin × 3B / pin = 150B individual reads, but with batching reduces to manageable).

**Optimizations Pinterest đã áp dụng**:

1. **Locality-aware partitioning**: shard pins by community detection trên graph → high-connectivity pins on same node → reduce cross-shard shuffle.

2. **Buffered shuffle**: instead of writing individual records, batch up many records per (target_shard) → fewer network ops.

3. **Incremental refresh**: nếu chỉ 5% pins change daily, recompute embedding chỉ cho changed pins + their 2-hop neighbors. ~10× speedup for daily incremental refresh vs full recompute.

4. **GPU acceleration for forward pass**: K80 era khó vì memory constraints; modern infra dùng A100 cho pass 1/2 compute, achieve ~3-5× speedup vs CPU-only.

**Production pipeline orchestration**:

```
Daily PinSage refresh job (Airflow DAG, runs at 02:00 UTC daily):

┌─────────────────────────────────────────────────────────┐
│ T+0h:    Snapshot graph state (pin-board adjacency)     │
│ T+0.5h:  Run random walk neighbor precomputation        │
│ T+2h:    [If model retrain day] Train PinSage on GPU    │
│           cluster, ~6-8 hours.                          │
│           Else: load latest checkpoint.                 │
│ T+10h:   MapReduce Pass 1 — compute h^1 for all pins    │
│ T+14h:   MapReduce Pass 2 — compute final z for all pins│
│ T+18h:   Quantize embeddings (fp32 → fp16/int8)         │
│ T+19h:   Build ANN indices (HNSW + IVF-PQ)              │
│ T+21h:   Recall@K validation on golden set              │
│           If recall drop > 2% vs yesterday: abort, alert│
│ T+22h:   Deploy new embeddings + indices to serving     │
│           (blue-green deployment via traffic split)     │
│ T+23h:   Monitor production metrics; auto-rollback if   │
│           CTR drops > 1% vs baseline.                   │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Production Deployment & Serving

### 6.1 Embedding storage

Sau khi MapReduce inference xong, output là **3B pin → 256-dim float32 embedding**.

| Format | Size per pin | Total (3B pins) |
|---|---|---|
| float32 raw | 256 × 4 = 1024 bytes | ~3 TB |
| float16 | 256 × 2 = 512 bytes | ~1.5 TB |
| int8 quantized | 256 × 1 = 256 bytes | ~750 GB |
| Product quantization (PQ-64) | ~64 bytes | ~192 GB |

Pinterest production typically dùng **fp16** hoặc **int8** cho embedding storage (balance accuracy vs storage). Cho **ANN index**, dùng PQ thêm để giảm hơn.

### 6.2 Quantization details

**FP16**: chỉ truncate mantissa, accuracy loss < 0.1% on recall@K. Default safe choice.

**INT8 quantization**:
- Compute per-dim min/max của embedding distribution trên sample 1M pins.
- Quantize: $q_i = \text{round}((e_i - \min_i) / (\max_i - \min_i) \times 255)$.
- Dequantize on the fly trước khi compute dot product.
- Accuracy loss ~1-2% recall@K — acceptable.

**Product Quantization (PQ)**:
- Chia 256-dim vector thành M sub-vectors (e.g. M=32, mỗi sub-vector 8-dim).
- Mỗi sub-vector: cluster thành 256 codebooks (k-means).
- Lưu mỗi sub-vector bằng 1 byte (256 = $2^8$ codebook IDs).
- Total: 32 bytes per embedding → 32× compression.
- Accuracy loss ~5-10%; thường dùng với reranking (PQ for first pass, full float for top-K rerank).

### 6.3 ANN serving

**Approximate Nearest Neighbor search** cho 3B vectors 256-dim:

| Algorithm | Pros | Cons | Pinterest use? |
|---|---|---|---|
| HNSW | Best recall, fast queries | High memory (~3-4x raw vectors) | Likely used for hot pin subset |
| IVF-PQ | Lower memory (PQ), reasonable recall | Slower than HNSW at same recall | Used at scale for full corpus |
| ScaNN (Google) | SOTA accuracy on benchmark | Less mature open-source | Possible |
| Custom partitioning | Pinterest-specific optimizations | Engineering complexity | Pinterest mentioned custom system "Manas" |

Pinterest production likely uses **hybrid**:
- Hot subset (most popular ~100M pins): in-memory HNSW for fast lookup.
- Cold tail: IVF-PQ on disk-backed, with periodic warm-up.
- Reranking: load full fp16 embeddings cho top-1000 candidates và rerank with exact dot product.

### 6.4 Hot pin caching

**Power law observation**: pin popularity follows Zipfian — top 1% pins account for ~50%+ of retrieval requests.

**Cache strategy**:
- **L1**: in-process memory cache top-100K pins (most popular last 24h) — ~25MB if int8.
- **L2**: Redis cluster cache top-100M pins — ~25GB if int8.
- **L3**: full embedding store (RocksDB / custom distributed KV).

**Eviction**: LRU within tier, but with popularity weighting (Pinterest internal).

### 6.5 Latency breakdown

For a single retrieval request (1 query pin → top-1000 similar pins):

```
Total budget: 50ms P99
├─ Network ingress + parse: 2ms
├─ Query embedding lookup (cache hit): 1ms
│   (cache miss + fetch from KV: +5-10ms — accept rarely)
├─ ANN search (HNSW or IVF-PQ): 15-25ms
│   - HNSW: traverses log(N) layers, each layer ~M edges
│   - IVF-PQ: lookup centroid + scan ~5% of corpus partitions
├─ Top-K candidates fetched (1000 pin metadata): 5ms parallel
├─ Optional rerank with full fp16 embeddings: 5ms
└─ Response serialization + egress: 2ms
                          Total ≈ 30-45ms (P99 50ms budget OK)
```

### 6.6 Online inference for new pins

Pin mới (created < 1 hour, không có trong MapReduce daily output) cần embedding ngay để xuất hiện trong Related Pins.

**Strategy**:
- Pin mới chưa có 2-hop neighbor history (board nó được save vào có thể có neighbors, nhưng signal ít).
- **Cold-start hack**: dùng visual + text features pass qua phần "self" của model (skip aggregation layers), output coarse embedding.
- Sau 1-7 ngày khi đã có enough board placements, MapReduce sẽ pickup pin và compute proper embedding ở daily batch.

Online forward pass cho pin mới: ~30-50ms (model relatively small) — acceptable cho trickle volume new pins.

### 6.7 Monitoring & observability

Production PinSage cần monitoring đa chiều:

**Model quality metrics** (daily):
- Recall@10, Recall@100, MRR trên golden human-labeled set (~10K pairs).
- Embedding distribution drift: PSI vs yesterday on sample 1M pins.
- Average L2 norm (should ~1.0), variance, % NaN.

**Serving metrics** (realtime):
- ANN search P50/P99 latency.
- Cache hit rate per tier (L1/L2/L3).
- QPS per surface (Related Pins, Home Feed, Search).
- Error rate (timeouts, KV store failures).

**Business metrics** (offline + online):
- CTR per surface, save rate, hide rate.
- Diversity: % distinct categories in top-K results.
- Coverage: % of corpus surfaced over 7-day window.
- A/B test wins/losses vs candidate models.

**Alert thresholds** (examples):
- Recall@10 drops >2% day-over-day → page on-call.
- P99 latency >100ms for 5 minutes → page.
- Cache hit rate drops >5% → investigate (possibly stale embedding deployment).
- CTR drops >1% in production over 30-min → auto-rollback embedding.

### 6.8 Capacity planning

Cho serving 3B pins ở scale Pinterest:
- **Embedding storage** (fp16 + PQ for ANN, fp16 for rerank): ~3 TB hot in-memory + ~30 TB cold disk-backed.
- **ANN serving nodes**: ~50-100 nodes per region, each handling ~20K QPS.
- **Replication**: 3x across availability zones for redundancy.
- **Cost** (estimated): ~$100-300K/month for serving infra at this scale (very rough, depends on cloud provider).

---

## 7. Improvements Over Time — PinnerSage, TransAct, multi-modal fusion

### 7.1 PinnerSage (KDD 2020) — Multi-embedding cho user

**Limitation phát hiện sau khi PinSage deploy**: PinSage chỉ output pin embedding, nhưng để recommend cho **user**, Pinterest cần **user embedding**. Approach đơn giản nhất: average tất cả PinSage embeddings của pins mà user đã engage.

**Vấn đề "interest dilution"**: user có nhiều interests rất khác nhau (e.g. recipes + woodworking + travel). Average → embedding "mờ" ở giữa các interest cluster, không recall được pins thuộc cluster nào.

**Concrete example**:
- User Alice engaged 100 pins: 40 về "vegan recipes" (cluster A), 30 về "DIY woodworking" (cluster B), 30 về "Iceland travel" (cluster C).
- Average embedding $z_{Alice}$ rơi vào vùng "trung tâm" giữa A, B, C.
- ANN với $z_{Alice}$ → return pins ở vùng trung tâm, thường là **không thuộc cluster nào** → quality kém.

**PinnerSage giải pháp**: multi-embedding per user via hierarchical clustering của user actions.

**Algorithm**:

```python
def compute_pinnersage_user_embeddings(user_id: UserID,
                                        time_window_days: int = 90,
                                        max_clusters: int = 3) -> List[ndarray]:
    """
    Hierarchical clustering của user's pin engagements.
    Output: list of cluster centroids = user's multi-embedding.
    """
    # Step 1: collect user's engaged pins in window
    actions = get_user_actions(user_id, days=time_window_days)
    pin_ids = [a.pin_id for a in actions]
    
    # Step 2: lookup PinSage embeddings
    embeddings = [pinsage_embed[p] for p in pin_ids]  # N × 256
    
    # Step 3: hierarchical agglomerative clustering với Ward linkage
    cluster_tree = ward_linkage(embeddings)
    
    # Step 4: cut tree at threshold (distance < tau)
    # tau tuned to produce avg 3-5 clusters per active user
    clusters = cut_tree(cluster_tree, threshold=0.4)
    
    # Step 5: pick top-K clusters by recency-weighted size
    cluster_scores = []
    for cluster in clusters:
        actions_in_cluster = [actions[i] for i in cluster.members]
        recency_weight = sum(exp(-0.1 * action.days_ago)
                             for action in actions_in_cluster)
        cluster_scores.append((cluster, recency_weight))
    
    top_clusters = sorted(cluster_scores, key=lambda x: -x[1])[:max_clusters]
    
    # Step 6: centroid (medoid actually — pin closest to cluster center)
    user_embeddings = []
    for cluster, _ in top_clusters:
        member_embeds = [embeddings[i] for i in cluster.members]
        centroid = mean(member_embeds, axis=0)
        # Medoid: actual pin embedding closest to centroid
        medoid = max(member_embeds, key=lambda e: dot(e, centroid))
        user_embeddings.append(medoid)
    
    return user_embeddings  # Length 1-3
```

**Serving với multi-embedding**:
- ANN query với mỗi user embedding riêng → top-K per cluster.
- Merge results, dedupe, blend với cluster weights.
- Mỗi "section" trong Home Feed có thể correspond với 1 cluster.

**Trade-off**:
- **Storage cost**: 3-5× user embeddings vs 1 (acceptable — user count < pin count).
- **Quality gain**: PinnerSage paper reports ~25% improvement in user engagement metrics over single-embedding baseline.
- **Engineering complexity**: clustering pipeline + multi-query ANN — non-trivial.

### 7.2 TransAct (KDD 2023) — Sequence-aware extension

**Limitation của PinSage + PinnerSage**: static embeddings, không capture **realtime user behavior**. User vừa click 5 pins về "halloween costume" trong 10 phút qua → muốn next recommendation reflect intent này. Daily-updated PinnerSage clusters sẽ slow to catch up.

**TransAct giải pháp**: combine **realtime action sequence** + **PinSage embeddings** qua transformer.

**Architecture (high-level)**:

```
Realtime user actions (last 100 actions, ~minutes):
[click_pin_1, save_pin_2, hide_pin_3, ...]
         │
         ▼
Action sequence → PinSage embeddings:
[z_pin_1, z_pin_2, z_pin_3, ...]  (each 256-dim)
         │
         ▼
Add positional encoding + action type embedding
         │
         ▼
Transformer encoder (2-4 layers, ~8 heads)
         │
         ▼
Pooled context vector (CLS token or mean of last K tokens)
         │
         ▼
Combine với:
- Long-term user embeddings (PinnerSage)
- Candidate pin features
         │
         ▼
Final ranking score
```

**Innovation**: TransAct dùng PinSage embedding **as input tokens** vào transformer (mỗi pin = 1 token = 256-dim vector). Transformer học **temporal attention** giữa các actions. Output = realtime user representation.

**Impact**: Pinterest reported TransAct gave significant improvement in Home Feed engagement (+11% saves rate per session in some experiments).

**Lineage tóm tắt**:

```
PinSage (2018): pin embeddings, static.
       │
PinnerSage (2020): user multi-embedding, daily refresh, no realtime.
       │
TransAct (2023): realtime sequence transformer on PinSage tokens.
       │
   (future): unified multi-modal foundation (image + text + graph + sequence)?
```

### 7.3 PinText / multi-modal fusion

Pinterest progressively integrated:
- **Visual features**: CNN → ResNet → CLIP-style image encoder.
- **Text features**: word2vec → BERT-based → multilingual text encoder.
- **Cross-modal alignment**: image-text contrastive training (CLIP-style).

These feed into PinSage as **input features $x_p$** — quality của input features ảnh hưởng trực tiếp quality output embeddings. Pinterest reports ~5-10% recall improvement mỗi lần upgrade visual backbone.

### 7.4 Cold-start improvements

For pin mới chưa có graph context:
- **Content-only initial embedding**: pass visual + text features through "self only" branch of PinSage (skip aggregation).
- **Bootstrap with similar pins**: find K nearest pins by visual feature (image embedding), borrow their PinSage embedding as proxy.
- **Active learning**: surface new pins in controlled exposure (low-volume), collect engagement signal, incorporate into graph next refresh.

### 7.5 Sequence-aware extensions (future direction)

Limitation phổ biến của static embedding: user's interest evolves. "Same-style" pins co-occurring during 2020 (pandemic baking trends) khác hẳn co-occurrence patterns 2024.

**Approaches từ literature (chưa rõ Pinterest deploy chưa)**:

- **Temporal GNN**: edge features include timestamp, model learns time-aware aggregation.
- **Dynamic GNN**: graph snapshots at intervals, embeddings change over time.
- **Continual learning**: incremental update without full retrain.

### 7.6 Improvement journey summary table

| Year | System | Key innovation | Quality lift | Cost trade-off |
|---|---|---|---|---|
| 2018 | PinSage | Random walk + importance pool + MapReduce inference | +150% Related Pins (human eval) vs Pixie | 1 day daily inference |
| 2019 | PinSage + better visual backbone | Upgrade VGG → ResNet50 | +5-8% recall | Training cost +20% |
| 2020 | PinnerSage | Multi-embedding user (hierarchical clustering) | +25% engagement (paper claim) | 3-5× user embedding storage |
| 2021-22 | PinSage v3 | Add board-level embedding output, multi-task heads | +3-5% home feed | More complex training pipeline |
| 2023 | TransAct | Transformer over realtime action sequence | +11% saves per session | Adds online inference cost |

**Pattern observation**: improvements không chỉ thay đổi architecture mà còn thay đổi **representation granularity** (single → multi embedding) và **time scale** (daily static → realtime sequence). Đây là evolution pattern phổ biến trong recsys industry.

---

## 8. Trade-offs & Design Decisions

### 8.1 PinSage vs GCN — scale-ability

| Criterion | GCN (Kipf-Welling 2017) | PinSage (Ying et al. 2018) |
|---|---|---|
| Setting | Transductive | Inductive |
| Sampling | None (full neighbor) | Random walk, top-T |
| Aggregator | Symmetric normalization | Importance pooling |
| Training | Full-batch SpMM | Mini-batch + producer-consumer |
| Max graph size (paper era) | ~10K nodes | 3B nodes |
| Handles new nodes? | No (must retrain) | Yes (forward pass on new neighbors) |
| Best for | Small academic graphs (Cora, Citeseer) | Web-scale graphs (Pinterest, Alibaba) |

### 8.2 Random walk sampling vs k-hop sampling

| Criterion | k-hop (GraphSAGE) | Random walk (PinSage) |
|---|---|---|
| Compute per node | $O(d^K)$ với $d$ = sampled degree | $O(\text{num\_walks} \times \text{walk\_len})$ |
| Provides importance? | No | Yes (visit count) |
| Handles high-degree nodes | Bad (must sample many to cover) | Good (concentrated on high-PPR nodes) |
| Locality | Strict hop boundary | Soft (PPR-style) |
| Quality empirical | Baseline | +15-20% recall vs uniform |

### 8.3 Importance pooling vs mean vs LSTM

| Aggregator | Compute | Discriminative power | Uses edge info? |
|---|---|---|---|
| Mean | Cheap (vectorized) | Low (loses neighbor identity) | No |
| LSTM | Slow (sequential) | High | No (treats as sequence) |
| Max pooling | Cheap | Medium | No |
| **Importance pooling** | Cheap (weighted sum) | High | Yes (visit counts) |

### 8.4 MapReduce inference vs online vs batch micro-batched

| Approach | Latency per pin | Total time 3B pins | Use case |
|---|---|---|---|
| Online per-pin | ~10-50ms | ~year (sequential) / impractical | Cold-start new pins only |
| Naive batch on GPU | ~1ms amortized | days (~7-14 days) | Small corpus (<100M) |
| **MapReduce two-pass** | n/a (offline) | ~1 day on cluster | 3B+ pins, daily refresh |
| Streaming incremental | ~10ms per update | Continuous | Future direction |

### 8.5 Single embedding vs multi-embedding (PinSage vs PinnerSage)

| Approach | Storage per user | Capture diverse interests? | Compute at serving |
|---|---|---|---|
| Single avg PinSage | 256 × 4 = 1KB | Bad (dilution) | 1 ANN query |
| PinnerSage (3-5 clusters) | 3-5 KB | Good | 3-5 ANN queries + merge |

Pinterest tolerated 3-5× storage cost vì engagement improvement >20% → clear win.

### 8.6 GNN vs Collaborative Filtering vs Deep Cross Network

| Approach | Cold-start | Captures graph? | Captures sequential? | Scale |
|---|---|---|---|---|
| Matrix Factorization (ALS) | Bad | No | No | Up to ~100M items |
| Two-tower DNN | OK (content features) | No | No | Up to billions |
| **PinSage / GNN** | OK (inductive) | Yes (multi-hop) | No | Up to billions+ |
| DIN/DIEN (Alibaba) | OK | No | Yes (attention over history) | Billions |
| Hybrid (GNN + sequence) | OK | Yes | Yes | TransAct, future |

GNN shines khi **graph structure carries signal beyond content** (Pinterest pin-board, Amazon item-co-purchase, Twitter follow graph). Less useful khi graph thưa hoặc no relational structure (single-feed apps).

### 8.7 Pinterest PinSage vs Alibaba GraphSAGE vs Uber graph-based reco

| System | Domain | Graph | Sampling | Aggregator | Notes |
|---|---|---|---|---|---|
| Pinterest PinSage | Visual reco | Pin-board bipartite | Random walk | Importance pool | First production web-scale GNN |
| Alibaba EGES (KDD 2018) | E-commerce | Item co-occurrence | Random walk + meta-path | Side-info embedding | Multi-meta-path graphs |
| Alibaba GraphSAGE-CTR | Ads | User-item bipartite | k-hop with sampling | Mean / attention | Integrated into ad ranking |
| Uber food / driver reco | Food | Food-user-restaurant tripartite | Heterogeneous random walk | Attention | Multi-entity graph |
| LinkedIn LiGNN (2021+) | Social/jobs | Member-content + member-member | k-hop | Attention | Cold-start focus |

Common thread: **random-walk sampling + learned aggregator** = dominant pattern cho production GNN.

---

## 9. Common Pitfalls When Implementing GNN-based Reco

### 9.1 Over-smoothing problem với deep GNN (K > 3)

**Symptom**: stack nhiều layers (K=4,5,6) → embeddings của các nodes converge tới very similar vectors → mất discriminative power.

**Cause**: mỗi GNN layer là averaging operation. Sau K layers, mỗi node thấy K-hop neighborhood. Nếu graph connected, K đủ lớn → mỗi node thấy "cả graph" → mọi node converge tới global average.

**Math intuition**: GCN layer là $H^{(l+1)} = \tilde{A} H^{(l)} W$. Iterating này tương đương random walk lên graph. Random walk dài → distribution converge tới stationary distribution (PageRank) — same cho mọi starting node.

**Mitigations**:
- Limit depth: K=2 hoặc K=3 max (PinSage chọn K=2).
- **Residual connections**: $h^{(l+1)} = \text{GNN}(h^{(l)}) + h^{(l)}$.
- **Jumping Knowledge Networks (JK-Net)**: concat representations từ nhiều layers.
- **PairNorm / GraphNorm**: normalize embeddings để prevent collapse.

### 9.2 Neighbor explosion ở high-degree nodes

**Symptom**: train time biến động lớn giữa batches — vì batch chứa high-degree node phải load nhiều neighbor features.

**Cause**: ở Pinterest graph, một pin có thể trong 100K boards (viral pin). Naive uniform sampling vẫn explosion.

**Mitigations**:
- **Cap neighbor count**: nếu degree > threshold, sample top-T quan trọng nhất (PinSage's random walk approach inherently does this).
- **Importance-based pruning**: pre-filter edges by weight, drop low-weight edges.
- **Hierarchical sampling**: sample at coarser granularity first.

### 9.3 Negative sampling bias

**Symptom**: model học bias systematic — recommend over-popular items hoặc under-recommend tail.

**Cause**: random negative sampling từ entire corpus → popular items often picked → model learns to "push them down" too much.

**Mitigations**:
- **Sampling by inverse popularity**: weight popular items lower as negatives.
- **In-batch negatives** (PinSage): negatives are from same batch (which contains positive examples) → naturally distribution-matched.
- **Hard negative curriculum** (PinSage): mix easy + hard.

### 9.4 Cold-start trap

**Symptom**: new pins / new boards get embedding của hệ chất lượng kém, vĩnh viễn underrepresented.

**Cause**: GNN cần graph context. New pin với 0-1 board placement → forward pass mostly relies on raw features (visual + text) → noisy.

**Mitigations**:
- **Content-only fallback**: precompute pure visual/text embedding cho new pins.
- **Bootstrap via similar items**: tìm K nearest by visual, borrow their PinSage embedding.
- **Active exposure**: surface new pins in 1-5% impression slots để collect early signal.
- **Faster refresh cho new pins**: micro-batch inference for pins < 7 days old, daily for rest.

### 9.5 Training/serving graph drift

**Symptom**: model performance degrades over time even without code change.

**Cause**: graph thay đổi liên tục (new pins, new boards, new edges). Embeddings trained on graph snapshot $G_t$ become stale for $G_{t+\Delta}$.

**Mitigations**:
- **Daily retrain (or finetune)**: PinSage retrain weekly initially, later daily.
- **Continual learning**: incremental update without full retrain.
- **Monitor drift metrics**: PSI on embedding distribution, recall@K on holdout, click-through delta vs production.

### 9.6 Other gotchas

- **Numerical instability**: importance weights $\alpha$ should sum to 1; if not normalized, gradient scale varies wildly across batches. Always normalize.
- **Feature scaling**: visual features (4096-dim VGG output) and text features (256-dim word2vec) có scale rất khác — must normalize each modality before concat.
- **Batch construction balance**: if randomly sample query pins, head pins dominate batch → tail underlearned. Use stratified sampling by popularity buckets.

### 9.7 Debugging recipe khi recall@K bị drop

**Step-by-step diagnostic flow** khi production team thấy recall@K giảm đột ngột (>2%) sau daily refresh:

1. **Verify input data integrity**:
   - Có shard nào fail upload không?
   - Pin-board edge count có thay đổi bất thường không (e.g. lỗi data pipeline upstream)?
   - Random walk precomputation outputs có nan/inf không?

2. **Check model checkpoint**:
   - Loss curve trong training run này có spike không?
   - Gradient norm có explode không?
   - Comparison với checkpoint cũ: weight diff norm có quá lớn không (>10% per layer)?

3. **Check intermediate embeddings**:
   - Sample 1000 pins, compute $h^1$ và compare với yesterday — distribution overlap >95%?
   - L2 norm distribution của final embeddings có stable không (mean ~1.0 sau normalize)?
   - Inter-pin similarity distribution có shift không?

4. **Check ANN index**:
   - Index build có complete không?
   - Recall của ANN index vs brute-force trên golden set bao nhiêu?
   - Có pin nào bị missing trong index không?

5. **Rollback policy**:
   - Nếu không tìm được root cause trong 1h → rollback về yesterday's embedding.
   - Auto-rollback trigger: production CTR drop >1% over 30-min window.

**Common root causes** (từ experience patterns):
- 40%: data pipeline issue (upstream edge data corrupted, missing shards).
- 25%: training instability (loss spike, NaN gradients).
- 20%: ANN index build issue (incomplete, wrong partitioning).
- 10%: feature distribution shift (e.g. visual backbone update without retraining downstream).
- 5%: actual model regression requiring investigation.

---

## 10. Lessons Learned & Best Practices

### 10.1 Concrete lessons từ PinSage journey

**Lesson 1 — Graph signal compounds with content signal**.
PinSage didn't replace visual/text features — it **fused** them via aggregation. Pin embedding = function of (own features, neighbor features). Best of both worlds.

**Lesson 2 — Random walks > uniform sampling cho importance-weighted graphs**.
Pinterest pin-board graph có natural importance via co-occurrence. Random walk surface this signal "for free". Khi graph có structure (community, popularity), exploit it.

**Lesson 3 — Inductive GNN > transductive cho production**.
Production graphs grow continuously. Any approach requiring "fixed node set" is dead on arrival at scale. GraphSAGE / PinSage's inductive setup is non-negotiable for web-scale.

**Lesson 4 — Inference pipeline matters as much as model**.
Pinterest's MapReduce two-pass inference is what made PinSage actually deployable. Many papers ignore inference cost — production teams cannot.

**Lesson 5 — Hard negative curriculum gives free quality**.
Loss function alone won't separate fine-grained classes if negatives are too easy. Curriculum (easy → hard) is cheap to implement, gives 10%+ gain.

**Lesson 6 — Single embedding is bottleneck for diverse interests**.
PinnerSage shows: even great embedding model needs **right representation granularity**. Single avg = compression loss. Multi-embedding = better fit for human interest diversity.

**Lesson 7 — Always plan for staleness**.
Graph evolves. Daily refresh is minimum; weekly is risky. Plan inference pipeline that can run end-to-end in < 24h.

### 10.2 Khi nào NÊN dùng GNN cho recsys

- ✅ Data có **explicit graph structure** (user-item, item-item, social).
- ✅ **Multi-hop relations carry signal** (not just direct neighbors).
- ✅ **Cold-start matters** AND content features available (inductive GNN exploits both).
- ✅ Scale > 10M items where embedding-based retrieval cần thiết.
- ✅ Engineering capacity để build inference pipeline (MapReduce / Spark).

### 10.3 Khi nào KHÔNG nên

- ❌ **Small graphs** (< 100K items): GNN overkill, simple CF / two-tower DNN sufficient.
- ❌ **Dense features dominate signal** (rich user features, item content): tree models hoặc DNN ranking enough.
- ❌ **No relational structure**: nếu chỉ có user-item interaction (no item-item co-occurrence), GNN không add value beyond MF.
- ❌ **Realtime sequence is the main signal**: nếu user behavior changes fast (TikTok-style), GNN's daily refresh too slow → cần online learning (xem S1-02 TikTok Monolith).

### 10.4 Engineering best practices

- **Start với simple baseline**: 2-tower DNN, then GNN as upgrade. Quantify lift before committing.
- **Build offline pipeline first**: MapReduce / Spark embedding refresh — get this stable before model improvements.
- **Quantize aggressively**: fp16 / int8 for serving, full fp32 only for training.
- **Monitor recall@K daily**: catch drift / training pipeline bugs early.
- **Maintain golden set**: 1K-10K human-labeled query-relevant pairs để benchmark.
- **Version everything**: model version, graph snapshot version, embedding output version.

### 10.5 Cross-references

- Foundations cho retrieval-ranking two-stage architecture: **[S1-01 YouTube Recommendation](S1-01_youtube_recommendation_end_to_end.md)**.
- Cho realtime / online learning bottleneck mà PinSage không giải quyết: **[S1-02 TikTok Monolith](S1-02_tiktok_monolith_realtime_recommendation.md)**.
- Ranking stage that consumes PinSage embeddings: **[S2-02 Wide & Deep / DeepFM / DCN evolution](S2-02_wide_deep_deepfm_dcn_evolution.md)**.
- ANN serving cho 3B PinSage vectors: xem **[S3-03 Vector Database Internals — HNSW vs IVF-PQ](../03-modern-stack/S3-03_vector_db_hnsw_ivf_pq.md)** cho deep dive về HNSW/IVF-PQ trade-off và lựa chọn index cho hot subset vs cold tail.
- Feature store cho graph features: **[S4-01 Uber Michelangelo](S4-01_uber_michelangelo_feature_store.md)**.

---

## 11. References

### Primary papers (read these first)

1. **Ying et al. (2018). "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"**. KDD 2018. Pinterest's PinSage paper.
   - arXiv: https://arxiv.org/abs/1806.01973
   - The canonical reference.

2. **Hamilton, Ying, Leskovec (2017). "Inductive Representation Learning on Large Graphs"**. NeurIPS 2017. GraphSAGE paper.
   - arXiv: https://arxiv.org/abs/1706.02216
   - Foundation for inductive GNN.

3. **Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"**. ICLR 2017. GCN paper.
   - arXiv: https://arxiv.org/abs/1609.02907
   - Foundation for convolution-on-graph idea.

### Pinterest follow-up papers

4. **Pal et al. (2020). "PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest"**. KDD 2020.
   - arXiv: https://arxiv.org/abs/2007.03634
   - Multi-embedding extension.

5. **Xia et al. (2023). "TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest"**. KDD 2023.
   - arXiv: https://arxiv.org/abs/2306.00248
   - Realtime sequence model fusing PinSage embeddings.

### Pinterest engineering blogs (semi-formal sources)

6. Eksombatchai et al. "PixieGen: Pinterest's Random Walk-Based Recommendation Engine".
   - Pinterest engineering blog (2018).
   - https://medium.com/pinterest-engineering — search "Pixie".

7. "Applying Deep Learning to Related Pins" — Pinterest engineering blog (2017).
   - Describes pre-PinSage state of Related Pins.

8. "PinSage: A new graph convolutional neural network for web-scale recommender systems" — Pinterest engineering blog (2018, public summary of paper).
   - Same content as paper but blog format.

### Related production GNN systems

9. **Wang et al. (2018). "Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba" (EGES)**. KDD 2018.
   - arXiv: https://arxiv.org/abs/1803.02349
   - Alibaba's concurrent GNN-style item embedding system.

10. **Liu et al. (2021). "Heterogeneous Graph Neural Networks for Large-Scale Bid Keyword Matching" — Alibaba/Microsoft**.
    - KDD 2021.
    - Heterogeneous GNN cho ads.

### Foundational background

11. **Perozzi, Al-Rfou, Skiena (2014). "DeepWalk: Online Learning of Social Representations"**. KDD 2014.
    - arXiv: https://arxiv.org/abs/1403.6652
    - Random walk + skip-gram (ancestor of PinSage's sampling idea).

12. **Grover & Leskovec (2016). "node2vec: Scalable Feature Learning for Networks"**. KDD 2016.
    - arXiv: https://arxiv.org/abs/1607.00653
    - Biased random walks, BFS/DFS interpolation.

13. **Page et al. (1999). "The PageRank Citation Ranking: Bringing Order to the Web"**.
    - Personalized PageRank is theoretical foundation for PinSage's random walk importance.

### Tutorials / surveys

14. **Stanford CS224W: Machine Learning with Graphs**. Jure Leskovec (one of PinSage authors).
    - https://web.stanford.edu/class/cs224w/
    - Best free university course covering GNN foundations.

15. **PyTorch Geometric documentation**.
    - https://pytorch-geometric.readthedocs.io/
    - For implementing GraphSAGE / similar in code.

16. **DGL (Deep Graph Library) tutorials**.
    - https://docs.dgl.ai/
    - Alternative PyG with focus on scalability.

### Note on internal numbers

PinSage paper (2018) gave specific numbers (3B pins, 18B edges, 7.5B training subgraph). Internal Pinterest scale has grown significantly since — actual current numbers not public but estimates suggest 10B+ pins, 50B+ edges. Architecture concepts remain valid but specific hardware / batch sizes / latencies have evolved.

---

## TL;DR (Vietnamese)

PinSage là **GNN production đầu tiên ở scale tỷ items**. Pinterest đã modify GraphSAGE qua 3 đổi mới quan trọng:
1. **Random-walk sampling** thay vì uniform/k-hop — exploit pin-board bipartite structure để có importance signal "miễn phí".
2. **Importance pooling aggregator** — weighted sum theo visit count, vừa rẻ compute vừa preserve discriminative info.
3. **MapReduce two-pass offline inference** — memoize layer-1 outputs để serve được 3B pins trong < 1 ngày.

Phía training: producer-consumer pipeline với CPU sampling + GPU forward pass, hard negative curriculum để học fine-grained boundaries.

Sau PinSage, Pinterest extend qua **PinnerSage** (multi-embedding user) và **TransAct** (realtime sequence transformer fusing PinSage tokens). Bài học chính: **graph signal compounds with content signal**, **inductive setup is non-negotiable cho production**, **inference pipeline matters as much as model architecture**.
