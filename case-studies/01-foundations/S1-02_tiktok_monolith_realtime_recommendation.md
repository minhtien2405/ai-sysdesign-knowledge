---
id: S1-02
title: "TikTok Monolith & Real-time Recommendation"
summary: "Collisionless embedding table, online learning, multi-channel retrieval, cold-start strategies."
slug: tiktok_monolith_realtime_recommendation
scope: 1
scope_name: foundations
difficulty: intermediate-advanced
status: done
tags:
  - recommendation systems
  - online learning
  - collisionless embedding
  - multi-channel retrieval
  - cold-start
cross_refs: [S1-01, S2-01, S4-01]
created: 2026-05-20
last_validated: 2026-05-20
line_count: 622
---

# S1-02 — TikTok Monolith & Real-time Recommendation

> **Scope 1** — Foundations | **Difficulty**: Intermediate–Advanced
> **Tham chiếu chéo**: [S1-01 YouTube Recommendation](S1-01_youtube_recommendation_end_to_end.md), [S2-01 Meta DLRM](S2-01_meta_dlrm_architecture.md), [S4-01 Uber Michelangelo](S4-01_uber_michelangelo_feature_store.md)

---

## 1. Tổng quan (Overview)

### 1.1 Bối cảnh business

TikTok (ByteDance) là một trong những hệ thống recommendation thành công nhất thập kỷ. Khác với YouTube — nơi user chủ động search hoặc click vào video — TikTok đẩy nội dung qua **For You Page (FYP)**, một stream gần như infinite của short-form videos. Toàn bộ trải nghiệm phụ thuộc vào chất lượng của recommendation system, và đây cũng chính là moat lớn nhất của ByteDance.

**Một vài con số public** (tính đến năm 2023–2024, theo ByteDance blog và talks tại MLSys, RecSys):
- **>1 tỷ MAU** (monthly active users) toàn cầu.
- **Hàng trăm tỷ video views mỗi ngày**, mỗi user xem trung bình 50–100 videos/session.
- **Average session length** ~45–60 phút (cao hơn rất nhiều so với feed của Instagram hay Facebook).
- **Latency target end-to-end cho FYP**: P99 < 200ms (theo Monolith paper).

### 1.2 Bài toán cốt lõi

TikTok có 3 đặc thù khiến reco system của họ phải design khác với YouTube/Netflix:

1. **Cold-start cực kỳ critical**: mỗi ngày có hàng chục triệu videos mới upload. Một video chỉ có giá trị trong vài giờ → vài ngày đầu. Nếu reco system phải đợi vài ngày để collect enough signals trước khi recommend → video đã hết "fresh window".
2. **Implicit feedback dominance**: user không click thumbnail như YouTube, mà signal đến từ **watch time, completion rate, replay, skip rate, like/comment/share** — tất cả collected liên tục trong session.
3. **Short feedback loop**: vì video chỉ dài 15s–3 phút, user phản hồi (skip hay xem hết) trong vài giây. Hệ thống phải **học ngay trong session** thay vì đợi batch retraining mỗi đêm.

→ Điều này dẫn đến kiến trúc **online learning + streaming feature updates** mà ByteDance gọi là **Monolith**.

### 1.3 Tại sao case study này quan trọng

- Đại diện cho paradigm **online learning at scale** — khác hẳn với batch retraining truyền thống.
- Mô tả cách handle **sparse parameter server** với hashing trick để xử lý vocabularies vô hạn (user IDs, video IDs, hashtags không bao giờ ngừng tăng).
- Là một trong số ít papers từ một top consumer-facing company (ByteDance) công bố chi tiết kiến trúc production của họ — Monolith paper xuất bản tại DLRS 2022.

---

## 2. Yêu cầu hệ thống (System Requirements)

### 2.1 Functional requirements

- Cho mỗi user request → trả về một sequence các video IDs để fill FYP feed (typically 6–10 videos per request, prefetch thêm khi user scroll).
- Personalize theo:
  - Past watch history (recent 100–500 videos)
  - Real-time session signals (videos đã xem trong 10 phút qua, watch %, skip)
  - Device/locale/time context
  - Social graph (follows, mutual interactions)
- Diversify: tránh show 5 videos cùng creator liên tiếp, mix genres/topics.
- Cold-start: handle (a) new user (<10 videos watched) và (b) new video (<1h since upload).

### 2.2 Non-functional requirements

| Yêu cầu | Target (public info) | Ghi chú |
|---|---|---|
| End-to-end latency (FYP request → response) | P99 < 200ms, P50 ~80–100ms | Bao gồm cả retrieval + ranking + safety filter |
| QPS (peak) | Hàng triệu QPS toàn cầu | Sharded theo region |
| Model update freshness | Minutes (not days) | Online learning, gradient cập nhật ~every batch |
| Feature freshness | Seconds | User vừa xem video xong → feature update trong < 5s |
| Training data throughput | Hàng PB/ngày | All user interaction events streamed |
| Storage cho embedding tables | 10s of TB | Sparse params nhiều hơn dense params nhiều bậc |
| Availability | 99.99%+ | Sharded, replicated |

### 2.3 Constraints quan trọng

- **No fixed vocabulary**: số lượng user IDs và video IDs tăng liên tục, không thể pre-define embedding table size cố định → phải dùng **dynamic hashing**.
- **Long-tail distribution**: 80% queries hit top 20% videos, nhưng tail vẫn phải serve tốt vì creator monetization.
- **Regulatory**: TikTok phải có content moderation pipeline (safety filter) trước khi recommend.

---

## 3. High-level Architecture

### 3.1 Sơ đồ tổng thể

```
                              ┌──────────────────────────────────┐
                              │       USER (mobile app)          │
                              │  scroll FYP, watch/skip/like     │
                              └──────────────┬───────────────────┘
                                             │
                                             │  Request: get_feed(user_id, ctx)
                                             ▼
                              ┌──────────────────────────────────┐
                              │      Recommendation Gateway      │
                              │  (load balance, auth, A/B route) │
                              └──────────────┬───────────────────┘
                                             │
                                             ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │                          RETRIEVAL LAYER                           │
       │  (multi-channel, parallel, each returns ~500-1000 candidates)      │
       │                                                                    │
       │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ ┌────────┐ │
       │  │ Two-tower│  │ Item-CF  │  │ Social   │  │ Trending │ │ Geo/   │ │
       │  │ DNN ANN  │  │  KNN     │  │ Graph    │  │  Pool    │ │ Topic  │ │
       │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ └───┬────┘ │
       │       └─────────────┴─────────────┴─────────────┴───────────┘      │
       │                              ↓ union + dedupe                     │
       │                       ~5000–10000 candidates                      │
       └────────────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │                       PRE-RANKING (light DNN)                      │
       │   nhỏ gọn, score nhanh để cắt xuống ~500–1000 candidates           │
       └────────────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │                  RANKING (Monolith — full DNN)                     │
       │   multi-task: predict (watch_time, finish, like, share, follow…)   │
       │   uses real-time + offline features from feature store             │
       └────────────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
       ┌────────────────────────────────────────────────────────────────────┐
       │             RE-RANKING / DIVERSITY / SAFETY / BUSINESS              │
       │   - MMR diversity                                                   │
       │   - safety filter (NSFW, hate speech, copyright)                    │
       │   - business rules (creator boost, ads injection slots)             │
       └────────────────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────────┐
                              │   FYP feed (6–10 videos)         │
                              └──────────────┬───────────────────┘
                                             │  watch / skip / like events
                                             ▼
                              ┌──────────────────────────────────┐
                              │  EVENT STREAM (Kafka / pub-sub)  │
                              └──────────┬───────────────────────┘
                                         │
                ┌────────────────────────┴───────────────────────┐
                ▼                                                ▼
   ┌────────────────────────┐                  ┌────────────────────────────┐
   │  Online Feature Store  │                  │  Online Training (Monolith)│
   │  (Redis/RocksDB-like)  │                  │  PS-worker, async gradient │
   │  per-user / per-item   │                  │  update embedding params   │
   └────────────────────────┘                  └────────────────────────────┘
```

### 3.2 Data flow tổng quát

1. **Request**: app gửi `get_feed` với user_id + device context (locale, time, network).
2. **Retrieval đa kênh**: chạy parallel ~5–10 channels (two-tower DNN, item-CF, social graph, geo, trending …). Mỗi channel trả ~500–1000 candidates, union + dedupe → 5k–10k.
3. **Pre-ranking**: một mạng nhỏ (vài layer MLP) score nhanh, cắt xuống ~500–1000.
4. **Ranking**: full DNN (Monolith), multi-task, output P(watch_time), P(finish), P(like), … → combine bằng weighted sum thành final score.
5. **Re-ranking**: apply MMR diversity, safety filters, business rules (ads injection slot at position 4 chẳng hạn).
6. **Response**: trả 6–10 videos về app.
7. **Event collection**: mọi tương tác (watch_time tick mỗi 100ms, skip, like, share, follow, comment, …) → streamed về Kafka.
8. **Streaming feature update**: feature store cập nhật user state (recent watches, session embedding) trong < 5s.
9. **Online training loop**: workers consume events, joined với features tại time of request (từ feature log), tạo training samples → SGD update parameters async qua parameter server.

→ Vòng feedback loop **kín trong vài phút**: user xem video → event log → feature update + gradient update → mô hình mới phục vụ chính user đó vài phút sau.

---

## 4. Deep dive các components chính

### 4.1 Monolith — Online Training System

**Monolith** là tên hệ thống training + serving của ByteDance, public qua paper *"Monolith: Real Time Recommendation System With Collisionless Embedding Table"* (Liu et al., DLRS 2022).

#### 4.1.1 Vấn đề: vocabulary vô hạn

Hệ thống recommendation cần embedding cho mỗi user_id, video_id, hashtag, … Nhưng:
- Mỗi ngày có hàng chục triệu users/videos mới.
- Không thể pre-allocate fixed-size embedding table như NLP (chỉ ~50k vocab).
- Nếu dùng standard hashing (mod prime number) → **hash collision** = 2 user khác nhau share embedding → noisy, hurt quality.

#### 4.1.2 Giải pháp: Collisionless HashTable

Monolith dùng **cuckoo hashing** + **dynamic resizing** để xây embedding table:

```
Embedding lookup pipeline:

  feature_value (e.g. user_id=12345)
        │
        ▼
  ┌─────────────────┐
  │  Hash function  │ → bucket index
  └─────┬───────────┘
        ▼
  ┌─────────────────────────────────────────┐
  │     Collisionless HashTable             │
  │  ┌──────────────────────────────────┐   │
  │  │ key → (slot_id, expire_ts)       │   │
  │  └──────────────────────────────────┘   │
  │  ┌──────────────────────────────────┐   │
  │  │ slot_id → embedding (float[d])   │   │
  │  └──────────────────────────────────┘   │
  └─────────────────────────────────────────┘
        │
        ▼
   embedding vector
```

**Key idea**:
- Mỗi feature_value (raw ID) được mapped tới một **slot** trong embedding table.
- Khi feature_value mới xuất hiện và bucket đã đầy → cuckoo eviction, hoặc dynamic resize.
- Khi feature ít gặp (long tail) → **probabilistic filter** (count-min sketch) decide có nên allocate slot không. Nếu user_id chỉ thấy 1 lần trong cả tuần → không cần dành slot riêng, fall back sang default embedding.

**Trade-off**: bộ nhớ extra cho hash table + bookkeeping (key → slot_id map), nhưng giảm hash collision noise đáng kể, dẫn đến AUC improvement ~0.5–1% (theo paper). Trong CTR world, 0.5% AUC ≈ revenue lift đáng kể.

#### 4.1.3 Online training architecture

```
                  ┌──────────────────────────────┐
                  │   Event Stream (Kafka)       │
                  │   user-video interactions    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
            ┌─────────────────────────────────────────┐
            │   Sample Joiner (stream processor)      │
            │   join event với feature snapshot       │
            │   tại thời điểm request                 │
            └────────────────┬────────────────────────┘
                             │  training samples
                             ▼
              ┌──────────────────────────────┐
              │   Training Workers (GPU)     │
              │   compute gradient cho dense │
              │   + sparse params            │
              └──────────┬───────────────────┘
                         │  sparse grads        │  dense grads
                         ▼                       ▼
        ┌────────────────────────────┐  ┌──────────────────────┐
        │   Sparse Parameter Server  │  │  Dense Param Server  │
        │   (sharded across nodes)   │  │  (all-reduce or PS)  │
        │   - collisionless table    │  │                      │
        │   - async updates          │  │                      │
        └──────────┬─────────────────┘  └──────────┬───────────┘
                   │                                │
                   ▼                                ▼
              ┌────────────────────────────────────────┐
              │   Snapshot / Sync to Serving Cluster   │
              │   (every few minutes, incremental)     │
              └────────────────────────────────────────┘
```

**Đặc điểm**:
- **Asynchronous SGD** cho sparse params: workers push gradients lên parameter server không cần đồng bộ. Lý do: sparse embeddings ít overlap giữa các mini-batch (user A và user B hiếm khi share embedding → conflict rate thấp).
- **Synchronous (all-reduce)** cho dense params: vì dense weights (MLP layers, attention) shared bởi mọi sample → cần đồng bộ để tránh stale gradient.
- **Incremental snapshot**: mỗi ~5 phút, sparse params được snapshot và sync tới serving cluster. Không cần full snapshot vì chỉ một phần nhỏ params (recently updated) thay đổi.

#### 4.1.4 Train-Serve consistency

Một challenge lớn: model đang train liên tục, serving model là một snapshot cũ vài phút. Làm sao tránh skew?

ByteDance giải bằng:
1. **Feature logging at request time**: khi serving model fetch features để inference, đồng thời log chính các features đó vào event stream với request_id. Khi event (watch_time) đến, sample joiner join event ↔ feature log theo request_id → đảm bảo training samples dùng đúng features mà serving model đã thấy.
2. **Online evaluation guard**: trước khi push snapshot mới lên serving, chạy quick eval trên một stream nhỏ. Nếu metric drop quá threshold → rollback.

---

### 4.2 Retrieval — Multi-channel

TikTok dùng **multiple retrieval channels** chạy song song, đặc trưng:

| Channel | Mô tả | Use case |
|---|---|---|
| **Two-tower DNN + ANN** | User tower + item tower, train với in-batch sampled softmax, serve qua ANN (HNSW or Faiss) | Personalized retrieval theo behavior |
| **Item-CF** | Co-engagement matrix, item-item similarity | User vừa xem video X → tìm các video Y mà user khác cũng xem cùng X |
| **Social graph** | Videos từ accounts user follow hoặc accounts mà follows của user follow | Strong signal cho engagement |
| **Trending pool** | Top-K videos theo region trong 1h, 6h, 24h | Diversity, fresh content |
| **Geo / locale** | Filter theo country/city/language | Localization |
| **Hashtag / topic** | Match interest profile của user với hashtags videos | Cold-start friendly |

**Tại sao multi-channel thay vì một big retrieval model?**
- Mỗi channel cover một aspect (behavior similarity vs trending vs social) → union diverse hơn.
- Robustness: nếu một channel down → các channel khác vẫn cover.
- Cold-start: new user có ít behavior → trending + geo channels gánh; new video có ít engagement → two-tower (qua content features) + hashtag channels gánh.

#### 4.2.1 Two-tower với streaming update

Two-tower model của TikTok cũng được train online (theo công bố tại RecSys 2022/2023):

```
   User tower input:                    Item tower input:
   - user_id embedding (sparse)         - video_id embedding (sparse)
   - recent watch history (sequence)    - creator_id embedding
   - session signals (last 30 min)      - hashtags, language, duration
   - device context                     - upload time (recency feature)
       │                                    │
       ▼                                    ▼
   [MLP/Transformer]                    [MLP/Transformer]
       │                                    │
       ▼                                    ▼
       u_emb (128-d)                        v_emb (128-d)
       └──────────── dot product ──────────┘
                       │
                       ▼
                 logit → P(positive)
```

**Item embeddings được index vào ANN** (HNSW hoặc IVF-PQ) định kỳ (mỗi 1–5 phút). Vì TikTok có hàng tỷ videos nhưng chỉ ~100M–500M video gần đây active → index chỉ chứa active subset (recency window).

**Cold-start video**: ngay khi upload, video chưa có engagement → embedding khởi tạo từ content features (visual via CV model, audio, text caption qua text encoder). Sau khi có vài chục interactions → fine-tune embedding qua online training.

---

### 4.3 Ranking — Monolith DNN

Ranking model là phần "nặng" nhất, predict nhiều targets cùng lúc.

#### 4.3.1 Architecture (high-level)

```
Inputs:
   ├── Sparse features (one-hot/multi-hot)
   │      user_id, video_id, creator_id, hashtags, geo, device, …
   │      → embedding lookup (collisionless table) → concat
   │
   ├── Dense features
   │      watch_time của user trong 7 ngày, finish_rate trung bình,
   │      session length so far, time-of-day, …
   │
   └── Sequence features
          recent N watched videos (mỗi cái có embedding + meta)
          → sequence encoder (DIN-style attention hoặc Transformer)

Sau concat → shared MLP backbone (vài layer 1024 → 512 → 256)
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │  Multi-task heads (mỗi task một MLP nhỏ)     │
   │  - P(watch_time > threshold)                 │
   │  - P(finish_video)                           │
   │  - P(like)                                   │
   │  - P(share)                                  │
   │  - P(comment)                                │
   │  - P(follow_creator)                         │
   │  - P(skip_within_3s)                         │
   └──────────────────────────────────────────────┘

Final score = Σ w_i * P_i   (weights learned + business-tuned)
```

#### 4.3.2 Tại sao multi-task?

Một single target (e.g. chỉ predict `like`) tạo bias rất rõ:
- Like rate thấp (~5%) → noisy signal.
- Mô hình chỉ predict like sẽ over-rank những video gây surprise/clickbait, hurt long-term watch time.

→ Multi-task giúp:
- Share representation across tasks (MLP backbone).
- Combine theo business priority (watch_time thường được weighted nhiều nhất, vì correlate với DAU retention).
- Một số tasks act như **regularizer** (e.g. P(skip_within_3s) acts như "don't recommend if too easily skipped").

#### 4.3.3 Sequence modeling — kế thừa từ DIN/DIEN

User history (recent 100–500 videos) là feature quan trọng nhất. TikTok dùng kỹ thuật tương tự DIN (Deep Interest Network của Alibaba, sẽ deep dive ở S2-03):

```
Recent watched videos: [v1, v2, …, v100]
                            │
                            ▼
                  embedding lookup
                            │
                            ▼
        ┌─────────── Attention with target video v_t ───────────┐
        │   weight_i = softmax(query=v_t, key=v_i)              │
        │   user_interest_for_v_t = Σ weight_i * v_i_embedding  │
        └────────────────────────────────────────────────────────┘
                            │
                            ▼
              Concat vào ranking input
```

→ User interest **dynamic theo target video** thay vì fixed user embedding. Một user thích cả cooking videos và sports → khi rank cooking video, attention sẽ highlight watched cooking videos; khi rank sports, highlight watched sports.

---

### 4.4 Online Feature Store

Khác với batch feature store của Uber Michelangelo (xem [S4-01](S4-01_uber_michelangelo_feature_store.md)) — vốn ưu tiên consistency offline ↔ online — TikTok feature store ưu tiên **freshness real-time**.

**Architecture**:

```
   Event stream (Kafka)
         │
         ▼
   ┌──────────────────────────┐
   │   Stream processors      │
   │   (Flink-like)           │
   │   - update user state    │
   │   - update item state    │
   │   - update counters      │
   └──────────┬───────────────┘
              │
              ▼
   ┌──────────────────────────────────────────┐
   │   Online Feature Store                   │
   │   - in-memory KV (Redis-like, sharded)   │
   │   - per-user key: u:<user_id>            │
   │       value: { recent_watches: [...],    │
   │                session_emb: [...],       │
   │                last_active_ts: ...,      │
   │                counts: {like, view, ...} │
   │              }                           │
   │   - per-item key: v:<video_id>           │
   │       value: { views_1h: ...,            │
   │                finish_rate: ...,         │
   │                like_rate: ...,           │
   │                trending_score: ...       │
   │              }                           │
   └──────────────────────────────────────────┘
              │
              ▼
       served to ranking model
       at inference time
```

**Latency**:
- Feature lookup (1 user + 1000 items): cần < 20ms (vì còn budget cho model inference + downstream).
- Solution: collocate feature store shards với ranking workers (cùng datacenter, cùng rack nếu có thể). Batched lookup, pre-fetched theo candidate list.

**Update latency** (event xảy ra → feature reflect):
- User-level features: <5s (Flink-like stream processor).
- Item-level aggregates (e.g. views_1h): vài giây tới ~30s.
- Counters cần consistency cao (e.g. exact view count cho creator dashboard) — chạy parallel batch pipeline để correct.

---

### 4.5 Cold-start handling

#### 4.5.1 Cold-start user (new account)

- **Step 1**: ngay khi onboard, ask user pick 3–5 interests (basic but effective).
- **Step 2**: serve một **exploration pool**: trending videos + diverse genres để probe interest.
- **Step 3**: feature initialization — user embedding khởi tạo từ initial preferences + device/geo cluster average.
- **Step 4**: bandit-style exploration: trong vài giờ đầu, 20–30% slots dùng cho exploration thay vì exploitation, để learn fast.

→ Trong vài giờ, user embedding đủ "ấm" để personalize.

#### 4.5.2 Cold-start video (new upload)

Đây là vấn đề khó hơn. TikTok dùng **traffic seeding strategy**:

```
Hour 0-1:    ┌─ Show video tới ~100-1000 carefully selected users ─┐
             │  (similar interest to creator's existing audience)   │
             └────────────────────────────────────────────────────-─┘
                              │
                              ▼ collect engagement signals
                              │
                  ┌───────────┴────────────┐
                  ▼                        ▼
            High engagement?         Low engagement?
                  │                        │
                  ▼                        ▼
       Hour 1-6: scale to             Cap exposure, treat
       wider audience                 as "filtered out"
       (~10K-100K users)
                  │
                  ▼
       Hour 6-24: if still strong → push to general FYP pool
```

**Embedding cho new video**:
- Khởi tạo từ content features: visual embedding (qua CV model như Vision Transformer pretrained), audio embedding, text từ caption/hashtags.
- Sau ~100 interactions → start updating qua online training như video bình thường.

---

### 4.6 Re-ranking, Diversity, Safety

#### 4.6.1 Diversity

Sau ranking, một re-ranker apply **MMR (Maximal Marginal Relevance)** hoặc **DPP (Determinantal Point Process)** để diversify:

```
Score_final(v_i) = λ * score_ranking(v_i)
                 - (1 - λ) * max_j∈selected sim(v_i, v_j)
```

Đảm bảo không recommend 5 cooking videos liên tiếp. Sim có thể đo qua embedding distance hoặc category overlap.

#### 4.6.2 Safety filter

Trước khi return feed, mỗi candidate phải pass:
- **NSFW classifier** (CV model + audio classifier).
- **Hate speech / misinfo** (text classifier trên caption + ASR transcript).
- **Copyright** (audio fingerprint match).
- **Age-appropriate** (cross-check user age vs content rating).

Safety pipeline thường chạy **offline trước khi video vào candidate pool**, nhưng còn một thin online filter để catch edge cases (e.g. video bị flag sau khi đã upload).

#### 4.6.3 Business rules

- **Ads injection**: position 4 và 11 trong FYP thường là ads (theo public reverse-engineering blog posts).
- **Creator boost**: monetization program creators được boost mild.
- **Compliance**: trong một số region (EU under DSA), phải provide "non-personalized" option cho user.

---

## 5. Trade-offs & Design decisions

### 5.1 Online learning vs batch retraining

| Approach | Pros | Cons | Khi nào dùng |
|---|---|---|---|
| **Batch retraining** (YouTube/Netflix style, daily/weekly) | Đơn giản, dễ A/B test, stable | Slow adaptation, cold-start không tốt | Khi content có lifecycle dài (movies, long videos), behavior thay đổi chậm |
| **Online learning** (TikTok Monolith) | Fast adaptation, tốt cho fresh content, capture trend nhanh | Complex infra, hard to debug, dễ bị training/serving skew, risk catastrophic forgetting | Short content lifecycle, real-time signals, large feedback volume |

**Tại sao TikTok chọn online**: content lifecycle ngắn (giờ → ngày), feedback signal cực dồi dào (mỗi user 50–100 interactions/session), trend xoay vòng nhanh.

### 5.2 Collisionless table vs standard hashing

| | Standard hash (mod prime) | Collisionless (cuckoo + filter) |
|---|---|---|
| Memory | Smaller, fixed | Lớn hơn (~1.5–2x do bookkeeping) |
| Collision noise | Có, hurt AUC | Không, AUC tốt hơn ~0.5–1% |
| Implementation | Simple | Complex (cuckoo + eviction + resize) |
| Long-tail handling | Bad (popular IDs collide với tail) | Tốt (filter loại bỏ long-tail spam IDs) |

→ Trade-off worth it ở scale TikTok, nhưng overkill cho team nhỏ.

### 5.3 Multi-channel retrieval vs single big model

| Approach | Pros | Cons |
|---|---|---|
| **Multi-channel** (TikTok, YouTube) | Robust, diverse, cold-start friendly | Complex ops, hard to A/B test single component |
| **Single big retrieval model** (some startups) | Simpler, single source of truth | Single point of failure, miss diversity, harder cold-start |

### 5.4 Async vs sync gradient update

TikTok pick **async cho sparse, sync cho dense**:
- Sparse updates rarely conflict (mỗi sample touch few embeddings) → async giúp throughput cao.
- Dense weights (MLP) shared bởi mọi sample → conflict cao → cần sync (all-reduce) để stable.

### 5.5 Multi-task ranking vs single-task

| | Single-task (chỉ predict CTR) | Multi-task (TikTok) |
|---|---|---|
| Easier to tune | ✓ | ✗ (cần balance loss weights) |
| Capture business intent | ✗ (CTR ≠ retention) | ✓ |
| Share representation | ✗ | ✓ (backbone shared) |
| Risk over-optimization | Cao (clickbait) | Thấp |

---

## 6. Lessons learned & Best practices

### 6.1 Lessons từ công bố của ByteDance

1. **Collisionless > clever hash**: cố gắng giảm collision bằng better hash function (e.g. murmurhash) không bằng accept memory overhead và build collisionless structure. Quality always wins ở scale top consumer apps.
2. **Train-serve consistency là 80% công việc**: nếu features tại training time khác features tại serving time → model degrade ngay cả khi offline AUC tốt. **Feature logging at request time** là pattern phổ biến (Meta, Google, ByteDance đều dùng).
3. **Online learning không thay thế batch training hoàn toàn**: TikTok vẫn có batch retraining pipeline để (a) bootstrap model mới, (b) recover khi online drift quá xa, (c) backfill dataset cho experiments. Hybrid approach.
4. **Cold-start cần multi-prong**: không chỉ một technique. Combine content embedding + traffic seeding + bandit exploration + user onboarding signal.
5. **Diversity > raw ranking quality**: một feed với AUC cao nhưng monotonous (toàn cooking) sẽ kill retention. Re-ranking cho diversity là non-negotiable.

### 6.2 Common pitfalls khi build hệ thống tương tự

- **Underestimate event volume**: TB-PB per day. Kafka cluster phải sized right, sample joiner phải scalable.
- **Forget about replay**: khi rollback model, training data từ giai đoạn đó cũng skewed (vì model B influence what users watched). Cần track model version trong event log.
- **Ignore long-tail creators**: nếu chỉ optimize cho engagement, head creators sẽ dominate. Long-tail creator monetization sẽ suffer → ecosystem chết. Cần explicit constraints.
- **Underestimate safety filter latency**: nếu safety check ở online path quá chậm → block ranking. Phải đẩy phần lớn safety check offline (pre-compute video safety score), chỉ keep thin online check.

### 6.3 Khi nào KHÔNG nên copy TikTok architecture

- Khi content lifecycle dài (>1 tuần): batch retraining mỗi đêm đủ tốt, không cần online learning.
- Khi data volume nhỏ (<1M events/day): không justify infra phức tạp. Daily Airflow + offline two-tower + LightGBM ranking là đủ.
- Khi team < 10 ML engineers: maintenance cost của parameter server + collisionless table + stream training là quá lớn.

→ Architecture của TikTok phù hợp với scale **rất rất lớn**. Đừng over-engineer.

---

## 7. References

### 7.1 Papers (chính thức)

- **Monolith** — Liu, Z. et al. *"Monolith: Real Time Recommendation System With Collisionless Embedding Table"*. arXiv:2209.07663 (2022). [https://arxiv.org/abs/2209.07663](https://arxiv.org/abs/2209.07663) — **độ tin cậy: rất cao**, paper chính thức từ ByteDance.
- **DIN** — Zhou, G. et al. *"Deep Interest Network for Click-Through Rate Prediction"*. KDD 2018. (Alibaba, nhưng kỹ thuật sequence attention được TikTok kế thừa.)
- **Two-tower model** — Yi, X. et al. *"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"*. RecSys 2019. (Google, foundation cho retrieval của TikTok.)

### 7.2 Engineering blogs / talks

- ByteDance Engineering Blog (Chinese, có English translations): [https://www.bytedance.com/en/](https://www.bytedance.com/en/) — periodic posts về Monolith, training infra. Độ tin cậy cao, nhưng số liệu cụ thể thường hạn chế.
- "How TikTok's algorithm works" — multiple journalistic deep dives (WSJ, NYT) — useful cho user-facing intuition, không phải technical source.

### 7.3 Cross-reference trong knowledge base

- [S1-01 YouTube Recommendation](S1-01_youtube_recommendation_end_to_end.md) — so sánh với batch-style reco.
- [S2-01 Meta DLRM](S2-01_meta_dlrm_architecture.md) — sparse embedding + parameter server pattern.
- [S2-03 Alibaba DIN/DIEN](#) (📋 planned) — sequence modeling deep dive.
- [S4-01 Uber Michelangelo](S4-01_uber_michelangelo_feature_store.md) — feature store offline-first vs TikTok feature store realtime-first.

### 7.4 Notes về độ tin cậy

- **Confirmed**: collisionless embedding table, async sparse + sync dense, multi-task ranking, feature logging at request time — all in Monolith paper.
- **Inferred from public info**: số liệu cụ thể về QPS, model size, snapshot interval — based on public talks, có thể khác internal số.
- **Speculation marked as such**: chi tiết về ads injection slots, exact bandit policy cho cold-start — based on industry pattern, không phải direct quote từ ByteDance.

---

**Tóm tắt 30s**: TikTok = **Monolith online learning** + **collisionless sparse embeddings** + **multi-channel retrieval** + **multi-task ranking với sequence attention** + **streaming feature store**. Architecture optimize cho short content lifecycle, dồi dào feedback signals, và cold-start nhanh. Đừng copy nếu data volume nhỏ — over-engineering sẽ giết team.
