---
id: S1-01
title: "YouTube Recommendation System End-to-End"
summary: "Two-stage architecture (candidate generation + ranking), DNN models, serving infra ở scale tỷ user."
slug: youtube_recommendation_end_to_end
scope: 1
scope_name: foundations
difficulty: foundational
status: done
tags:
  - recommendation systems
  - two-stage architecture
  - candidate generation
  - ranking
  - deep learning
  - large-scale serving
cross_refs: [S1-02, S2-01, S2-02, S4-01, S4-02]
created: 2026-05-20
last_validated: 2026-05-20
line_count: 520
---

# S1-01 — YouTube Recommendation System End-to-End

> **Scope**: AI/ML System Design Foundations
> **Difficulty**: Foundational
> **Tags**: recommendation systems, two-stage architecture, candidate generation, ranking, deep learning, large-scale serving
> **Primary sources**: Covington et al. "Deep Neural Networks for YouTube Recommendations" (RecSys 2016), Zhao et al. "Recommending What Video to Watch Next: A Multitask Ranking System" (RecSys 2019), các engineering talks công khai.

---

## 1. Overview

YouTube là một trong những recommendation system lớn nhất thế giới — recommend video từ **corpus hàng tỷ videos** cho **2+ tỷ logged-in users mỗi tháng** (số liệu công bố của Google năm 2019, hiện tại có thể cao hơn). Mỗi lần user mở app/web, YouTube phải sinh ra một feed cá nhân hoá trong **vài trăm milliseconds**, trong khi vẫn đảm bảo:

- **Relevance** — video user thực sự muốn xem.
- **Freshness** — videos mới upload (cold-start items) cũng phải có cơ hội xuất hiện.
- **Diversity** — không spam một chủ đề.
- **Long-term user satisfaction** — không chỉ optimize CTR (Click-Through Rate) ngắn hạn (sẽ dẫn đến clickbait).

### Business problem

Trước thời recommendation, YouTube dùng search + trending để drive views. Vấn đề: phần lớn users không biết họ muốn xem gì cụ thể — họ mở YouTube để "tiêu thụ" content. Recommendation đóng vai trò **discovery engine**, chiếm hơn **70% watch time** trên YouTube (con số được Goodrow công bố tại Netflix Research Summit 2018, sau đó được nhiều bài blog Google trích lại). Đây là một con số khổng lồ — nó biến reco từ "feature" thành **core business**.

### Tại sao YouTube là case study tốt để học foundations?

1. **Kiến trúc two-stage (candidate generation → ranking)** đã trở thành **template chuẩn** cho industrial-scale reco (Pinterest, TikTok, Instagram, LinkedIn đều dùng pattern này).
2. **Bridging từ classical methods (matrix factorization) sang deep learning** — paper 2016 là một trong những paper đầu tiên public về DNN reco ở scale tỷ users.
3. Có **multi-objective ranking** (CTR, watch time, satisfaction) — minh hoạ trade-off realistic chứ không phải toy problem.
4. Public sources rất dồi dào: 2 paper RecSys flagship + nhiều talks ở MLSys, KDD.

---

## 2. System Requirements

### 2.1 Functional requirements

- Sinh ra một **ranked list** vài chục videos cho một user, mỗi khi user mở homepage hoặc xem xong một video (watch-next).
- Hỗ trợ **multiple surfaces**: homepage feed, watch-next sidebar, shorts feed, search re-ranking.
- Có thể **personalize theo context**: time of day, device, location, recent watches.
- Hỗ trợ **fresh content** — video upload trong vài giờ qua phải có cơ hội được recommend.

### 2.2 Non-functional requirements

| Metric | Target (public approximations) | Ghi chú |
|---|---|---|
| Corpus size | ~10^9 videos (billions) | Phải prune trước khi candidate gen |
| MAU | 2B+ (2019 figure) | |
| End-to-end serving latency | < 100 ms (homepage), < 200 ms (watch-next) | Bao gồm cả candidate gen + ranking + post-processing |
| Candidate gen latency | < 10 ms cho top-K (K~hundreds) | ANN-based |
| Ranking latency | < 50 ms cho ~hundreds candidates | DNN inference batched |
| QPS | Hàng triệu QPS toàn cục | Distributed across regions |
| Model refresh cadence | Ranking model: daily/weekly retrain; Candidate gen model: weekly | Inferred from talks |
| Training data volume | Petabyte-scale watch logs | |

### 2.3 Constraints quan trọng

- **Cold-start cho items**: video mới upload chưa có engagement signal → embedding chưa được học.
- **Position bias**: user click vào video ở vị trí #1 nhiều hơn vị trí #10 dù relevance ngang nhau. Phải debias trong training data.
- **Implicit feedback only**: không có rating 1–5 sao, chỉ có watch time / skip / like / share → đòi hỏi nghĩ kỹ về label.
- **Survival bias**: training data chỉ chứa videos mà system **đã từng recommend**. Videos chưa bao giờ recommend → không có signal → khó break ra khỏi local optimum.

---

## 3. High-level Architecture

YouTube dùng **two-stage architecture** kinh điển:

```
                        ┌─────────────────────────────────────────────┐
                        │            Video corpus (~10^9)             │
                        └──────────────────────┬──────────────────────┘
                                               │
                                               ▼
                        ┌─────────────────────────────────────────────┐
        Stage 1         │      CANDIDATE GENERATION (Retrieval)        │
      "Recall"          │  Goal: corpus → ~hundreds candidates         │
   Latency: ~10ms       │  Method: ANN over learned embeddings         │
                        │  Models: two-tower / neural CF / co-watch   │
                        └──────────────────────┬──────────────────────┘
                                               │  ~hundreds candidates
                                               ▼
                        ┌─────────────────────────────────────────────┐
        Stage 2         │              RANKING                         │
     "Precision"        │  Goal: score precisely, sort                 │
   Latency: ~50ms       │  Method: deep DNN with rich features         │
                        │  Output: predicted watch time / engagement   │
                        └──────────────────────┬──────────────────────┘
                                               │  ranked list
                                               ▼
                        ┌─────────────────────────────────────────────┐
        Stage 3         │       POST-PROCESSING / RE-RANKING           │
   (heuristics +        │  Diversity, freshness boost, business rules │
   constraints)         │  Dedup, channel cap, exploration injection  │
                        └──────────────────────┬──────────────────────┘
                                               │  final feed
                                               ▼
                                          [User device]
```

Lý do dùng two-stage: **không thể chạy DNN với 100+ features trên 10^9 videos trong 100ms**. Phải tách: candidate gen làm việc với **embeddings nhẹ + ANN** (sub-linear), ranking chạy DNN nặng nhưng chỉ trên hundreds candidates (linear nhưng N nhỏ).

### Data flow ở high-level

```
[Watch logs / impressions]  →  Kafka / pub-sub
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  [Streaming feature             [Batch ETL                 [Realtime
   computation]                   (Spark / FlumeJava /       counters
                                  Beam)]                     (Bigtable)]
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                     [Training data warehouse (Bigtable / Spanner / GCS)]
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
        [Candidate gen training]                [Ranking training]
                │                                       │
                ▼                                       ▼
        [Embedding index built                  [Ranking model
         via ScaNN/FAISS-like ANN]              checkpoint → TF Serving]
                │                                       │
                └───────────────┬───────────────────────┘
                                ▼
                        [Online serving stack]
```

ScaNN là Google's open-source ANN library (paper Guo et al. ICML 2020), được dùng nội bộ cho retrieval ở Google scale.

---

## 4. Deep dive các components chính

### 4.1 Candidate generation (retrieval) — paper 2016

**Mục tiêu**: từ corpus ~10^9 videos → vài trăm candidates trong < 10 ms, optimize **recall** chứ không phải precision.

#### Model design

Covington et al. (2016) mô hình hoá như **extreme multi-class classification**:
- Mỗi video là một **class** (10^9 classes!).
- Input: user embedding `u` (function of user history + context).
- Output: P(watch video v_i | u) = softmax.

Vì 10^9 classes không thể compute softmax đầy đủ → dùng **sampled softmax** trong training (sample ~vài nghìn negatives mỗi step) và **approximate nearest neighbor (ANN)** trong serving:

```
P(v_i | u) ∝ exp(u · v_i)   ⟹   top-K candidates  ≈  ANN(u, V)
```

Trong đó `V` là ma trận video embeddings (10^9 × d, d~256).

#### Tower architecture (current generation)

Paper 2016 dùng kiến trúc single-tower MLP. Generation hiện tại (theo các talks gần đây) đã chuyển sang **two-tower model**:

```
           User features                       Video features
                │                                    │
   ┌────────────┴────────────┐         ┌────────────┴────────────┐
   │ watch history (avg of   │         │ video ID embedding       │
   │  video embeddings)      │         │ topic / category         │
   │ search history          │         │ language / age           │
   │ demographics, geo       │         │ uploader info            │
   │ device, time            │         │ (no engagement features  │
   └────────────┬────────────┘         │  to avoid feedback loop) │
                ▼                       └────────────┬────────────┘
        ┌───────────────┐                            ▼
        │ User tower    │                    ┌───────────────┐
        │  MLP (3-4     │                    │ Item tower    │
        │   layers)     │                    │  MLP          │
        └───────┬───────┘                    └───────┬───────┘
                │                                    │
                ▼                                    ▼
            u (vector d)                         v (vector d)
                │                                    │
                └─────────  dot(u, v)  ───────────────┘
                              │
                              ▼
                       sampled softmax
                       (1 positive +
                        N sampled negatives)
```

**Tại sao two-tower?** Để serving được — chỉ user tower phải chạy realtime (per request), item tower chạy offline để build ANN index.

#### Training details (paper 2016)

- **Label**: predict next video user sẽ watch (sequence modeling style), không phải video user click rồi không xem.
- **Training data**: chỉ lấy **complete watches** (user xem hết hoặc > threshold) để filter clickbait.
- **Example weighting**: trick quan trọng — predict **expected watch time** thay vì click probability (xem section ranking).
- **Negative sampling**: random negatives + hard negatives (in-batch negatives — items khác trong cùng batch).

#### Embedding của user history

Một feature quan trọng: **average embedding của các videos user đã xem gần đây**. Đây là dạng đơn giản của user-item interaction encoding. Kiến trúc hiện đại có thể dùng **transformer over watch sequence** (như SASRec, BERT4Rec), nhưng public docs của YouTube chưa confirm chính xác sequence model dùng version nào.

#### Serving: ANN over embeddings

Sau khi train xong, build index của 10^9 video embeddings. Khi request đến:
1. User tower chạy → compute `u`.
2. ANN library (ScaNN / FAISS-like) → top-K videos theo `dot(u, v_i)`.
3. ScaNN có thể đạt < 10ms cho top-K trên billion-scale corpus với recall > 0.9 (theo Guo et al. 2020).

```python
# Pseudo-code serving (per request)
def candidate_generation(user_id, context):
    # 1. Fetch user features (history, demographics) - từ feature store
    user_features = feature_store.get_online(user_id, context)
    
    # 2. Forward pass user tower (small MLP, ms-level)
    u_vec = user_tower.forward(user_features)  # shape: [d]
    
    # 3. ANN lookup (ScaNN/FAISS) - dưới 10ms cho billion items
    candidate_ids = ann_index.search(u_vec, top_k=500)
    
    # 4. (Optional) Filter — remove already watched, blocked channels
    candidate_ids = filter_seen(candidate_ids, user_id)
    return candidate_ids
```

#### Vấn đề thường gặp ở candidate gen

- **Feedback loop**: model học từ những gì system đã show → biased. Mitigation: dùng **multiple candidate sources** (collaborative, content-based, exploration) và **propensity correction**.
- **Cold-start items**: video mới upload chưa có embedding tốt. Workaround: dùng **content-based features** (uploader, topic, language, thumbnail embeddings) — gọi là **content-based candidate source**.
- **Fresh videos boost**: thường có một **freshness candidate source** riêng (videos uploaded trong N giờ qua, ranked theo uploader popularity + topic match).

### 4.2 Ranking — paper 2019 (multi-task)

**Mục tiêu**: nhận ~hundreds candidates từ retrieval → score chính xác → sort. Optimize **multiple objectives** đồng thời.

#### Tại sao ranking khác candidate gen?

- Số candidates nhỏ (hundreds) → có thể chạy DNN nặng.
- Có thể dùng **nhiều features hơn** — cross features (user × video), realtime counters, position features.
- Cần precision cao hơn — sai một vài positions đầu sẽ trực tiếp impact user experience.

Lineage của CTR ranking architectures: từ Wide & Deep (Google 2016) → DeepFM (Huawei 2017) → DCN-V2 (Google 2020) — xem [S2-02 Wide & Deep / DeepFM / DCN evolution](../02-model-development/S2-02_wide_deep_deepfm_dcn_evolution.md). Khi scale lên hàng chục triệu features + 100GB+ embedding tables (như Meta Ads), kiến trúc tiến hoá thành DLRM với hybrid model + data parallelism — xem [S2-01 Meta DLRM](../02-model-development/S2-01_meta_dlrm_architecture.md).

#### Multi-task objectives (Zhao et al. 2019)

YouTube không optimize một mục tiêu duy nhất. Paper 2019 mô tả MMoE (Multi-gate Mixture-of-Experts) cho **multi-task ranking**:

```
                    Shared bottom features
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Expert 1         Expert 2  ...   Expert N
              │               │               │
              └───────┬───────┴───────────────┘
                      │ (gating mechanism per task)
        ┌─────────────┼─────────────┬───────────────┐
        ▼             ▼             ▼               ▼
    CTR head     Watch-time     "Like" head     Dismiss head
    (binary)      (regression)   (binary)       (binary, neg)
        │             │             │               │
        └─────────────┴──────┬──────┴───────────────┘
                             ▼
                  Weighted combination
                  at serving time
                  (business-tuned weights)
```

**Tasks điển hình**:
- **Engagement tasks**: CTR, watch time, completion rate, share.
- **Satisfaction tasks**: survey rating, like, follow, dismiss (negative signal).
- Mỗi task có **head** riêng nhưng share bottom layers — giúp regularize và share representation.

#### Position bias debiasing

Paper 2019 dùng kiến trúc **"shallow tower" cho position bias**:

```
Main features ──→ Main tower ──┐
                                ├──→ score
Position features ──→ Shallow ──┘     (trained joint)
(serving position,  tower
 device, etc.)
```

Tại training time, model học cả ảnh hưởng của position. Tại serving time, **set position feature = constant** (ví dụ position=1) → loại bỏ position bias khi predict.

```python
# Pseudo-code ranking model với position debiasing
class RankingModel(nn.Module):
    def __init__(self):
        self.main_tower = MLP(...)
        self.shallow_tower = MLP(...)  # cho position features
        self.task_heads = {
            'ctr': MLP(...),
            'watch_time': MLP(...),
            'like': MLP(...),
        }
    
    def forward(self, main_features, position_features, mode='train'):
        main_logit = self.main_tower(main_features)
        if mode == 'train':
            pos_logit = self.shallow_tower(position_features)
        else:  # serving — set position = 1 (top)
            pos_logit = self.shallow_tower(get_const_pos_features())
        return {
            task: head(main_logit + pos_logit)
            for task, head in self.task_heads.items()
        }
```

#### Predict expected watch time (paper 2016 trick)

Một insight kinh điển từ paper 2016: thay vì predict click probability, model predict **expected watch time**. Cách làm:

- Trong training data, **positive examples weighted by watch time** (T_i seconds).
- Negative examples weight = 1.
- Loss: weighted logistic regression.
- Tại serving, output `e^x` (với x là logit) ≈ expected watch time.

Lý do: optimize click sẽ dẫn đến **clickbait** (thumbnail giật gân, user click rồi tắt sau 5s). Optimize watch time encourages content thật sự engaging.

#### Features ở ranking

Ranking dùng nhiều features hơn candidate gen:
- **Item features**: video age, channel quality, topic, language.
- **User features**: long-term interests, demographics.
- **User-item cross features**: số lần user đã xem channel này, time-since-last-watch của channel/topic.
- **Realtime features**: số impressions trong session, position features.
- **Context features**: device, time of day, geography.

Một feature đặc biệt impactful (paper 2016 highlight): **"time since user last watched a video from this channel"** — nó capture pattern user thường quay lại channel yêu thích sau một khoảng thời gian nhất định.

### 4.3 Post-processing & re-ranking

Sau ranking model, có thêm một layer **heuristics + constraints**:

- **Diversity**: không cho 5 videos từ cùng 1 channel xuất hiện liên tiếp (MMR — Maximal Marginal Relevance hoặc determinantal point processes).
- **Freshness boost**: multiply score của fresh videos với một factor.
- **Business rules**: ad insertion slots, age-appropriate filtering, regional restrictions.
- **Exploration**: inject một số videos với uncertainty cao để collect signal (epsilon-greedy hoặc Thompson sampling).

```python
def post_process(scored_candidates, user_context):
    # 1. Business filters (age, region, blocked)
    filtered = [c for c in scored_candidates if passes_policy(c, user_context)]
    
    # 2. Diversity reranking — MMR style
    final = []
    for _ in range(20):  # final feed size
        best = max(filtered, key=lambda c: mmr_score(c, final, lambda_=0.7))
        final.append(best)
        filtered.remove(best)
    
    # 3. Inject exploration items (~5% probability)
    if random() < 0.05:
        final[random_position] = sample_exploration_item(user_context)
    
    return final
```

### 4.4 Training infrastructure

YouTube train ở scale petabyte data. Public info cho thấy họ dùng:
- **TensorFlow** (Google's flagship framework — YouTube là internal user lớn).
- **TPU** cho training (recsys models large enough để dùng TPU pods).
- **Mesh-TensorFlow / GSPMD** cho model parallelism (suy đoán dựa trên Google papers).
- **Bigtable / Spanner / GCS** cho data storage.

Training cadence: theo các talks, ranking model được retrain **hàng ngày**, candidate gen model retrain **hàng tuần**. Online learning (continuous training) cũng đã được experiment nhưng public details chưa rõ. TikTok lại go all-in cho online learning paradigm — model weights cập nhật trong vài phút từ realtime watch events; xem [S1-02 TikTok Monolith](S1-02_tiktok_monolith_realtime_recommendation.md) cho comparison.

### 4.5 Serving infrastructure

```
              [Load balancer / front-end]
                       │
              ┌────────┴────────┐
              ▼                 ▼
        [Reco service]    [Reco service]   ... (replicated)
              │
              │  fan out
   ┌──────────┼──────────┐
   ▼          ▼          ▼
[Cand. gen][Cand. gen][Cand. gen]  ← multiple sources
 (ScaNN)    (co-watch) (fresh)
   │          │          │
   └─────┬────┴──────────┘
         ▼
     [Dedup + merge]
         │
         ▼
  [Ranking service]  ← TF Serving / Servomatic / custom
         │
         ▼
  [Post-processing]
         │
         ▼
     [Response]
```

Serving constraints (inferred from talks):
- Ranking inference ~50ms cho hundreds candidates trên TPU/GPU pools.
- Multiple replicas across regions cho latency + availability.
- Feature fetching từ online feature store (low-latency KV store, có thể là Bigtable hoặc Spanner).

---

## 5. Trade-offs & Design decisions

### 5.1 Two-stage vs one-stage

| Approach | Pros | Cons |
|---|---|---|
| **Two-stage** (chọn) | Scale to billion items; có thể dùng heavy DNN ở ranking; mỗi stage optimize riêng | Lỗi propagate (recall bottleneck); maintain 2 models; offline/online consistency phức tạp |
| **One-stage** (DNN over all items) | Simpler stack; consistent objective | Không khả thi với 10^9 items và 100ms latency |

YouTube chọn two-stage vì scale của họ buộc phải vậy. Các công ty nhỏ hơn (< vài triệu items) có thể chạy one-stage và đơn giản hơn nhiều.

### 5.2 Predict watch time vs predict CTR

| Metric | Pros | Cons |
|---|---|---|
| **CTR** | Đơn giản, label rõ ràng | Clickbait, không capture quality |
| **Watch time** (YouTube chọn) | Align với business (ad revenue, satisfaction) | Bias về long-form content; cần normalize theo video length |
| **Multi-task** (current gen) | Balance multiple signals | Phức tạp hơn, cần tune weights |

### 5.3 Sampled softmax vs full softmax vs negative sampling

YouTube dùng **sampled softmax** ở candidate gen. So sánh:

| Method | Pros | Cons |
|---|---|---|
| **Full softmax** | Theoretically correct | O(N) per step, N=10^9 → infeasible |
| **Sampled softmax** (chọn) | Unbiased estimator (importance weighted); scale tốt | Cần sample distribution Q tốt |
| **Negative sampling** (word2vec style) | Đơn giản, fast | Biased — không phải proper probability estimator |
| **In-batch negatives** | Free (reuse batch) | Bias về popular items có trong batch |

### 5.4 Feature engineering vs end-to-end learning

YouTube vẫn dùng **feature engineering nặng** (cross features, realtime counters) thay vì pure end-to-end. Lý do:
- Latent representations từ DNN không capture được hết business signals.
- Một số features (như "time since last watch") chứa strong prior — explicit hơn dễ train.
- Debug + monitor dễ hơn khi có named features.

Trade-off: nhiều feature → nhiều pipeline → maintenance cost cao. YouTube đầu tư mạnh vào feature platform để giảm cost này (tương tự Uber Michelangelo — xem [S4-01 Michelangelo feature store](../04-production/S4-01_uber_michelangelo_feature_store.md)).

### 5.5 Personalization vs popularity baseline

Một insight quan trọng từ paper 2016: **đơn thuần recommend trending videos** đã là baseline rất mạnh. Personalization phải beat baseline này — không trivial. Vì vậy, big tech reco luôn có **A/B test rigorous** với holdout là popularity baseline hoặc previous model — xem [S4-02 A/B testing & experimentation platforms](../04-production/S4-02_ab_testing_experimentation_platforms.md) cho CUPED variance reduction, mSPRT sequential testing, MAB exploration, SRM guardrails.

---

## 6. Lessons learned & Best practices

1. **Two-stage architecture là default cho large-scale reco**. Bắt đầu với pattern này, customize sau.

2. **Đừng optimize CTR alone** — sẽ dẫn đến clickbait. Optimize proxy của user satisfaction (watch time, dwell time, follow-up actions).

3. **Position bias là vấn đề lớn, nhưng có solution đơn giản** — train với position feature, serve với position=1.

4. **Multi-task learning thường win single-task** ở reco — share representation across CTR, watch time, like → regularize tốt hơn.

5. **Candidate diversity > candidate volume** — có nhiều candidate sources khác nhau (collaborative, content, fresh, exploration) tốt hơn là một source với top-1000.

6. **Cold-start cần candidate source riêng** — content-based hoặc upload-based, không kì vọng main retrieval handle được.

7. **Feature freshness có giá** — realtime features (counters cập nhật mỗi phút) thường impactful hơn features được precompute hàng ngày.

8. **A/B test mọi thứ, đo long-term metrics** — không chỉ click rate; đo retention, satisfaction surveys, weekly active users.

9. **Offline metrics chỉ là proxy** — AUC, NDCG offline không luôn correlate với A/B win. Luôn phải A/B test.

10. **Đầu tư vào platform** — engineering productivity của ML team quan trọng hơn một số tweaks model. YouTube đầu tư vào TFX, Vizier (hyperparameter tuning), feature platform, model registry.

---

## 7. References

### Papers

1. **Covington, Adams, Sargin (Google).** "Deep Neural Networks for YouTube Recommendations." RecSys 2016. [Link](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) — paper foundational cho candidate gen + ranking architecture.
2. **Zhao et al. (Google).** "Recommending What Video to Watch Next: A Multitask Ranking System." RecSys 2019. [Link](https://daiwk.github.io/assets/youtube-multitask.pdf) — MMoE multi-task ranking + position debiasing.
3. **Guo et al. (Google).** "Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)." ICML 2020. [arxiv 1908.10396](https://arxiv.org/abs/1908.10396) — ANN library Google open-sourced.

### Engineering blogs / talks

4. **Google AI Blog.** ["Announcing ScaNN: Efficient Vector Similarity Search"](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) (2020).
5. **Cristos Goodrow (YouTube).** Various talks về YouTube recommendation, đặc biệt "On YouTube's Recommendation System" (YouTube Blog, 2021). [Link](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/)
6. **Baylor et al. (Google).** ["TFX: A TensorFlow-Based Production-Scale Machine Learning Platform"](https://dl.acm.org/doi/10.1145/3097983.3098021) (KDD 2017) — mô tả Google's TFX (TensorFlow Extended) production ML platform mà YouTube là user internal.

### Related case studies (đọc tiếp)

- **Pinterest PinSage** (Ying et al. KDD 2018) — graph-based retrieval, alternative cho two-tower.
- **Instagram Reels** (Meta Engineering Blog 2022) — apply two-stage cho short-video reco.
- **TikTok Monolith** (Liu et al. 2022 arxiv) — real-time, sparse params, online learning.

### Lưu ý độ tin cậy

- Paper 2016 và 2019 là source chính thức (Google authors).
- Architecture chi tiết của generation hiện tại (2023+) chưa public hoàn toàn → một số phần (như sequence transformer cho user history) là **inferred from industry trends**, không phải confirmed.
- Số liệu QPS, latency là **approximations** từ talks/blogs công khai; internal numbers có thể khác đáng kể.
