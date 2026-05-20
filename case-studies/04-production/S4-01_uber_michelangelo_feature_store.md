# S4-01 — Uber Michelangelo: End-to-End ML Platform & Feature Store

> **Scope**: Production AI Systems
> **Difficulty**: Advanced
> **Tags**: ML platform, feature store, online/offline serving, training pipeline, model registry, monitoring, MLOps
> **Primary sources**: Uber Engineering Blog series ("Meet Michelangelo" 2017, "Scaling Michelangelo" 2019, "Michelangelo Palette: A Feature Engineering Platform at Uber" 2018), Hermann & Del Balso talks (QCon, MLConf 2017-2019), follow-up Uber Engineering posts 2020-2023.

---

## 1. Tổng quan (Overview)

Michelangelo là **Uber's internal end-to-end Machine Learning Platform**, launch internally năm 2015 và được public hoá ở blog post nổi tiếng tháng 9/2017 ("Meet Michelangelo: Uber's Machine Learning Platform"). Đây là một trong những **internal ML platform được document nhiều nhất** trong industry, và đặc biệt nổi tiếng vì **giới thiệu khái niệm "feature store"** ra industry (mặc dù Uber không phải người đầu tiên có ý tưởng — Booking, LinkedIn cũng có internal feature systems trước đó, nhưng Uber là người publicize và đặt tên khái niệm này).

Sau khi Michelangelo public, **feature store đã trở thành standard component** trong mọi ML platform serious: Tecton (founded by ex-Uber Michelangelo team), Feast (open-source), Vertex AI Feature Store (Google), SageMaker Feature Store (AWS), Databricks Feature Store, Hopsworks.

### Tại sao Michelangelo là case study quan trọng cho production AI?

- **Coverage end-to-end**: từ data ingestion → feature engineering → training → deployment → monitoring → retraining. Không phải case study chỉ một component.
- **Concept "feature store" lần đầu được crystallize**: solve một problem rất thật — **online/offline training-serving skew**.
- **Battle-tested ở scale**: Uber có hàng nghìn ML models in production (ETA prediction, surge pricing, fraud detection, UberEats restaurant ranking, driver-rider matching, …).
- **Tài liệu rất phong phú**: Uber blog 2017-2023 cover Palette (feature store), Pyro (probabilistic programming), Manifold (model debugging), DataK9 (data quality), …

### Bối cảnh business

Trước Michelangelo (~2015), data scientists ở Uber:
- Mỗi team build pipeline riêng → fragmented, không reusable.
- Train trên Jupyter, deploy bằng cách… không có process chuẩn.
- Features compute lại nhiều lần ở nhiều places → expensive + inconsistent.
- Không có model monitoring → models drift in silence.

Michelangelo ra đời để standardize: **một platform, một set of APIs, một set of best practices**, cho phép data scientists focus on modeling thay vì infra.

---

## 2. System Requirements

### 2.1 Functional requirements

- **Train models** trên multiple frameworks (XGBoost, TensorFlow, PyTorch, Spark MLlib) ở scale (TB-PB data).
- **Serve models** với **both batch và realtime** modes, support 10K+ QPS per model.
- **Feature store**: serve features cho training và serving **consistently** (không skew).
- **Model registry**: version models, track lineage (data → features → model → deployment).
- **Monitoring**: track prediction distribution, feature distribution, model performance, drift.
- **Retraining**: trigger retrain automatically khi drift detected hoặc schedule.
- **Experimentation**: A/B test models, shadow mode deployment.

### 2.2 Non-functional requirements

| Metric | Target (approximated, public) | Notes |
|---|---|---|
| Models in production | **hundreds → thousands** | Uber 2019 talks mention thousands |
| Features in catalog | **10K+ canonical features** | Across teams |
| Online feature lookup latency | **< 10 ms P99** | Cho complex prediction (e.g. ETA) |
| Online serving QPS | 100K+ QPS aggregate | Across all models |
| Batch training data | TB-PB scale | |
| Feature freshness (streaming) | seconds to minutes | Realtime aggregations |
| Feature freshness (batch) | hours | Daily ETL |

### 2.3 Constraints quan trọng

- **Training/serving skew là enemy #1**: features compute slightly khác giữa training và serving → silent accuracy degradation.
- **Multi-tenant**: hundreds of teams share platform → cần isolation, quotas, access control.
- **Heterogeneous models**: từ simple linear/tree-based (XGBoost) đến deep learning. Platform phải support cả hai.
- **Geographically distributed**: Uber operate globally, models deploy nhiều regions.

---

## 3. High-level Architecture

Michelangelo's 6-step workflow (per Uber 2017 blog):

```
   ┌────────────────────────────────────────────────────────────────┐
   │                  6-step ML Workflow                              │
   └────────────────────────────────────────────────────────────────┘

   1. Manage data         →  Palette (Feature Store) + DataK9 (quality)
                              ┌────────────────────┐
                              │ Offline (Hive/HDFS)│
                              │ Online (Cassandra) │
                              └────────────────────┘
   
   2. Train models        →  Distributed training framework
                              (Spark MLlib, XGBoost, TF, Horovod)
   
   3. Evaluate models     →  Model report, performance metrics
                              Manifold (debugging)
   
   4. Deploy models       →  Online (low-latency serving), Batch,
                              Library (embedded in service)
   
   5. Make predictions    →  Prediction service
                              (gRPC, REST endpoints)
   
   6. Monitor predictions →  Online metrics, feature distribution,
                              drift detection, alerts
```

### Architecture stack

```
                         ┌──────────────────────────────┐
                         │       User-facing UI / SDK    │
                         │   (Web UI, Python SDK, gRPC)  │
                         └───────────────┬──────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
  ┌──────────┐                    ┌──────────────┐                ┌──────────┐
  │ Palette  │                    │  Training    │                │  Model   │
  │ (Feature │                    │  service     │                │  Registry│
  │  Store)  │                    │              │                │          │
  └─────┬────┘                    └──────┬───────┘                └────┬─────┘
        │                                │                              │
   ┌────┴────────────┐                   │                              │
   ▼                 ▼                   ▼                              ▼
 ┌──────┐      ┌──────────┐        ┌──────────┐                  ┌──────────┐
 │Hive/ │      │Cassandra │        │ Spark    │                  │ Model    │
 │HDFS  │      │ (online  │        │ cluster, │                  │  store   │
 │(offl)│      │  KV)     │        │ Horovod  │                  │ (S3/HDFS)│
 └──────┘      └──────────┘        └──────────┘                  └──────────┘
        │           ▲
        │           │
        │           │ (sync via Kafka / ETL)
        └───────────┘

                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │      Prediction service       │
                         │   (low-latency online serving)│
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │   Monitoring + Drift Detect   │
                         │   (Manifold, M3, alerts)      │
                         └──────────────────────────────┘
```

---

## 4. Deep dive các components chính

### 4.1 Palette — Feature Store

Đây là component **iconic** nhất của Michelangelo. Palette giải quyết hai vấn đề core:

1. **Reusability**: data scientist nào đó compute "average trip duration of driver in last 7 days" → feature này nên reusable cho team khác, không phải re-implement.
2. **Training/serving consistency**: feature dùng training phải y hệt feature dùng serving — same code path, same data source.

#### Two-store design

Palette có **two stores**:

```
                  Feature definition (single source of truth)
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
       OFFLINE STORE                    ONLINE STORE
       (Hive / HDFS)                    (Cassandra)
       
       For training:                    For serving:
       - bulk reads                     - point lookups by key
       - point-in-time correct          - low latency (<10ms P99)
       - historical (years)              - fresh (latest value)
       - batch updates                   - streaming + batch updates
```

**Critical property**: cả hai stores **populated từ cùng một feature definition** → guaranteed consistency.

#### Feature types

Palette support nhiều loại features:

| Feature type | Compute | Storage | Example |
|---|---|---|---|
| **Batch features** | Spark job daily/hourly | Hive (offline) → Cassandra (online sync) | "Driver's 7-day avg rating" |
| **Streaming features** | Kafka + streaming aggregator | Cassandra directly | "Rider's trips in last 1 hour" |
| **Real-time features** | Compute at request time | Not stored, computed on-the-fly | "Distance from current pickup point to driver" |
| **Lookup features** | Static dimensional data | Cassandra | "Restaurant cuisine type" |

#### Feature DSL

Uber engineers define features bằng **DSL (Domain-Specific Language)** giống SQL+Python:

```python
# Pseudo-code feature definition (DSL-like)
@feature(
    name="driver_avg_rating_7d",
    entity="driver_id",
    backfill_from="2020-01-01",
    refresh="daily",
)
def driver_avg_rating_7d(spark_df):
    return (
        spark_df
        .filter(F.col("event_time") > F.current_date() - F.expr("INTERVAL 7 DAYS"))
        .groupBy("driver_id")
        .agg(F.avg("rating").alias("driver_avg_rating_7d"))
    )

@feature(
    name="driver_trips_last_hour",
    entity="driver_id",
    type="streaming",
    refresh="1 minute",
)
def driver_trips_last_hour(kafka_stream):
    return (
        kafka_stream
        .filter(...)
        .groupBy("driver_id", window("event_time", "1 hour"))
        .agg(F.count("*").alias("driver_trips_last_hour"))
    )
```

Definitions go into **central catalog** với metadata: owner, schema, description, freshness SLO, popularity (number of consumers).

#### Point-in-time correctness

Khi training, một issue rất subtle: **feature snapshot phải reflect state AT THE TIME of label**. Nếu không, **label leakage**:

```
Bad (label leakage):
  Label time: 2024-01-15 10:00:00 (user click)
  Feature "user_total_orders": computed as of NOW (2024-05-20)
  → Training sees future state, will overfit.

Good (point-in-time):
  Label time: 2024-01-15 10:00:00
  Feature "user_total_orders": computed as of 2024-01-15 09:59:59
  → Training sees same state as serving would see.
```

Palette enforces point-in-time correctness through **time-versioned features** trong offline store. Generating training datasets là **temporal join** giữa labels và features by entity + timestamp:

```sql
-- Conceptual temporal join
SELECT 
    label.driver_id, label.event_time, label.trip_completed,
    features.driver_avg_rating_7d,
    features.driver_trips_last_hour,
    ...
FROM labels AS label
LEFT JOIN driver_features AS features
    ON label.driver_id = features.driver_id
    AND features.feature_time <= label.event_time
    AND features.feature_time = (
        SELECT MAX(f2.feature_time) 
        FROM driver_features f2 
        WHERE f2.driver_id = features.driver_id 
          AND f2.feature_time <= label.event_time
    )
```

(Trong thực tế dùng asof-join trong Spark hoặc Snowflake.)

#### Online serving — Cassandra

Cho online lookup, Palette dùng **Cassandra** (KV store distributed, low-latency):
- Key: entity_id (e.g. driver_id).
- Value: latest feature values (denormalized — multiple features per row).
- Replication factor + multi-region cho availability.

Lookup pattern:

```python
# Online feature lookup at prediction time
def predict(driver_id, rider_id, pickup_location):
    # Single Cassandra get (or batched gets via multi-key)
    driver_features = palette.online_lookup(
        entity_type="driver",
        entity_id=driver_id,
        feature_names=["driver_avg_rating_7d", "driver_trips_last_hour", ...],
    )
    rider_features = palette.online_lookup(
        entity_type="rider", entity_id=rider_id, ...
    )
    
    # Compute realtime features
    distance = compute_distance(driver_features["last_location"], pickup_location)
    
    # Forward through model
    feature_vector = build_vector(driver_features, rider_features, distance, ...)
    return model.predict(feature_vector)
```

Latency budget cho online lookup là **<10ms P99** cho complex predictions.

### 4.2 Training service

Michelangelo support **multiple frameworks** thông qua a unified training API:

| Framework | Use case | Notes |
|---|---|---|
| **Spark MLlib** | Classical ML at scale (linear, RF) | Default for tabular |
| **XGBoost** | Gradient boosting | Most popular ở Uber per blog 2017 |
| **TensorFlow + Horovod** | Deep learning, distributed training | For DL workloads |
| **PyTorch** (added later) | Research + production DL | |

Workflow:

```
1. User submits training job spec (YAML/JSON):
   - feature list (from Palette)
   - label definition
   - time range
   - hyperparameters
   - framework
   
2. Training service:
   - Materialize training dataset (point-in-time join via Palette)
   - Spin up Spark/Horovod cluster
   - Run training
   - Save model artifact to S3
   
3. Output:
   - Model file (e.g. .pkl, .pb)
   - Model report (AUC, RMSE, feature importance)
   - Lineage record (data version, code version, hyperparams)
```

Hyperparameter tuning: Bayesian optimization (via Vizier-like internal tool).

### 4.3 Model registry

Mỗi model trained được register với:
- **Unique model ID** + version
- **Metadata**: owner, project, framework, training data hash, feature schema
- **Artifacts**: model file, eval metrics, feature transformations
- **Lineage**: link đến training job, dataset, feature definitions

Deployment workflow:

```
Model trained → registered → reviewed (eval metrics) → promoted to staging 
              → A/B tested (shadow mode) → promoted to production
```

Registry là **single source of truth** cho deployment — không ai deploy model nào không qua registry.

### 4.4 Deployment & prediction service

Michelangelo support 3 deployment modes:

| Mode | Use case | Latency | Throughput |
|---|---|---|---|
| **Online** | Realtime predictions (surge, ETA) | < 50 ms | High QPS |
| **Batch** | Periodic scoring (fraud sweeps, marketing) | Minutes-hours | Massive throughput |
| **Library** (embedded) | Critical-path low-latency (e.g. driver-rider match) | < 5 ms | Embedded in service |

**Online prediction service**:
- Models loaded into prediction server (one model per process or shared).
- gRPC API.
- Multi-replica for HA.
- Cache for frequently-queried entities (optional).

Sample architecture:

```
[Client app] ──gRPC──→ [Load balancer] ──→ [Prediction service replicas]
                                                    │
                                                    ▼
                                            [Online feature lookup
                                             (Palette Cassandra)]
                                                    │
                                                    ▼
                                            [Model inference]
                                                    │
                                                    ▼
                                            [Response + log to Kafka]
```

Predictions được **log to Kafka** for monitoring (xem 4.5).

### 4.5 Monitoring & drift detection

Một insight quan trọng từ Michelangelo: **deployment is not the end**. Models drift, data distribution shifts, world changes.

Components:

1. **Prediction logging**: mỗi prediction (input features + output + groundtruth khi có) log to Kafka → Hive.
2. **Distribution monitoring**:
   - Per-feature distribution drift (PSI, KS test).
   - Per-prediction-bucket distribution.
   - Compare current vs training baseline.
3. **Performance monitoring** (when labels available):
   - AUC, RMSE, MAE on rolling windows.
   - Calibration plots.
4. **Alerting**: triggers Slack/PagerDuty when:
   - Feature distribution shifts beyond threshold.
   - Prediction distribution shifts.
   - Performance drops.

```
Daily monitoring report (conceptual):
  Model: eats_eta_v3.2
  Date range: 2024-05-19
  
  Feature drift:
    - restaurant_avg_prep_time: PSI = 0.18  ⚠ (threshold 0.15)
    - rider_distance_to_restaurant: PSI = 0.05  ✓
  
  Prediction drift:
    - mean predicted ETA: 18.4 min (training baseline: 16.2)  ⚠ +14%
    - distribution KS test p-value: 0.001  ⚠
  
  Performance (lagged labels):
    - MAE: 3.2 min (last week: 3.0)  ⚠ +7%
    - AUC for "late delivery" classifier: 0.83 (last week: 0.85)  ⚠
  
  Recommended action: investigate restaurant_avg_prep_time drift, 
                       possibly retrain.
```

#### Manifold — model debugging tool

Uber built **Manifold** (open-sourced 2019) — a model-agnostic debugging UI. Slices predictions by feature buckets, shows where model performs poorly. Helps identify bias / failure modes (e.g. model bad for rides starting between 2-4 AM, model bad for restaurants in specific cuisine).

### 4.6 Retraining & continuous learning

Triggered retraining:
- **Schedule-based**: daily, weekly per model SLA.
- **Drift-triggered**: PSI threshold breached → kick off retrain.
- **Manual**: data scientist trigger after investigation.

Each retrain follows same pipeline → same training service → registry → A/B test before full promotion. **No raw cron jobs hidden in someone's account** — everything goes through platform.

---

## 5. Trade-offs & Design decisions

### 5.1 Centralized feature store vs decentralized

| Approach | Pros | Cons |
|---|---|---|
| **Centralized (Palette)** | Reuse, consistency, governance | Single point of contention, ownership disputes |
| **Decentralized (per-team)** | Team autonomy, fast iteration | Duplication, skew, no reuse |

Uber went centralized. Vital cho a company với hundreds of ML teams nhưng cost: cần dedicated platform team, governance process, SLA negotiation.

### 5.2 Two-store (offline + online) vs single store

| Approach | Pros | Cons |
|---|---|---|
| **Two-store** (Palette) | Optimize each store for its use case | Sync complexity, possible delays |
| **Single store for both** | Simpler | Either slow training reads or slow online lookups |

Two-store is standard now. Sync mechanism (Kafka pub-sub) is the tricky part.

### 5.3 Online sync: push vs pull

- **Push**: streaming pipeline computes feature, pushes to Cassandra immediately.
- **Pull**: at lookup time, query offline store (slow).

Uber chose **push** for low online lookup latency. Trade-off: must handle out-of-order, late-arriving events, exactly-once semantics in streaming pipeline.

### 5.4 Materialize training dataset vs on-the-fly join

| Approach | Pros | Cons |
|---|---|---|
| **Materialize** (Uber default) | Fast training iteration, reproducible | Storage cost, can be stale |
| **On-the-fly** | Fresh, no storage | Slow training, expensive query |

Materialize is standard. Datasets versioned for reproducibility.

### 5.5 Heterogeneous frameworks vs single standard

Uber support multiple (XGBoost, TF, PyTorch, Spark). Trade-off:

| Approach | Pros | Cons |
|---|---|---|
| **Multiple frameworks** | Right tool for job | Maintenance burden, more deployment paths |
| **Single framework** | Simpler platform | Forces wrong tool sometimes |

Uber chose multiple because XGBoost dominates tabular while TF/PyTorch needed cho DL. Platform abstracts deployment uniformly even if training framework differs.

### 5.6 Build vs buy

In 2015 nothing public matched Uber's scale → had to build. Today there are alternatives:

| Option | Pros | Cons |
|---|---|---|
| **Build internal (Uber 2015 choice)** | Fit exact needs, IP | Engineering cost, maintenance |
| **Tecton** (commercial, from ex-Uber team) | Battle-tested, less to build | Cost, vendor dependency |
| **Feast** (open-source) | Free, community | Less mature, more glue code |
| **Cloud provider (Vertex/SageMaker/Databricks)** | Integrated with cloud | Lock-in, sometimes less flexible |

For new orgs today, **Feast or commercial feature store is usually right**. Building from scratch is rarely justified.

---

## 6. Lessons learned & Best practices

1. **Training/serving skew is the #1 silent killer in production ML**. Even 1% feature difference → significant accuracy drop. A feature store with single source of truth is non-negotiable for serious ML orgs.

2. **Point-in-time correctness must be enforced by the platform, not by humans**. Asking data scientists to "remember to filter by timestamp" → guaranteed bugs.

3. **Feature reuse compounds**. First feature for a new team is 100% effort. After 10 features, reuse becomes 50% of features. After 100, reuse > 80%. Investing early is high-ROI.

4. **Online feature latency budget < 10ms P99 is achievable** with proper KV store (Cassandra, Redis, ScyllaDB), denormalization, and connection pooling. Don't allow lookups via slow API calls.

5. **Lineage is gold**. Knowing "this model was trained on this data version with this feature definition" enables: reproducibility, debugging, compliance audits (GDPR, etc.), retraining safely.

6. **Monitoring must include feature drift, not just model accuracy**. Often you discover drift weeks before accuracy degrades (because labels are lagged).

7. **Multi-mode deployment is necessary**: not every model fits online serving. ETA prediction = online. Marketing scoring = batch. Driver-rider match = embedded library. Platform must support all three.

8. **A/B testing must be platform-native**. Don't allow models to "go to prod" without canary/shadow mode. Built-in traffic splitting per model reduces accidents dramatically.

9. **Feature definitions outlive models**. A model from 2018 is retired; a "user_total_orders_lifetime" feature definition from 2018 still drives 50 active models. Treat features as a first-class durable asset.

10. **Governance matters at scale**. Who owns feature X? What's its SLA? Can it be deleted? Who consumes it? Without governance, feature catalog becomes a graveyard.

11. **Allow framework heterogeneity but enforce platform uniformity**. Don't force XGBoost teams to switch to TF, but require all models go through same registry + serving infra.

12. **Documentation & discoverability are platform features**. A feature store with 10K features and no search UI is unusable. Invest in metadata, search, examples.

---

## 7. References

### Engineering blogs (chính)

1. **Hermann & Del Balso (Uber).** "Meet Michelangelo: Uber's Machine Learning Platform." Uber Engineering Blog, Sept 2017. [Link](https://www.uber.com/blog/michelangelo-machine-learning-platform/) — paper foundational.
2. **Del Balso, Hermann (Uber).** "Scaling Michelangelo." Uber Engineering Blog, 2019.
3. **Uber Engineering.** "Michelangelo Palette: A Feature Engineering Platform at Uber." Approx. 2018-2019 talks.
4. **Uber Engineering.** "Productionizing Machine Learning Models at Uber" — series various authors 2018-2022.
5. **Uber Engineering.** "Manifold: A Model-Agnostic Visual Debugging Tool" 2019. (Manifold open-source: [github.com/uber/manifold](https://github.com/uber/manifold)).

### Talks

6. **Jeremy Hermann.** "Michelangelo: ML Platforms at Uber." Various venues: QCon SF 2017, MLConf 2018, Strata 2018-2019.
7. **Mike Del Balso.** "Feature Stores for ML" — multiple talks 2019-2020, foundational for the feature store concept becoming industry-wide.

### Open-source ecosystem (post-Michelangelo)

8. **Feast (Feature Store).** Open-source, founded by ex-Gojek/Google. [feast.dev](https://feast.dev/). Influenced by Michelangelo.
9. **Tecton.** Commercial feature platform, founded by ex-Uber Michelangelo team (Del Balso, Hermann). [tecton.ai](https://tecton.ai/).
10. **Hopsworks.** Feature store + ML platform from Logical Clocks. [hopsworks.ai](https://hopsworks.ai/).

### Related papers

11. **Polyzotis et al. (Google).** "Data Lifecycle Challenges in Production Machine Learning." SIGMOD Record 2018 — generalized lessons.
12. **Baylor et al. (Google).** "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform." KDD 2017 — Google's analogous internal platform.

### Related case studies (đọc tiếp)

- **S1-01 YouTube recommendation** — feature store concepts apply equally to retrieval + ranking.
- **S2-01 Meta DLRM** — recsys models served via platforms like Michelangelo.
- **S4-02 A/B Testing & Experimentation Platforms** — natural complement to Michelangelo.
- **S4-03 Data & Model Drift Detection** — deep dive into the monitoring piece of Michelangelo.

### Độ tin cậy

- Michelangelo 2017 blog post + follow-up Uber blogs là **chính thức**, high confidence cho architecture.
- Số liệu cụ thể (QPS, latency, model count) — Uber blogs có mention một số con số nhưng có thể đã thay đổi nhiều từ 2017-2019 đến nay (2025-2026).
- Một số detail về DSL, framework support có thể đã evolve (Michelangelo 2025 chắc chắn khác Michelangelo 2017) — bài này focus vào **conceptual architecture** vẫn áp dụng được, ít focus vào exact 2025 implementation.
- "Public information may not reflect current internal state" — Uber không update blog public với latest architecture chi tiết.
