# S4-02 — A/B Testing & Experimentation Platforms (Booking, Airbnb, LinkedIn, Netflix)

> **Scope**: Production AI Systems
> **Difficulty**: Intermediate
> **Tags**: A/B testing, experimentation platform, CUPED, variance reduction, sequential testing, interleaving, causal inference, MLOps
> **Primary sources**:
> - Kohavi et al. "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing" (Cambridge, 2020) — bible của field.
> - Deng et al. "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED paper, Microsoft, WSDM 2013).
> - Booking.com, Airbnb, LinkedIn, Netflix, Uber engineering blogs (cited inline).

---

## 1. Tổng quan (Overview)

A/B testing — hay đầy đủ là **online controlled experimentation** — là cơ chế **đo lường chính** giúp các công ty internet xác nhận một change (ML model, UI tweak, ranking algorithm, pricing strategy) thực sự cải thiện business metrics, chứ không chỉ "looks better in offline eval".

Đây là **closure của ML loop**:
- Bạn train một model (xem [S2-01 DLRM](S2-01_meta_dlrm_architecture.md)).
- Bạn deploy nó qua feature store + serving infra ([S4-01 Michelangelo](S4-01_uber_michelangelo_feature_store.md)).
- Nhưng cuối cùng, **làm sao biết model mới tốt hơn baseline?** Offline AUC tăng 0.5% có translate sang +revenue không? Câu trả lời chỉ đến từ **A/B test**.

Các big tech có experimentation culture mature thường chạy **hàng nghìn experiments concurrent**:
- **Microsoft Bing**: ~20,000+ experiments/year (Kohavi 2020).
- **Booking.com**: ~1,000 concurrent experiments thường xuyên, đã chạy >100,000 experiments tổng.
- **Netflix**: hàng trăm concurrent, mỗi feature/personalization model đều được A/B test.
- **Facebook/Meta, Google, Amazon**: số liệu nội bộ không public nhưng known >1000 concurrent.

Case study này design một **experimentation platform** end-to-end, cover:
- Statistical foundations (sample size, hypothesis test, multiple testing).
- Variance reduction techniques (CUPED, stratification).
- Sequential testing, interleaving.
- Advanced: Bayesian, MAB (multi-armed bandits), CUPAC.
- Platform architecture cho **trustworthy** experiments at scale (consistent hashing, randomization, metric pipelines, traps detection).

### Tại sao A/B testing là kỹ năng critical cho AI engineer?

- **Counter-intuitive results phổ biến**: ~80% experiments ở Bing/Microsoft không cải thiện metric mong muốn (Kohavi 2017). Without A/B test, bạn ship false improvements liên tục.
- **Offline metrics khác online metrics**: AUC/NDCG tăng có thể giảm user retention (e.g. quá personalize → echo chamber → boring).
- **Network effects, novelty effects, primacy effects** — chỉ A/B test phát hiện được.
- **Multi-objective trade-offs**: model mới tăng CTR nhưng giảm revenue → cần đo cả hai.

---

## 2. System Requirements

### 2.1 Functional requirements

- **Experiment definition**: PM/engineer khai báo experiment với hypothesis, metrics, target population, traffic %, duration.
- **Random assignment**: mỗi user (hoặc unit) được assign vào variant deterministic (consistent across requests).
- **Traffic allocation**: split traffic theo % configurable (e.g. 50/50, 10/10/10/70).
- **Metrics computation**: aggregate event logs → user-level metrics → group statistics → significance tests.
- **Dashboard**: visualize metric lift, confidence intervals, p-values, segment breakdowns.
- **Interaction detection**: alert khi 2 experiments concurrent có interaction effect.
- **Guardrails**: monitor key metrics không degrade (latency, error rate, revenue).
- **Stopping criteria**: support early stopping (with proper sequential test) hoặc fixed-duration.
- **Holdout groups**: long-term holdout để detect compounding effects.

### 2.2 Non-functional requirements (typical big tech platform)

| Metric | Target |
|---|---|
| Concurrent experiments supported | 100 - 10,000 |
| Assignment latency | < 10ms (in-line với serving) |
| Assignment consistency | 100% deterministic given user_id |
| Event logging volume | Billion events/day |
| Metric computation latency | Daily snapshots (preferred) hoặc near-realtime (4-hour windows) |
| Statistical correctness | False positive rate < α (typically 5%) under null hypothesis |
| Onboarding effort | < 1 hour for new experiment (config + go live) |

### 2.3 Constraints quan trọng

- **Trustworthiness > novelty**: tốt hơn 1 platform A/B test đúng còn hơn 10 features fancy mà số liệu sai. Đây là principle Kohavi nhấn mạnh xuyên suốt sách.
- **Sample size limits**: cho dù user base 1B, statistical power vẫn limited bởi effect size + variance. Nhiều experiments cần weeks → tháng.
- **Multiple testing**: chạy 1000 experiments với α=0.05 → expect 50 false positives. Cần FDR (False Discovery Rate) control hoặc Bonferroni.
- **SUTVA violation** (Stable Unit Treatment Value Assumption): user A bị ảnh hưởng bởi user B's treatment (e.g. social network, marketplace) → naive A/B test biased.

---

## 3. High-level Architecture

```
                    EXPERIMENTATION PLATFORM ARCHITECTURE

   ┌─────────────────── CONFIG PLANE ──────────────────────────────┐
   │                                                               │
   │  Experiment metadata store (Postgres / DynamoDB)              │
   │  - exp_id, owner, hypothesis, metrics, status                 │
   │  - variant definitions, traffic %                             │
   │  - targeting rules (segments, geo, app version)               │
   │  - guardrail metrics + thresholds                             │
   │                                                               │
   │  + UI for PM/engineer to create/edit experiments              │
   └───────────────────────────────────────────────────────────────┘
                              │
                              │ (config push)
                              ▼
   ┌─────────────────── ASSIGNMENT SERVICE (online) ───────────────┐
   │                                                               │
   │   Incoming request (user_id, context)                         │
   │             │                                                 │
   │             ▼                                                 │
   │   ┌───────────────────────────────────────────┐               │
   │   │ For each active experiment:               │               │
   │   │   - Check targeting rules → in audience?  │               │
   │   │   - Hash(user_id, exp_id, salt) % 100     │               │
   │   │     → assign variant                       │               │
   │   │   - Log assignment event                  │               │
   │   └───────────────────────────────────────────┘               │
   │             │                                                 │
   │             ▼                                                 │
   │   Return: {exp_id: variant} dict to client/app                │
   │                                                               │
   │   Latency budget: < 10ms (often in feature store call)        │
   └───────────────────────────────────────────────────────────────┘
                              │
                              │ (assignment + business events)
                              ▼
   ┌─────────────────── EVENT PIPELINE ────────────────────────────┐
   │                                                               │
   │   Kafka → Spark/Flink → Data lake (Parquet on S3/GCS/HDFS)    │
   │   - assignment_events (user, exp, variant, ts)                │
   │   - business_events (clicks, conversions, watch_time, ...)    │
   └───────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
   ┌─────────────────── METRIC COMPUTATION ────────────────────────┐
   │                                                               │
   │   Daily / hourly batch jobs (Spark, Airflow, dbt)             │
   │     ├─► Join assignments × events on user_id                  │
   │     ├─► Compute per-user metric: ctr, watch_time, revenue     │
   │     ├─► Aggregate by variant: mean, variance, count           │
   │     ├─► CUPED adjustment using pre-experiment covariate       │
   │     └─► Statistical test: t-test / Mann-Whitney / etc.        │
   │                                                               │
   │   Output: metric tables (variant, metric, lift, CI, p-value)  │
   └───────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
   ┌─────────────────── ANALYSIS & REPORTING ──────────────────────┐
   │                                                               │
   │   Dashboard (web UI):                                         │
   │     - Per-experiment summary (lift, CI, decision)             │
   │     - Segment breakdowns (country, platform, new vs returning)│
   │     - Time series (daily metric evolution)                    │
   │     - Guardrail status (any red metric?)                      │
   │     - Sample ratio mismatch (SRM) detector                    │
   │     - Multiple testing correction status                      │
   │                                                               │
   │   Alerting:                                                   │
   │     - SRM detected → immediate page                           │
   │     - Guardrail breach → escalate                             │
   │     - p-value < 0.001 (clear win/loss) → notify owner         │
   └───────────────────────────────────────────────────────────────┘
```

### Companies and their platform names (public references)

| Company | Platform name | Key references |
|---|---|---|
| Microsoft | ExP (Experimentation Platform) | Kohavi book, KDD talks |
| Google | Overlapping Experiment Infrastructure | Tang et al. 2010 KDD paper |
| Facebook/Meta | PlanOut + Deltoid (internal) | PlanOut paper, Bakshy 2014 |
| Netflix | XP / ABlaze | Netflix Tech Blog |
| Airbnb | ERF (Experimentation Reporting Framework) | Airbnb Engineering blog |
| LinkedIn | XLNT | LinkedIn Engineering blog |
| Booking.com | (internal, no public name) | Lukas Vermeer talks |
| Uber | XP | Uber Engineering blog |
| Spotify | (internal, "Spotify experimentation platform") | Spotify R&D blog |

---

## 4. Deep dive các components chính

### 4.1 Random assignment — đảm bảo unbiased split

#### Hash-based deterministic assignment

```python
import hashlib

def assign_variant(user_id, exp_id, salt, num_variants=2):
    """
    Deterministic, balanced assignment via hashing.
    """
    hash_input = f"{user_id}_{exp_id}_{salt}".encode()
    hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
    return hash_val % num_variants
```

**Key properties**:
- **Deterministic**: same user → same variant across requests (consistent UX).
- **Independent across experiments**: different `salt` mỗi exp → no correlation between assignments.
- **Balanced**: với good hash function, distribution gần uniform.

#### Layered experiments (Google's overlapping infrastructure)

Vấn đề: nếu 1000 experiments concurrent, mỗi exp lấy 5% traffic → cần 5000% traffic, impossible.

**Solution — layered design**:
- Phân traffic thành **layers** (mỗi user thuộc 1 bucket trong mỗi layer).
- Experiments cùng layer **mutually exclusive** (nếu có interaction risk).
- Experiments khác layer **orthogonal** (independent randomization).

```
Layer "Ranking algorithm":  exp_A (50%) | exp_B (50%)
Layer "UI color":           exp_C (33%) | exp_D (33%) | control (34%)
Layer "Recommendation":     exp_E (10%) | control (90%)
```

Một user là (exp_A, exp_C, control) → có thể participate trong 3 experiments cùng lúc, không bị interaction.

#### Sample Ratio Mismatch (SRM) — must monitor

**Definition**: nếu config split 50/50 nhưng actual data shows 49.5/50.5 với p-value < 0.001 → SRM detected.

**Causes**:
- Bot traffic skewed.
- Variant-specific bug (e.g. variant B crash → user reload → re-assigned).
- Filtering bias (e.g. only counted "engaged users" sau treatment).
- Logging asymmetry.

**Implication**: SRM **invalidates all metrics**. Không trust kết quả cho đến khi root cause.

```python
from scipy.stats import chi2_contingency

def check_srm(observed_counts, expected_ratios):
    expected = [sum(observed_counts) * r for r in expected_ratios]
    chi2, p, _, _ = chi2_contingency([observed_counts, expected])
    return p < 0.001   # alert if SRM detected
```

### 4.2 Statistical foundations

#### Sample size calculation

For two-sided t-test, normal approximation:

```
n_per_variant ≈ (z_{α/2} + z_β)² · 2 · σ² / Δ²
```

- `z_{α/2}` = 1.96 cho α=0.05.
- `z_β` = 0.84 cho power = 0.80.
- `σ²` = variance của metric (per user).
- `Δ` = minimum detectable effect (MDE).

Ví dụ: metric mean = 5 minutes, σ = 10, want detect Δ = 0.05 (1%):
```
n ≈ (1.96 + 0.84)² · 2 · 100 / 0.0025 ≈ 627,200 per variant
```

→ Cần **1.25M users total**. Nếu DAU = 1M và experiment lấy 10% traffic → cần 12 days.

**Python pseudo-code**:
```python
from scipy import stats

def sample_size(sigma, mde, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return int(((z_alpha + z_beta) ** 2 * 2 * sigma ** 2) / mde ** 2)
```

#### Hypothesis test

- **Continuous metrics** (revenue, watch time): **Welch's t-test** (handles unequal variance).
- **Binary metrics** (CTR, conversion): **Z-test for proportions** or **chi-square test**.
- **Skewed metrics** (revenue with heavy tail): consider **log transformation**, or **Mann-Whitney U** non-parametric.

```python
from scipy import stats

# Welch's t-test
t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)

# Z-test for proportions (CTR)
def two_proportion_z(x1, n1, x2, n2):
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value
```

#### Confidence interval

CI cho lift:
```
lift = (treatment_mean - control_mean) / control_mean
SE_lift ≈ √(SE_treat²/control_mean² + ... delta method)
CI_95% = lift ± 1.96 · SE_lift
```

**Best practice**: luôn show CI thay vì chỉ p-value. CI cho phép thấy magnitude + uncertainty.

### 4.3 CUPED — Variance Reduction (must-have technique)

**CUPED** = **C**ontrolled-experiment **U**sing **P**re-**E**xperiment **D**ata. Microsoft paper (Deng et al. WSDM 2013).

#### Intuition

Variance của metric (Y) lớn → cần sample size lớn. Nếu có covariate X **correlated với Y** mà **không bị affected bởi treatment** (e.g. user behavior trước experiment), có thể "trừ bớt" variance giải thích bởi X.

#### Math

```
Y_adjusted = Y - θ · (X - X_mean)

where θ = Cov(Y, X) / Var(X)
```

`Y_adjusted` có **mean giống Y** nhưng **variance nhỏ hơn**:
```
Var(Y_adjusted) = Var(Y) · (1 - ρ²)
```
với `ρ` = correlation between Y và X.

#### Practical example

Y = revenue trong 7 days của experiment.
X = revenue 28 days **trước** experiment start (pre-period).

Typical: ρ ≈ 0.5-0.7 → variance giảm 25-50% → sample size needed giảm tương ứng → experiment kết thúc nhanh hơn ~30-50%.

#### Pseudo-code

```python
import numpy as np

def cuped(y, x):
    """
    y: post-experiment metric per user (numpy array)
    x: pre-experiment covariate per user
    Returns adjusted y with same mean, lower variance.
    """
    theta = np.cov(y, x)[0, 1] / np.var(x)
    return y - theta * (x - x.mean())

# Trong analysis:
y_adj = cuped(y_all_users, pre_period_revenue)
t_stat, p_val = stats.ttest_ind(
    y_adj[treatment_mask], y_adj[control_mask], equal_var=False
)
```

#### CUPAC (extension)

CUPAC = **C**UPED with **A**ny **C**ovariate. Replace `X` bằng **predicted Y** từ ML model trên pre-experiment features.

```
X = model.predict(user_features_pre_experiment)
```

Improvement: có thể giảm variance thêm 10-20% so với CUPED, vì ML model uses more covariates.

### 4.4 Sequential testing (early stopping done right)

#### Vấn đề với "peeking" naive

Nếu PM check kết quả mỗi ngày và stop khi p < 0.05 → **false positive rate inflate**:
- α=0.05, 14 peeks → actual false positive rate ~30%.
- Đây là lý do nhiều "wins" không reproduce.

#### Group sequential testing (Pocock, O'Brien-Fleming)

Adjust p-value threshold dựa trên số peeks dự kiến trước.

#### Always Valid Inference (mSPRT — used by Optimizely, Netflix)

Johari et al. 2017: sử dụng **mixture Sequential Probability Ratio Test**, cho phép **stop anytime** với valid p-value.

Key formula (high level):
```
Λ(t) = ∫ exp(λ · n(t) · effect(t) - λ² · n(t) · variance / 2) dπ(λ)

Reject H0 if Λ(t) > 1/α
```

`dπ(λ)` = prior over effect sizes (typically Gaussian centered at 0).

```python
# Simplified mSPRT implementation
def msprt_pvalue(diff_mean, diff_var, n, prior_sigma=1.0, alpha=0.05):
    # See Johari et al. 2017 for exact formula
    # Returns "always-valid p-value" — can stop anytime
    ...
```

#### Bayesian approach

Compute **posterior probability** P(treatment > control | data). Stop khi posterior > 0.95.

```python
import pymc as pm

with pm.Model() as model:
    p_control = pm.Beta('p_control', alpha=1, beta=1)
    p_treat = pm.Beta('p_treat', alpha=1, beta=1)
    obs_c = pm.Binomial('obs_c', n=n_c, p=p_control, observed=clicks_c)
    obs_t = pm.Binomial('obs_t', n=n_t, p=p_treat, observed=clicks_t)
    trace = pm.sample(2000)

prob_treat_better = (trace['p_treat'] > trace['p_control']).mean()
```

Companies using Bayesian: VWO, Optimizely. Frequentist (with sequential): Netflix, Microsoft, LinkedIn.

### 4.5 Interleaving — paired comparison cho ranking

Vấn đề với A/B test cho search/reco ranking:
- Cần **huge sample** vì variance cao (click rates ~5-10%, lift ~0.5%).
- Có thể tốn weeks chỉ để decide một ranking change.

**Interleaving solution** (Joachims 2003, Chapelle 2012):
- Mỗi query, **merge results từ ranker A và ranker B** thành single list.
- Track clicks: click vào doc từ A → vote A win, click vào doc từ B → vote B win.
- Power: ~10-100x cao hơn A/B test (cùng sample size).

#### Team Draft Interleaving (TDI)

```
Ranker A:  [a1, a2, a3, a4, a5]
Ranker B:  [b1, b2, b3, b4, b5]

Coin flip: A first
Interleaved: [a1, b1, a2, b2, a3, b3, ...]
(Skip duplicates if overlap)
```

#### Probabilistic Interleaving (PI)

Mỗi vị trí, random chọn từ A hoặc B với probability tỷ lệ rank score.

#### Khi nào dùng interleaving?

- **Khi**: ranking changes (search, reco, ads ranking).
- **Không khi**: UI changes, end-to-end metrics (revenue), policy changes.
- Interleaving đo **ranking quality**, không đo direct business metrics.

Production usage:
- **Yandex**: standard cho ranking experiments.
- **Bing**: complementary cho A/B (interleaving early signal, A/B confirm).
- **Netflix**: cho ranker selection.

### 4.6 Multi-armed bandits (MAB) — adaptive allocation

Khác với A/B test (fixed allocation), MAB **shift traffic** sang variant winning dần dần. Trade-off: explore vs exploit.

#### Thompson Sampling

```python
def thompson_sample(successes, trials, num_variants):
    samples = []
    for i in range(num_variants):
        s, n = successes[i], trials[i]
        # Sample from Beta posterior
        samples.append(np.random.beta(s + 1, n - s + 1))
    return np.argmax(samples)
```

Mỗi request, sample từ posterior của mỗi variant, chọn argmax. Variant đang win → sampled nhiều hơn.

#### Khi nào MAB > A/B test?

- **Headline optimization, ad creative selection**: 100s variants, want fast convergence.
- **Loss-averse situations**: tránh expose users tới poor variant lâu.
- **Short-term decisions** (e.g. cho mỗi user session).

#### Khi nào A/B test > MAB?

- **Long-term effects** (retention, LTV): MAB allocate ít cho variant initially poor → skew long-term measurement.
- **Multiple metrics**: MAB simple version optimize single metric. Multi-metric → A/B test thông thường better.
- **Statistical rigor needed** (regulated industry, scientific publication).

### 4.7 Advanced — Heterogeneous Treatment Effects (HTE)

Average Treatment Effect (ATE) có thể null nhưng có **heterogeneity**: works for some users, not others.

#### Subgroup analysis

```python
for segment in ['new_user', 'power_user', 'mobile', 'desktop']:
    seg_data = data[data['segment'] == segment]
    lift = compute_lift(seg_data)
    print(f"{segment}: lift={lift:.2%}")
```

**Caveat**: multiple testing → spurious. Apply Bonferroni hoặc FDR.

#### Causal forests, Meta-learners

Wager & Athey 2018, Künzel et al. 2019: ML models for personalized treatment effects.

```python
from econml.metalearners import XLearner

learner = XLearner(models=GradientBoostingRegressor())
learner.fit(Y=y, T=treatment, X=user_features)
te_per_user = learner.effect(user_features)
```

Production usage: identify users who benefit most từ treatment → roll out selectively (targeted launch).

---

## 5. Trade-offs & Design decisions

### 5.1 Fixed-horizon vs sequential testing

| Aspect | Fixed horizon | Sequential (mSPRT) |
|---|---|---|
| Sample size calc upfront | Required | Not strict |
| Stop early on clear win | No (would inflate FP) | Yes, valid |
| Implementation complexity | Simple | Higher |
| Best for | Default, statistical purity | Time-sensitive, expensive experiments |

### 5.2 Frequentist vs Bayesian

| Aspect | Frequentist | Bayesian |
|---|---|---|
| Output | p-value, CI | Posterior probability |
| Interpretability cho non-stat audience | Confusing ("95% CI" ≠ 95% chance) | More intuitive ("P(B > A) = 0.92") |
| Prior choice | None | Required, can bias if poor |
| Standard practice | Most big tech | Optimizely, VWO |

### 5.3 Variance reduction techniques

| Technique | Variance reduction | Implementation cost |
|---|---|---|
| CUPED with 1 covariate | 20-50% | Low |
| CUPAC (ML covariate) | 30-60% | Medium |
| Stratification | 10-30% | Low |
| Variance-weighted estimation | 5-15% | Low |
| Trigger-based analysis | Varies | Medium |

**Recommendation**: implement **CUPED as default** trong platform. Auto-apply when pre-period data available.

### 5.4 Common pitfalls

**1. SUTVA violation (network effects)**
- Marketplace: A's price change affects B's purchase decision.
- Social network: A sees ad → tells B about product.
- Solutions: cluster randomization (treat groups together), switch-back design (Uber dùng cho pricing), graph-based randomization.

**2. Novelty / primacy effects**
- New UI: users curious → click more initially → effect fades.
- New feature: existing users primacy effect → reject change.
- Mitigation: run experiment ≥ 14 days, look at trajectory.

**3. Survivorship bias**
- Only count "engaged users" sau treatment → exclude users churned vì treatment.
- Solution: define cohort BEFORE treatment, include everyone.

**4. Twyman's law**
- "Any data that looks interesting is probably wrong."
- Lift > 20% trong A/B test → 95% là bug, không phải miracle.
- Always investigate suspicious wins.

**5. Multiple testing**
- 100 experiments × α=0.05 → expect 5 false positives.
- Need: Bonferroni (conservative) hoặc Benjamini-Hochberg FDR.

**6. Cherry picking**
- Run 5 metrics, only report the one significant → false positive.
- Solution: **pre-register** primary metric. Secondary metrics labeled clearly.

### 5.5 Guardrails

Mỗi experiment cần monitor **guardrail metrics** — không phải để maximize, mà để **không degrade**:

| Guardrail | Typical threshold | Why |
|---|---|---|
| Latency P99 | < +10ms vs control | UX |
| Error rate | < +0.1% vs control | Stability |
| Revenue | > -1% vs control | Business |
| Crash rate (mobile) | < +0.05% | App health |

Nếu guardrail breach → **auto-stop experiment** hoặc page on-call immediately.

---

## 6. Lessons learned & Best practices

### 6.1 Lesson 1: Trust > Speed

Theo Kohavi (Microsoft, then Airbnb): nhiều product teams pressure platform team để show wins. **Resist**. A platform shows reliable losses cũng valuable bằng shows reliable wins. Một false win shipped = nguồn của technical debt + flawed mental model.

**Investment**: dedicate 20-30% platform engineering time vào **detection of bugs in experiments themselves** (SRM, AA tests, trap detection).

### 6.2 Lesson 2: Most experiments fail

Booking.com data (Lukas Vermeer talk, ~2019): chỉ ~10% experiments cải thiện primary metric statistically significantly. Microsoft Bing: ~30%. Default mindset phải là "this probably won't work, prove me wrong".

**Implication**: experimentation culture phải celebrate **learning** from null/negative results, không chỉ wins. Otherwise PMs sẽ p-hack.

### 6.3 Lesson 3: AA tests — sanity check platform

**AA test**: 2 variants identical, expect no significant diff. Run nhiều AA tests:
- Distribution of p-values should be uniform [0, 1].
- ~5% should have p < 0.05 (by definition).

Nếu platform có bug (randomization broken, metrics computed wrong), AA test sẽ phát hiện trước experiments thật.

**Booking.com practice**: continuous AA tests trong background, monitor p-value distribution daily.

### 6.4 Lesson 4: ML model A/B test có nuances

Khi A/B test cho ML model (recommendation, ranking):

**Counterfactual logging**: log không chỉ what user did, mà cả what ranker scored các items NOT shown. Cần cho off-policy evaluation later.

**Holdback group**: maintain 1-5% users always seeing baseline model — measure compounding effect over time.

**Cold start consideration**: model mới chưa có exploration data → kém tự nhiên trong early days. Don't conclude after 1 week.

**Feedback loops**: model rankings → user behavior → training data → next model. Có thể create runaway effects (filter bubbles, popularity bias). Long-term holdback giúp detect.

### 6.5 Lesson 5: Document protocol BEFORE running

For each experiment, pre-register:
- **Primary metric** (THE metric for decision).
- **Secondary metrics** (informational, không decision-critical).
- **Guardrails**.
- **Target population** (segment, geo, platform).
- **Duration / sample size**.
- **Stopping criteria**.

Nếu ad-hoc change protocol mid-experiment → high risk of bias. Even unintentional.

### 6.6 Lesson 6: Pretty dashboards lie. Look at raw data.

Dashboards typically show "lift = +2.3%, p < 0.05" — green check. But:
- Was there SRM?
- Did metric trajectory move monotonically or spike?
- Outliers? (One user with $1M revenue dominate?)
- Segment differences?

**Rule**: before declaring victory, **always plot daily metric per variant**, check distribution, run diagnostics.

### 6.7 Lesson 7: Outliers can dominate

Revenue, watch time có heavy tails. One outlier (bot, whale user, fraud) can shift mean dramatically.

Solutions:
- **Winsorization**: cap top 0.1% to 99.9 percentile.
- **Trimmed mean**: drop top/bottom 1%.
- **Log transformation**: report lift on log(metric).
- **Median + non-parametric tests**.

### 6.8 Lesson 8: Cultural factors matter as much as tech

Companies với strong experimentation culture:
- Senior leadership cite "data from experiment X" trong decisions.
- "I think Y" met với "let's test it".
- Failures discussed openly without blame.
- PMs trained trên statistical literacy.

Companies với weak culture: experiments run for show, decisions still HiPPO-driven (Highest Paid Person's Opinion). Platform tốt không cứu nổi.

### 6.9 Best practice — production checklist

**Platform side**:
- [ ] Deterministic hash-based assignment, salt per experiment.
- [ ] Layered/orthogonal experiment design.
- [ ] SRM detection automatic, alert on breach.
- [ ] CUPED variance reduction enabled by default.
- [ ] Sequential or fixed-horizon clearly chosen per experiment.
- [ ] Guardrail metrics auto-monitored.
- [ ] AA tests running continuously.
- [ ] Multiple testing correction at portfolio level.
- [ ] Holdback group maintained for long-term effects.

**Experiment side**:
- [ ] Primary metric pre-registered.
- [ ] Sample size / duration calculated upfront.
- [ ] Hypothesis stated explicitly.
- [ ] Segment analysis defined upfront (not exploratory).
- [ ] Plot daily trajectory before concluding.
- [ ] Check distribution + outliers.
- [ ] Document decision in writing.

### 6.10 Cross-references đến knowledge base

- ML model deployment leading to A/B test: [S4-01 Michelangelo](S4-01_uber_michelangelo_feature_store.md).
- Ranking model under A/B test: [S1-01 YouTube](S1-01_youtube_recommendation_end_to_end.md), [S2-01 DLRM](S2-01_meta_dlrm_architecture.md).
- Drift detection (complementary monitoring): S4-03 (planned).
- Specific case study: how Netflix uses A/B test for personalization — see S1-04 (planned).

---

## 7. References

### Foundational books & papers

1. **Kohavi, Tang, Xu "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing"** — Cambridge University Press, 2020. **The single most important book**. Read this if you do any A/B testing.

2. **Deng, Xu, Kohavi, Walker "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"** — WSDM 2013. Original CUPED paper.

3. **Tang, Agarwal, O'Brien, Meyer "Overlapping Experiment Infrastructure"** — KDD 2010. Google's layered design.

4. **Johari, Pekelis, Walsh "Always Valid Inference: Bringing Sequential Analysis to A/B Testing"** — arXiv 2017, [1512.04922](https://arxiv.org/abs/1512.04922). mSPRT for sequential testing.

5. **Bakshy, Eckles, Bernstein "Designing and Deploying Online Field Experiments"** — WWW 2014. Facebook's PlanOut.

6. **Chapelle, Joachims, Radlinski, Yue "Large-scale Validation and Analysis of Interleaved Search Evaluation"** — TOIS 2012. Foundational for interleaving.

7. **Wager, Athey "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"** — JASA 2018. Causal forests.

8. **Künzel, Sekhon, Bickel, Yu "Metalearners for Estimating Heterogeneous Treatment Effects"** — PNAS 2019. X-learner, T-learner, S-learner.

### Engineering blog posts

9. **Microsoft ExP team — "ExP Platform"** — multiple posts at [exp-platform.com](https://exp-platform.com). Practical wisdom.

10. **Netflix Tech Blog — "Experimentation Platform"** series: [netflixtechblog.com](https://netflixtechblog.com/?gi=ed18a78ddb86). Look for tags "experimentation", "A/B testing".

11. **Airbnb Engineering — "Designing Experimentation Guardrails"**, "Experimentation Reporting Framework": [medium.com/airbnb-engineering](https://medium.com/airbnb-engineering).

12. **LinkedIn Engineering — "A/B Testing at LinkedIn", "XLNT" series**: [engineering.linkedin.com](https://engineering.linkedin.com).

13. **Booking.com — Lukas Vermeer talks "Big Lessons from Booking.com"**: search on YouTube. Real numbers on experiment success rates.

14. **Uber Engineering — "Under the Hood of Uber's Experimentation Platform"**: [eng.uber.com](https://www.uber.com/blog/engineering/).

15. **Spotify R&D — "Spotify's Experimentation Platform"** series.

16. **Twitch — "Multi-Armed Bandits in the Wild"**.

### Tools / libraries

17. **GrowthBook** — open source experimentation platform: [growthbook.io](https://www.growthbook.io).

18. **Statsig** — managed experimentation: [statsig.com](https://www.statsig.com).

19. **Optimizely** — managed: [optimizely.com](https://www.optimizely.com).

20. **EconML (Microsoft)** — causal inference toolkit (HTE): [github.com/microsoft/EconML](https://github.com/microsoft/EconML).

21. **CausalML (Uber)** — uplift modeling: [github.com/uber/causalml](https://github.com/uber/causalml).

### Talks (must-watch)

22. **Ron Kohavi — multiple keynotes at KDD, WSDM, RecSys**: search "Ron Kohavi A/B testing" YouTube.

23. **Lukas Vermeer — "Big Lessons from Booking.com Experiments"**: classic talk.

24. **Stitch Fix — "Causal Inference in Industry"**: Eric Colson and team.

### Other useful references

25. **Hill "Experiments at Airbnb"** — early Airbnb blog post on experimentation principles.

26. **Drutsa, Gusev, Serdyukov "Practical Aspects of Sensitivity in Online Experimentation with User Engagement Metrics"** — CIKM 2015 (Yandex). Variance reduction for engagement.

27. **Xu, Chen, Kazemi, Geng "Avoid Common Pitfalls in Causal Effect Estimation"** — Snap Inc, KDD 2018.

---

> **Tóm tắt 1 dòng**: Experimentation platform là closure của ML loop — design phải prioritize **trustworthiness** (deterministic assignment + SRM detection + AA tests + guardrails) trước **velocity**, áp dụng **CUPED variance reduction** mặc định, **pre-register protocol**, và treat **most experiments failing** là norm chứ không phải dấu hiệu của broken platform.
