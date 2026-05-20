---
proposed_id: S3-09
proposed_title: Test-Time Compute & Reasoning Model Serving — o1/o3, DeepSeek R1, s1, Parallel Sampling
proposed_scope: 3
proposed_scope_name: modern-stack
proposed_difficulty: advanced
proposed_summary: "Inference-time scaling cho reasoning models: long CoT generation, best-of-N + verifier, majority voting, parallel tree search, budget forcing (s1), serving architecture cho long-output high-variance workload."
proposed_at: 2026-05-20
proposed_by: topic-researcher
status: pending-review
---

# Proposal: Test-Time Compute & Reasoning Model Serving — o1/o3, DeepSeek R1, s1, Parallel Sampling

## Why this topic NOW

OpenAI o1 (Sep 2024) → o3 (early 2025) → DeepSeek R1 (Jan 2025, Nature 2025) → Kimi k1.5, QwQ, Qwen3-Thinking đã định nghĩa lại scaling law: **"compute spent at inference is the new training compute"**. R1 generates **10-100× more tokens per query** than non-reasoning models; Jensen Huang công khai: next-gen reasoning models cần "up to 100× more computational resources" tại inference. Analyst projection: inference sẽ chiếm **75% of total AI compute by 2030** (vs ~50% trước reasoning era).

Vấn đề serving của reasoning models hoàn toàn khác serving của instruction-tuned chat models:
- **Output length distribution heavy-tail**: median ~2k tokens nhưng p99 >32k tokens (DeepSeek-R1 training rollout max_len=32,768). Buffer sizing và preemption policy phải rethink.
- **Latency budget khác**: user chấp nhận chờ 30s-3min cho complex problem, nhưng cần TTFT < 1s để cảm giác "thinking visible".
- **Parallel sampling pattern**: best-of-N với N=4-64 (R1 dùng N=16 trong rollout, AIME experiments dùng N=64); majority voting / consensus cần ensemble logic ở serving layer.
- **Tree search inference**: dynamic parallel tree search (DPTS, ACL 2025), Tree-of-Thoughts orchestration. Khó parallelize trên GPU vì retrospective + recursive nature.
- **Budget forcing**: s1 (Stanford, ICLR 2025) — append "Wait" token để extend thinking, hoặc force-terminate. Cần serving primitive cho controlled stopping.
- **Confidence-aware early exit**: thinking-short-and-right (May 2025) — nhiều CoT trajectories có thể vote sớm khi confident.

Đây là **architecture shift đang định hình production stack 2025-2026**. Một case study deep-dive trả lời: kiến trúc serving thay đổi như thế nào? Best-of-N implementation ở batch level vs request level? Cost-per-correct-answer thay vì cost-per-token? Khi nào parallel scaling > sequential CoT?

## Sources to draw from

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI (Jan 2025, Nature 2025). https://arxiv.org/abs/2501.12948 (Nature version: https://www.nature.com/articles/s41586-025-09422-z)
   - Why useful: full RL training recipe + inference characteristic (max 32k thinking tokens, N=16 rollout, majority voting boost AIME from 71% → 86.7%). Foundational reference cho open reasoning models.
2. **s1: Simple Test-Time Scaling** — Muennighoff et al. (Stanford/UW, Jan 2025, EMNLP 2025). https://arxiv.org/abs/2501.19393
   - Why useful: minimal recipe (1000 examples SFT) + **budget forcing** primitive (append "Wait" to extend, EOS suppression / forced termination); SOTA cho test-time-scaling control. Open code/data/model.
3. **OpenAI o1 / o3 system card + Learning to Reason with LLMs blog** — OpenAI (Sep 2024 / 2025). https://openai.com/index/learning-to-reason-with-llms/
   - Why useful: defining article cho test-time compute paradigm; reveals inference latency budget (10s-many minutes), hidden CoT, cost scaling. Public reference cho proprietary frontier.
4. **Dynamic Parallel Tree Search for Efficient LLM Reasoning** — Ding et al. (Feb 2025, ACL 2025). https://arxiv.org/abs/2502.16235
   - Why useful: parallel exploration của reasoning tree với dynamic priority; addresses GPU-unfriendly retrospective/recursive nature. Concrete serving algorithm cho ToT-style inference.
5. **Thinking Short and Right Over Thinking Long: Serving LLM Reasoning Efficiently and Accurately** — (May 2025). https://arxiv.org/abs/2505.13326
   - Why useful: confidence-based early exit policy; tradeoff giữa correctness và token budget; production-relevant serving knob.
6. **Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters** — Snell et al. (Google DeepMind, Aug 2024). https://arxiv.org/abs/2408.03314
   - Why useful: thiết lập framework "compute-optimal test-time scaling"; quantifies when N samples + verifier > larger model. Foundational scaling law paper.
7. **Generative Verifiers: Reward Modeling as Next-Token Prediction** — Zhang et al. (Google DeepMind, Aug 2024). https://arxiv.org/abs/2408.15240
   - Why useful: trained verifier để rank N candidates; serving implication — cần extra forward pass per candidate. Critical cho understanding cost của best-of-N production.
8. **SGLang v0.4 + Reasoning Optimization Blog** — LMSYS (Dec 2024 - 2025). https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/ + reasoning-specific posts
   - Why useful: production serving framework's native support cho long-output workloads, branching/forking primitives, ensemble inference.
9. **Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures** — DeepSeek (May 2025). https://arxiv.org/html/2505.09343v1
   - Why useful: production hardware reflection từ team đã serve R1 ở scale; throughput/latency trade-off cho reasoning workload.
10. **NVIDIA Dynamo + reasoning inference** — NVIDIA Developer Blog (GTC 2025). https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
    - Why useful: explicitly framed cho reasoning workload; 30× throughput cho DeepSeek R1 trên Blackwell.

## Estimated depth

- Lines target: 1700-1900
- Key mechanisms / diagrams:
  - Sequential CoT vs parallel sampling vs tree search — taxonomy + diagram cho 3 modes.
  - Token-length distribution: histogram cho regular chat vs reasoning model output (heavy-tail @ p99).
  - Best-of-N serving topology: shared prefill (one prompt) → N parallel decodes → verifier scoring → selection. Cost model: prefill_cost + N × decode_cost + N × verifier_cost.
  - Majority voting vs best-of-N với verifier vs weighted voting — accuracy/cost Pareto.
  - Budget forcing state machine (s1): default decode → on EOS attempt at iteration i < budget, suppress EOS + append "Wait" → continue → at budget, force EOS.
  - Tree-of-Thoughts serving challenge: GPU batch alignment khi branches có different depths; DPTS solution.
  - Cost-per-correct-answer formula: `cost_per_token × tokens_per_attempt × attempts_per_correct`. So sánh GPT-4 vs o1 vs R1 ở fixed accuracy target.
  - Confidence-based early exit: per-step entropy thresholding để cut short.
  - Verifier architecture: discriminative (PRM, ORM) vs generative verifiers — serving cost difference.
- R&D evolution thread: chain-of-thought (Wei et al., 2022) → self-consistency majority voting (Wang et al., 2022) → Tree-of-Thoughts (Yao, 2023) → STaR / self-improve (2023) → o1 (Sep 2024, hidden CoT) → Snell scaling-law (Aug 2024) → DeepSeek-R1 (Jan 2025, open weights + recipe) → s1 budget forcing (Jan 2025) → DPTS parallel tree search (Feb 2025) → confidence-aware early exit (May 2025).

## Cross-references to existing studies

- Should link FROM: **S3-01** (vLLM PagedAttention) — reasoning workload's long output stresses KV cache mgmt; preemption + recompute policy critical. S3-09 extends to ensemble/branching serving.
- Should link FROM: **S3-02** (Production RAG) — RAG + reasoning (e.g. agentic search-then-reason) is canonical hybrid workload; eval methodology cho RAG cần update khi LLM is reasoning model.
- Should link FROM: **S3-04** (Agent Framework, planned) — agents use reasoning models as core engine; serving constraint propagates lên agent loop design.
- Should link TO: **S2-04** (LLM Pretraining at Scale, planned) — reasoning model training (RL stage) shares infra với inference (rollout = inference at training time); cross-link cho training-inference duality.
- Should link TO: **S2-05** (Finetune vs RAG vs Prompt, planned) — decision framework cần include: "khi nào dùng reasoning model" vs "khi nào finetune + thường chat model".
- Should link TO: **S3-06** (Speculative Decoding — proposal) — speculative decoding particularly helps reasoning models (long decode), nhưng acceptance rate có pattern khác (CoT có high local entropy).
- Should link TO: **S4-04** (GPU Cluster Management, planned) — reasoning workload changes GPU pooling + autoscaling significantly (long tail outputs need preemption-aware scheduling).

## Risks / open questions

- **Public info coverage:** Excellent. R1 (Nature 2025), s1 (EMNLP 2025), Snell (peer-reviewed), o1 system card. ≥4 peer-reviewed papers, ≥3 production blogs. All ≤24 months.
- **Internal vs public details:** OpenAI o1/o3 internal serving infra is opaque — sẽ infer behavior từ system card + cost. Anthropic's "extended thinking" Claude 3.7/4.x analogous nhưng undisclosed. DeepSeek + s1 + Qwen-Thinking sufficient cho concrete numbers.
- **Controversial decisions to highlight:**
  - **Best-of-N vs MCTS vs simple long CoT**: R1 paper explicitly rejects MCTS at inference ("exponential search space, token vocabulary too large"). When is each better?
  - **Verifier quality bottleneck**: best-of-N caps at verifier accuracy. When does a weak verifier hurt more than help?
  - **Budget forcing extrapolation**: s1 paper shows "Wait"-extending boosts AIME 50%→57%, but at what point does it plateau or hurt? Length-bias trade-off.
  - **Cost-per-correct-answer as the right metric**: many production teams still optimize cost-per-token. When does this misalign?
  - **Sequential CoT vs parallel scaling Pareto**: at fixed total compute budget B, allocate to longer single trace vs N parallel traces? Snell paper claims problem-dependent.
  - **Privacy + hidden CoT**: OpenAI hides CoT trace; users get only final answer. Open models expose full trace → debugging vs IP trade-off.
  - **Streaming UX for reasoning**: users watch thinking trace token-by-token (like ChatGPT o1) — does this trigger different latency requirements vs batched final answer?
- **Source quality:** Reasoning serving literature is fast-moving (multiple papers per month). Some 2025-Q2/Q3 candidates còn ở preprint stage — sẽ check publication status khi drafting.

## Recommendation

PROPOSE — đây là **defining shift của 2025** mà current scope-3 không cover. Distinct với S3-06 (speculative decoding: same answer, faster) — S3-09 là different answer quality, more compute. 10 sources, mix peer-reviewed papers (R1 ở Nature, s1 ở EMNLP, Snell, Zhang) + frontier system cards (o1) + production blogs (LMSYS, NVIDIA). Có evolution thread sạch, controversial design decisions phong phú, và direct production relevance (every team building agents needs to decide: reasoning model hay không?).
