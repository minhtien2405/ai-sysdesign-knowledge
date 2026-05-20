---
proposed_id: S3-07
proposed_title: Disaggregated Prefill-Decode LLM Serving — DistServe, Splitwise, Mooncake, Dynamo
proposed_scope: 3
proposed_scope_name: modern-stack
proposed_difficulty: advanced
proposed_summary: "Tách prefill và decode lên different GPU pools để tối ưu goodput dưới SLO; KV-cache transfer engine, scheduler, production deployment (Kimi, NVIDIA Dynamo, llm-d)."
proposed_at: 2026-05-20
proposed_by: topic-researcher
status: pending-review
---

# Proposal: Disaggregated Prefill-Decode LLM Serving — DistServe, Splitwise, Mooncake, Dynamo

## Why this topic NOW

Năm 2025 là năm disaggregated serving chuyển từ research paper thành **default production architecture**. Quote từ Hao AI Lab retrospective (Mar 2026): *"Almost every production-grade LLM serving framework — NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, MoonCake — runs on disaggregation."* Mooncake powers Kimi của Moonshot AI ở scale **>100 tỷ tokens/ngày**, deploy Kimi K2 trên 128 H200 GPUs với prefill throughput 224k tok/s, decode throughput 288k tok/s. NVIDIA Dynamo công bố tại GTC 2025 đạt **30× throughput boost** trên DeepSeek-R1 với Blackwell. Red Hat khởi xướng llm-d community tại Red Hat Summit 2025.

Cốt lõi insight: prefill là compute-bound (parallel attention over context), decode là memory-bound (sequential, KV-cache I/O dominant). Chạy chung trên 1 GPU pool gây **prefill-decode interference**, làm vỡ cả TTFT và TPOT SLO. Disaggregate ra 2 pools (có thể heterogeneous hardware như Splitwise: prefill trên H100, decode trên A100) cho phép co-optimize parallelism độc lập, đạt **up to 7× higher request rate** dưới cùng SLO (DistServe original). Đây là kiến trúc shift lớn nhất trong LLM serving từ continuous batching.

## Sources to draw from

1. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** — Zhong et al. (UCSD/PKU, OSDI 2024). https://arxiv.org/abs/2401.09670
   - Why useful: foundational paper; định nghĩa TTFT vs TPOT vs goodput; simulator-driven search cho parallelism + placement; 7× request rate baseline.
2. **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — Patel et al. (Microsoft Azure Research/ISCA 2024). https://www.researchgate.net/publication/382806162_Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting
   - Why useful: heterogeneous hardware angle — prefill trên H100, decode trên A100; energy efficiency optimization; complement DistServe.
3. **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — Qin et al. (Moonshot AI, FAST 2025 / ACM TOS 2025). https://arxiv.org/abs/2407.00079
   - Why useful: production system serving Kimi; KVCache pool tận dụng CPU+DRAM+SSD+RDMA của GPU cluster; Transfer Engine đạt 87 GB/s (4×200 Gbps RoCE) đến 190 GB/s (8×400 Gbps); 75% more requests vs baseline.
4. **NVIDIA Dynamo, A Low-Latency Distributed Inference Framework** — NVIDIA Developer Blog (GTC 2025). https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
   - Why useful: production-grade open-source framework; LLM-aware routing, dynamic GPU scheduling, NIXL accelerated data transfer; 30× DeepSeek-R1 throughput trên Blackwell.
5. **Disaggregated Inference: 18 Months Later** — Hao AI Lab @ UCSD retrospective (2026). https://haoailab.com/blogs/distserve-retro/
   - Why useful: tổng kết các production deployments; what worked, what didn't; comparison table giữa Dynamo / llm-d / Mooncake / vLLM Disagg / SGLang.
6. **Throughput is Not All You Need: Maximizing Goodput in LLM Serving** — Hao AI Lab blog (DistServe explainer). https://haoailab.com/blogs/distserve/
   - Why useful: pedagogical explanation của goodput vs throughput; visual diagrams cho prefill-decode interference.
7. **Deploying Kimi K2 with PD Disaggregation and Large-Scale Expert Parallelism on 128 H200 GPUs** — LMSYS Blog (Jul 2025). https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep/
   - Why useful: real-world deployment ở scale; tương tác giữa PD disaggregation và MoE expert parallelism (relevant cho DeepSeek-V3, Kimi K2).
8. **Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving (TaiChi)** — Wang et al. (Aug 2025). https://arxiv.org/abs/2508.01989
   - Why useful: phản biện — aggregation tốt hơn khi tight TTFT + relaxed TPOT; disaggregation excel khi strict TPOT. TaiChi hybrid approach.

## Estimated depth

- Lines target: 1800-2000
- Key mechanisms / diagrams:
  - Prefill vs decode characterization (compute-bound vs memory-bound). FLOPs/byte ratio.
  - Interference diagram: tại sao 1 prefill request kéo dài TPOT của 50 decode requests đang chạy continuous batching.
  - Disaggregated architecture: prefill pool ↔ KV cache transfer ↔ decode pool.
  - Mooncake Transfer Engine: RDMA topology, multi-NIC striping, 87→190 GB/s scaling.
  - KV cache pool tier hierarchy (HBM → DRAM → NVMe SSD → remote object store) — Mooncake's Disaggregated KVCache Pool.
  - Goodput vs throughput vs latency relationship; per-GPU goodput formula.
  - Dynamo scheduling diagram: LLM-aware request routing dựa trên prefix hash cho cache hit.
- R&D evolution thread: continuous batching trên 1 pool (Orca/vLLM 2022-2023) → DistServe disaggregate (Jan 2024) → Splitwise heterogeneous (2024) → Mooncake KVCache pool (Jul 2024) → Dynamo + llm-d ecosystem (2025) → TaiChi hybrid PD (Aug 2025).

## Cross-references to existing studies

- Should link FROM: **S3-01** (vLLM PagedAttention) — disaggregation là next-generation architecture sau continuous batching; S3-01 explains prefill-decode mechanics ở level GPU memory, S3-07 sẽ extend lên cluster level.
- Should link FROM: **S3-02** (Production RAG) — RAG workload có rất nhiều prefill (long context retrieval) + ít decode; chính là sweet spot của disaggregation.
- Should link TO: **S4-04** (GPU Cluster Management for LLM Inference, planned) — disaggregated serving thay đổi GPU pooling strategy; multi-tenant complexity tăng.
- Should link TO: **S3-06** (Speculative Decoding, đề xuất ở proposal khác) — speculative decoding chạy ở decode pool; tương tác cần document.

## Risks / open questions

- **Public info coverage:** Excellent — 5 papers, 2 production engineering blogs (Mooncake/Kimi, LMSYS), 1 retrospective. Tất cả ≤24 tháng.
- **Internal vs public details:** Mooncake công bố rất chi tiết (Kimi production numbers). OpenAI/Anthropic không publish chi tiết nhưng có thể infer từ open-source equivalent (Dynamo, llm-d).
- **Controversial decisions to highlight:**
  - PD disagg always better? TaiChi paper argues NO — aggregation thắng ở vài SLO regime. Khi nào pick which?
  - Network bandwidth bottleneck: 200 Gbps RoCE đủ cho prefill→decode KV cache transfer ở scale 70B+ model? Mooncake shows multi-NIC needed.
  - Tương tác với MoE: Kimi K2 deployment combine PD disagg + expert parallelism — orchestration complexity?
  - Cold cache penalty khi KV cache evict khỏi HBM xuống DRAM — Mooncake's tiered approach có overhead gì?
- **Source citation:** TaiChi paper publication venue chưa confirm; cần verify ở stage drafting.

## Recommendation

PROPOSE — đây là **highest-priority gap** trong scope-3. S3-01 dừng ở mức single-GPU/single-node optimization; S3-07 sẽ là cluster-level architecture story với production deployment numbers from Kimi (>100B tokens/day). 8 sources, đủ depth cho 1800-2000 dòng. Có thể là case study scope-3 đắt giá nhất sau S3-01.
