---
proposed_id: S3-08
proposed_title: Long-Context LLM Serving — Ring Attention, Context Parallelism, Prefix Caching, Sparse Attention
proposed_scope: 3
proposed_scope_name: modern-stack
proposed_difficulty: advanced
proposed_summary: "Kiến trúc serving cho 1M–10M context: ring/striped attention, context parallelism, prefix/radix cache sharing, attention sinks (StreamingLLM), query-aware sparse attention (Quest/LServe), hierarchical KV cache (Strata)."
proposed_at: 2026-05-20
proposed_by: topic-researcher
status: pending-review
---

# Proposal: Long-Context LLM Serving — Ring Attention, Context Parallelism, Prefix Caching, Sparse Attention

## Why this topic NOW

2025 là năm context window bùng nổ ra commercial scale: **Gemini 2.5 Pro (2M tokens), Claude Sonnet 4 (1M), GPT-5 family, Qwen2.5-1M, Llama 4 Scout (10M)** — tất cả ship trong 12 tháng qua. Đồng thời các production workload thực tế (codebase indexing, long-document analysis, multi-hour agent conversations) đẩy median prompt length lên >30k tokens. Vấn đề là: KV cache cho 1M tokens trên Llama-3-8B FP16 cần **~137 GB** — vượt một H100 80GB. Prefill 1M tokens trên Llama 3 405B mất **~77 giây** trên 128 H100 với context parallelism (Meta, Nov 2024); 2 phút cho max context. Single-GPU paged attention (S3-01) đã không còn đủ — cần kiến trúc serving riêng cho long context.

Trong 18 tháng qua, một họ kỹ thuật mới đã consolidate thành **standard long-context stack**: (i) Ring Attention / Context Parallelism cho prefill phân tán; (ii) Prefix/Radix caching cho shared system prompt + RAG documents (SGLang ở 400k+ GPU production); (iii) Hierarchical KV cache tier (HBM → DRAM → NVMe — Strata, Mooncake); (iv) Query-aware sparse attention cho decode (Quest, LServe, DuoAttention — accepted MLSys 2025 / ICLR 2025); (v) Attention sinks (StreamingLLM, ICLR 2024) cho infinite-length streaming. Đây là điểm hợp lý để document toàn bộ R&D thread như một case study riêng — distinct với S3-01 (single-node KV cache mechanics) và S3-07 (cluster-level prefill/decode disaggregation).

## Sources to draw from

1. **Ring Attention with Blockwise Transformers for Near-Infinite Context** — Liu, Zaharia, Abbeel (UC Berkeley, 2023, ICLR 2024). https://arxiv.org/abs/2310.01889
   - Why useful: foundational paper định nghĩa ring topology cho phân tán attention; overlapping SendRecv với compute; foundation cho mọi context parallelism implementation sau này.
2. **Context Parallelism for Scalable Million-Token Inference** — Meta AI (Jianyu Huang et al., Nov 2024). https://arxiv.org/abs/2411.01783
   - Why useful: production-grade implementation; pass-KV và pass-Q ring attention variants; **1M context prefill cho Llama 3 405B trong 77s trên 128 H100, 93% parallel efficiency, 63% FLOPS utilization**. Benchmark trên RDMA + TCP — generalizable infra.
3. **Efficient Streaming Language Models with Attention Sinks (StreamingLLM)** — Xiao, Tian, Chen, Han, Lewis (MIT/Meta/CMU, ICLR 2024). https://arxiv.org/abs/2309.17453
   - Why useful: phát hiện hiện tượng "attention sink"; enable infinite-length streaming với window attention + sink tokens; 22.2× speedup so với sliding window recomputation. Foundational concept áp dụng vào mọi long-context serving system.
4. **Fast and Expressive LLM Inference with RadixAttention and SGLang** — Zheng et al. (UC Berkeley/LMSYS, Jan 2024) + SGLang v0.4 update (Dec 2024). https://www.lmsys.org/blog/2024-01-17-sglang/ + https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
   - Why useful: radix tree-indexed KV cache cho automatic prefix sharing; **400,000+ GPU production deployment** (xAI, NVIDIA, AMD, LinkedIn); 75-95% cache hit rate ở workload có ≥60% prefix overlap; 1.9× throughput + 3.8× hit rate improvement với cache-aware load balancer.
5. **Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference** — Tang et al. (MIT, NeurIPS 2024). https://arxiv.org/abs/2406.10774
   - Why useful: query-aware page selection cho KV cache; lưu min/max key per page → estimate page criticality; top-K page loading → **2.23× attention speedup, 7.03× latency reduction** với negligible accuracy loss. Standard reference cho decode-time sparse attention.
6. **LServe: Efficient Long-Sequence LLM Serving with Unified Sparse Attention** — Yang et al. (MIT/NVIDIA/SJTU, MLSys 2025). https://arxiv.org/abs/2502.14866
   - Why useful: thống nhất prefill + decode sparsity vào one framework; block-wise skip cho less important tokens; **2.9× prefill, 1.3-2.1× decode speedup vs vLLM**; evaluated trên LongBench + Needle-in-a-Haystack.
7. **DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads** — Xiao et al. (MIT, ICLR 2025). https://arxiv.org/abs/2410.10819
   - Why useful: phân loại attention heads thành "retrieval heads" (cần full context) vs "streaming heads" (chỉ cần local window) — chỉ giữ full KV cache cho retrieval heads → significant memory saving without accuracy loss.
8. **Strata: Hierarchical Context Caching for Long Context Language Model Serving** — (Aug 2025). https://arxiv.org/abs/2508.18572
   - Why useful: multi-tier KV cache (HBM/DRAM/NVMe); offloading + prefetch policy cho long-context multi-tenant. Complement Mooncake's Transfer Engine.
9. **Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism** — (Oct 2025). https://arxiv.org/html/2510.17896v1
   - Why useful: comprehensive comparison giữa Ring, USP, Ulysses, hybrid SP approaches; kernel-level numbers cho FlashAttention-3, kernel + distributed combined.

## Estimated depth

- Lines target: 1700-1900
- Key mechanisms / diagrams:
  - Long-context cost model: KV cache size = `2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes`. Cụ thể cho Llama 3 8B (137 GB @ 1M FP16), 70B, 405B.
  - Ring attention block diagram: each device holds 1 chunk of Q/K/V; KV blocks rotate around ring; overlap with compute. Diff giữa pass-KV vs pass-Q variants.
  - Context parallelism vs sequence parallelism vs tensor parallelism — when to combine which.
  - RadixAttention diagram: radix tree với token sequence là edge, KV cache pointer là leaf; LRU eviction; cache-aware routing decision.
  - StreamingLLM attention sink visualization: attention score heatmap trước/sau khi giữ first 4 tokens + sliding window.
  - Quest page selection: per-page (min_key, max_key) bounding box; query attention score upper bound = max(q·min_k, q·max_k); top-K selection.
  - DuoAttention head classification: training-time identify which heads need full vs local context; runtime cost.
  - Hierarchical KV cache tier latency budget: HBM ~1μs, DRAM ~100μs, NVMe ~100-500μs, RDMA remote ~10μs.
- R&D evolution thread: FlashAttention (single-GPU memory-efficient kernel, 2022) → Ring Attention (distributed, 2023) → StreamingLLM (infinite length via sink, 2023) → Context Parallelism @ Meta (1M production, Nov 2024) → SGLang RadixAttention (prefix sharing, Jan 2024) → Quest (query-aware sparsity, Jun 2024) → DuoAttention (head specialization, Oct 2024) → LServe (unified sparse, Feb 2025) → Strata (hierarchical, Aug 2025).

## Cross-references to existing studies

- Should link FROM: **S3-01** (vLLM PagedAttention) — S3-01 covers single-GPU KV management; S3-08 extends lên distributed + multi-million-token regime với new techniques.
- Should link FROM: **S3-02** (Production RAG) — RAG là canonical long-context use case (retrieved docs concatenated); prefix caching đặc biệt impactful cho RAG.
- Should link FROM: **S3-07** (Disaggregated Prefill-Decode — proposal) — disaggregation và context parallelism combine ở Mooncake/Kimi production; reference S3-08 cho prefill-side context parallelism details.
- Should link TO: **S2-04** (LLM Pretraining at Scale, planned) — long-context training uses similar Ring/Ulysses SP techniques; cross-link cho training-vs-inference SP comparison.
- Should link TO: **S3-06** (Speculative Decoding — proposal) — long context + speculative decoding interaction (verify cost scales với prefix length).
- Should link TO: **S4-04** (GPU Cluster Management for LLM Inference, planned) — long-context workload changes GPU pooling / SLO targets dramatically.

## Risks / open questions

- **Public info coverage:** Excellent — 4 peer-reviewed papers (ICLR 2024, NeurIPS 2024, MLSys 2025, ICLR 2025), 2 production engineering blogs (LMSYS, Meta), 2 recent arXiv preprints (2025). All ≤24 months except foundational Ring Attention (Oct 2023) and StreamingLLM (Sep 2023) which are tier-1 prerequisites.
- **Internal vs public details:** Google (Gemini 2M), Anthropic (Claude 1M), OpenAI (GPT-5) không publish chi tiết long-context infra. Sẽ infer từ open implementations (Meta CP, SGLang, vLLM) — clearly flag inference vs known facts.
- **Controversial decisions to highlight:**
  - Approximate sparse attention (Quest, LServe) vs exact context parallelism (Meta CP) — accuracy/latency Pareto: when "good enough" is good enough?
  - Prefix caching benefit collapses ở high-cardinality workload (mỗi user prompt unique) — cách đo cache-hit floor.
  - DuoAttention's "retrieval head" identification: model-specific, có generalize không? Training-time finetune required.
  - StreamingLLM's attention sink: cần re-pretraining với placeholder sink token để optimal — feasible trong production?
  - Ring topology nhạy cảm với straggler/network jitter — recovery strategy?
  - 10M context Llama 4 Scout: dùng hybrid attention (interleave full + sliding window) — bài học thiết kế gì?
- **Source citation:** Strata (2508.18572) là preprint Aug 2025 — chưa peer-reviewed, cần verify khi drafting.

## Recommendation

PROPOSE — đây là **defining infrastructure topic của 2025** trong long-context serving. Gap rõ ràng giữa S3-01 (single-GPU KV) và S3-07 (cluster PD disagg). 9 sources (4 peer-reviewed, 5 papers + blogs), tổng cộng dư cho 1700-1900 dòng. Có evolution thread sạch (Ring → CP → sparse → hierarchical), production data point đầy đủ (Meta 128 H100, SGLang 400k GPU), và controversial trade-offs đáng phân tích.
