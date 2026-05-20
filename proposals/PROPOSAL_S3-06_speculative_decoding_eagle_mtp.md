---
proposed_id: S3-06
proposed_title: Speculative Decoding in Production — EAGLE-3, Medusa, DeepSeek MTP
proposed_scope: 3
proposed_scope_name: modern-stack
proposed_difficulty: advanced
proposed_summary: "Draft-and-verify speculative decoding (EAGLE-3, Medusa, MTP) cho LLM inference: tree attention, acceptance rate, training pipeline, integration với vLLM/SGLang/TensorRT-LLM."
proposed_at: 2026-05-20
proposed_by: topic-researcher
status: pending-review
---

# Proposal: Speculative Decoding in Production — EAGLE-3, Medusa, DeepSeek MTP

## Why this topic NOW

Trong 12-18 tháng qua, speculative decoding đã chuyển từ research curiosity thành **standard optimization layer** cho mọi production LLM serving stack. EAGLE-3 (Mar 2025, NeurIPS 2025) đạt 3.0-6.5× speedup; DeepSeek-V3's Multi-Token Prediction (MTP) module — released Dec 2024 — đẩy throughput của một 671B MoE model lên 1.8× ở production và đến 60% higher output throughput trong serving frameworks gần đây. NVIDIA TensorRT-LLM ghi nhận **up to 3.6× throughput boost**; Baseten đo được **94-122% tokens/sec improvement** với Medusa trên Llama 3. Khi vLLM, SGLang, TensorRT-LLM đều ship speculative decoding as default path và SpecForge (Jul 2025) trở thành open-source training framework cho EAGLE-3 draft models, đây là điểm hợp lý để document toàn bộ R&D thread.

Crucially, đây là kỹ thuật **không tăng cost mà giảm latency** — economics rất khác quantization (quality trade-off) hoặc disaggregation (infra complexity). Một case study chuyên sâu sẽ trả lời: khi nào nên dùng EAGLE vs Medusa vs MTP? acceptance rate quyết định gì? làm sao train draft model phù hợp với target model?

## Sources to draw from

1. **EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test** — Li, Wei et al. (Mar 2025, NeurIPS 2025). https://arxiv.org/abs/2503.01840
   - Why useful: định nghĩa state-of-the-art draft model architecture; direct token prediction + multi-layer feature fusion (early/middle/late layer); training-time test technique. Báo cáo 3.0-6.5× speedup trên LLaMA, Vicuna, Qwen.
2. **DeepSeek-V3 Technical Report** — DeepSeek-AI (Dec 2024). https://arxiv.org/html/2412.19437v1
   - Why useful: MTP module thiết kế dual-purpose (training auxiliary loss + inference speculative head); ≥80% acceptance rate cho MTP1; 1.8× generation throughput trong production. Foundation cho hiểu MTP vs EAGLE difference.
3. **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** — Cai et al. (Princeton/Together, 2024). https://github.com/FasterDecoding/Medusa
   - Why useful: tree-based attention mechanism, multiple "Medusa heads"; reference architecture cho framework đầu tiên đi vào TensorRT-LLM/vLLM mainline.
4. **SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding** — LMSYS team (Jul 2025). https://arxiv.org/abs/2603.18567 (note: arXiv id likely mis-listed; canonical source https://www.lmsys.org/blog/2025-07-25-spec-forge/)
   - Why useful: target-draft decoupling, hybrid parallelism, optimized training kernels; up to 9.9× faster EAGLE-3 training trên Qwen3-235B-A22B. SpecBundle catalog of production-grade draft models.
5. **TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6×** — NVIDIA Developer Blog (2024-2025). https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/
   - Why useful: production-engineering perspective; benchmark numbers, integration patterns, draft model selection guidelines.
6. **How to double tokens per second for Llama 3 with Medusa** — Baseten Engineering Blog (2024). https://www.baseten.co/blog/how-to-double-tokens-per-second-for-llama-3-with-medusa/
   - Why useful: real production deployment numbers (94-122% tokens/sec lift); MMLU validation methodology; gotchas khi finetune draft heads.
7. **Accelerating SGLang with Multiple Token Prediction** — LMSYS Blog (Jul 2025). https://www.lmsys.org/blog/2025-07-17-mtp/
   - Why useful: production integration của DeepSeek MTP vào SGLang; benchmark + tuning advice.

## Estimated depth

- Lines target: 1700-1900
- Key mechanisms / diagrams:
  - Sequential decoding baseline (memory-bound vs compute-bound). Tại sao decode stage chỉ dùng <5% GPU FLOPs.
  - Draft-and-verify loop: draft model proposes k tokens → target model verifies in one parallel forward pass → accept longest-matching prefix.
  - Tree-based attention (Medusa, EAGLE): multiple candidate paths verified simultaneously via a verification mask.
  - EAGLE-3 multi-layer feature fusion diagram: early (syntax) + middle (semantics) + late (probability) features feeding draft model.
  - MTP head architecture trong DeepSeek-V3: shared embedding + per-position transformer block + output head.
  - Acceptance rate vs draft model size trade-off curve.
- R&D evolution thread: speculative sampling (Leviathan/Chen 2023) → Medusa (multiple heads, no separate model) → EAGLE-1/2 (feature-level draft, dynamic tree) → EAGLE-3 (training-time test) → MTP (built-in từ pretraining) → SpecForge (production training infra).

## Cross-references to existing studies

- Should link FROM: **S3-01** (vLLM PagedAttention) — vLLM ship continuous batching + speculative decoding as the two main throughput levers; new study sẽ explain phần speculative.
- Should link FROM: **S3-02** (Production RAG) — RAG đặc biệt benefit từ speculative decoding khi context dài và output ngắn.
- Should link TO: **S2-04** (LLM Pretraining at Scale, planned) — MTP module được embed vào pretraining objective; có mối liên hệ chặt với pretraining recipe.
- Should link TO: **S4-04** (GPU Cluster Management for LLM Inference, planned) — speculative decoding tương tác phức tạp với batching + quantization; cost-per-token impact đáng kể.

## Risks / open questions

- **Public info coverage:** Rất đầy đủ — papers, NVIDIA blogs, LMSYS/Baseten/HuggingFace tutorials đều public. Risk hallucination thấp.
- **Internal vs public details:** OpenAI/Anthropic/Google không publish exact speculative configs; sẽ tập trung vào open models (DeepSeek-V3, LLaMA, Qwen) + open serving stacks (vLLM/SGLang/TensorRT-LLM).
- **Controversial decisions to highlight:**
  - EAGLE vs Medusa vs MTP — production teams thực sự pick which? Trade-off training cost vs serving simplicity.
  - Acceptance rate distribution có heavy-tail không? (long-tail queries bị hurt vì verify cost > gain.)
  - Tương tác giữa speculative decoding và FP8/INT4 quantization — có bù trừ hoặc xung đột không?
  - Batch size > 32 thì speculative decoding mất lợi thế (compute-bound rồi) — khi nào tắt?

## Recommendation

PROPOSE — đây là gap rõ nhất trong scope-3 hiện tại. S3-01 vLLM nhắc đến speculative decoding nhưng không deep-dive; toàn bộ R&D thread Medusa → EAGLE-3 → MTP chưa được document. Tất cả 7 sources đều ≤24 tháng, ≥3 trong số đó là peer-reviewed papers hoặc tier-1 industry blogs.
