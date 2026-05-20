---
proposed_id: S3-10
proposed_title: Structured Output & Constrained Decoding — XGrammar, llguidance, Outlines, JSON Schema in Production
proposed_scope: 3
proposed_scope_name: modern-stack
proposed_difficulty: intermediate
proposed_summary: "Grammar-guided generation cho LLM: FSM/CFG-based logit masking, XGrammar (vLLM/SGLang/TRT-LLM default), llguidance (OpenAI Structured Outputs backend), compressed FSM, function calling reliability, latency overhead engineering."
proposed_at: 2026-05-20
proposed_by: topic-researcher
status: pending-review
---

# Proposal: Structured Output & Constrained Decoding — XGrammar, llguidance, Outlines, JSON Schema in Production

## Why this topic NOW

Mọi production LLM application cần structured output — function calling cho agents, JSON cho API integration, code generation tuân thủ syntax. Trước Q4 2024, JSON reliability ở mức 75-85% via prompt-based methods; sau khi constrained decoding ship vào production frameworks, **failure rate giảm xuống <0.1%** (OpenAI Structured Outputs strict mode, May 2025).

12 tháng qua chứng kiến **consolidation cực nhanh** của open-source constrained decoding stack:
- **XGrammar** (CMU/MLC, Nov 2024) — default backend cho **vLLM (Dec 2024), SGLang (Nov 2024), TensorRT-LLM (Jan 2025), MLC-LLM (Nov 2024)**. Đạt **<40 microseconds/token, gần-zero overhead** cho JSON.
- **llguidance** (Microsoft Research, 2023-2025) — Rust-based Earley parser, ~50μs/token. **OpenAI Structured Outputs chuyển sang llguidance từ May 2025** — OpenAI public credit on the underlying tech.
- **Outlines** (.txt, 2023-2024) — first widely-adopted FSM-based library; định nghĩa technique cho Compressed FSM (LMSYS, Feb 2024) đạt **2× latency reduction, 2.5× throughput**.
- **Anthropic Claude tool use, OpenAI Structured Outputs, Gemini structured generation** — all production APIs ship strict-mode schema enforcement.

Tại sao đây là một case study riêng dù có vẻ "narrow"? Bởi vì:
1. **Universal substrate**: mọi agent (S3-04), mọi RAG-with-extraction (S3-02), mọi function-calling production app đều phụ thuộc nó.
2. **Performance engineering deep**: bitmask precompute, context-independent vs context-dependent token split, FSM compression, kernel fusion với logit_processor — tất cả non-trivial.
3. **Subtle correctness vs latency trade-offs**: strict mode có thể hurt quality (model bị ép vào unreachable states); cần engineering knob để balance.
4. **Tier-1 production data point đầy đủ**: OpenAI public credit llguidance, Anthropic discusses tool use bias, vLLM/SGLang benchmark results — đủ cho case study chi tiết.

## Sources to draw from

1. **XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models** — Dong, Xu, Liu, Chen (CMU/MLC, Nov 2024, MLSys 2025). https://arxiv.org/abs/2411.15100
   - Why useful: foundational paper; core insight về **context-independent (99% of vocab) vs context-dependent (1%)** token split — precompute bitmask tables cho 99% case → **up to 100× speedup** so với traditional grammar-guided. Default backend cho 4 major serving frameworks. Open source: https://github.com/mlc-ai/xgrammar.
2. **Efficient Guided Generation for Large Language Models (Outlines)** — Willard, Louf (.txt, Jul 2023). https://arxiv.org/abs/2307.09702
   - Why useful: foundational FSM-of-regex technique; O(1) average per-step lookup; defined paradigm cho FSM-based logit masking. Reference cho mọi work sau này.
3. **Fast JSON Decoding for Local LLMs with Compressed Finite State Machine** — LMSYS Blog (Feb 2024). https://www.lmsys.org/blog/2024-02-05-compressed-fsm/
   - Why useful: compressed FSM — collapse singular transition paths để decode multiple tokens per step; **2× latency reduction, 2.5× throughput** so với guidance+llama.cpp và outlines+vLLM. Production-relevant benchmark.
4. **LLGuidance: Super-Fast Structured Outputs** — Microsoft Research (guidance-ai org, 2023-2025). https://github.com/guidance-ai/llguidance + https://guidance-ai.github.io/llguidance/llg-go-brrr
   - Why useful: Rust-based Earley parser cho CFG; ~50μs/token với negligible startup. **OpenAI Structured Outputs backend từ May 2025** — credited publicly. Sister project llgtrt cho TensorRT-LLM.
5. **Generating Structured Outputs from Language Models: Benchmark and Studies** — (Jan 2025). https://arxiv.org/abs/2501.10868
   - Why useful: comprehensive benchmark giữa XGrammar, llguidance, Outlines, guidance ở dimension: schema coverage, latency, accuracy. Reference cho framework selection.
6. **Guided Decoding Performance on vLLM and SGLang** — SqueezeBits Tech Blog (2025). https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang
   - Why useful: production-grade benchmark, integration patterns trong vLLM vs SGLang; gotchas, kernel-level numbers. Engineering perspective.
7. **OpenAI Structured Outputs Announcement + GPT-4o + GPT-5 docs** — OpenAI (Aug 2024 + May 2025 backend swap). https://openai.com/index/introducing-structured-outputs-in-the-api/
   - Why useful: production frontier API; strict mode + schema preprocessing; <0.1% failure rate claim; constraints (schema subset, recursion depth).
8. **Guided Decoding and Its Critical Role in Retrieval-Augmented Generation** — (Sep 2025). https://arxiv.org/abs/2509.06631
   - Why useful: cross-link với S3-02 (RAG) — structured extraction trong RAG pipeline; ablation showing quality lift.
9. **Anthropic Tool Use Documentation + Behavior Notes** — Anthropic Docs. https://docs.anthropic.com/en/docs/build-with-claude/tool-use
   - Why useful: Anthropic's approach (model-trained tool use, no hard logit constraint by default); contrast với OpenAI strict mode. Discussion of trade-offs.

## Estimated depth

- Lines target: 1500-1700
- Key mechanisms / diagrams:
  - Logit masking primitive: for each generation step, compute boolean mask over vocab → apply −∞ to invalid → sample from constrained distribution.
  - FSM-of-regex compilation: regex → NFA → DFA → token-level FSM. Precompute next-token-allowed table per state.
  - CFG (Context-Free Grammar): Earley parser state representation; production rule application; advantage over regex (recursive structures like nested JSON, code).
  - XGrammar's split insight: **context-independent tokens** (e.g. comma, brace — valid based only on current state) → precomputed bitmask; **context-dependent tokens** (depend on parse stack, e.g. closing brace must match opening) → runtime check. 99/1 split → 100× speedup.
  - Compressed FSM: detect single-path runs (e.g. literal string `"name":"`) → emit all those tokens in one step.
  - Integration with continuous batching: where does logit_processor sit in vLLM/SGLang pipeline? GPU vs CPU implementation; latency-hiding via async mask prep.
  - JSON Schema preprocessing pipeline: schema → grammar → FSM/parser state machine → load into runtime.
  - Function calling reliability stack: schema definition → grammar compilation → runtime constraint → result parsing → validation.
  - Strict mode vs guided mode vs soft mode (prompt-only) — accuracy vs flexibility Pareto.
- R&D evolution thread: prompt-based JSON (2022-2023, 75-85% reliability) → guidance library (Microsoft, 2023) → Outlines + FSM-of-regex (Willard/Louf, 2023) → Compressed FSM (LMSYS, Feb 2024) → OpenAI Structured Outputs (Aug 2024, in-house impl) → XGrammar (CMU/MLC, Nov 2024) → integration spree Nov 2024 - Jan 2025 (vLLM, SGLang, TRT-LLM, MLC) → OpenAI switches to llguidance (May 2025) → benchmark consolidation (Sep 2025+).

## Cross-references to existing studies

- Should link FROM: **S3-01** (vLLM PagedAttention) — XGrammar is now vLLM's default structured generation backend; logit_processor sits next to sampling step trong vLLM pipeline.
- Should link FROM: **S3-02** (Production RAG) — structured extraction là common RAG output (e.g. extract entities, citations). Strong cross-link.
- Should link FROM: **S3-04** (Agent Framework, planned) — function calling = structured output. S3-10 sẽ là foundational for understanding agent's tool-call reliability.
- Should link FROM: **S3-05** (Document AI, planned) — Donut + structured table extraction need structured output. Cross-link cho extraction pipeline.
- Should link TO: **S3-06** (Speculative Decoding, proposal) — constrained decoding + speculative interaction: draft tokens phải pass through constraint check → can hurt acceptance rate.
- Should link TO: **S3-09** (Test-Time Compute / Reasoning, proposal) — reasoning models often output structured final answer (e.g. boxed expression); how does constrained decoding interact với CoT reasoning?

## Risks / open questions

- **Public info coverage:** Excellent. XGrammar paper (MLSys 2025), Outlines (peer-reviewed). 3 production-grade benchmarks, OpenAI's public credit on llguidance, GitHub source for all 3 libraries. All ≤24 months.
- **Internal vs public details:** OpenAI's exact constraint pipeline (pre-llguidance) not fully public — sẽ frame xung quanh open implementations + công khai swap. Anthropic không expose constraint primitive publicly — sẽ mention as a design choice.
- **Controversial decisions to highlight:**
  - **Strict mode hurts quality?** Forcing model into low-probability paths (e.g. continuation that doesn't match training distribution) can degrade reasoning. When to use strict vs soft constraint?
  - **CFG vs FSM**: CFG handles nested structures (XGrammar) but slower than DFA-only (Outlines). Choose based on schema complexity?
  - **OpenAI's choice of llguidance over XGrammar**: technical reasons? Earley parser flexibility for OpenAI's specific schema preprocessing?
  - **Anthropic's tool use without hard logit constraint**: rely on training, claim better quality. Validated?
  - **Schema design as the contract**: bad schema (e.g. enum with 1000 values) breaks performance. Best practices?
  - **Constrained decoding + speculative decoding interaction**: draft model proposals must satisfy constraint → acceptance rate likely drops. Production data?
  - **Constraint quality of tokens vs characters**: tokenizer artifacts (e.g. `{"` vs `{`+`"`) impact mask correctness. Subtle bugs.
- **Source citation:** Compressed FSM is LMSYS blog (not peer-reviewed) but tier-1 source given LMSYS's production track record. Some 2025 benchmarks (Sep 2025) ở preprint stage — verify khi drafting.

## Recommendation

PROPOSE — đây là **highly practical, production-defining topic** mà current scope-3 totally chưa cover (kb_find_cross_refs returned only S3-04 score 1). Difficulty intermediate (vs advanced for S3-06/07/08/09) → balance scope-3 portfolio. 9 sources (2 peer-reviewed papers + MLSys 2025, 3 production blogs, 1 benchmark study, 3 frontier API docs). Direct cross-link foundation cho S3-04 (agents), S3-02 (RAG), S3-05 (Document AI). Engineering depth (bitmask precompute, FSM compression, integration with batching) đủ cho 1500-1700 dòng.
