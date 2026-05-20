# S3-01 — vLLM Deep Dive: PagedAttention & Continuous Batching

> **Scope**: Modern Tech Stack (LLM serving)
> **Difficulty**: Advanced
> **Tags**: LLM serving, KV cache, PagedAttention, continuous batching, throughput optimization, GPU memory management
> **Primary sources**: Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023), vLLM GitHub + docs (UC Berkeley Sky Computing Lab), Yu et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022).

---

## 1. Tổng quan (Overview)

vLLM là một **LLM inference engine** open-source release năm 2023 bởi nhóm Sky Computing Lab (UC Berkeley, Kwon, Li et al.). Nó nhanh chóng trở thành de-facto standard cho self-hosted LLM serving vì hai innovation chính:

1. **PagedAttention** — quản lý KV cache theo block, lấy cảm hứng từ **virtual memory + paging** của OS, giảm memory fragmentation từ **60-80% xuống ~4%**.
2. **Continuous batching** — dynamic batching ở token level (kế thừa idea từ Orca, OSDI 2022), tăng throughput **2-4x** so với static batching.

Vào thời điểm release (giữa 2023), vLLM cho throughput **gấp 14-24x** so với HuggingFace Transformers naive serving và **2-5x** so với HuggingFace Text Generation Inference (TGI) thời đó (per paper Kwon et al. 2023). TGI sau đó đã adopt continuous batching và gap rút lại, nhưng vLLM vẫn giữ vị trí leading về throughput cho nhiều workloads.

### Tại sao vLLM là case study quan trọng cho modern AI stack?

- **LLM serving khác hoàn toàn classical ML serving**: stateful (KV cache), variable-length outputs, autoregressive (sequential decoding), memory-bound chứ không compute-bound.
- **PagedAttention là một insight đẹp**: apply OS concept (paging) vào ML system — cross-discipline thinking.
- **Continuous batching đang thành standard** cho mọi LLM serving framework: TGI, TensorRT-LLM, SGLang, MLC-LLM đều adopt.
- Hiểu vLLM sẽ giúp bạn debug throughput/latency issues trong RAG/agent systems thực tế.

### LLM serving khác classical ML serving như thế nào?

| Aspect | Classical ML (CTR, image classifier) | LLM serving |
|---|---|---|
| Output length | Fixed (1 scalar / fixed-dim vector) | **Variable** (1 → 4096+ tokens) |
| State per request | Stateless | **Stateful** (KV cache, growing) |
| Compute pattern | One forward pass | **Many forward passes** (one per token) |
| Bottleneck | Compute (matmul) | **Memory bandwidth** (KV cache reads) |
| Batching | Static batches OK | Static batching wastes GPU on padding |
| Latency profile | Single latency number | **TTFT** (time to first token) vs **TPOT** (time per output token) |

Vì những khác biệt này, các technique của classical serving (TF Serving style batching) không apply tốt cho LLM. vLLM giải quyết exactly các vấn đề này.

---

## 2. System Requirements

### 2.1 Functional requirements

- Serve LLM (7B-70B+ parameters) cho **concurrent requests** với variable prompt length và output length.
- Support **streaming output** (token-by-token).
- Support **sampling parameters** per request (temperature, top-p, top-k, stop tokens).
- Support **multi-GPU** (tensor parallelism) cho large models.
- Support **quantization** (INT8, INT4, AWQ, GPTQ, FP8).

### 2.2 Non-functional requirements

| Metric | Target | Notes |
|---|---|---|
| Throughput | **Maximize tokens/sec** | Trade-off với latency |
| TTFT (time to first token) | < 1 sec cho prompt ~1K tokens | Phụ thuộc prompt length |
| TPOT (time per output token) | 20-50 ms (cho 7B model trên A100) | Memory-bandwidth bound |
| Memory efficiency | KV cache fragmentation < 5% | vs ~60-80% trên naive impl |
| GPU utilization | > 80% during peak | |
| Tail latency (P99) | < 2x median | Dưới load |

### 2.3 Constraints quan trọng

- **GPU memory là tài nguyên scarce**: 7B model FP16 chiếm ~14GB model weights, KV cache có thể chiếm thêm 10-50GB tuỳ batch + sequence length.
- **Memory bandwidth-bound**: token generation chỉ ~5-10% GPU compute utilization; bottleneck là moving KV cache + weights từ HBM lên SM.
- **Variable-length sequences**: padding lãng phí; cần dynamic shapes.

---

## 3. High-level Architecture

### 3.1 Lifecycle của một LLM inference request

```
1. PREFILL phase (compute on full prompt)
   Input: prompt P (length L_p)
   Compute: forward pass on P
   Output: KV cache for P, first generated token
   Cost: O(L_p^2 * d) — quadratic in prompt length (attention)

2. DECODE phase (autoregressive generation)
   Repeat T times (T = output length):
     Input: previous KV cache + last token
     Compute: forward pass on 1 token (attending to all previous)
     Output: next token, updated KV cache
     Cost per step: O((L_p + t) * d) — grows linearly
```

Hai phase này có **compute characteristics khác hẳn nhau**:

| Phase | Compute pattern | Bottleneck |
|---|---|---|
| **Prefill** | Big matmul (L_p × hidden), high arithmetic intensity | Compute-bound |
| **Decode** | Small matmul (1 × hidden) per step, low arithmetic intensity | Memory-bound (HBM bandwidth) |

vLLM phải handle cả hai phases hiệu quả → continuous batching cho phép mix prefill + decode trong cùng batch (xem section 4.2).

### 3.2 vLLM architecture overview

```
            ┌─────────────────────────────────────────┐
            │       Request queue (incoming)          │
            └────────────────┬────────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────────┐
            │            Scheduler                     │
            │  - Continuous batching policy           │
            │  - Preemption when memory full          │
            │  - Mix prefill + decode requests        │
            └────────────────┬────────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────────┐
            │   Block Manager (KV cache allocator)    │
            │  - Maintains free block pool            │
            │  - Allocates blocks per sequence        │
            │  - Block table maps logical→physical    │
            └────────────────┬────────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────────┐
            │      Model executor (one or more GPUs)   │
            │  - PagedAttention CUDA kernel            │
            │  - Tensor parallelism (TP) if multi-GPU  │
            │  - Sampling kernels                      │
            └────────────────┬────────────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────────┐
            │       Streaming response to client       │
            └─────────────────────────────────────────┘
```

---

## 4. Deep dive các components chính

### 4.1 PagedAttention — KV cache management

#### Vấn đề: KV cache memory fragmentation

KV cache là **K, V tensors của tất cả tokens đã sinh, cho mỗi layer**:
- Shape: `[num_layers, 2 (K and V), num_heads, seq_len, head_dim]`
- Cho LLaMA-7B (32 layers, 32 heads, head_dim 128): mỗi token tốn ~256 KB KV.
- Sequence 2048 tokens → ~512 MB per sequence.
- Batch 32 sequences → 16 GB chỉ riêng KV cache.

**Naive implementation** (như HuggingFace early versions): allocate KV cache theo **max_seq_len** cho mỗi request. Vấn đề:

```
Request 1: max_len=2048, actual output=100 tokens
  → allocated: 2048 tokens worth of KV space
  → used: 100 tokens
  → wasted: 1948 tokens worth (~95% waste!)
```

vLLM paper đo thấy **chỉ 20-40% KV cache memory thực sự được dùng** trong serving system trước đây. Đây là internal fragmentation và reservation waste.

#### Solution: PagedAttention

Inspired by **virtual memory + paging** trong OS:
- Chia KV cache thành **fixed-size blocks** (e.g. 16 tokens per block).
- Mỗi sequence có một **block table** (giống page table) map logical token positions → physical block IDs.
- Allocate blocks **on-demand**, free khi sequence finish.
- Blocks không cần contiguous → no external fragmentation.

```
Logical view of sequence S1 (length 50):
  Token 0 .. 49
  
Physical view:
  Block table for S1: [block_42, block_17, block_8, block_103]
  Each block holds 16 tokens.
  Token 0-15  → block_42
  Token 16-31 → block_17
  Token 32-47 → block_8
  Token 48-49 → block_103 (only first 2 slots used)
```

Memory layout in GPU HBM:

```
   GPU HBM (KV cache region):
   ┌──────────────────────────────────────────────────┐
   │ block_0 │ block_1 │ block_2 │ block_3 │ ... ...   │
   │ (free)  │ (S2:0-15)│ (free) │ (S1:48-49)│ ...    │
   └──────────────────────────────────────────────────┘
   
   Free block pool: [0, 2, 5, 6, ...]
   
   Block tables:
     S1: [42, 17, 8, 103]   (length=50, 4 blocks)
     S2: [1, 23, 99]        (length=33, 3 blocks)
     S3: [11]               (length=12, 1 block — block partially full)
```

**Fragmentation drops to < 5%** (internal fragmentation chỉ trong block cuối, max 15 tokens worth = ~4MB per sequence).

#### PagedAttention CUDA kernel

Standard attention assumes contiguous KV. PagedAttention modifies attention kernel để:
- Take block table as input.
- Gather K, V from non-contiguous blocks during attention compute.
- Maintain efficiency thông qua optimized memory access patterns.

```python
# Pseudo-code conceptual (kernel thật là CUDA)
def paged_attention(query, block_table, key_cache_blocks, value_cache_blocks, block_size):
    # query: [num_heads, head_dim] — query của token mới
    # block_table: list of physical block IDs cho sequence này
    # key_cache_blocks[block_id]: shape [block_size, num_heads, head_dim]
    
    output = zeros_like(query)
    scores = []
    
    for logical_pos in range(seq_len):
        block_idx = logical_pos // block_size
        offset_in_block = logical_pos % block_size
        physical_block = block_table[block_idx]
        
        k = key_cache_blocks[physical_block][offset_in_block]
        scores.append(dot(query, k) / sqrt(head_dim))
    
    attn_weights = softmax(scores)
    
    for logical_pos in range(seq_len):
        block_idx = logical_pos // block_size
        offset = logical_pos % block_size
        physical_block = block_table[block_idx]
        v = value_cache_blocks[physical_block][offset]
        output += attn_weights[logical_pos] * v
    
    return output
```

Trong thực tế CUDA kernel highly optimized với:
- Shared memory caching.
- Block-level parallelism cho multiple queries.
- Coalesced memory access.

#### Block size trade-off

Block size = 16 tokens là default của vLLM. Trade-off:

| Block size | Pros | Cons |
|---|---|---|
| Small (e.g. 4) | Less internal fragmentation | More overhead (more block table entries, more kernel ops) |
| Medium (16) — default | Good balance | |
| Large (e.g. 64) | Lower overhead | More fragmentation cho short sequences |

### 4.2 Continuous batching (iteration-level scheduling)

#### Vấn đề: static batching wastes compute

**Static batching** (như TF Serving): collect N requests, batch them, run until **all finish**, return.

Vấn đề: requests có **output length khác nhau**. Request short finish sớm, nhưng phải đợi request longest finish → wasted GPU.

```
Static batching timeline (4 requests in batch):
  
  Step 0   1   2   3   4   5   6   7   8
  Req A:  ▓   ▓   ▓   ▓   ✓   ─   ─   ─   ─   (finished at step 4, waiting)
  Req B:  ▓   ▓   ✓   ─   ─   ─   ─   ─   ─   (finished at step 2)
  Req C:  ▓   ▓   ▓   ▓   ▓   ▓   ▓   ✓   ─   (finished at step 7)
  Req D:  ▓   ▓   ▓   ✓   ─   ─   ─   ─   ─   (finished at step 3)
  
  Idle slots = ▓ count missing in steps 3-8 ⇒ WASTED GPU compute
```

#### Continuous batching solution (Orca → vLLM)

**Iteration-level scheduling**: sau **mỗi token step**, scheduler:
1. Remove finished requests from batch.
2. Add new pending requests if memory available.
3. Continue next step.

```
Continuous batching timeline:
  
  Step 0   1   2   3   4   5   6   7   8
  Req A:  ▓   ▓   ▓   ▓   ✓                    
  Req B:  ▓   ▓   ✓                    
  Req C:  ▓   ▓   ▓   ▓   ▓   ▓   ▓   ✓  
  Req D:  ▓   ▓   ▓   ✓                    
  Req E:          ▓   ▓   ▓   ▓   ▓   ✓   (added at step 2)
  Req F:              ▓   ▓   ▓   ▓   ▓   ✓
  Req G:              ▓   ▓   ▓   ✓
  Req H:                  ▓   ▓   ✓
  ...
  
  GPU stays full — high utilization
```

vLLM scheduler mỗi step có thể có batch size khác nhau. Mỗi step:

```python
# Pseudo-code scheduler step
def schedule_next_step(running_requests, waiting_queue, kv_cache_budget):
    # 1. Remove finished requests
    running = [r for r in running_requests if not r.is_finished()]
    
    # 2. Free their KV blocks
    for r in running_requests:
        if r.is_finished():
            block_manager.free(r.block_table)
    
    # 3. Try to add new requests (if memory available)
    while waiting_queue and block_manager.can_allocate(estimated_blocks):
        new_req = waiting_queue.pop()
        block_manager.allocate(new_req)
        running.append(new_req)
    
    # 4. If memory tight, may need to PREEMPT a running request
    if block_manager.is_full() and len(running) > min_batch:
        victim = select_preempt_victim(running)  # e.g. lowest priority
        block_manager.swap_or_recompute(victim)
        running.remove(victim)
    
    return running
```

#### Mix prefill + decode trong cùng batch

vLLM (sau 0.2.0) hỗ trợ **mixed batches**: prefill (cho new requests) cùng với decode (cho running). Lý do:

- Prefill compute-bound, decode memory-bound → mix nhau utilize GPU tốt hơn.
- Tránh "head-of-line blocking" — request mới đến không phải đợi tất cả decode finish.

Trade-off: prefill thường latency cao hơn decode-only step → TPOT của decode requests sẽ bị ảnh hưởng. vLLM có flag `--enable-chunked-prefill` để **chunk prefill** thành nhiều steps, smoother latency.

### 4.3 Preemption và recovery

Khi nhiều requests share GPU memory, có thể xảy ra **out-of-memory** giữa run. vLLM xử lý bằng preemption:

**Option 1: Swap to CPU** — move KV cache của victim sang CPU RAM, swap back later. Cost: PCIe bandwidth.

**Option 2: Recompute** — drop KV cache, recompute prefill khi resume. Cost: extra compute, nhưng cho short prompts có thể faster than swap.

vLLM default chọn **recompute** vì với prompts vừa phải, recompute nhanh hơn PCIe swap.

### 4.4 Tensor parallelism cho large models

Cho models > GPU memory (e.g. LLaMA-70B FP16 = 140GB > A100 80GB), vLLM dùng **tensor parallelism (TP)**:

- Shard mỗi linear layer along output dimension across N GPUs.
- All-reduce activations giữa GPUs.

```
Linear layer W of shape [hidden, hidden] across 4 GPUs (TP=4):
  
  GPU 0: W[:, 0:hidden/4]      → output column 0..hidden/4
  GPU 1: W[:, hidden/4:hidden/2] → output column hidden/4..hidden/2
  ...
  
  After matmul → all-reduce to gather full output.
```

Trade-off:
- TP=2: cần GPU connected via fast interconnect (NVLink ideal, PCIe acceptable).
- TP=4 hoặc 8: typically requires NVLink + NVSwitch (single DGX node).
- Cross-node TP slow.

vLLM cũng support **pipeline parallelism** cho models cực lớn, nhưng less common.

### 4.5 Quantization support

vLLM support nhiều quantization formats:

| Format | Bits | Notes |
|---|---|---|
| FP16 / BF16 | 16 | Default, no quantization |
| FP8 | 8 | H100+ native support, low accuracy loss |
| INT8 (W8A8) | 8 | Weights + activations |
| AWQ | 4 | Weight-only, activation-aware |
| GPTQ | 4 | Weight-only, GPTQ algorithm |
| INT4 | 4 | Various flavors |

Quantization weights: giảm memory (4-bit = 1/4 of FP16), tăng throughput (memory-bandwidth bound!). Accuracy loss thường < 1% nếu dùng AWQ/GPTQ properly cho 7B-70B models.

### 4.6 Optimizations bổ sung

- **CUDA graphs**: capture decode step thành CUDA graph để giảm kernel launch overhead. Big win cho small batch sizes.
- **Speculative decoding**: dùng small draft model dự đoán nhiều tokens, verify bằng big model — giảm latency.
- **Prefix caching**: cache KV của prompt prefix dùng chung (e.g. system prompt) → tránh recompute prefill nếu prefix giống.
- **Chunked prefill**: chia prefill thành chunks small để mix với decode requests smoothly.
- **FlashAttention / FlashInfer**: high-performance attention kernels mà vLLM integrate.

---

## 5. Trade-offs & Design decisions

### 5.1 Throughput vs Latency

LLM serving không thể tối ưu cả hai — **fundamental trade-off**:

| Goal | Batch size | TTFT | TPOT | Total throughput |
|---|---|---|---|---|
| Min latency | Small (e.g. 1) | Low | Low | Low |
| Max throughput | Large (e.g. 64+) | High (queue + compute) | Higher (more contention) | High |

vLLM defaults tối ưu throughput (assume bulk workload). Cho latency-sensitive serving (interactive chat), config:
- `max_num_seqs` thấp hơn
- Disable chunked prefill nếu prompts ngắn
- Có thể dùng speculative decoding

### 5.2 PagedAttention vs contiguous KV cache

| Approach | Pros | Cons |
|---|---|---|
| **Contiguous** (naive) | Simpler kernel | 60-80% memory waste |
| **PagedAttention** (vLLM) | < 5% waste | Kernel complexity, slight overhead per attention call |

Overhead của PagedAttention ~5-10% trong attention kernel, nhưng memory saved → **larger effective batch size** → throughput net positive 2-4x.

### 5.3 Continuous batching vs static batching

| Aspect | Static | Continuous |
|---|---|---|
| Implementation | Simple | Complex (scheduler + iteration-level) |
| Throughput | Low (idle GPU) | High |
| Latency variance | Predictable | Variable (depends on batch composition) |
| Fairness | First-come-first-served | Need explicit fairness policy |

Continuous batching đã thành standard — không có lý do dùng static batching trừ khi workload uniform length.

### 5.4 Quantization aggressiveness

| Setting | Quality loss | Memory savings | Throughput |
|---|---|---|---|
| FP16 (no quant) | 0% | 0 | Baseline |
| FP8 | ~0.1-0.3% | 2x | 1.5-2x |
| INT8 W8A8 | ~0.5-1% | 2x | 1.5-2x |
| AWQ 4-bit | ~0.5-1% | 4x | 1.5-2.5x (memory-bound win) |
| INT4 naive | 1-5%+ | 4x | similar |

Production thường chọn FP8 (nếu H100) hoặc AWQ 4-bit cho deployment, với careful eval. Quality loss khó measure — cần task-specific eval, không chỉ perplexity.

### 5.5 vLLM vs alternatives

| Framework | Strengths | Weaknesses | Best for |
|---|---|---|---|
| **vLLM** | Throughput, PagedAttention, mature | Less specialized features | General-purpose self-host |
| **TensorRT-LLM** | NVIDIA-optimized, fastest on H100 | Vendor lock-in, complex build | NVIDIA shops, max perf |
| **TGI (HuggingFace)** | Easy deploy, HF integration | Throughput slightly lower than vLLM | HF ecosystem |
| **SGLang** | Structured outputs, RadixAttention prefix cache | Newer, less battle-tested | RAG, structured workloads |
| **MLC-LLM** | Cross-platform (mobile, edge) | Less focused on server perf | Edge/mobile |
| **llama.cpp** | CPU + GPU, lightweight | Lower throughput | Single-user, edge |

---

## 6. Lessons learned & Best practices

1. **Memory bandwidth là dominant bottleneck cho decode, không phải compute**. Tối ưu cho memory access pattern (KV cache layout, quantization) impactful hơn tối ưu matmul kernels.

2. **PagedAttention pattern là transferable** — bất kì khi nào bạn có dynamic memory allocation pattern variable-length, paging analogy work tốt.

3. **Đo TTFT và TPOT riêng**, không phải overall latency. User experience: TTFT < 1s + smooth streaming > 1 single big latency number.

4. **Batch size lớn ≠ throughput cao luôn** — qua một ngưỡng, attention compute (O(L^2)) dominate và throughput diminish. Sweet spot tùy model + GPU + workload.

5. **Prefix caching big win cho RAG/chat** — system prompt + few-shot examples thường share giữa requests. Prefix caching tránh re-prefill, giảm TTFT đáng kể.

6. **Chunked prefill = smoother latency** — đặc biệt khi mix prefill + decode workload. Latency P99 cải thiện rõ.

7. **Continuous batching không free** — scheduler overhead per token step. Cho models very large (70B+ TP=8), step time đủ lớn để scheduler không phải bottleneck. Cho models nhỏ (7B single GPU), scheduler có thể chiếm 5-10% time.

8. **Đừng over-provision GPU memory cho max batch size**. Better: tune `gpu_memory_utilization` (vLLM flag) ~0.9-0.95 → để vLLM tự manage block pool optimally.

9. **Quantization phải eval task-specific**. Perplexity drop có thể nhỏ nhưng task accuracy (e.g. code generation, reasoning) có thể drop nhiều hơn. Luôn benchmark trên downstream task.

10. **Speculative decoding lý thuyết đẹp, production hard**. Cần draft model tốt + careful tuning. Gain ~1.5-2x cho latency, nhưng implementation overhead lớn — chỉ nên invest khi latency critical.

11. **Multi-GPU setup**: tensor parallelism với NVLink fast, PCIe slow. Cho 70B model: TP=2 trên A100 80GB (vừa đủ với quant), TP=4 cho FP16, TP=8 nếu cần more memory cho KV cache.

12. **Monitor GPU memory + cache hit rate** trong production. vLLM expose metrics via Prometheus — quan trọng cho debug throughput regression.

---

## 7. References

### Papers

1. **Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica.** "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — paper gốc vLLM.
2. **Yu, Jeong, Kim, Park, Chun, Kim (Seoul National Univ.).** "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022. [Link](https://www.usenix.org/conference/osdi22/presentation/yu) — paper introduce iteration-level scheduling (continuous batching).
3. **Dao et al.** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
4. **Lin et al.** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
5. **Frantar et al.** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).
6. **Leviathan et al. (Google).** "Fast Inference from Transformers via Speculative Decoding." ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).

### Engineering blogs & docs

7. **vLLM official docs.** [docs.vllm.ai](https://docs.vllm.ai/)
8. **vLLM GitHub.** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — codebase, đọc `csrc/attention/paged_attention_kernels.cu` để hiểu kernel.
9. **Anyscale Blog.** "Continuous Batching for LLM Inference." Tháng 6/2023.
10. **NVIDIA Blog.** "Mastering LLM Techniques: Inference Optimization." 2023-2024 series.
11. **HuggingFace TGI docs.** [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference).

### Related case studies (đọc tiếp)

- **S3-02 Production RAG System Architecture** — vLLM ở phía generator + vector DB ở phía retriever.
- **S4-04 GPU Cluster Management & Cost Optimization** — multi-tenant LLM serving ở scale.
- **S3-04 Agent Framework Architecture** — workload pattern agent (multi-turn, tool calling) khác với chat.

### Độ tin cậy

- PagedAttention paper SOSP 2023 là **peer-reviewed**, high confidence.
- Continuous batching gain numbers (2-4x throughput) là theo paper Kwon et al., có replicated trong production deployments — high confidence.
- Số cụ thể (TPOT 20-50ms cho 7B trên A100) là **typical numbers from community benchmarks**, sẽ vary theo workload + tuning.
- Comparison table giữa vLLM/TGI/TensorRT-LLM phản ánh **state ở thời điểm 2024-2025**, gap có thể đã thay đổi.
