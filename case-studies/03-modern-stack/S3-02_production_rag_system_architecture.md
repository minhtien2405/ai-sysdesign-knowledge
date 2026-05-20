# S3-02 — Production RAG System Architecture

> **Scope**: Modern Tech Stack (LLM / RAG / Agent / CV)
> **Difficulty**: Intermediate
> **Tags**: RAG, retrieval-augmented generation, chunking, dense retrieval, hybrid retrieval, reranker, LLM, evaluation
> **Primary sources**:
> - Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020, arXiv 2005.11401) — original RAG paper.
> - Karpukhin et al. "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020, arXiv 2004.04906) — DPR.
> - Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (arXiv 2309.15217, 2023).
> - LangChain, LlamaIndex documentation; production blogs từ Pinecone, Weaviate, Anthropic, OpenAI.

---

## 1. Tổng quan (Overview)

RAG (Retrieval-Augmented Generation) là **kiến trúc thông dụng nhất** để build LLM applications với knowledge specific (internal docs, knowledge base, codebase) mà không cần finetune model. Lý do RAG thắng finetune trong 80%+ use cases enterprise:

- **Knowledge tươi** — update KB không cần retrain model, chỉ cần re-index.
- **Citation/traceability** — có thể trỏ thẳng tới source document, critical cho compliance/legal.
- **Cost-effective** — không cần GPU training, chỉ cần vector DB + embedding model.
- **Modular** — có thể thay LLM (Claude → GPT-4 → Llama) mà không động đến KB.

Nhưng "RAG đơn giản" (chunk + embed + cosine search + LLM) chỉ là **demo level**. Production RAG ở các công ty serious cần xử lý:

- **Chunking strategies** chuyên sâu (semantic, hierarchical, recursive).
- **Hybrid retrieval** (BM25 + dense + metadata filtering) — không chỉ dense alone.
- **Reranker** (cross-encoder hoặc LLM-based) để boost relevance.
- **Query understanding** (rewriting, expansion, decomposition).
- **Evaluation framework** thực sự đo được — không chỉ "looks good".
- **Caching, observability, cost monitoring** ở production scale.

Case study này design một production RAG system end-to-end, dùng kinh nghiệm thực tế từ các talks và blog posts của Anthropic (Contextual Retrieval), OpenAI (Cookbook), Pinecone, Weaviate, và LangChain/LlamaIndex production deployments.

### Khi nào dùng RAG, khi nào KHÔNG?

| Trường hợp | Nên dùng RAG? | Lý do |
|---|---|---|
| Q&A trên internal docs | **Yes** | Knowledge update, citation cần thiết |
| Customer support với KB | **Yes** | Update articles thường xuyên |
| Code completion từ codebase | **Yes** | Context-aware, knowledge specific |
| Style/personality của assistant | **No** | Finetune hoặc system prompt phù hợp hơn |
| Math reasoning improvement | **No** | Cần training (tool use cũng OK) |
| Latency-critical (< 100ms) | **Maybe** | Cache rất nhiều, hoặc dùng embedding-only retrieval |
| Privacy: data không được leave premise | **Yes** | Self-host embedding + vector DB |

Xem thêm so sánh ở **S2-05 Finetune vs RAG vs Prompt Engineering** (planned).

---

## 2. System Requirements

### 2.1 Functional requirements

- Ingest documents từ multiple sources (PDFs, HTML, Confluence, Notion, code repos, databases).
- Process documents: chunk, embed, store metadata.
- Serve queries: receive user question → retrieve relevant chunks → generate answer với citations.
- Support **multi-turn conversations** với conversation context.
- Support **filtering** by metadata (e.g., "search only in 2024 docs", "only in product team space").
- **Citations**: response phải include reference đến source chunks.
- **Feedback loop**: user thumbs up/down để cải thiện retrieval.

### 2.2 Non-functional requirements (target cho production system mid-scale)

| Metric | Target | Notes |
|---|---|---|
| Document corpus size | 100K - 10M chunks | After chunking, e.g. ~10M tokens of raw text |
| Total embedding storage | ~vài chục GB | 1M chunks × 1536 dim × 4 bytes ≈ 6 GB |
| Query latency P50 | < 1 s (end-to-end) | Retrieval ~100ms + LLM ~600ms |
| Query latency P99 | < 3 s | Bao gồm reranker, multi-step |
| QPS target | 10-100 sustained | Cao hơn cần caching layer |
| Indexing throughput | 100-1000 docs/sec | Phụ thuộc embedding model batch size |
| Retrieval recall@10 | > 90% | Tỷ lệ ground-truth chunk có trong top-10 retrieved |
| Faithfulness score | > 0.85 | RAGAS metric — answer không hallucinate |
| Cost per query | $0.001 - $0.01 | LLM call dominate; embedding cheap |

### 2.3 Constraints quan trọng

- **Embedding model context window**: thường 512 - 8192 tokens → chunk size phải fit.
- **LLM context window**: dù GPT-4 / Claude có 200K+ context, **đưa quá nhiều chunks vào context giảm performance** ("lost in the middle" — Liu et al. 2023). Sweet spot thường 5-20 chunks.
- **Latency**: mỗi step (embed query, retrieve, rerank, generate) add latency. Phải parallelize và cache aggressively.
- **Cost**: LLM tokens dominate. Mỗi query có thể tốn $0.01-0.05 nếu không quản lý context size.

---

## 3. High-level Architecture

```
                           PRODUCTION RAG ARCHITECTURE

   ┌─────────────── INGESTION PIPELINE (offline / batch) ────────────────┐
   │                                                                     │
   │  Sources (PDFs, HTML, Confluence, Notion, GitHub, DBs, ...)         │
   │           │                                                         │
   │           ▼                                                         │
   │  ┌────────────────┐   ┌─────────────┐   ┌──────────────┐            │
   │  │ Doc parsers    │──►│  Chunker    │──►│  Enrichment  │            │
   │  │ (unstructured, │   │ (semantic / │   │ (metadata,   │            │
   │  │  pdfplumber)   │   │  recursive) │   │  summary)    │            │
   │  └────────────────┘   └─────────────┘   └──────┬───────┘            │
   │                                                │                    │
   │                       ┌────────────────────────┴───┐                │
   │                       ▼                            ▼                │
   │              ┌────────────────┐         ┌─────────────────┐         │
   │              │ Embedding      │         │ BM25 index      │         │
   │              │ model          │         │ (Elasticsearch/ │         │
   │              │ (OpenAI/local) │         │  OpenSearch)    │         │
   │              └────────┬───────┘         └────────┬────────┘         │
   │                       ▼                          ▼                  │
   │              ┌────────────────┐         ┌─────────────────┐         │
   │              │ Vector DB      │         │ Inverted index  │         │
   │              │ (Pinecone /    │         │                 │         │
   │              │  Weaviate /    │         └─────────────────┘         │
   │              │  pgvector)     │                                     │
   │              └────────────────┘                                     │
   └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │
   ┌──────────── QUERY PIPELINE (online / per request) ──────────────────┐
   │                                                                     │
   │   User question                                                     │
   │       │                                                             │
   │       ▼                                                             │
   │  ┌──────────────┐       ┌────────────────┐                          │
   │  │ Conversation │──────►│ Query rewriting│                          │
   │  │ history      │       │ + expansion    │                          │
   │  └──────────────┘       │ (LLM call)     │                          │
   │                         └────────┬───────┘                          │
   │                                  │                                  │
   │            ┌─────────────────────┴─────────────────────┐            │
   │            ▼                                           ▼            │
   │  ┌─────────────────┐                        ┌──────────────────┐    │
   │  │ Embed query     │                        │ Keyword extract  │    │
   │  │ (same model as  │                        │ for BM25         │    │
   │  │  ingestion)     │                        │                  │    │
   │  └────────┬────────┘                        └─────────┬────────┘    │
   │           ▼                                           ▼             │
   │  ┌─────────────────┐                        ┌──────────────────┐    │
   │  │ Dense retrieval │                        │ Sparse retrieval │    │
   │  │ (vector DB,     │                        │ (BM25, top 100)  │    │
   │  │  top 100)       │                        │                  │    │
   │  └────────┬────────┘                        └─────────┬────────┘    │
   │           └────────────────────┬──────────────────────┘             │
   │                                ▼                                    │
   │                       ┌─────────────────┐                           │
   │                       │ Fusion (RRF)    │                           │
   │                       │ → top 50        │                           │
   │                       └────────┬────────┘                           │
   │                                ▼                                    │
   │                       ┌─────────────────┐                           │
   │                       │ Reranker        │                           │
   │                       │ (cross-encoder) │                           │
   │                       │ → top 5-10      │                           │
   │                       └────────┬────────┘                           │
   │                                ▼                                    │
   │                       ┌─────────────────┐                           │
   │                       │ Prompt assembly │                           │
   │                       │ + LLM call      │                           │
   │                       │ (Claude/GPT-4)  │                           │
   │                       └────────┬────────┘                           │
   │                                ▼                                    │
   │                       ┌─────────────────┐                           │
   │                       │ Answer +        │                           │
   │                       │ citations to    │                           │
   │                       │ source chunks   │                           │
   │                       └─────────────────┘                           │
   └─────────────────────────────────────────────────────────────────────┘
```

### Data flow summary

1. **Offline ingestion**: docs → parsed → chunked → enriched → embedded → stored ở vector DB + BM25 index.
2. **Online query**: question → rewritten → embedded + keyword extract → hybrid retrieval → reranked → top-k chunks → LLM với citations.

---

## 4. Deep dive các components chính

### 4.1 Chunking strategies — đừng coi thường

Chunking là **single most impactful decision** trong RAG. Sai chunking strategy → mọi thứ downstream sai theo.

#### Naive fixed-size chunking (KHÔNG nên dùng cho production)

```python
def naive_chunk(text, size=500, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks
```

**Vấn đề**:
- Cắt giữa câu → mất ngữ nghĩa.
- Mất layout structure (headings, tables, code blocks).
- Overlap không đảm bảo context boundary.

#### Recursive character chunking (LangChain RecursiveCharacterTextSplitter)

Tách theo separators hierarchy: `["\n\n", "\n", ". ", " ", ""]`. Cố gắng giữ đoạn văn hoàn chỉnh.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(doc)
```

**Tốt cho**: prose documents (articles, books).
**Yếu**: code, structured docs (PDFs với tables).

#### Semantic chunking (LlamaIndex SemanticSplitterNodeParser)

Embed từng câu, group các câu adjacent có cosine similarity cao thành một chunk. Threshold dùng percentile.

```python
def semantic_chunk(sentences, embeddings, percentile=95):
    distances = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        distances.append(1 - sim)
    threshold = np.percentile(distances, percentile)
    
    chunks, current = [], [sentences[0]]
    for i, d in enumerate(distances):
        if d > threshold:
            chunks.append(" ".join(current))
            current = [sentences[i+1]]
        else:
            current.append(sentences[i+1])
    chunks.append(" ".join(current))
    return chunks
```

**Tốt cho**: long-form docs với topic shifts. Compute cost cao hơn (embed mọi sentence).

#### Document-aware chunking — recommended cho production

- **Markdown/HTML**: split theo headings (H1, H2, H3). Preserve hierarchy trong metadata.
- **PDFs**: dùng `unstructured` library hoặc `pdfplumber` để extract layout. Tables → keep as one chunk. Figures → captions + alt text.
- **Code**: split theo function/class boundaries, không cắt giữa function.
- **Slides**: 1 slide = 1 chunk usually.

#### Hierarchical chunking (advanced — LlamaIndex sentence window, auto-merging)

Idea: tạo **multiple chunk granularities**.
- Small chunks (200 tokens): tốt cho retrieval (precise match).
- Large chunks (1000 tokens): cung cấp cho LLM khi generate.

Workflow:
1. Retrieve dựa trên small chunks.
2. Khi return, replace bằng parent chunk lớn hơn (or expand window around hit).

```python
# Pseudo-code
def auto_merging_retrieval(query, vector_db, k=10):
    small_hits = vector_db.search(query, k=k)
    # Group by parent_id
    parent_counts = Counter(h.metadata['parent_id'] for h in small_hits)
    # If >threshold% siblings hit same parent, return parent
    chunks_to_return = []
    for parent_id, count in parent_counts.items():
        if count >= 3:
            chunks_to_return.append(get_parent(parent_id))
        else:
            chunks_to_return.extend([h for h in small_hits if h.metadata['parent_id'] == parent_id])
    return chunks_to_return
```

#### Contextual retrieval (Anthropic, 2024)

Insight: chunk thuần thường mất context (e.g. "It increased by 3%" — IT là gì?). Solution: **prepend LLM-generated context** cho mỗi chunk.

```python
def contextualize_chunk(doc, chunk):
    prompt = f"""<document>{doc}</document>
    
    Here is the chunk we want to situate within the whole document:
    <chunk>{chunk}</chunk>
    
    Please give a short succinct context to situate this chunk within the
    overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else."""
    
    context = llm.generate(prompt)
    return f"{context}\n\n{chunk}"
```

Anthropic report: **-35% retrieval failures** với contextual retrieval, **-49%** khi combine với BM25 (Anthropic blog "Introducing Contextual Retrieval", 2024-09).

**Cost trade-off**: prepend chunk vào prompt mỗi lần index → expensive. Mitigate bằng **prompt caching** (Anthropic) — cache document portion.

### 4.2 Embedding model selection

| Model | Dim | Context | Strength | Cost (per 1M tokens) |
|---|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 (configurable) | 8191 | General, cheap, fast | $0.02 |
| `text-embedding-3-large` (OpenAI) | 3072 | 8191 | Higher quality, English | $0.13 |
| `voyage-3` (Voyage AI) | 1024 | 32K | High quality, long context | $0.06 |
| `cohere-embed-v3` | 1024 | 512 | Multilingual strong | $0.10 |
| `bge-m3` (BAAI, open source) | 1024 | 8192 | Self-host, multilingual | Free (compute cost) |
| `nomic-embed-text-v1.5` (Nomic) | 768 (configurable) | 8192 | Open source, Matryoshka | Free |

**Production recommendations**:
- **Default**: OpenAI text-embedding-3-small. Cost-effective, latency thấp.
- **Self-host**: BGE-M3 hoặc Nomic — single GPU đủ cho >1000 QPS.
- **Domain-specific**: finetune embedding model trên domain data (e.g. medical, legal) — typical +5-15% recall.

**Matryoshka embeddings** (Kusupati et al. 2022): train embedding model sao cho prefix của vector vẫn có ý nghĩa. Cho phép trade-off accuracy vs storage dynamically (e.g. dùng dim 256 cho retrieval coarse, expand 1536 cho rerank fine).

### 4.3 Hybrid retrieval — BM25 + Dense + Fusion

#### BM25 (sparse retrieval) — keyword-based

BM25 (Best Matching 25) là biến thể TF-IDF với term frequency saturation:

```
BM25(q, d) = Σ_i IDF(qi) · [tf(qi, d) · (k1 + 1)] / [tf(qi, d) + k1 · (1 - b + b · |d|/avgdl)]
```

**Khi BM25 thắng dense retrieval**:
- Exact phrase match (e.g. error codes "ERR_408", model numbers "GPT-4o").
- Out-of-domain terms không có trong embedding model training.
- Very long queries (multiple keywords).

#### Dense retrieval — semantic match

Dense thắng khi:
- Paraphrasing ("how to cook chicken" → "chicken recipes").
- Cross-lingual (query EN, docs FR).
- Concept-level matching ("vehicle" → "car", "truck", "motorcycle").

#### Hybrid — combine via Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(rankings_list, k=60):
    """
    rankings_list: list of [doc_id, ...] sorted by rank from each retriever.
    """
    scores = {}
    for ranking in rankings_list:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**RRF advantages**:
- Không cần normalize scores giữa retrievers (different scales).
- Robust, parameter k=60 hầu như universal.
- Hoạt động tốt cho **bất kỳ số lượng retrievers** (BM25 + dense + sparse-dense like SPLADE).

#### Metadata filtering — pre-filter

Quan trọng cho production: filter trước khi rank.

```python
results = vector_db.search(
    query_embedding,
    top_k=100,
    filter={
        "year": {"$gte": 2023},
        "team": "product",
        "doc_type": {"$in": ["spec", "design_doc"]},
    },
)
```

Vector DBs hỗ trợ filter ở mức khác nhau:
- **Pinecone, Weaviate, Qdrant**: native metadata filtering, integrated với ANN search.
- **pgvector**: SQL WHERE clause.
- **FAISS**: không built-in, phải post-filter (kém efficient).

### 4.4 Reranking — boost quality của top results

#### Tại sao cần reranker?

Dense retrieval dùng **bi-encoder** (encode query và doc độc lập, compute cosine). Trade-off: scale tốt (precompute doc embeddings) nhưng **kém precise**.

**Cross-encoder** đưa cả query và doc qua model cùng lúc, output relevance score → quality cao hơn, nhưng phải re-compute mỗi pair → **không scale cho retrieval**, chỉ rerank top-N.

Pipeline chuẩn: retrieve top-100 với bi-encoder → rerank với cross-encoder → top-5-10 đưa vào LLM.

#### Reranker options

| Model | Type | Latency (top 100) | Quality |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Open source, MiniLM | ~50ms (GPU) | Baseline |
| `BAAI/bge-reranker-large` | Open source, BERT-large | ~100ms (GPU) | Strong |
| `Cohere Rerank v3` | API | ~150ms | Best (multilingual) |
| **LLM-as-reranker** (GPT-4 score) | API | ~1-2s | Best but expensive |

**Production tip**: Cohere Rerank API là default lựa chọn cho most teams. Nếu self-host, BGE reranker.

#### Pseudo-code rerank pipeline

```python
def rerank(query, candidates, reranker, top_n=5):
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.score(pairs)   # cross-encoder forward
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_n]]
```

#### MMR (Maximal Marginal Relevance) — diversify top results

MMR = Maximal Marginal Relevance. Sau khi rerank, có thể rất similar chunks dồn lên top → redundant context. MMR balance relevance vs diversity:

```
MMR(d) = λ · sim(d, q) - (1-λ) · max_{d' in selected} sim(d, d')
```

Greedy select chunk maximize MMR score. λ=0.7 là default tốt.

```python
def mmr_select(query_emb, candidate_embs, candidates, k=5, lam=0.7):
    selected_idx = []
    remaining = list(range(len(candidates)))
    for _ in range(k):
        if not remaining:
            break
        scores = []
        for i in remaining:
            rel = cosine(query_emb, candidate_embs[i])
            if selected_idx:
                div = max(cosine(candidate_embs[i], candidate_embs[j])
                          for j in selected_idx)
            else:
                div = 0
            scores.append(lam * rel - (1 - lam) * div)
        best = remaining[np.argmax(scores)]
        selected_idx.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected_idx]
```

### 4.5 Query understanding & rewriting

#### Query rewriting

Direct embedding của user query thường suboptimal:
- Conversational queries ("what about it?") — không có context.
- Vague queries ("tell me more").
- Multi-intent queries ("compare X and Y and find which is cheaper").

Pre-processing với LLM:

```python
def rewrite_query(question, conversation_history):
    prompt = f"""Given this conversation history:
{conversation_history}

And the user's latest question: "{question}"

Rewrite the question as a standalone search query that captures the user's intent.
Be specific and include any context from the conversation."""
    return llm.generate(prompt)
```

#### Query expansion (HyDE — Hypothetical Document Embedding)

Gao et al. 2022: thay vì embed query, ask LLM generate hypothetical answer, embed answer.

```python
def hyde_query(question):
    hypothetical = llm.generate(f"Write a passage that answers: {question}")
    return embed(hypothetical)
```

Intuition: hypothetical answer ở "ngôn ngữ giống answers trong KB" → better semantic match. Trade-off: extra LLM call latency.

#### Query decomposition (multi-step)

Cho complex queries ("compare X and Y in dimensions A, B, C"):

```python
def decompose_query(question):
    prompt = f"""Break down this question into sub-questions:
"{question}"

Output a JSON list of sub-questions, each searchable independently."""
    sub_qs = json.loads(llm.generate(prompt))
    return sub_qs

# Retrieve for each sub-q, merge contexts, generate final answer
```

### 4.6 Prompt assembly & generation

#### Standard prompt template

```python
PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the
provided context. If the context doesn't contain enough information, say so.
Always cite sources using [source_id] notation.

<context>
{context}
</context>

<question>{question}</question>

Answer:"""

def assemble_prompt(question, chunks):
    context = "\n\n".join([
        f"[source_{i}] (from {c.metadata['source']}):\n{c.text}"
        for i, c in enumerate(chunks)
    ])
    return PROMPT.format(context=context, question=question)
```

#### Anti-hallucination techniques

1. **Explicit grounding instruction**: "Answer ONLY based on context".
2. **Force citations**: model phải cite source_id cho mỗi claim.
3. **Refusal training**: nếu context không đủ, return "I don't have enough information".
4. **Self-check pass** (Anthropic Citation API style): post-process answer, verify mỗi claim có support trong context.

#### Context window management

Vấn đề "lost in the middle" (Liu et al. 2023): LLM attention biased về đầu và cuối context. Mitigations:

- **Limit chunks**: 5-10 chunks là sweet spot (không quá 20 chunks).
- **Reorder by relevance**: most relevant ở đầu và cuối context (LLM-friendly).
- **Compression**: summarize lower-ranked chunks instead of dropping.

---

## 5. Trade-offs & Design decisions

### 5.1 Chunk size

| Chunk size | Pros | Cons |
|---|---|---|
| Small (100-300 tokens) | Precise retrieval, high recall@1 | Mất context, có thể cần merge |
| Medium (500-800 tokens) | Balanced — **production default** | OK trade-off |
| Large (1000-2000 tokens) | Self-contained, less merge | Recall lower (multiple topics per chunk dilute embedding) |

**Recommendation**: start với 500-800 tokens, 50-100 token overlap. A/B test trên domain data.

### 5.2 Vector DB choice

| DB | Type | Sweet spot | Notes |
|---|---|---|---|
| **pgvector** | Postgres extension | < 10M vectors, đã có Postgres | Easy ops, transactional |
| **Pinecone** | Managed | Production, scale | Closed source, mature filtering |
| **Weaviate** | Open source, managed option | Mid-large, hybrid built-in | Good hybrid native |
| **Qdrant** | Open source, Rust | Self-host, filtering heavy | Strong filter perf |
| **Milvus** | Open source | Very large scale (billion+) | Complex ops |
| **FAISS** | Library | < 100M, embed in app | Không phải DB, không filter native |

Xem [S3-03 Vector DB Internals](S3-03_vector_database_internals.md) (planned) cho deep dive HNSW vs IVF-PQ.

### 5.3 Open source vs API stack

| Component | Open source path | API path |
|---|---|---|
| Embedding | BGE-M3, Nomic | OpenAI, Voyage, Cohere |
| Vector DB | Qdrant, Milvus, pgvector | Pinecone, Weaviate Cloud |
| Reranker | BGE reranker | Cohere Rerank |
| LLM | Llama, Mistral (vLLM serve) | Claude, GPT-4 |
| Orchestration | LlamaIndex, LangChain | LlamaIndex Cloud, LangSmith |

**Hybrid is common**: open-source embedding + managed vector DB + Claude/GPT-4 for generation.

### 5.4 Latency vs quality trade-offs

| Optimization | Latency saved | Quality impact |
|---|---|---|
| Skip query rewriting | ~500ms | -3-5% quality on multi-turn |
| Skip reranker | ~100-200ms | -5-10% precision @ 5 |
| Smaller embedding (1536 → 256 Matryoshka) | ~10ms (negligible) | -2-3% recall |
| Cache top-K retrieved chunks | ~200ms (cache hit) | 0 if cache fresh |
| Stream LLM response | TTFT improves | None |
| Smaller LLM (Claude Sonnet → Haiku) | -50% generation time | -10-15% answer quality |

### 5.5 Evaluation — measure cái gì?

#### Retrieval metrics (offline, with labeled QA pairs)

- **Recall@k**: % ground-truth chunks trong top-k retrieved. Target > 0.9 at k=10.
- **MRR (Mean Reciprocal Rank)**: 1 / rank của first correct chunk.
- **NDCG@k**: Discounted gain, support multi-level relevance.

#### Generation metrics

**RAGAS framework** (Es et al. 2023):
- **Faithfulness**: % statements trong answer có support từ context. Use LLM judge.
- **Answer relevance**: answer có address question không?
- **Context precision**: chunks retrieved có relevant không?
- **Context recall**: chunks retrieved cover ground-truth answer không?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

scores = evaluate(
    dataset=evaluation_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
```

#### Online metrics

- **Thumbs up/down ratio** từ users.
- **Citation click-through**: user có click vào source link?
- **Conversation turn count**: ít turn = answer tốt (one-shot resolution).
- **Escalation rate**: % conversations escalated tới human support.

### 5.6 Cost optimization

Per-query cost breakdown (typical):
- Embedding query: ~$0.00002 (negligible).
- Vector DB query: ~$0.0001 - $0.001 (managed).
- Reranker: ~$0.001 (Cohere) hoặc $0 (self-host).
- LLM generation: ~$0.005 - $0.05 (dominant).
- **Total: ~$0.01 per query**.

Optimization:
1. **Prompt caching** (Anthropic Claude): cache system prompt + common context portions. -90% input cost.
2. **Semantic cache**: cache query → answer pairs. Hit rate có thể 20-40% trong support contexts.
3. **Smaller LLM cho easy queries**: use Haiku/GPT-4o-mini for simple Q, escalate to Sonnet/GPT-4 for complex.
4. **Truncate context**: chỉ pass relevant chunks, không full context.

---

## 6. Lessons learned & Best practices

### 6.1 Lesson 1: Demo RAG ≠ Production RAG

Phổ biến: team build demo "simple RAG" trong 1 ngày, deploy → users complain hallucinations + bad answers. Root cause:
- Naive chunking cắt vỡ context.
- Dense-only retrieval miss exact-match queries (product codes, names).
- Không có reranker → low precision.
- LLM hallucinate khi context không đủ.

**Solution**: design production RAG từ đầu với hybrid retrieval + reranker + grounding + evaluation framework. Đừng iterate từ "simple → complex" — sẽ tốn thời gian gấp 3.

### 6.2 Lesson 2: Chunking + retrieval > LLM choice

Empirical: thay LLM tốt hơn (GPT-3.5 → GPT-4) cho ~5% improvement. Improve chunking + hybrid retrieval cho **15-25%** improvement (theo nhiều internal reports + Pinecone benchmarks).

**Implication**: dành 70% effort vào ingestion + retrieval, 30% vào LLM/prompt.

### 6.3 Lesson 3: Hybrid retrieval is non-negotiable

Pure dense retrieval **luôn miss** một số query types (exact match, OOD terms). BM25 alone miss semantic queries. Hybrid là **default**, không phải optimization.

Anthropic Contextual Retrieval blog: dense alone có ~5.7% failure rate, hybrid (dense + BM25) giảm xuống 4.4%, contextual hybrid giảm xuống 2.9% (top-20 retrieval failures).

### 6.4 Lesson 4: Eval datasets là asset đắt giá nhất

Build evaluation dataset (50-200 question-answer pairs labeled by domain experts) là **single most valuable** investment. Cho phép:
- A/B test config changes objectively.
- Detect regression khi update embedding model, chunking, LLM.
- Convince stakeholders về improvements.

**Build process**:
1. Sample 100 real user queries từ production logs.
2. Domain expert label: golden answer + golden chunks (which docs should be retrieved).
3. Versioning: tag dataset v1.0, v1.1, ... — never delete questions.

### 6.5 Lesson 5: Observability từ ngày 1

Log mọi step:
- Query input + rewriting output.
- Retrieved chunks (IDs + scores).
- Reranker scores.
- Final prompt sent to LLM.
- LLM output.
- User feedback.

**Tools**: LangSmith (LangChain), Arize Phoenix, Helicone, Weights & Biases. Self-host: log to ClickHouse + Grafana.

Without observability, debug "why answer wrong?" là impossible at scale.

### 6.6 Lesson 6: Citations là feature, không phải nice-to-have

Users **không trust** RAG answers without citations. Citations:
- Build trust ("model said X, source confirms X").
- Allow user verification.
- Critical cho compliance/legal use cases.

Implementation: enforce LLM cite source IDs trong prompt, hyperlink trong UI. Anthropic Citation API (2024) provide structured citations natively.

### 6.7 Lesson 7: Update strategy matters

KB không tĩnh — docs add/update/delete liên tục. Production system cần:
- **Incremental indexing**: không re-embed entire KB mỗi đêm.
- **Document versioning**: track which version of doc was used for each answer.
- **Stale detection**: nếu user feedback negative trên answer citing old doc, flag for review.
- **Delete handling**: hard delete trong vector DB + BM25 index (tombstones không đủ).

### 6.8 Lesson 8: Privacy & access control

Nếu RAG over corporate docs, **mỗi user có thể chỉ xem 1 subset KB**. Solutions:
- **Metadata filtering at query time**: filter chunks by user permissions.
- **Row-level security** trong pgvector / Qdrant.
- **Per-user / per-team vector DB namespaces** (Pinecone namespaces).

Sai approach: filter post-retrieval — risk leak chunks vào LLM context.

### 6.9 Best practice — production checklist

- [ ] Chunking strategy match document type (markdown / PDF / code).
- [ ] Embedding model có context window đủ cho chunks.
- [ ] Hybrid retrieval (BM25 + dense + metadata).
- [ ] Reranker on top-50-100.
- [ ] Top 5-10 chunks vào LLM context (not more).
- [ ] Citations enforced trong prompt + UI.
- [ ] Evaluation dataset versioned.
- [ ] Observability/logging mỗi step.
- [ ] Access control filtering at query time.
- [ ] Incremental update pipeline.
- [ ] Semantic cache for common queries.
- [ ] Prompt caching (if Anthropic/OpenAI support).
- [ ] Latency SLA defined and measured (P50 < 1s, P99 < 3s).
- [ ] Cost monitoring per query.

### 6.10 Cross-references đến knowledge base

- LLM serving infrastructure underneath RAG: [S3-01 vLLM](S3-01_vllm_paged_attention_continuous_batching.md).
- Vector DB internals: S3-03 (planned).
- RAG vs Finetune decision: S2-05 (planned).
- Production ML monitoring (analogous patterns): [S4-01 Michelangelo](S4-01_uber_michelangelo_feature_store.md), S4-03 drift detection (planned).

---

## 7. References

### Foundational papers

1. **Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** — NeurIPS 2020. arXiv: [2005.11401](https://arxiv.org/abs/2005.11401). Original RAG paper, intro framework.

2. **Karpukhin et al. "Dense Passage Retrieval for Open-Domain Question Answering"** — EMNLP 2020. arXiv: [2004.04906](https://arxiv.org/abs/2004.04906). DPR, foundation cho dense retrieval.

3. **Robertson & Zaragoza "The Probabilistic Relevance Framework: BM25 and Beyond"** — 2009. Reference cho BM25.

4. **Liu et al. "Lost in the Middle: How Language Models Use Long Contexts"** — 2023, arXiv: [2307.03172](https://arxiv.org/abs/2307.03172). Why ordering chunks matters.

5. **Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels"** — 2022, arXiv: [2212.10496](https://arxiv.org/abs/2212.10496). HyDE method.

6. **Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation"** — 2023, arXiv: [2309.15217](https://arxiv.org/abs/2309.15217). Eval framework.

7. **Kusupati et al. "Matryoshka Representation Learning"** — NeurIPS 2022, arXiv: [2205.13147](https://arxiv.org/abs/2205.13147).

### Industry / production blogs

8. **Anthropic — "Introducing Contextual Retrieval"** (2024-09): [anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval). Contextual chunking, prompt caching, benchmark results.

9. **OpenAI Cookbook — RAG examples**: [github.com/openai/openai-cookbook](https://github.com/openai/openai-cookbook). Practical patterns.

10. **Pinecone Learning Center — RAG**: [pinecone.io/learn](https://www.pinecone.io/learn/). Series về chunking, hybrid, reranker.

11. **Weaviate Blog — Hybrid search, RAG patterns**: [weaviate.io/blog](https://weaviate.io/blog).

12. **LlamaIndex Documentation**: [docs.llamaindex.ai](https://docs.llamaindex.ai). Patterns cho production RAG (auto-merging, sentence window, multi-query).

13. **LangChain RAG cookbook**: [python.langchain.com/docs/use_cases/question_answering](https://python.langchain.com/docs/use_cases/question_answering).

### Tools / libraries

14. **LlamaIndex** — orchestration framework, mạnh về indexing patterns: [github.com/run-llama/llama_index](https://github.com/run-llama/llama_index).

15. **LangChain** — orchestration: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain).

16. **Cohere Rerank** — managed reranker API: [docs.cohere.com/docs/rerank](https://docs.cohere.com/docs/rerank).

17. **RAGAS** — evaluation: [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas).

18. **Unstructured** — document parsing: [github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured).

### Talks / videos

19. **Jerry Liu (LlamaIndex) — "Building Production-Grade LLM Apps"**: nhiều talks YouTube, search "Jerry Liu LlamaIndex production".

20. **Anthropic Engineering — Contextual Retrieval walkthrough**: search "Anthropic contextual retrieval" trên YouTube.

21. **Pinecone — "RAG Fundamentals" series**: YouTube channel của Pinecone.

### Benchmarks & datasets

22. **MS MARCO** — passage retrieval benchmark, dataset cho train/eval embedding & rerankers.

23. **BEIR benchmark** — Thakur et al. 2021, arXiv: [2104.08663](https://arxiv.org/abs/2104.08663). Zero-shot retrieval benchmark across 18 datasets.

24. **MTEB (Massive Text Embedding Benchmark)** — Muennighoff et al. 2022, arXiv: [2210.07316](https://arxiv.org/abs/2210.07316). Leaderboard cho embedding models.

---

> **Tóm tắt 1 dòng**: Production RAG = **hybrid retrieval (BM25 + dense + metadata)** + **smart chunking** (contextual hoặc hierarchical) + **reranker** + **grounded generation với citations** + **eval framework**. Demo RAG (chunk + cosine + LLM) không scale lên production — cần design hybrid + reranker từ ngày 1.
