---
name: lint-bilingual
description: Quick lint of a case study or draft for bilingual VI-EN consistency. Checks for over-translated technical terms (e.g. "vector nhúng" instead of "embedding"), missing acronym expansions, and reference quality. Fast (no LLM call needed — pure grep). Use whenever user asks to "lint", "check terminology", or after editing a case study.
---

# lint-bilingual skill

Static check for bilingual style violations. Pure grep, no LLM reasoning.

## Inputs

- **path** (required): path to a `.md` file in `case-studies/` or `drafts/`.

## Steps

1. **Existence check**:
   - File must exist and be under `case-studies/` or `drafts/`
   - Else error

2. **Over-translation check** — grep for forbidden Vietnamese phrases:

   ```bash
   grep -nE 'vector nhúng|hai tháp|bộ đệm khoá|bộ nhớ đệm khoá-giá|huấn luyện trước|mô hình ngôn ngữ lớn|kho đặc trưng|cửa hàng đặc trưng|tăng cường bằng truy xuất|sinh tăng cường|độ trễ phân vị' <file>
   ```

   For each match, print: `<line>: forbidden phrase "<phrase>" → use "<correct EN>" instead`

   Mapping:
   - "vector nhúng" → "embedding"
   - "hai tháp" → "two-tower"
   - "bộ đệm khoá" / "bộ nhớ đệm khoá-giá" → "KV cache"
   - "huấn luyện trước" → "pretraining"
   - "mô hình ngôn ngữ lớn" → "LLM"
   - "kho đặc trưng" / "cửa hàng đặc trưng" → "feature store"
   - "tăng cường bằng truy xuất" / "sinh tăng cường" → "RAG"
   - "độ trễ phân vị" → "P99 latency"

3. **Acronym first-use check**:

   Use Python to find first occurrences of these acronyms and verify they are expanded (followed by `(...)` within 50 chars):

   `CTR, CVR, AUC, NDCG, QPS, P99, TTFT, TPOT, MMoE, FM, GNN, GCN, RAG, HNSW, IVF-PQ, BM25, RRF, MMR, CUPED, SRM, MDE, MAB, HTE, ATE, FDR, PSI`

   Skip if file is `S1-01` (foundations, allow assumed knowledge).

   For each acronym that appears WITHOUT expansion on its first occurrence:
   `<line>: acronym <X> not expanded on first use — write "<X> (<full form>)"`

4. **Frontmatter quick check**:

   Just call `python scripts/validate_refs.py <file>` and pipe its output. This already covers frontmatter + structure + line count.

5. **Output format**:

   ```
   # lint-bilingual: <path>

   ## Over-translation: <N> issues
   <list>

   ## Acronym expansion: <N> issues
   <list>

   ## validate_refs.py: <PASS|FAIL>
   <verbatim output>

   Verdict: CLEAN | NEEDS FIXES (<total> issues)
   ```

## Notes

- This is a STATIC linter — does not modify the file.
- Acronym detection ignores code blocks (skip lines inside ```...```).
- If the file is < 100 lines, skip the acronym check (likely a stub).
- Exit 0 if clean, 1 if any issues — but in this skill context, just print and let the user decide.
