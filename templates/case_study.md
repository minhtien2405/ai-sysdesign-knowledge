---
# === Identity (immutable after creation) ===
id: SX-YY                              # SX = scope (1-4), YY = sequence (01, 02, …)
title: Descriptive Title Here
slug: descriptive_slug_snake_case      # matches filename suffix

# === Classification ===
scope: 0                               # 1=foundations, 2=model-dev, 3=modern-stack, 4=production
scope_name: foundations                # foundations | model-development | modern-stack | production
difficulty: foundational               # foundational | intermediate | advanced
tags:
  - tag-1
  - tag-2

# === Lifecycle ===
status: planned                        # planned | draft | done
created: 2026-05-20
last_validated: 2026-05-20             # auto-updated by validate-refs
line_count: 0                          # auto-updated by build_index

# === Knowledge graph ===
cross_refs: []                         # list of case study IDs, e.g. [S1-01, S2-01]
primary_sources:
  - type: paper                        # paper | blog | talk | book | docs
    title: ""
    url: ""
    year: 2024
    org: ""                            # author/org (e.g. "Google", "Naumov et al.")
---

# SX-YY — Title

> **Scope**: <Scope name> | **Difficulty**: <foundational|intermediate|advanced>
> **Tham chiếu chéo**: [SX-YY ...](../path/to.md)
> **Key insight (1-2 câu)**: …

---

## 1. Overview

Bối cảnh, business problem, tại sao quan trọng. 1-2 đoạn ngắn nhưng dense.

## 2. System Requirements

### Functional
- …

### Non-functional
- QPS target, latency budget (P50/P99), throughput, memory footprint.
- Concrete numbers — ghi rõ source/year.

### Constraints
- Hardware, cost, regulatory, latency hard limits.

## 3. High-level Architecture

```text
┌──────────────┐      ┌──────────────┐
│  Component A │ ───▶ │  Component B │
└──────────────┘      └──────────────┘
```

Data flow narrative — explain mỗi arrow nghĩa là gì.

## 4. Deep dive — <component / mechanism>

### 4.1 …

Algorithm step-by-step, intuition, math khi cần.

```python
def example():
    """Pseudo-code, comment tiếng Việt."""
    pass
```

### 4.2 R&D evolution

- Predecessor system X (year): tại sao thay thế
- Paper lineage: A → B → C
- Alternatives đã thử và bị loại

### 4.3 Improvements over time

- Failure mode 1 → fix
- Successor system Y

## 5. Trade-offs & Design decisions

| Option | Approach | Pros | Cons | Used when |
|---|---|---|---|---|
| A | … | … | … | … |
| B | … | … | … | … |

## 6. Lessons learned & Best practices

- Lesson 1 (với citation nếu lấy từ post-mortem)
- Lesson 2

## 7. References

### Papers
- Author et al. "Title" (Venue Year, arXiv:XXXX.XXXXX)

### Engineering blogs
- "Title" — Org Engineering Blog (Year), URL

### Talks
- "Title" — Conference Year, URL
