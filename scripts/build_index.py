#!/usr/bin/env python3
"""Regenerate INDEX.md + docs/progress.md from case-study frontmatter + docs/planned.yaml.

Single source of truth: filesystem. Run after adding/promoting/deleting a case study.

Usage:
    python scripts/build_index.py                       # regenerate INDEX.md + docs/progress.md
    python scripts/build_index.py --check               # exit 1 if anything would change (CI mode)
    python scripts/build_index.py --check-duplicates    # exit 1 if any ID appears in > 1 source
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML not installed. Run: pip install pyyaml\n")
    sys.exit(2)

REPO = Path(__file__).resolve().parent.parent
CASE_STUDIES_DIR = REPO / "case-studies"
DRAFTS_DIR = REPO / "drafts"
PLANNED_FILE = REPO / "docs" / "planned.yaml"
INDEX_FILE = REPO / "INDEX.md"
PROGRESS_FILE = REPO / "docs" / "progress.md"

SCOPE_FOLDERS = {
    1: "01-foundations",
    2: "02-model-development",
    3: "03-modern-stack",
    4: "04-production",
}

SCOPE_INFO = {
    1: {
        "name": "AI/ML System Design Foundations",
        "intent": "hiểu cấu trúc end-to-end của một ML system production (data → training → serving → monitoring), nắm vững các thuật ngữ và kiến trúc cơ bản trước khi đào sâu vào component cụ thể.",
    },
    2: {
        "name": "Model Development & Training",
        "intent": "hiểu cách big tech chọn architecture, train mô hình ở scale, debug overfit/underfit, evaluation methodology. Focus vào model design trade-offs hơn là infra.",
    },
    3: {
        "name": "Modern Tech Stack (LLM / RAG / Agent / CV)",
        "intent": "làm chủ tech stack hiện đại — LLM serving, retrieval, agent frameworks, vector DB, OCR/CV pipelines.",
    },
    4: {
        "name": "Production AI Systems",
        "intent": "vận hành ML system ở production — scaling, latency, A/B test, drift, cost, GPU management.",
    },
}

DIFFICULTY_LABEL = {
    "foundational": "Foundational",
    "intermediate": "Intermediate",
    "intermediate-advanced": "Intermediate–Advanced",
    "advanced": "Advanced",
}

STATUS_ICON = {
    "done": "✅",
    "draft": "🚧",
    "planned": "📋",
}


@dataclass
class CaseStudy:
    id: str
    title: str
    scope: int
    scope_name: str
    difficulty: str
    status: str
    summary: str
    line_count: int = 0
    path: Path | None = None  # relative to repo root, None for planned
    tags: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)
    last_validated: str | None = None


def parse_frontmatter(md_path: Path) -> tuple[dict, str]:
    """Return (frontmatter dict, body)."""
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{md_path}: missing frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{md_path}: unterminated frontmatter")
    fm = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5:]
    return fm, body


def load_case_studies() -> list[CaseStudy]:
    cases: list[CaseStudy] = []

    # 1. Done + draft from case-studies/ and drafts/
    for root in [CASE_STUDIES_DIR, DRAFTS_DIR]:
        if not root.exists():
            continue
        for md in sorted(root.glob("**/*.md")):
            try:
                fm, body = parse_frontmatter(md)
            except ValueError as e:
                sys.stderr.write(f"WARN: {e}\n")
                continue
            line_count = len(md.read_text(encoding="utf-8").splitlines())
            cases.append(CaseStudy(
                id=fm["id"],
                title=fm["title"],
                scope=fm["scope"],
                scope_name=fm["scope_name"],
                difficulty=fm["difficulty"],
                status=fm["status"],
                summary=fm.get("summary", ""),
                line_count=line_count,
                path=md.relative_to(REPO),
                tags=fm.get("tags", []) or [],
                cross_refs=fm.get("cross_refs", []) or [],
                last_validated=fm.get("last_validated"),
            ))

    # 2. Planned from docs/planned.yaml
    if PLANNED_FILE.exists():
        data = yaml.safe_load(PLANNED_FILE.read_text(encoding="utf-8")) or {}
        for entry in data.get("planned", []):
            cases.append(CaseStudy(
                id=entry["id"],
                title=entry["title"],
                scope=entry["scope"],
                scope_name=entry["scope_name"],
                difficulty=entry["difficulty"],
                status="planned",
                summary=entry.get("summary", ""),
            ))

    return cases


def find_duplicates(cases: list[CaseStudy]) -> dict[str, list[CaseStudy]]:
    """Return {id: [cases]} for IDs appearing in > 1 source.

    Invariant: every case-study id must exist in EXACTLY ONE of {case-studies/, drafts/, planned.yaml}.
    Violations are usually caused by /new-case-study not pruning planned.yaml or a manual edit.
    """
    by_id: dict[str, list[CaseStudy]] = {}
    for c in cases:
        by_id.setdefault(c.id, []).append(c)
    return {k: v for k, v in by_id.items() if len(v) > 1}


def report_duplicates(dups: dict[str, list[CaseStudy]]) -> str:
    lines = ["ERROR: duplicate case-study IDs detected — INDEX would be ambiguous.\n"]
    for cid in sorted(dups):
        lines.append(f"  {cid}:")
        for c in dups[cid]:
            source = c.path.as_posix() if c.path else "docs/planned.yaml"
            lines.append(f"    - status={c.status:7s} source={source}")
    lines.append(
        "\nResolve: an id must exist in EXACTLY ONE source. Either delete the planned entry, "
        "the draft, or the done file. /new-case-study should remove the planned.yaml entry "
        "when scaffolding a draft; if it didn't, run /promote-draft or edit manually."
    )
    return "\n".join(lines)


def sort_key(c: CaseStudy):
    # Sort by id: SX-YY → (X, YY)
    m = re.match(r"S(\d+)-(\d+)", c.id)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 99)


def render_scope_table(cases: list[CaseStudy], scope: int) -> str:
    rows = ["| # | Case study | Difficulty | Mô tả |", "|---|---|---|---|"]
    for c in sorted([x for x in cases if x.scope == scope], key=sort_key):
        icon = STATUS_ICON.get(c.status, "❓")
        if c.path:
            title_cell = f"[{c.title}]({c.path.as_posix()})"
        else:
            title_cell = c.title
        diff = DIFFICULTY_LABEL.get(c.difficulty, c.difficulty)
        rows.append(f"| {icon} {c.id} | {title_cell} | {diff} | {c.summary} |")
    return "\n".join(rows)


def render_progress_stats(cases: list[CaseStudy]) -> str:
    total = len(cases)
    by_status = {s: sum(1 for c in cases if c.status == s) for s in ["done", "draft", "planned"]}
    return (
        f"- **Total planned**: {total} case studies\n"
        f"- **Completed**: {by_status['done']}\n"
        f"- **In progress (drafts)**: {by_status['draft']}\n"
        f"- **Planned**: {by_status['planned']}"
    )


def render_done_list(cases: list[CaseStudy]) -> str:
    done = sorted([c for c in cases if c.status == "done"], key=sort_key)
    lines = []
    for i, c in enumerate(done, 1):
        lines.append(f"{i}. ✅ {c.id} {c.title} — {c.summary}")
    return "\n".join(lines)


def render_index(cases: list[CaseStudy]) -> str:
    today = date.today().isoformat()
    sections = []
    sections.append(f"""# AI System Design Knowledge Base — Case Study Roadmap

> Knowledge base cho AI/ML engineer ôn luyện system design qua case studies thực tế của big tech.
> Phong cách viết: **bilingual VI-EN** (giữ nguyên technical terms tiếng Anh, diễn giải tiếng Việt).
> Depth target: **1500–2000 dòng/case study** (từ S1-03 trở đi) với ASCII diagrams, concrete numbers, pseudo-code, references thật.

<!-- AUTO-GENERATED by scripts/build_index.py — DO NOT EDIT MANUALLY -->
<!-- Last generated: {today} -->
<!-- To regenerate: `python scripts/build_index.py` or `/update-index` skill -->

---

## Cách dùng knowledge base này

1. **Đọc theo scope** nếu bạn muốn tập trung một mảng (foundations / model dev / modern stack / production).
2. **Đọc theo difficulty** (foundational → intermediate → advanced) nếu bạn muốn build kiến thức tuần tự.
3. **Mỗi file** đều có structure 7 sections cố định: Overview → Requirements → Architecture → Deep dive → Trade-offs → Lessons learned → References.
4. **Status icons**:
   - ✅ — đã viết, đã review
   - 🚧 — đang viết (draft)
   - 📋 — planned, chưa viết

---
""")

    for scope in (1, 2, 3, 4):
        info = SCOPE_INFO[scope]
        table = render_scope_table(cases, scope)
        sections.append(
            f"## Scope {scope} — {info['name']}\n\n"
            f"Mục tiêu: {info['intent']}\n\n"
            f"{table}\n"
        )

    progress = render_progress_stats(cases)
    done_list = render_done_list(cases)

    sections.append(f"""---

## Progress tracker

{progress}

**Đã viết (sorted theo id):**

{done_list}

---

## Conventions sử dụng trong knowledge base

- **Numbers / scale**: trích từ engineering blog hoặc paper, ghi rõ năm. Nếu không chắc → ghi "based on public information, internal numbers may differ".
- **Diagram**: ASCII art (preferred) hoặc Mermaid. Diagram phải show data flow, không chỉ static boxes.
- **Pseudo-code**: Python-flavored, có comment tiếng Việt giải thích intuition.
- **Comparison tables**: System | Approach | Pros | Cons | Use case.
- **References**: prioritize engineering blogs > arXiv papers > conference talks > textbook. Ghi rõ URL và năm publication.

## Project structure

```text
ai-sysdesign-knowledge/
├── README.md                          # Project overview (start here)
├── INDEX.md                           # This file — auto-generated roadmap
├── case-studies/                      # Published case studies
│   ├── 01-foundations/
│   ├── 02-model-development/
│   ├── 03-modern-stack/
│   └── 04-production/
├── drafts/                            # WIP case studies (status:draft)
├── proposals/                         # Topic proposals (researcher output)
├── templates/                         # Case-study skeleton
├── docs/                              # Style guide, terminology, planned.yaml
├── scripts/                           # build_index.py, validate_refs.py
└── .claude/                           # Agentic infra (agents, skills, hooks, MCP)
```

## Bilingual writing style

Xem [docs/style-guide.md](docs/style-guide.md) và [docs/terminology.md](docs/terminology.md).

**Ví dụ câu chuẩn**:

> "YouTube sử dụng kiến trúc two-tower model để xử lý candidate generation, trong đó user tower và item tower được train riêng biệt nhưng share chung embedding space."
""")

    return "\n".join(sections)


def render_progress(cases: list[CaseStudy]) -> str:
    today = date.today().isoformat()
    done = sorted([c for c in cases if c.status == "done"], key=sort_key)
    planned = sorted([c for c in cases if c.status == "planned"], key=sort_key)
    drafts = sorted([c for c in cases if c.status == "draft"], key=sort_key)

    total_lines = sum(c.line_count for c in done)

    out = [f"""# Progress Snapshot — AI System Design Knowledge Base

<!-- AUTO-GENERATED by scripts/build_index.py — DO NOT EDIT MANUALLY -->
<!-- Last generated: {today} -->

> Snapshot trạng thái. Để xem real-time, đọc [`../INDEX.md`](../INDEX.md).

## Scope (4 mảng)

- **Scope 1** — AI/ML System Design Foundations
- **Scope 2** — Model Development & Training
- **Scope 3** — Modern Tech Stack (LLM / RAG / Agent / CV)
- **Scope 4** — Production AI Systems

## File naming convention

`Sx-yy_<topic_slug>.md` — x = scope (1-4), yy = sequence (01, 02, …), slug = snake_case.

## Completed ({len(done)}) — case studies ✅
"""]
    out.append("| File | Scope | Title | Lines |")
    out.append("|---|---|---|---|")
    for c in done:
        out.append(f"| `{c.path.as_posix() if c.path else ''}` | {c.scope} | {c.title} | {c.line_count} |")
    out.append(f"\n**Total**: {total_lines} dòng across {len(done)} files.\n")

    if drafts:
        out.append(f"## Drafts ({len(drafts)}) 🚧\n")
        out.append("| ID | Title | Scope | Lines |")
        out.append("|---|---|---|---|")
        for c in drafts:
            out.append(f"| {c.id} | {c.title} | {c.scope} | {c.line_count} |")
        out.append("")

    out.append(f"## Planned ({len(planned)}) 📋\n")
    out.append("| ID | Title | Scope | Difficulty |")
    out.append("|---|---|---|---|")
    for c in planned:
        out.append(f"| {c.id} | {c.title} | {c.scope} | {DIFFICULTY_LABEL.get(c.difficulty, c.difficulty)} |")
    out.append("")

    out.append("""## Depth target (từ S1-03 trở đi)

Target line count = **1500-2000 dòng/case study** với:

1. **Mechanism deep-dive** — algorithm step-by-step, intuition, concrete examples.
2. **R&D evolution** — predecessor systems, paper lineage, alternatives đã thử và bị loại.
3. **Improvements over time** — failure modes, fixes, successor systems.
4. **Implementation depth** — đủ chi tiết để reader có thể tự build.

## Structure cố định mỗi case study

1. **Overview** — bối cảnh, business problem.
2. **System Requirements** — functional, non-functional với concrete numbers.
3. **High-level Architecture** — ASCII diagram + data flow.
4. **Deep dive các components chính.**
5. **Trade-offs & Design decisions** — comparison tables.
6. **Lessons learned & Best practices.**
7. **References** — papers, blogs, talks với URL.
""")
    return "\n".join(out)


def update_frontmatter_line_counts(cases: list[CaseStudy]) -> int:
    """Update line_count + last_validated stays untouched. Return # files changed."""
    changed = 0
    for c in cases:
        if not c.path:
            continue
        full = REPO / c.path
        text = full.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r"^line_count: \d+",
            f"line_count: {c.line_count}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n and new_text != text:
            full.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any output would change")
    ap.add_argument("--check-duplicates", action="store_true", help="exit 1 if any id appears in > 1 source")
    args = ap.parse_args()

    cases = load_case_studies()

    dups = find_duplicates(cases)
    if dups:
        sys.stderr.write(report_duplicates(dups) + "\n")
        sys.exit(1)

    if args.check_duplicates:
        print(f"OK: no duplicate ids across {len(cases)} entries.")
        return

    new_index = render_index(cases)
    new_progress = render_progress(cases)

    old_index = INDEX_FILE.read_text(encoding="utf-8") if INDEX_FILE.exists() else ""
    old_progress = PROGRESS_FILE.read_text(encoding="utf-8") if PROGRESS_FILE.exists() else ""

    drift = (new_index != old_index) or (new_progress != old_progress)

    if args.check:
        if drift:
            print("DRIFT: INDEX.md or docs/progress.md is out of sync. Run `python scripts/build_index.py`.")
            sys.exit(1)
        print("OK: INDEX + progress in sync.")
        return

    INDEX_FILE.write_text(new_index, encoding="utf-8")
    PROGRESS_FILE.write_text(new_progress, encoding="utf-8")
    changed = update_frontmatter_line_counts(cases)

    n_done = sum(1 for c in cases if c.status == "done")
    n_draft = sum(1 for c in cases if c.status == "draft")
    n_planned = sum(1 for c in cases if c.status == "planned")
    print(f"Built INDEX.md ({n_done} done, {n_draft} draft, {n_planned} planned)")
    print(f"Built docs/progress.md")
    print(f"Updated line_count in {changed} frontmatters")


if __name__ == "__main__":
    main()
