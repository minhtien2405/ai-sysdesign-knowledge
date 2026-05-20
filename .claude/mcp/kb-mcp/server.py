"""kb-mcp — Knowledge Base MCP server for agentic-aisys-wiki repo.

Exposes 6 tools that let Claude Code agents query the wiki state in-context:

    kb_get_terminology(en_term)       — VI mapping for a tech term
    kb_list_case_studies(...)         — list case studies with filters
    kb_get_case_study(id)             — fetch frontmatter + body
    kb_find_cross_refs(topic)         — keyword search across studies
    kb_propose_next_topic(scope?)     — gap analysis vs docs/planned.yaml
    kb_validate_refs(id, full?)       — run scripts/validate_refs.py

Repo root is inferred from this file's location (3 dirs up: .claude/mcp/kb-mcp/).

Run with:
    uv run --directory .claude/mcp/kb-mcp python server.py
or (after `uv sync`):
    python server.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_STUDIES_DIR = REPO_ROOT / "case-studies"
DRAFTS_DIR = REPO_ROOT / "drafts"
PLANNED_FILE = REPO_ROOT / "docs" / "planned.yaml"
TERMINOLOGY_FILE = REPO_ROOT / "docs" / "terminology.md"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_refs.py"

mcp = FastMCP("kb-mcp")


# ============================================================
# Helpers — frontmatter + terminology parsing
# ============================================================

def _parse_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter")
    fm = yaml.safe_load(text[4:end]) or {}
    return fm, text[end + 5:]


def _case_id_sort(c: dict) -> tuple[int, int]:
    m = re.match(r"S(\d+)-(\d+)", c.get("id", ""))
    return (int(m.group(1)), int(m.group(2))) if m else (99, 99)


def _load_all_cases() -> list[dict]:
    cases: list[dict] = []
    for root in (CASE_STUDIES_DIR, DRAFTS_DIR):
        if not root.exists():
            continue
        for md in root.glob("**/*.md"):
            try:
                fm, _ = _parse_frontmatter(md)
            except ValueError:
                continue
            cases.append({
                "id": fm.get("id"),
                "title": fm.get("title"),
                "scope": fm.get("scope"),
                "scope_name": fm.get("scope_name"),
                "difficulty": fm.get("difficulty"),
                "status": fm.get("status"),
                "summary": fm.get("summary", ""),
                "tags": fm.get("tags", []) or [],
                "cross_refs": fm.get("cross_refs", []) or [],
                "line_count": fm.get("line_count", 0),
                "path": str(md.relative_to(REPO_ROOT)),
            })
    if PLANNED_FILE.exists():
        data = yaml.safe_load(PLANNED_FILE.read_text(encoding="utf-8")) or {}
        for entry in data.get("planned", []):
            cases.append({
                "id": entry.get("id"),
                "title": entry.get("title"),
                "scope": entry.get("scope"),
                "scope_name": entry.get("scope_name"),
                "difficulty": entry.get("difficulty"),
                "status": "planned",
                "summary": entry.get("summary", ""),
                "tags": [],
                "cross_refs": [],
                "line_count": 0,
                "path": None,
            })
    return sorted(cases, key=_case_id_sort)


_TERM_CACHE: list[dict] | None = None


def _load_terminology() -> list[dict]:
    """Parse docs/terminology.md tables. Returns list of {en, vi, notes, category}."""
    global _TERM_CACHE
    if _TERM_CACHE is not None:
        return _TERM_CACHE

    text = TERMINOLOGY_FILE.read_text(encoding="utf-8") if TERMINOLOGY_FILE.exists() else ""
    entries: list[dict] = []
    category = "general"

    in_table = False
    sep_seen = False
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):
            category = line[3:].strip()
            in_table = False
            sep_seen = False
            continue
        if line.startswith("| EN") or line.startswith("|EN"):
            in_table = True
            sep_seen = False
            continue
        if in_table and re.match(r"^\|\s*-+\s*\|", line):
            sep_seen = True
            continue
        if in_table and sep_seen and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 2:
                entries.append({
                    "en": parts[0],
                    "vi": parts[1] if len(parts) > 1 else "",
                    "notes": parts[2] if len(parts) > 2 else "",
                    "category": category,
                })
        elif not line:
            # blank line — keep table state but if we hit another non-table line, exit
            continue
        elif not line.startswith("|"):
            in_table = False
            sep_seen = False

    _TERM_CACHE = entries
    return entries


# ============================================================
# MCP tools
# ============================================================

@mcp.tool()
def kb_get_terminology(en_term: str) -> dict:
    """Look up the VI translation/explanation for an English technical term.

    Performs case-insensitive substring match against the EN column of
    docs/terminology.md. Use this BEFORE writing a new term in a case study
    to ensure consistency with the established glossary.

    Args:
        en_term: English term to look up (e.g. "embedding", "two-tower", "PagedAttention").

    Returns:
        {"matches": [{"en", "vi", "notes", "category"}, ...], "query": "..."}.
        Empty matches list if not in glossary — propose adding it.
    """
    term_lower = en_term.lower().strip()
    matches = [
        e for e in _load_terminology()
        if term_lower in e["en"].lower()
    ]
    return {"query": en_term, "matches": matches}


@mcp.tool()
def kb_list_case_studies(scope: int | None = None, status: str | None = None) -> dict:
    """List case studies in the knowledge base with optional filters.

    Args:
        scope: filter by scope (1-4). None = all scopes.
        status: filter by lifecycle status ("done", "draft", "planned"). None = all.

    Returns:
        {"count": N, "cases": [{id, title, scope, ...}, ...]} sorted by id.
    """
    cases = _load_all_cases()
    if scope is not None:
        cases = [c for c in cases if c.get("scope") == scope]
    if status is not None:
        cases = [c for c in cases if c.get("status") == status]
    return {"count": len(cases), "cases": cases}


@mcp.tool()
def kb_get_case_study(case_id: str) -> dict:
    """Fetch a case study's full content + frontmatter by id.

    Args:
        case_id: e.g. "S1-01".

    Returns:
        {"id", "frontmatter", "body", "path", "line_count", "found": bool}.
        If not found: {"found": false, "id": case_id}.
    """
    for root in (CASE_STUDIES_DIR, DRAFTS_DIR):
        for md in root.glob(f"**/{case_id}_*.md"):
            try:
                fm, body = _parse_frontmatter(md)
            except ValueError as e:
                return {"found": False, "id": case_id, "error": str(e)}
            return {
                "found": True,
                "id": case_id,
                "frontmatter": fm,
                "body": body,
                "path": str(md.relative_to(REPO_ROOT)),
                "line_count": len(md.read_text(encoding="utf-8").splitlines()),
            }
    return {"found": False, "id": case_id}


@mcp.tool()
def kb_find_cross_refs(topic: str, limit: int = 5) -> dict:
    """Find existing case studies relevant to a topic — for auto-cross-linking.

    Keyword-based scoring across title (×3), tags (×2), summary (×1).
    Use this BEFORE writing about a concept to check if a related case study
    already exists and should be linked.

    Args:
        topic: free-form text (e.g. "graph neural network", "LLM serving").
        limit: max number of matches (default 5).

    Returns:
        {"query": "...", "matches": [{id, title, score, matched_field, summary}, ...]}.
    """
    tokens = [t.lower() for t in re.split(r"\W+", topic) if len(t) > 2]
    if not tokens:
        return {"query": topic, "matches": []}

    scored: list[tuple[int, str, dict]] = []
    for c in _load_all_cases():
        title = (c.get("title") or "").lower()
        tags = " ".join(c.get("tags") or []).lower()
        summary = (c.get("summary") or "").lower()
        score = 0
        matched_field = []
        for tok in tokens:
            if tok in title:
                score += 3
                matched_field.append("title")
            if tok in tags:
                score += 2
                matched_field.append("tags")
            if tok in summary:
                score += 1
                matched_field.append("summary")
        if score > 0:
            scored.append((score, ",".join(sorted(set(matched_field))), c))

    scored.sort(key=lambda x: (-x[0], _case_id_sort(x[2])))
    matches = [
        {
            "id": c["id"],
            "title": c["title"],
            "score": score,
            "matched_field": matched,
            "summary": c.get("summary", ""),
            "status": c.get("status"),
        }
        for score, matched, c in scored[:limit]
    ]
    return {"query": topic, "matches": matches}


@mcp.tool()
def kb_propose_next_topic(scope: int | None = None) -> dict:
    """List planned case studies (📋) that have not been written yet.

    Use this to pick a topic for the next case study, optionally filtered by scope.

    Args:
        scope: filter to a specific scope (1-4). None = all scopes.

    Returns:
        {"count": N, "proposals": [{id, title, scope, difficulty, summary}, ...]}.
    """
    planned = [c for c in _load_all_cases() if c.get("status") == "planned"]
    if scope is not None:
        planned = [c for c in planned if c.get("scope") == scope]
    return {
        "count": len(planned),
        "proposals": [
            {
                "id": c["id"],
                "title": c["title"],
                "scope": c["scope"],
                "scope_name": c["scope_name"],
                "difficulty": c["difficulty"],
                "summary": c["summary"],
            }
            for c in planned
        ],
    }


@mcp.tool()
def kb_validate_refs(case_id: str, full: bool = False) -> dict:
    """Run scripts/validate_refs.py on a case study and return the result.

    Args:
        case_id: e.g. "S1-01".
        full: if True, also check URL liveness (slow, ~seconds per URL).

    Returns:
        {"passed": bool, "output": str, "stderr": str, "exit_code": int, "case_id": str}.
    """
    info = kb_get_case_study(case_id)
    if not info.get("found"):
        return {"passed": False, "output": "", "stderr": f"case {case_id} not found", "exit_code": 2, "case_id": case_id}

    path = REPO_ROOT / info["path"]
    cmd = [sys.executable, str(VALIDATE_SCRIPT), str(path)]
    if full:
        cmd.append("--full")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return {
        "passed": proc.returncode == 0,
        "output": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
        "case_id": case_id,
    }


# ============================================================
# Entrypoint
# ============================================================

def main() -> None:
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
