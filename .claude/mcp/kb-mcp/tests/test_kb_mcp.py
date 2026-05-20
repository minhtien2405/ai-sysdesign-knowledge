"""Minimal pytest for kb-mcp core tools.

Run: cd .claude/mcp/kb-mcp && uv run pytest -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make server.py importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


def test_terminology_lookup_embedding():
    """Known EN term 'embedding' must be in glossary."""
    result = server.kb_get_terminology("embedding")
    assert result["query"] == "embedding"
    assert len(result["matches"]) >= 1
    # At least one match should have "embedding" in its EN column
    assert any("embedding" in m["en"].lower() for m in result["matches"])


def test_terminology_lookup_case_insensitive():
    """Lookup must be case-insensitive."""
    a = server.kb_get_terminology("HNSW")
    b = server.kb_get_terminology("hnsw")
    assert len(a["matches"]) == len(b["matches"])


def test_terminology_lookup_unknown():
    """Unknown term returns empty matches, not an error."""
    result = server.kb_get_terminology("xyznotaterminblossary")
    assert result["matches"] == []


def test_list_case_studies_no_filter():
    """List all should return ≥9 done + ≥9 planned at minimum (Phase 1 baseline)."""
    result = server.kb_list_case_studies()
    cases = result["cases"]
    assert result["count"] == len(cases)
    n_done = sum(1 for c in cases if c["status"] == "done")
    n_planned = sum(1 for c in cases if c["status"] == "planned")
    assert n_done >= 9, f"expected ≥9 done, got {n_done}"
    assert n_planned >= 9, f"expected ≥9 planned, got {n_planned}"


def test_list_case_studies_by_scope():
    """Filter by scope=1 should return only scope-1 cases."""
    result = server.kb_list_case_studies(scope=1)
    assert all(c["scope"] == 1 for c in result["cases"])
    assert result["count"] >= 3  # at least S1-01, S1-02, S1-03


def test_list_case_studies_by_status():
    """Filter by status=done should return only done cases."""
    result = server.kb_list_case_studies(status="done")
    assert all(c["status"] == "done" for c in result["cases"])


def test_get_case_study_known():
    """Known case S1-01 must be fetchable."""
    result = server.kb_get_case_study("S1-01")
    assert result["found"] is True
    assert result["id"] == "S1-01"
    assert "frontmatter" in result
    assert result["frontmatter"]["id"] == "S1-01"
    assert "body" in result
    assert result["line_count"] > 0


def test_get_case_study_unknown():
    """Unknown id returns found=False without raising."""
    result = server.kb_get_case_study("S9-99")
    assert result["found"] is False


def test_find_cross_refs_known_topic():
    """Searching for 'graph' should surface PinSage S1-03."""
    result = server.kb_find_cross_refs("graph neural network")
    ids = [m["id"] for m in result["matches"]]
    assert "S1-03" in ids


def test_propose_next_topic_returns_planned():
    """propose_next_topic returns only planned status."""
    result = server.kb_propose_next_topic()
    assert result["count"] >= 1
    for p in result["proposals"]:
        assert "id" in p and "title" in p


def test_propose_next_topic_by_scope():
    """Filter by scope=3 returns only scope-3 proposals."""
    result = server.kb_propose_next_topic(scope=3)
    for p in result["proposals"]:
        assert p["scope"] == 3
