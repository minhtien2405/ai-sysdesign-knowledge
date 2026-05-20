#!/usr/bin/env python3
"""Validate case-study quality. Two modes:

    --quick  : frontmatter completeness + section headers + line count (fast, for hook)
    --full   : --quick + URL liveness check via HEAD (slow, for cron)

Usage:
    python scripts/validate_refs.py path/to/file.md           # quick by default
    python scripts/validate_refs.py path/to/file.md --full
    python scripts/validate_refs.py --all                     # validate everything

Exit code: 0 = pass, 1 = fail. Failures printed to stderr.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML not installed.\n")
    sys.exit(2)

REPO = Path(__file__).resolve().parent.parent

REQUIRED_FRONTMATTER = {"id", "title", "scope", "scope_name", "difficulty", "status"}
SECTION_PATTERN = re.compile(r"^## \d+\.\s+(.+)$", re.MULTILINE)
REFERENCES_PATTERN = re.compile(r"^##\s+(\d+\.\s+)?References\b", re.MULTILINE)
URL_PATTERN = re.compile(r"https?://[^\s\)\]]+")
# Note: depth-target check (≥1500 lines) is enforced by the case-study-reviewer agent
# on new drafts, NOT by this structural validator. Legacy files pre-restructure are
# grandfathered.


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"{path}: unterminated frontmatter")
    fm = yaml.safe_load(text[4:end]) or {}
    return fm, text[end + 5:]


def case_id_seq(case_id: str) -> int:
    """Return YY part of SX-YY."""
    m = re.match(r"S\d+-(\d+)", case_id or "")
    return int(m.group(1)) if m else 0


def validate_file(path: Path, *, full: bool = False) -> list[str]:
    """Return list of error messages (empty = pass)."""
    errors: list[str] = []
    try:
        fm, body = parse_frontmatter(path)
    except ValueError as e:
        return [str(e)]

    # Frontmatter completeness
    missing = REQUIRED_FRONTMATTER - set(fm)
    if missing:
        errors.append(f"missing frontmatter fields: {sorted(missing)}")

    if fm.get("status") not in {"planned", "draft", "done"}:
        errors.append(f"invalid status: {fm.get('status')!r}")

    # Section headers — must have all 7 numbered sections (only enforced for done/draft)
    if fm.get("status") in {"draft", "done"}:
        section_titles = SECTION_PATTERN.findall(body)
        if len(section_titles) < 7:
            errors.append(f"only {len(section_titles)} sections found, expected ≥7")

    # Must have a References section + at least one URL
    if fm.get("status") in {"draft", "done"}:
        if not REFERENCES_PATTERN.search(body):
            errors.append("missing References section (## References or ## N. References)")
        urls = URL_PATTERN.findall(body)
        if not urls:
            errors.append("no URLs found in body — references required")

    # Full mode: check URL liveness
    if full and fm.get("status") in {"draft", "done"}:
        try:
            import urllib.request
            urls = list({u.rstrip(".,)") for u in URL_PATTERN.findall(body)})
            for url in urls:
                try:
                    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
                    urllib.request.urlopen(req, timeout=10)
                except Exception as e:
                    errors.append(f"dead URL: {url} ({type(e).__name__})")
        except Exception as e:
            errors.append(f"URL check failed: {e}")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="Path to a single case study .md")
    ap.add_argument("--all", action="store_true", help="Validate all case studies + drafts")
    ap.add_argument("--full", action="store_true", help="Also check URL liveness (slow)")
    ap.add_argument("--quick", action="store_true", help="Frontmatter + sections only (default)")
    args = ap.parse_args()

    if not args.path and not args.all:
        ap.error("provide a path or --all")

    files: list[Path]
    if args.all:
        files = sorted((REPO / "case-studies").glob("**/*.md")) + sorted((REPO / "drafts").glob("**/*.md"))
    else:
        files = [Path(args.path).resolve()]

    total_errors = 0
    for f in files:
        errors = validate_file(f, full=args.full)
        if errors:
            print(f"FAIL  {f.relative_to(REPO) if f.is_relative_to(REPO) else f}", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            total_errors += len(errors)
        else:
            print(f"OK    {f.relative_to(REPO) if f.is_relative_to(REPO) else f}")

    if total_errors:
        print(f"\n{total_errors} issue(s) found.", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(files)} file(s) passed.")


if __name__ == "__main__":
    main()
