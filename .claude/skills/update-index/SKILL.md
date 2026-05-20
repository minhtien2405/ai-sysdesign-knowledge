---
name: update-index
description: Regenerate INDEX.md and docs/progress.md from the filesystem (case-studies/ + drafts/ + docs/planned.yaml). Use whenever a case study is added, promoted, or deleted, or when the user says "update index" / "refresh index" / "rebuild INDEX".
---

# update-index skill

Run the build script that regenerates INDEX.md and docs/progress.md from frontmatter.

## Steps

1. Run: `python scripts/build_index.py`
2. Capture stdout — it reports counts of done / draft / planned + line_count updates.
3. Run: `git status --short -- INDEX.md docs/progress.md case-studies/` to show what changed.
4. Print the script output + a one-line summary.

## Drift check mode

If user asks "is INDEX up to date?" or "check INDEX drift", instead run:

```bash
python scripts/build_index.py --check
```

This exits 1 if anything would change. Report verbatim.

## Notes

- This skill should be IDEMPOTENT — running twice with no changes should be a no-op.
- Do NOT commit the changes automatically. User decides when to commit.
- If `pip install pyyaml` is missing (script exits with code 2), tell the user to install PyYAML.
