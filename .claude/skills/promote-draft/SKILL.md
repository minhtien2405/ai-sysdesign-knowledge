---
name: promote-draft
description: Promote a draft case study from drafts/ to case-studies/<scope-folder>/. Runs the case-study-reviewer agent first; only promotes if review passes. Then regenerates INDEX, updates frontmatter status to done, and suggests cross-ref backfill. Use when the user says "promote", "publish draft", or "/promote-draft <id>".
---

# promote-draft skill

Move a draft from `drafts/` into `case-studies/<scope-folder>/` after passing review.

## Inputs

- **id** (required): e.g. `S3-03`. Or full path to draft.

## Steps

1. **Locate draft**:
   - Try `drafts/S<id>_*.md` — must match exactly one file
   - If no match, error: "No draft found for `<id>`"
   - If multiple matches, error: "Multiple drafts found for `<id>` — be specific"

2. **Run review** (BLOCKING gate):
   - Invoke `case-study-reviewer` subagent with the draft path
   - If verdict is FAIL → print review report verbatim, STOP. Do NOT promote.
   - If verdict is PASS → continue

3. **Determine target folder** from frontmatter `scope`:
   - 1 → `case-studies/01-foundations/`
   - 2 → `case-studies/02-model-development/`
   - 3 → `case-studies/03-modern-stack/`
   - 4 → `case-studies/04-production/`

4. **Update frontmatter** in the draft file:
   - `status: draft` → `status: done`
   - `last_validated: <today>`

5. **Move file**: `mv drafts/S<id>_<slug>.md case-studies/<folder>/S<id>_<slug>.md`

6. **Remove from `docs/planned.yaml`**: delete the entry where `id == <id>`. If not present, skip silently.

7. **Regenerate INDEX**:
   - Run `python scripts/build_index.py`
   - Capture output

8. **Print summary**:
   ```
   Promoted S<id> → case-studies/<folder>/S<id>_<slug>.md
   Review: PASS
   INDEX regenerated (X done, Y draft, Z planned).
   Suggested next: run cross-ref-finder agent to backfill links in older case studies.
   ```

9. **DO NOT commit**. User commits explicitly.

## Failure modes — explicit handling

- Review FAIL → do not modify any file, print review report, exit
- Build fails → print error, leave promoted file in place but warn user INDEX may be stale
- `docs/planned.yaml` write fails → warn but don't roll back the move

## Tools used

- Bash (mv, python)
- Read, Edit (for frontmatter update + planned.yaml update)
- Agent (subagent_type: case-study-reviewer)
