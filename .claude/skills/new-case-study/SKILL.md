---
name: new-case-study
description: Scaffold a new case study draft in drafts/ from templates/case_study.md. Use when the user asks to start a new case study (e.g. "viết case study về X", "/new-case-study S3-03"). Picks the next available SX-YY slot based on docs/planned.yaml + case-studies/, writes a skeleton file with YAML frontmatter pre-filled.
---

# new-case-study skill

Scaffold a new case study draft, pre-filled with frontmatter from `docs/planned.yaml`.

## Inputs (parse from user prompt)

- **id** (required): e.g. `S3-03`. If not given, ask the user which planned id to start.
- **slug** (optional): snake_case, ≤40 chars. Derive from title in `docs/planned.yaml` if not given.

## Steps

1. **Confirm slot is free in published / drafts**:
   - `ls case-studies/**/S<id>_*.md drafts/S<id>_*.md` — should return nothing
   - If a file exists, STOP and tell user the slot is taken
2. **Check `docs/planned.yaml` for the same id**:
   - If the id exists in `planned.yaml`, the scaffold would create a **duplicate** (both planned + draft) → `build_index.py` will abort. Resolve by asking the user via `AskUserQuestion`:
     - **"Ghi đè cái cũ" (claim slot, Recommended)** — proceed: scaffold the draft AND remove the planned.yaml entry, so the id transitions cleanly `planned → draft`.
     - **"Bỏ cái mới" (abort)** — STOP: do not scaffold; the id stays in `planned.yaml`. Tell the user the slot is still planned.
   - If the id is NOT in `planned.yaml`, the user is creating a brand-new topic — proceed and warn that the metadata (title/scope/difficulty/summary) must be supplied by the user since there's no planned entry to pre-fill from.
3. **Load metadata** from the `planned.yaml` entry (title, scope, scope_name, difficulty, summary)
4. **Derive filename**: `S<id>_<slug>.md` (slug from title, snake_case, drop punctuation, ≤40 chars)
5. **Read** `templates/case_study.md`
6. **Generate frontmatter** by replacing placeholders:
   - id → from input
   - title, scope, scope_name, difficulty, summary → from planned.yaml
   - slug → derived
   - status → `draft`
   - tags → empty list (user fills)
   - cross_refs → empty list (cross-ref-finder will populate later)
   - created → today (use `date +%Y-%m-%d`)
   - last_validated → today
   - line_count → 0
7. **Write** to `drafts/S<id>_<slug>.md`
8. **Remove the planned.yaml entry** for this id (if "Ghi đè cái cũ" was chosen in step 2). Use `Edit` to delete the entire `- id: S<id> ... summary: ...` block plus its trailing blank line. This is the invariant: an id exists in EXACTLY ONE of `case-studies/`, `drafts/`, `planned.yaml`.
9. **Run `python scripts/build_index.py --check-duplicates`** to confirm no drift remains. If it fails, surface the error and ask the user to inspect.
10. **Print**:
    ```
    Scaffolded drafts/S<id>_<slug>.md
    Removed planned.yaml entry for S<id> (slot transitioned planned → draft).
    Next: invoke @ai-sysdesign-knowledge-writer to fill the 7 sections (depth target 1500-2000 lines).
    ```

## Important

- DO NOT scaffold if `<id>` is already done or has a draft — block with clear message.
- DO NOT scaffold without resolving the planned.yaml overlap — that's how duplicate INDEX rows happen.
- Always use `draft` status. Promotion to `done` happens via `/promote-draft`.
- Do NOT modify INDEX.md — `build_index.py` will pick the new draft up automatically.
- Suggest the user run `/update-index` after the file appears (or trust the post-write hook to remind them).
