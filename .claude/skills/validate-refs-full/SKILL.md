---
name: validate-refs-full
description: Network-validates all URLs and references in one or all case studies. Slower than the auto-hook (which only checks structure). Spawns the reference-validator agent. Use when user says "/validate-refs-full <id>", "check all URLs", "freshness audit", or for quarterly maintenance.
---

# validate-refs-full skill

Spawn the `reference-validator` agent to hit-the-network check all references.

## Inputs

- **target** (required, parse from user prompt): single id ("S1-01") or literal "all".

## Steps

1. **Validate target**:
   - If id: confirm it exists via `kb_get_case_study(<id>)`
   - If "all": no validation needed

2. **Spawn reference-validator agent**:
   - Pass target argument
   - Allow up to 5 minutes runtime for "all" (multiple cases × network round-trips)

3. **After agent returns**:
   - Print the report verbatim
   - Suggest: if dead URLs found, manually fix and re-run

## Tools used

- Agent (subagent_type: reference-validator)
- kb-mcp (for pre-flight check on single-id target)

## Hard constraints

- Do NOT auto-fix dead URLs — too easy to wrong-replace. Human reviews + edits.
- Do NOT run `scripts/validate_refs.py --full` directly — the agent does that internally with proper rate limiting.
