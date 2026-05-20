---
name: propose-topic
description: Research and propose new case study topic(s) for the ai-sysdesign-knowledge repo. Spawns the topic-researcher agent which mines arXiv + engineering blogs and writes proposal files in proposals/ folder. Use when user says "propose topic", "research topic ideas", "/propose-topic <scope>", or wants to grow the planned backlog with fresh research.
---

# propose-topic skill

Spawn the `topic-researcher` agent to mine sources + write proposal files.

## Inputs

- **scope** (optional, parse from user prompt): 1 | 2 | 3 | 4
- **focus_hint** (optional, free text): e.g. "agentic AI", "drift detection", "LLM cost"

## Steps

1. **Confirm scope** if not given:
   - Show user kb_propose_next_topic() result first so they see what's already planned
   - Ask: "Bạn muốn nghiên cứu thêm cho scope nào (1-4), hoặc cứ broad?"

2. **Spawn topic-researcher agent** with:
   - scope (or "any" if broad)
   - focus_hint (or none)
   - Explicit instruction: produce 1-3 proposal files in `proposals/`

3. **After agent returns**:
   - List the new files in `proposals/`
   - Show user: "N proposals written. Review each → if approve, run `/new-case-study <id>` to scaffold draft."
   - Do NOT auto-promote anything to planned.yaml or drafts/.

## Tools used

- Agent (subagent_type: topic-researcher)
- Bash (ls proposals/ for verification)

## Hard constraints

- Do NOT spawn writer agent in this skill. Researcher produces proposals only.
- Do NOT modify planned.yaml automatically. Human reviews proposal then decides whether to promote.
