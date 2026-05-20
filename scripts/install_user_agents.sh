#!/usr/bin/env bash
# OPTIONAL — Sync agent definitions from project-scope `.claude/agents/` into
# user-scope `~/.claude/agents/` so they're available across ALL your Claude
# Code projects (not just this repo).
#
# When you need this:
#   - You want `@ai-sysdesign-knowledge-writer` to work in OTHER projects too.
#   - You're maintaining multiple sysdesign repos and want shared agents.
#
# When you DON'T need this:
#   - You only use these agents inside this repo. Open Claude Code AT the repo
#     root and project-scope `.claude/agents/` will load automatically.
#
# Common pitfall: if you open Claude Code at a PARENT folder (e.g. `/projects/`
# instead of `/projects/agentic-aisys-wiki/`), project-scope `.claude/`
# is not discovered — but in that case use this script OR open at the right
# folder, don't mix both.
#
# Usage:
#   ./scripts/install_user_agents.sh              # copy all
#   ./scripts/install_user_agents.sh --dry        # preview only
#   ./scripts/install_user_agents.sh writer       # copy only ai-sysdesign-knowledge-writer
#
# Idempotent — safe to re-run after updating in-repo agents.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/.claude/agents"
DST="$HOME/.claude/agents"

DRY_RUN=0
FILTER=""
for arg in "$@"; do
  case "$arg" in
    --dry) DRY_RUN=1 ;;
    --help|-h)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) FILTER="$arg" ;;
  esac
done

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: $SRC not found. Run from a clone of agentic-aisys-wiki."
  exit 1
fi

mkdir -p "$DST"

echo "Syncing agents: $SRC → $DST"
[[ -n "$FILTER" ]] && echo "Filter: matches *$FILTER*"
echo

for f in "$SRC"/*.md; do
  name="$(basename "$f")"
  if [[ -n "$FILTER" ]] && [[ "$name" != *"$FILTER"* ]]; then
    continue
  fi
  target="$DST/$name"

  if [[ -f "$target" ]] && diff -q "$f" "$target" >/dev/null 2>&1; then
    echo "  ✓ $name (already synced)"
    continue
  fi

  if (( DRY_RUN )); then
    echo "  → $name (would copy)"
  else
    cp "$f" "$target"
    echo "  → $name (copied)"
  fi
done

echo
echo "Done. Restart Claude Code session to register changes."
