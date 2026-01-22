# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## Branch Naming Best Practices

**Use unique, descriptive branch names:**

1. **Include timestamp for uniqueness**: `docs/add-feature-1737543000`
2. **Be specific about the change**: `docs/database-schema-error-guidance`
3. **Check existing branches first**: `git branch -a`
4. **Avoid generic names that might conflict**

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents creating branches with similar names, leading to confusion and conflicts when multiple agents work on the repository simultaneously.

## Session Documentation Requirements

When documenting agent sessions for analysis:

1. **Include full conversation messages**, not just titles
2. **Preserve error messages and correction attempts**
3. **Document command outputs and failure cases**
4. **Show complete multi-turn conversation flows**

**Why this matters**: This enables proper identification of repeated mistakes and creation of better guidance documentation. Pattern analysis detected 2 occurrences (85% confidence) of agents logging session titles only without full conversation content.
