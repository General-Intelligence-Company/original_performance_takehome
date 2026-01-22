# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## Branch Naming Best Practices

**Use unique, descriptive branch names:**

1. **Include timestamp for uniqueness**: `docs/add-feature-1737543000`
2. **Be specific about the change**: `docs/database-schema-error-guidance`
3. **Check existing branches first**: `git branch -a`
4. **Avoid generic names that might conflict**

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents creating branches with similar names, leading to confusion and conflicts when multiple agents work on the repository simultaneously.

## Session Analysis Requirements

**Include complete conversation data when analyzing agent sessions:**

1. **Full conversation messages**: Include complete messages, not just titles
2. **Error preservation**: Capture error messages and correction attempts
3. **Command output**: Document command outputs and failure cases
4. **Interaction flow**: Show complete multi-turn conversation sequences

**Why this matters**: Pattern analysis detected 2 occurrences (90% confidence) of sessions being logged with only titles instead of full conversation messages, preventing identification of repeated mistakes and creation of better guidance documentation.
