# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## Branch Naming Best Practices

**Use unique, descriptive branch names:**

1. **Include timestamp for uniqueness**: `docs/add-feature-1737543000`
2. **Be specific about the change**: `docs/database-schema-error-guidance`
3. **Check existing branches first**: `git branch -a`
4. **Avoid generic names that might conflict**

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents creating branches with similar names, leading to confusion and conflicts when multiple agents work on the repository simultaneously.

## E2B Sandbox Limits

If you encounter '429: Rate limit exceeded... maximum number of concurrent E2B sandboxes (600)' errors:
1. This indicates the E2B service is at capacity
2. Wait and retry after a few minutes
3. Contact E2B support if the issue persists: https://e2b.dev/docs/getting-help
4. Consider using alternative approaches that don't require sandbox execution
