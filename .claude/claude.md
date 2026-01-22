# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## File Location Standards

### Claude Documentation File
- **Standard Path**: `.claude/claude.md`
- **Always use this path** - do not check for existence or search alternative locations
- **If missing**: Create the `.claude/` directory and `claude.md` file at this standard location
- **Never search** for alternative locations like root `CLAUDE.md` or other paths

## Branch Naming Best Practices

**Use unique, descriptive branch names:**

1. **Include timestamp for uniqueness**: `docs/add-feature-1737543000`
2. **Be specific about the change**: `docs/database-schema-error-guidance`
3. **Check existing branches first**: `git branch -a`
4. **Avoid generic names that might conflict**

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents creating branches with similar names, leading to confusion and conflicts when multiple agents work on the repository simultaneously.
