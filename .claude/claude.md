# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## Branch Naming Best Practices

**Before creating a new branch:**
1. Check existing branches: `git branch -r | grep <prefix>/`
2. Use unique branch names with timestamps for common tasks: `docs/topic-$(date +%s)`
3. If branch already exists, check if you can reuse it: `git checkout existing-branch`

**Common branch naming conflicts:**
- `docs/add-database-prerequisites`
- `docs/github-rate-limits`
- `docs/prevent-duplicate-*`

Use descriptive, unique names to avoid conflicts.

**Why this matters**: Multiple agents working concurrently often create branches with identical names, causing 422 errors when pushing. Using timestamps ensures uniqueness.
