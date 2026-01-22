# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## File Location Standards

### Claude Documentation File
- **Standard Path**: `.claude/claude.md`
- **Always use this path** - do not check for existence or search alternative locations
- **If missing**: Create the `.claude/` directory and `claude.md` file at this standard location
- **Never search** for alternative locations like root `CLAUDE.md` or other paths

## File Operations

**Direct File Access**: When reading or updating files, attempt the operation directly rather than pre-checking existence with commands like `ls -la`, `cat`, or `find`. Tools like `Read` gracefully handle non-existent files by returning an error, which is more efficient than a separate existence check.

**Why this matters**: Pre-checking file existence before reading/editing is an anti-pattern that:
- Adds unnecessary overhead (two operations instead of one)
- Creates race conditions in concurrent environments
- Provides no benefit since file operation tools already handle missing files gracefully

**Do**:
- Directly read/edit files and handle any "file not found" errors
- Create `.claude/` directory and `claude.md` if they don't exist

**Don't**:
- Run `ls -la .claude/` to check if directory exists before reading
- Run `cat .claude/claude.md` to verify existence before updating
- Search for alternative locations when the standard path doesn't exist

## Branch Naming Best Practices

**Use unique, descriptive branch names:**

1. **Include timestamp for uniqueness**: `docs/add-feature-1737543000`
2. **Be specific about the change**: `docs/database-schema-error-guidance`
3. **Check existing branches first**: `git branch -a`
4. **Avoid generic names that might conflict**

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents creating branches with similar names, leading to confusion and conflicts when multiple agents work on the repository simultaneously.

## Pull Request Creation

When creating PRs:
1. Try `gh pr create` command first
2. If it fails due to permissions, provide manual PR creation link:
   `https://github.com/OWNER/REPO/pull/new/BRANCH_NAME`
3. Always include suggested PR title and description
4. Confirm the branch has been pushed before attempting PR creation

## GitHub API Rate Limits

1. **Check rate limit status** - Use `gh api rate_limit` to check current limits before operations
2. **Wait if rate limited** - If you hit limits, wait for the reset time shown in the response
3. **Use authenticated requests** - Ensure you're using authenticated GitHub CLI for higher limits
4. **Implement backoff** - Use exponential backoff for retries on rate limit errors

**Common Error**: '429: Rate limit exceeded' - Wait for reset time before retrying

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents not checking rate limits before making GitHub API calls, leading to failures and blocked operations.
