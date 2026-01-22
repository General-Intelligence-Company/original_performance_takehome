# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## Branch Naming Best Practices

**Use unique, descriptive branch names:**

1. **Include timestamp for uniqueness**: `docs/add-feature-1737543000`
2. **Be specific about the change**: `docs/database-schema-error-guidance`
3. **Check existing branches first**: `git branch -a`
4. **Avoid generic names that might conflict**

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents creating branches with similar names, leading to confusion and conflicts when multiple agents work on the repository simultaneously.

## GitHub API Rate Limits

1. **Check rate limit status** - Use `gh api rate_limit` to check current limits before operations
2. **Wait if rate limited** - If you hit limits, wait for the reset time shown in the response
3. **Use authenticated requests** - Ensure you're using authenticated GitHub CLI for higher limits
4. **Implement backoff** - Use exponential backoff for retries on rate limit errors

**Common Error**: '429: Rate limit exceeded' - Wait for reset time before retrying

**Why this matters**: Pattern analysis detected 2 occurrences (80% confidence) of agents not checking rate limits before making GitHub API calls, leading to failures and blocked operations.
