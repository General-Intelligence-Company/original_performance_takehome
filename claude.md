# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## ⚠️ CRITICAL: Always Pull Before Push

**This is the #1 cause of push failures.** Always pull latest changes before any git operations.

**Before any git operations:**
1. **Always pull latest**: `git pull origin main`
2. Check status: `git status`
3. If push fails with 'Updates were rejected': `git pull --rebase origin main`
4. Use `git push --force-with-lease` after rebase if needed
5. Never use `git push --force` unless absolutely necessary

**Common Issue**: Push failures usually mean you haven't pulled the latest changes from remote.

**Why this matters**: Other agents may have pushed changes while you were working. Always sync with remote before pushing to avoid conflicts.

## ⚠️ CRITICAL: Database Prerequisites

**Before any database operations:**
1. Verify database connection and schema are properly configured
2. **Known Issue**: If you encounter `column users.onboarding_integration_step does not exist`:
   - This is a system-level database schema issue
   - Contact system administrators for database migration
   - Do NOT attempt to fix the schema yourself
   - This error prevents agent execution entirely until resolved
3. Wait for infrastructure team to resolve schema mismatches

**Why this matters**: Database schema issues are infrastructure problems that cannot be resolved at the agent level and require admin intervention.

## GitHub API Rate Limits

**Before any GitHub operations:**
1. Check rate limit status: `gh api rate_limit` or `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit`
2. If rate limited (HTTP 403), use git commands directly instead of GitHub CLI
3. For repository operations, prefer `git clone` over GitHub API calls
4. Wait for rate limit reset if GitHub API is required

**Fallback when rate limited:**
- Use `git` commands instead of `gh` commands
- Clone via HTTPS: `git clone https://github.com/owner/repo.git`
- Create branches and commits locally, push when rate limit resets

**Error Pattern**: If you see 'HTTP 403: API rate limit exceeded', pause operations and retry later.

**Why this matters**: Pattern analysis detected 9 occurrences (95% confidence) of agents failing due to GitHub API rate limit exceeded errors when trying to perform operations.

## File Naming Conventions

**Before creating documentation files**:
1. Check existing files: `ls *.md`
2. Use consistent case - this repository uses lowercase: `claude.md`, `agents.md`
3. Never create both `CLAUDE.md` and `claude.md` - use lowercase only
4. If file exists, read it first: `cat claude.md`

**Common mistake**: Creating `CLAUDE.md` when `claude.md` already exists
**Solution**: Always use lowercase and check existing files first

### Why This Matters

This prevents creating duplicate files with different casing, which causes confusion and content fragmentation when multiple agents work on the repository.

## File Naming and Case Sensitivity

**Important**: This repository uses `claude.md` (lowercase) as the canonical documentation file.

**Before editing documentation:**
1. Check existing files: `ls -la | grep -i claude` or `git ls-files | grep -i claude`
2. Always use `claude.md` (lowercase) as the main documentation file
3. If both `claude.md` and `CLAUDE.md` exist, work with the lowercase version
4. File systems are case-sensitive - always use the correct case

**Why this matters**: Inconsistent file naming creates confusion about which file is authoritative.

## Pre-work Checklist for Documentation Updates

Before modifying `claude.md` or other documentation:

1. Run `git fetch origin main` to get latest changes
2. Run `git pull origin main` if on main branch
3. Run `gh pr list` to check for open PRs that might conflict
4. Only then create your feature branch
5. When conflicts occur, read both versions carefully and preserve all valuable content

This prevents merge conflicts and reduces manual conflict resolution work.

## Database Prerequisites

**Before any database operations:**
1. Verify database connection: Check that DATABASE_URL is set and accessible
2. Test connectivity: `psql $DATABASE_URL -c 'SELECT 1'` or equivalent
3. Verify schema: Ensure required tables and columns exist before proceeding
4. Common error: 'column users.onboarding_integration_step does not exist' indicates incomplete schema
5. Do not proceed with code changes until database is properly configured

**Always test database queries before implementing features that depend on them.**

### Quick Reference

| Do | Don't |
|----|-------|
| `ls *.md` before creating | Create files without checking |
| Use `claude.md` (lowercase) | Use `CLAUDE.md` |
| Read existing content first | Overwrite without reading |
| Merge content if duplicates exist | Keep duplicate files |

## Repository Exploration Checklist

Before starting any task, explore the repository to understand its current state:

1. **List all files and directories**: `ls -la`
2. **Check all branches**: `git branch -a`
3. **Review recent commits**: `git log --oneline -5`
4. **Find existing documentation**: `ls *.md`
5. **Understand the current state** before making changes

**Why**: Prevents duplicate work and ensures awareness of existing code/documentation.

## File Investigation

**Before assuming file naming issues or creating documentation for problems**:

1. **Check what files actually exist**: `ls -la | grep -i filename`
2. **Search case-insensitively**: `find . -iname "*filename*" -type f`
3. **Don't create documentation for problems that don't exist**: Verify the actual state before proposing solutions
4. **Read the file before assuming its contents**: `cat filename` or `head -50 filename`

**Common mistake**: Creating fixes or documentation for issues (like "wrong file casing") without first verifying the issue exists.

**Why this matters**: Agents often waste time creating PRs and documentation to fix problems that don't exist, because they assumed an issue without investigating first.

## Repository Assessment Checklist

Before creating branches or making changes:

1. **Check repository permissions**: `git remote -v` and test with small commit
2. **Examine existing branches**: `git branch -a`
3. **Review recent activity**: `git log --oneline -5`
4. **Check for existing similar branches** to avoid duplication
5. **Understand the workflow**: Are there existing PRs? What's the branch naming convention?
6. **Verify you can push** to the intended branch

**Why**: Pattern analysis detected 7 occurrences (85% confidence) of agents creating branches and pushing without verifying repository permissions or understanding branch structure.

## GitHub API Rate Limits

**Before using GitHub CLI or API operations:**
1. Check current rate limit status: `gh api rate_limit`
2. If rate limited, use fallback approach:
   - Use local git operations instead of `gh` commands
   - Push branch manually: `git push origin branch-name`
   - Create branches and commits locally, push when API is available
   - Create PR via GitHub web interface if needed
   - For urgent changes, use manual git push instead of `gh pr create`
   - Wait for rate limit reset (typically within an hour)
3. For bulk operations, space out API calls to avoid hitting limits

**Error pattern**: If you see 'HTTP 403: API rate limit exceeded', immediately switch to local git workflow.

**If you encounter HTTP 403 'API rate limit exceeded':**
- Stop making GitHub API calls immediately
- Use git CLI commands instead: `git push`, `git pull`, `git checkout`
- For PR operations, use `gh` CLI with authentication or manual git workflows
- Wait before retrying API operations
- Consider using `git push --set-upstream origin branch-name` instead of API calls

**Fallback commands when rate limited:**
| Instead of | Use |
|------------|-----|
| `gh pr create` | `git push -u origin branch && # create PR via web UI` |
| `gh pr list` | `git branch -r` to see remote branches |
| `gh pr view` | `git log origin/main..HEAD` to see your commits |
| `gh api` calls | Direct git commands or wait for rate limit reset |

**Why this matters**: Pattern analysis detected 7 occurrences (95% confidence) of agents failing due to GitHub API rate limit exceeded errors. Using git CLI as a fallback allows work to continue.

## Environment Setup

Before running the Claude Code agent:

1. **Ensure database is properly initialized**: Verify all required tables exist
2. **Run all database migrations**: Check that all migrations have been applied
3. **Ensure environment variables are set**: Verify all necessary configuration is in place

**Note**: For database schema errors (e.g., `column users.onboarding_integration_step does not exist`), see the **"⚠️ CRITICAL: Database Prerequisites"** section at the top of this file.

## Before Adding Documentation

**Always check for existing content first:**

1. **Read the entire target file** to understand current structure: `cat claude.md` or `git show main:claude.md`
2. **Check section headings**: `grep '^##' claude.md`
3. **Search for similar existing sections** using grep: `grep -i "keyword1\|keyword2" claude.md`
4. **Check recent commits** for related changes: `git log --oneline -5 -- claude.md`
5. **Check for open PRs** that might add similar content: `gh pr list --search "docs" --state=all --limit=10`
6. **If similar content exists**, enhance the existing section instead of creating a duplicate
7. **Only add new sections** when content is genuinely missing
8. **Use unique branch names** to avoid conflicts with concurrent work (e.g., `docs/add-X-guidelines-username`)
9. **Use consistent formatting**: Match existing markdown style and section hierarchy

**Common sections that already exist** (check before adding):
- Git workflow, branch management, merge conflicts
- File naming and management
- PR creation and merging
- Testing guidelines
- Troubleshooting (database, rate limits)

**Why this matters**: Creating PRs for documentation that already exists wastes review time and can cause merge conflicts. Multiple agents working concurrently may have already added similar content.

## Before Requesting Documentation Changes

**Always verify current state first:**
1. Read the target file completely: `cat claude.md` or `cat AGENTS.md`
2. Search for existing similar content: `grep -i "keyword" filename`
3. Check recent commits: `git log --oneline -10 -- filename`
4. Only request new sections if they don't already exist or need significant updates

**Avoid**: Requesting duplicate sections that create redundant or conflicting documentation.

## Experimental Code Management

When making experimental changes to this codebase:

1. **Use version control, not file copies**: Make changes directly and use git for version control
2. **Use git branches for different approaches**: Instead of creating multiple files like `file_v2.py`, create feature branches
3. **Commit working versions before experiments**: Always `git commit` a working state before trying major changes
4. **If creating experimental files**: Use a consistent naming pattern (e.g., `*_experimental.py`)
5. **Clean up promptly**: Remove experimental files when done: `rm -f *_experimental.py`

This prevents accumulation of files like `perf_takehome_interleaved.py`, `perf_takehome_roundmajor.py`, etc.

## Before Any Git Operations

1. **Always pull latest changes first**: `git pull origin main`
2. **If push fails with 'remote has moved'**: Run `git pull --rebase` then push again
3. **Common Issue**: Push failures occur when other agents have pushed commits while you were working

This prevents merge conflicts and ensures you're working with the latest code.

## Merge Conflict Prevention and Resolution

**Prevention (Always do before starting work):**
1. `git fetch origin main` - Get latest changes
2. `git show main:filename` - Check if target files exist and their current state
3. Read existing content to understand structure before making changes

**When conflicts occur:**
1. `git status` - See which files have conflicts
2. Open conflicted files and look for `<<<<<<<`, `=======`, `>>>>>>>` markers
3. Manually edit to combine desired content from both sides
4. Remove all conflict markers completely
5. `git add filename` - Mark as resolved
6. `git rebase --continue` or `git merge --continue`
7. Verify final content is coherent

## Git Workflow

**Before creating any branch or making changes:**

1. Run `git fetch origin main` to get latest changes
2. Check if target files already exist: `git show main:filename`
3. If the file exists, read its contents to understand current structure
4. Plan your changes to complement existing content
5. Only then create your feature branch

**For documentation files (claude.md, README.md):** Always assume they may have been updated by others. Merge conflicts in documentation are common and preventable.

**Before creating branches or pushing:**

1. Always run `git pull origin main` to get latest changes
2. Check `git status` to ensure clean working directory
3. List existing branches: `git branch -a`
4. Check if you have push permissions: `git remote -v`
5. Only then create feature branches or push changes
6. Commit changes locally before attempting to push: `git add . && git commit -m "description"`
7. If push fails due to remote changes, use `git pull --rebase` to resolve

**Common Issue**: Push failures due to new commits on main that weren't pulled first.

**Important**: Do not automatically create branches like `cofounder/optimization-v1` or push to origin without first verifying repository permissions and existing branch structure.

## Git Conflict Resolution

When encountering merge conflicts:
1. Run `git status` to see conflicted files
2. Open conflicted files and look for `<<<<<<<`, `=======`, `>>>>>>>` markers
3. Manually edit to keep desired content from both sides
4. Remove all conflict markers
5. Run `git add <filename>` to mark as resolved
6. Continue with `git rebase --continue` or `git merge --continue`
7. Always verify the final merged content is coherent before continuing

## After Resolving Merge Conflicts

**Always verify merged content before continuing:**
1. Read the entire merged file: `cat filename`
2. Check that sections flow logically and don't contradict each other
3. Look for duplicate or redundant information that should be consolidated
4. Ensure all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) are removed
5. Test that any commands or examples in the documentation still work

**Why**: Mechanical conflict resolution can create inconsistent or duplicate content. Always review the final result as a coherent whole.

## Merge Conflict Resolution Best Practices

**When resolving conflicts:**
1. Read both versions of conflicted content carefully
2. Preserve ALL valuable information from both sides
3. Merge complementary sections intelligently rather than choosing one side
4. After resolving, read the entire merged file: `cat filename`
5. Ensure sections flow logically and don't contradict each other
6. Remove all conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
7. Test that any commands or examples still work

**Why this matters**: Mechanical conflict resolution creates inconsistent or duplicate content.

## Branch Management

**Before creating branches:**

1. Check repository permissions: Can you push to this repo?
2. Review existing branches: `git branch -a`
3. Understand the branching strategy from existing PRs
4. Ensure you have the latest main: `git pull origin main`
5. Only create branches after understanding the codebase and requirements

## Git Workflow for Documentation

When modifying documentation files (especially shared ones like this file), follow these steps to avoid merge conflicts:

1. Always run `git pull origin main` before creating branches
2. Check if the target file already exists: `git show main:filename`
3. If file exists, read current content first
4. Plan to merge/extend existing content rather than overwrite
5. When resolving merge conflicts, preserve all useful sections from both versions

**Note**: Multiple agents may work on documentation simultaneously. Always assume `claude.md` or `AGENTS.md` may have been modified since you last checked.

## Handling Merge Conflicts

When encountering merge conflicts:

1. Run `git pull origin main --rebase` to rebase your changes
2. If conflicts occur, edit the conflicted files to resolve them
3. Look for conflict markers `<<<<<<<`, `=======`, `>>>>>>>`
4. Keep the best parts from both versions
5. Run `git add <filename>` to mark conflicts as resolved
6. Run `git rebase --continue` to complete the rebase
7. Use `git push --force-with-lease` to update your branch safely

## Safe Force Pushing

When you need to force push after a rebase, always use `git push --force-with-lease` instead of `git push --force`. This prevents accidentally overwriting commits that were pushed by others since your last fetch.

## When PR Merges Fail

If GitHub API returns 405 errors or PRs show 'dirty' mergeable state:

1. **Check PR status**: `gh pr view <number> --json mergeable,mergeStateStatus`
2. **If conflicts exist, rebase the branch**: `git fetch origin main && git rebase origin/main`
3. **Resolve any conflicts** following the conflict resolution steps above
4. **Force push safely**: `git push --force-with-lease origin <branch>`
5. **Only push directly to main as a last resort** when PR workflow is completely broken

**Important**: Never push directly to main just because PR creation or merging failed. Always exhaust the PR workflow options first.

## Testing

**Before running tests**:
1. Check `README.md` for specific test instructions
2. Look for existing test files: `ls tests/` or `find . -name '*test*.py'`
3. Check for requirements: `ls requirements*.txt` or `pyproject.toml`

**Standard test command for this repo**: `python tests/submission_tests.py`

If tests fail, check for:
- Missing dependencies
- Wrong Python version
- Required environment setup

- Always examine test files first: `ls tests/` and `head tests/*.py`
- Check for specific test runners or requirements
- Verify test commands work before documenting them
- Look for existing test documentation in README.md

Before running tests, always verify the test setup:

1. **Check that test files exist**: `ls tests/`
2. **Run the test suite**: `python tests/submission_tests.py`
3. **If direct execution fails**, try running from the repository root or importing programmatically:
   ```bash
   cd /path/to/repo && python tests/submission_tests.py
   ```
4. **Alternative approach** if module imports fail:
   ```bash
   python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/submission_tests.py').read())"
   ```
5. **Verify test output**: Tests report cycle counts and speedup metrics. A successful run shows `CYCLES: <number>` and test pass/fail status.

The test file `tests/submission_tests.py` uses Python's unittest framework and tests both correctness and performance against various benchmarks.

## Git Workflow Best Practices

Before creating branches or pushing changes:

1. **Always fetch latest main**: `git fetch origin main`
2. **Check if main has new commits**: `git log --oneline origin/main..HEAD`
3. **If main has moved, rebase your branch**: `git rebase origin/main`
4. **Resolve any conflicts before pushing**
5. **Use `--force-with-lease` for force pushes**: This avoids overwriting others' work

**Why this matters**: Pushing without checking if main has moved can cause merge conflicts and force other collaborators to resolve issues that could have been handled locally.

## Git Push Best Practices

**Before pushing any branch**:
1. `git fetch origin main` - Check for remote updates
2. `git pull origin main --rebase` - Get latest changes
3. Resolve any conflicts that arise
4. Then push your branch

**If push fails with 'Updates were rejected'**:
1. `git pull origin main --rebase`
2. Resolve conflicts if any
3. `git push origin BRANCH_NAME`
4. For force push: use `--force-with-lease` for safety

## PR Creation Best Practices

**After creating any PR:**

1. **Verify mergeable status**: `gh pr view <pr-number> --json mergeable,mergeStateStatus`
2. **Check for conflicts**: Ensure `mergeStateStatus` is `CLEAN` not `DIRTY`
3. **If conflicts exist**: Rebase with `git rebase origin/main` and force push
4. **Always test locally** before marking complete

**Important**: Do not consider a PR task complete until it shows as mergeable without conflicts.

## PR Merging Best Practices

**When GitHub API merge fails (405 Method Not Allowed)**:

This often happens when PRs have conflicts or the repository has merge restrictions.

1. **Try the GitHub CLI first**: `gh pr merge PR_NUMBER --merge`
2. **If that fails, fallback to manual merge**:
   ```bash
   git checkout main && git pull origin main && git merge BRANCH_NAME && git push origin main
   ```
3. **If conflicts exist, rebase first**:
   ```bash
   git checkout BRANCH_NAME
   git rebase origin/main
   # Resolve any conflicts
   git checkout main && git pull origin main && git merge BRANCH_NAME && git push origin main
   ```
4. **Always use `--force-with-lease` for force pushes** to avoid overwriting others' work
5. **Verify merge was successful**: `git log --oneline -3`
6. **Clean up feature branch** after successful merge: `git branch -d BRANCH_NAME`

**Prevention**:
- Check PR mergeable status before attempting merge: `gh pr view PR_NUMBER --json mergeable,mergeStateStatus`
- Ensure branch is up to date with main before creating PR
- Rebase against main if the PR shows conflicts

## Performance Measurement

To get consistent performance measurements:

1. Use a standardized test script: `python -c "from problem import *; # run your test here"`
2. Run measurements multiple times to check for consistency
3. Clear any cached state between runs if needed
4. Document your current best result and the method used to achieve it
5. Always verify correctness before measuring performance

## Additional File Management Notes

- See "File Naming Conventions" section at the top of this file for the canonical guidance
- Always verify file case sensitivity before creating new documentation files

## Merge Conflict Resolution

When encountering merge conflicts in `claude.md` or other documentation files:

1. **Read the entire conflicted file** to understand both versions
2. **Preserve all existing sections** and documentation content
3. **Add new content as additional sections** rather than replacing existing content
4. **Test that the merged content is well-formatted** before completing the merge
5. **Complete the resolution**: Use `git add <file>` and `git rebase --continue` (or `git merge --continue`) to finalize

## File Management

**Before creating or modifying claude.md:**
1. Check if the file already exists: `git ls-files claude.md`
2. Check for similar files with different cases: `find . -name "*claude*" -type f`
3. If it exists, read the current content first
4. Always merge new content with existing content rather than overwriting
5. Use `git status` to check for uncommitted changes before making modifications
6. Follow existing naming conventions in the repository

### Before Creating Documentation Files
1. **Always check if the file exists first**: `ls claude.md` or `cat claude.md`
2. **If file exists**: Read it completely, then append or merge your changes
3. **If creating new file**: Use lowercase `claude.md` (not `CLAUDE.md`)
4. **When merging content**: Preserve existing sections and add new sections in logical order
5. **After creating/updating**: Verify the final content with `cat claude.md`

## Git Workflow Verification

**Before any git operations:**

1. Run `git status` to check current state
2. Run `git branch -a` to see all existing branches
3. Check repository permissions and settings
4. Verify you're on the correct base branch before creating feature branches
5. Use descriptive branch names that follow project conventions

## Working with Shared Documentation

Before modifying `claude.md` or other shared docs:

1. Run `git fetch origin main` to get latest changes
2. Check for open PRs: `gh pr list | grep claude.md`
3. If other PRs are modifying the same file, coordinate or wait
4. When multiple agents work simultaneously, expect merge conflicts
5. Always read existing content before adding new sections
6. Use descriptive branch names to indicate what you're adding

## Multi-Agent Documentation Workflow

When multiple agents may be working on documentation simultaneously:

1. **Check file existence**: `ls claude.md` before creating/editing
2. **Read current content**: `cat claude.md` to understand existing structure
3. **Plan additions**: Identify where new content fits logically
4. **Merge, don't replace**: Preserve existing sections when adding new ones
5. **Expect conflicts**: Multiple agents may edit documentation simultaneously

**Note**: Always assume `claude.md` may have been modified since you last checked.

## Database Schema Issues

**Critical Error: `column users.onboarding_integration_step does not exist`**

This is a known infrastructure issue that prevents agent execution.

**Required Actions:**
1. STOP all operations immediately
2. This is NOT a code issue - it's a database schema/migration issue
3. Contact system administrators or development team
4. Do NOT attempt to fix database schema or run migrations
5. Wait for infrastructure team to resolve before proceeding with any tasks

**This error blocks all agent functionality and requires admin intervention.**

## Database Connection Issues

### System-Level Errors

If you encounter database errors like 'column users.onboarding_integration_step does not exist':

1. **This is a system-level database schema issue requiring human intervention**
2. **Do NOT attempt to continue with the task** - this will fail
3. **Report the error immediately**: 'Database schema issue detected - requires human intervention'
4. The error typically indicates missing database migrations or schema updates
5. Contact the development team to resolve the database schema before proceeding

### Database Requirements

Before running Claude Code agent:

1. **Ensure database is properly initialized**: Verify the database exists and has the required schema
2. **Check that all required columns exist**: Particularly in the `users` table
3. **Run schema migrations**: If encountering 'column does not exist' errors, run pending migrations first
4. **Verify database connection string**: Ensure the connection configuration is correct

### Troubleshooting Database Errors

If you encounter database connection errors (such as 'column does not exist' or similar schema errors):

1. **Check if the database is properly initialized**: Verify the database exists and has been set up correctly
2. **Verify schema migrations are up to date**: Run any pending migrations before attempting database operations
3. **Consider using offline mode**: If database operations are not essential for the current task, proceed without them
4. **Report issues immediately**: Do not attempt to proceed with tasks that require database access if connections are failing

**Common Error**: `column users.onboarding_integration_step does not exist`

If you see this error:
- This indicates missing database schema or outdated migrations
- Contact repository maintainer for proper setup instructions
- **Do not proceed with code changes until database is properly configured**
- Running code with an improperly configured database will lead to cascading errors

**Why this matters**: Attempting to continue with broken database connections leads to cascading errors and wasted effort.

### Handling Database Schema Errors

**If you encounter database schema errors:**

1. Errors like `column users.onboarding_integration_step does not exist` indicate missing database migrations
2. This is a system-level issue that prevents agent execution
3. Do not attempt to continue with the task - report the error and terminate gracefully
4. These errors require human developer intervention to fix database schema

**Why**: These are infrastructure issues outside the agent's scope of control.

## Before Requesting Documentation Updates

**Always check existing content first:**
1. Read current file: `cat claude.md` or view on GitHub
2. Check recent commits: `git log --oneline -10 main -- claude.md`
3. Review open/recent PRs: `gh pr list --search "docs"`
4. Search for similar content: `grep -i "keyword" claude.md`
5. If similar content exists, enhance it rather than creating duplicates

**Common topics already covered:**
- Git workflow and pull-before-push guidance
- Merge conflict prevention and resolution
- Documentation best practices
- File naming conventions
- Branch management
- PR creation/merging best practices
- GitHub API rate limits

**Why this matters**: Multiple coding agent sessions often add similar documentation sections with overlapping content. By checking first, you prevent redundant PRs and reduce maintenance burden.

## GitHub API Rate Limits

**Critical**: If you encounter 'HTTP 403: API rate limit exceeded' errors:
1. Stop using GitHub CLI commands (gh) immediately
2. Wait 60+ seconds before retrying API operations (use exponential backoff)
3. Use git CLI commands instead:
   - Use `git push origin branch-name` instead of `gh pr create`
   - Use `git log --oneline` instead of `gh pr list`
   - Use `git merge` for merging operations
4. For PR creation, push the branch first, then create PR manually when API is available
5. Always check rate limit status before starting: `gh api rate_limit` or `gh auth status`
6. Always include request ID and timestamp when reporting rate limit errors

**Why this matters**: GitHub API has rate limits that can cause agent failures, but most operations can be completed using local git commands as fallbacks.

## GitHub API Rate Limit Handling

**Before any GitHub operations:**
1. Check rate limit status: `gh api rate_limit`
2. If rate limited, wait 60+ seconds before retrying
3. Use alternative approaches when API unavailable:
   - Use `git` commands instead of `gh` CLI
   - Perform operations locally and sync when API is available
   - Document changes for manual PR creation
4. Always include request ID in error reports for GitHub support

**Common Error**: HTTP 403 API rate limit exceeded means you need to wait before retrying operations.

## GitHub API Rate Limits - Quick Reference

**Before using GitHub CLI commands:**
1. Check API status: `gh api rate_limit`
2. If rate limited, use fallback approaches:
   - Push with git: `git push origin branch-name`
   - Create PR manually via web interface
   - Wait for rate limit reset (usually within an hour)

**Common error patterns:**
- `API rate limit exceeded` - Use manual git operations
- `HTTP 403` errors - Usually indicates rate limiting
