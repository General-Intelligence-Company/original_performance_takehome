# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

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

## Pre-work Checklist for Documentation Updates

Before modifying `claude.md` or other documentation:

1. Run `git fetch origin main` to get latest changes
2. Run `git pull origin main` if on main branch
3. Run `gh pr list` to check for open PRs that might conflict
4. Only then create your feature branch
5. When conflicts occur, read both versions carefully and preserve all valuable content

This prevents merge conflicts and reduces manual conflict resolution work.

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

## Before Adding Documentation

Before creating new documentation:
1. Check if `claude.md` or `AGENTS.md` already exists
2. Read existing content to understand current guidelines
3. Add to existing sections rather than creating duplicate guidance
4. Use consistent formatting and style with existing documentation

## Before Adding New Documentation Sections

**Critical**: Always verify content doesn't already exist on main before creating PRs.

1. **Check main branch state**: Run `git show main:claude.md` to see the current file content
2. **Search for existing sections**: Read through all headings and content to find similar topics
3. **Enhance existing content**: If similar content exists, improve it rather than adding new sections
4. **Only create new sections**: When the content is genuinely missing from the file

**Why this matters**: Creating PRs for documentation that already exists wastes review time and can cause merge conflicts. Multiple agents working concurrently may have already added similar content.

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

Always verify the final merged content is coherent before continuing.

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

## PR Creation Best Practices

**After creating any PR:**

1. **Verify mergeable status**: `gh pr view <pr-number> --json mergeable,mergeStateStatus`
2. **Check for conflicts**: Ensure `mergeStateStatus` is `CLEAN` not `DIRTY`
3. **If conflicts exist**: Rebase with `git rebase origin/main` and force push
4. **Always test locally** before marking complete

**Important**: Do not consider a PR task complete until it shows as mergeable without conflicts.

## PR Merging Best Practices

**When GitHub API merge fails (405 errors)**:
1. Try `gh pr merge PR_NUMBER --merge` first
2. If that fails, use git commands:
   - `git fetch origin main`
   - `git checkout main`
   - `git merge BRANCH_NAME`
   - `git push origin main`
3. Always verify merge was successful with `git log --oneline -3`
4. Clean up feature branch after successful merge

**Prevention**:
- Check PR mergeable status before attempting merge
- Ensure branch is up to date with main

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

Before modifying `claude.md` or other shared files:

1. **Pull latest changes**: Run `git pull origin main` to get the most recent version
2. **Check existing content**: Read the file's current content before making changes
3. **Append, don't replace**: If adding new sections, append to existing content rather than replacing
4. **Use descriptive branches**: Use branch names that indicate the type of documentation being added (e.g., `docs/feature-name`)

## Multi-Agent Documentation Workflow

When multiple agents may be working on documentation simultaneously:

1. **Check file existence**: `ls claude.md` before creating/editing
2. **Read current content**: `cat claude.md` to understand existing structure
3. **Plan additions**: Identify where new content fits logically
4. **Merge, don't replace**: Preserve existing sections when adding new ones
5. **Expect conflicts**: Multiple agents may edit documentation simultaneously

**Note**: Always assume `claude.md` may have been modified since you last checked.
