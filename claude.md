# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

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

## Git Workflow

**Before creating any branch or making changes:**

1. Run `git fetch origin main` to get latest changes
2. Check if target files already exist: `git show main:filename`
3. If the file exists, read its contents to understand current structure
4. Plan your changes to complement existing content
5. Only then create your feature branch

**For documentation files (claude.md, README.md):** Always assume they may have been updated by others. Merge conflicts in documentation are common and preventable.

**Before creating branches or pushing:**

1. Check current status: `git status`
2. List existing branches: `git branch -a`
3. Check if you have push permissions: `git remote -v`
4. Work on the main branch first, only create feature branches if explicitly needed
5. Commit changes locally before attempting to push: `git add . && git commit -m "description"`

**Important**: Do not automatically create branches like `cofounder/optimization-v1` or push to origin without first verifying repository permissions and existing branch structure.

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

## Testing

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

## Performance Measurement

To get consistent performance measurements:

1. Use a standardized test script: `python -c "from problem import *; # run your test here"`
2. Run measurements multiple times to check for consistency
3. Clear any cached state between runs if needed
4. Document your current best result and the method used to achieve it
5. Always verify correctness before measuring performance

## File Naming Conventions

When creating or updating documentation files:

1. **Use lowercase `claude.md`** for agent guidelines - NOT `CLAUDE.md`
2. **Check if the file already exists** before creating a new one: `ls -la | grep -i filename`
3. **If unsure about naming**, check existing files in the repository root
4. **Always use exact case sensitivity** when working with files

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
