# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## Git Workflow

Before creating branches or pushing:

1. Check current status: `git status`
2. List existing branches: `git branch -a`
3. Check if you have push permissions: `git remote -v`
4. Work on the main branch first, only create feature branches if explicitly needed
5. Commit changes locally before attempting to push: `git add . && git commit -m "description"`

**Important**: Do not automatically create branches like `cofounder/optimization-v1` or push to origin without first verifying repository permissions and existing branch structure.

## Before Creating Documentation PRs

When creating or modifying documentation files (like `claude.md`), follow these steps to prevent merge conflicts:

1. Run `git fetch origin main` to get latest changes
2. Check if target file exists: `ls claude.md` or `git show main:claude.md`
3. If file exists, read current content before making changes
4. Consider rebasing branch against main before pushing

This prevents merge conflicts when multiple agents work on the same documentation files simultaneously.

## Testing

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

## Performance Measurement

To get consistent performance measurements:

1. Use a standardized test script: `python -c "from problem import *; # run your test here"`
2. Run measurements multiple times to check for consistency
3. Clear any cached state between runs if needed
4. Document your current best result and the method used to achieve it
5. Always verify correctness before measuring performance
