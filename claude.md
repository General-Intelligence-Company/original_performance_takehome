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

## File Naming Conventions

**Before creating documentation files:**

1. Check existing files: `ls *.md` and `find . -name "*.md"`
2. Use consistent casing - if `claude.md` exists, don't create `CLAUDE.md`
3. Follow the repository's established patterns
4. When in doubt, use lowercase for documentation files

This prevents file casing inconsistencies that can cause confusion and duplicate content.

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
