# Project Guidelines for Claude

## File Management

When making experimental changes to this codebase:

1. **Use version control, not file copies**: Make changes directly in the main file (`perf_takehome.py`) and use git for version control
2. **Use git branches for different approaches**: Instead of creating multiple files like `perf_takehome_v2.py`, create feature branches
3. **Commit working versions before experiments**: Always `git commit` a working state before trying major optimizations
4. **If creating experimental files**: Use a consistent naming pattern (e.g., `*_experimental.py`)
5. **Clean up promptly**: Remove experimental files when done: `rm -f *_experimental.py`

This prevents accumulation of files like `perf_takehome_interleaved.py`, `perf_takehome_roundmajor.py`, etc.

## Git Workflow for Documentation Updates

Before making changes to `claude.md` or `CLAUDE.md`:

1. **Fetch latest changes**: `git fetch origin main`
2. **Check current state**: `git status`
3. **Check for open PRs**: `gh pr list` - Look for PRs that might affect the same files
4. **Pull latest if on main**: `git pull origin main`
5. **Only then create your branch and make changes**

This prevents merge conflicts when multiple agents update documentation simultaneously.

## Pull Request Best Practices

### After Creating a PR
1. **Verify PR status**: Check that mergeable_state is 'clean' not 'dirty'
2. **Review the changes**: Use `git show` or check the PR diff online
3. **If conflicts exist**: Rebase against main and force push
4. **Test locally**: Run any relevant tests before marking complete
5. **Document what was added**: Clearly state what sections/content were added
