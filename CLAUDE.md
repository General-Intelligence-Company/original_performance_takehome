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

## Before Adding New Sections to claude.md

1. **Read the entire file first**: `cat claude.md` or open in editor
2. **Look for existing sections**: Search for topics that might cover similar content
3. **Enhance existing sections**: If similar content exists, update that section instead of creating a new one
4. **Place new sections logically**: Position relative to existing content flow
5. **Use consistent formatting**: Match header levels and style with the rest of the file

This prevents duplicate or conflicting sections from accumulating.

## Pull Request Best Practices

### After Creating a PR
1. **Verify PR status**: Check that mergeable_state is 'clean' not 'dirty'
2. **Review the changes**: Use `git show` or check the PR diff online
3. **If conflicts exist**: Rebase against main and force push
4. **Test locally**: Run any relevant tests before marking complete
5. **Document what was added**: Clearly state what sections/content were added

## Branch and PR Management

### When Changes Become Redundant

1. **Close the PR with explanation**: Clearly explain why the PR is no longer needed (e.g., changes superseded, feature dropped, or content merged elsewhere)
2. **Delete both local and remote branches**: `git branch -D branch-name` and `git push origin --delete branch-name`
3. **Document partial incorporations**: If parts of the content were useful, mention what was incorporated and where

This ensures clean branch history and prevents confusion from stale PRs.

### When Conflicts Are Complex

1. **Evaluate necessity**: Consider if the change is still needed given recent updates to main
2. **Document resolution clearly**: If proceeding, describe what conflicts were resolved and how in the commit message
3. **Update PR description**: Reflect the conflict resolution approach so reviewers understand the changes

This prevents unintentional regressions and helps reviewers understand merged content.

## Safe Force Pushing

When you need to force push (e.g., after rebasing or amending pushed commits):

1. **Use --force-with-lease**: Always use `git push --force-with-lease` instead of `--force`
2. **Why it matters**: Protects against overwriting commits pushed by others since your last fetch
3. **Common scenarios**: After rebasing against main, amending commit messages, or squashing commits

This prevents accidentally destroying other agents' work on shared branches.

## Before Creating Branches

1. **Check repository permissions**: Can you push to main? Do you need to fork?
2. **Examine existing branches**: Run `git branch -a` to see all branches
3. **Understand the workflow**: Are there existing PRs? What's the branch naming convention?
4. **Check for existing similar branches**: Avoid duplication of effort
5. **Only create branches after understanding the structure**: Verify you understand the repository layout

This prevents permission errors and duplicate work.
