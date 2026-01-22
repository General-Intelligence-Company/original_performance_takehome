# Claude Code Agent Guidelines

This file provides guidance for AI coding agents working on this repository.

## File Operations

**Standard Location**: Always use `.claude/claude.md` - do not check for existence first. If the file doesn't exist, create the `.claude/` directory and `claude.md` file at this location. Do not search for alternative locations.

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
