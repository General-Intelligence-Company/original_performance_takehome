# Project Guidelines for Claude

## File Management

When making experimental changes to this codebase:

1. **Use version control, not file copies**: Make changes directly in the main file (`perf_takehome.py`) and use git for version control
2. **Use git branches for different approaches**: Instead of creating multiple files like `perf_takehome_v2.py`, create feature branches
3. **Commit working versions before experiments**: Always `git commit` a working state before trying major optimizations
4. **If creating experimental files**: Use a consistent naming pattern (e.g., `*_experimental.py`)
5. **Clean up promptly**: Remove experimental files when done: `rm -f *_experimental.py`

This prevents accumulation of files like `perf_takehome_interleaved.py`, `perf_takehome_roundmajor.py`, etc.
