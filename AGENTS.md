# Agent Guidelines

This document contains troubleshooting information and guidelines for Claude agents working with this repository.

## Agent Startup Issues

If you encounter 'Claude Code agent failed' with database errors:

1. **This is a known infrastructure issue**: The agent startup process may fail due to backend service problems
2. **Database schema mismatch**: The error `column users.onboarding_integration_step does not exist` indicates the database schema is out of sync
3. **Contact infrastructure team**: Report the issue or try again later - this is typically a transient issue
4. **No workaround available**: This error prevents the agent from performing any repository operations until resolved

This is not a code issue - it requires infrastructure-level intervention.
