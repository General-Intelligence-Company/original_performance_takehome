# Agent Guidelines

This document contains troubleshooting information and guidelines for Claude agents working with this repository.

## System Requirements

Before running agents:

1. **Ensure database is properly initialized** with all required columns
2. **Check agent system status** before beginning tasks
3. **If agent fails to start**, verify database schema is up to date

Common error: `column users.onboarding_integration_step does not exist` indicates database migration needed.

## Agent Startup Issues

If you encounter 'Claude Code agent failed' with database errors:

1. **This is a known infrastructure issue**: The agent startup process may fail due to backend service problems
2. **Database schema mismatch**: The error `column users.onboarding_integration_step does not exist` indicates the database schema is out of sync
3. **Contact infrastructure team**: Report the issue or try again later - this is typically a transient issue
4. **No workaround available**: This error prevents the agent from performing any repository operations until resolved

This is not a code issue - it requires infrastructure-level intervention.

## Fallback Options

If agent startup fails:
- Use manual git commands instead of agent automation
- Perform tasks step-by-step with verification at each stage

## Database Issues

If you encounter database errors like 'column users.onboarding_integration_step does not exist':
1. This indicates a schema mismatch in the agent system
2. Report this error to system administrators
3. This is not a repository-specific issue but an agent platform issue
4. Try using a different agent or manual git commands as fallback