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

## Agent Startup Requirements

Before beginning work, ensure:

1. **Database schema is properly initialized**: Verify the database has been set up correctly before executing queries
2. **All required columns exist in the users table**: Check that expected columns are present before running operations that depend on them
3. **Handle missing column errors gracefully**: If encountering `column users.onboarding_integration_step does not exist` error, contact system administrator rather than attempting to create or modify the schema
4. **Graceful degradation for database issues**: Agent should handle database connection issues gracefully rather than failing completely - log the error and notify the user instead of crashing

**Common Issue**: Agents fail completely when encountering database schema mismatches. Always verify schema compatibility before proceeding with database operations.

## Known Issues

If you encounter database errors like `column users.onboarding_integration_step does not exist`, this is a system-level issue. Contact the development team as this prevents agent execution.

## Known Infrastructure Issues

**Database Column Error**: `column users.onboarding_integration_step does not exist`

- This is a known system-level database schema issue
- **Do not attempt to fix this yourself**
- Contact development team or system administrator
- Requires database migrations to be run by admin
- Agent cannot proceed with any tasks until resolved
- This error prevents agent execution entirely
