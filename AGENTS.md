# Agent Guidelines

## System Requirements

Before running agents:

1. **Ensure database is properly initialized** with all required columns
2. **Check agent system status** before beginning tasks
3. **If agent fails to start**, verify database schema is up to date

Common error: `column users.onboarding_integration_step does not exist` indicates database migration needed.

## System Prerequisites

**Before starting Claude Code agent**:
1. Verify database schema is properly initialized
2. Check that all required database columns exist
3. If encountering 'users.onboarding_integration_step does not exist' error, contact system administrator
4. Use alternative approaches if agent startup fails

**Fallback options**:
- Use manual git commands instead of agent automation
- Perform tasks step-by-step with verification at each stage
