# Agent Guidelines

## System Requirements

Before running agents:

1. **Ensure database is properly initialized** with all required columns
2. **Check agent system status** before beginning tasks
3. **If agent fails to start**, verify database schema is up to date

Common error: `column users.onboarding_integration_step does not exist` indicates database migration needed.
