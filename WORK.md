# Active Ideas / In-Progress Concepts

### Habit Pattern Miner (Jarvis-style automation)
- Continuously log structured behavior events (intents, reminders, traumas, app launches).
- Nightly/periodic miner scans for repeated patterns (e.g., “set Monday 6 AM alarm” every Sunday night).
- When confidence crosses a threshold, auto-synthesize a `HabitRule`:
  - Condition: e.g., Tuesday 8 PM
  - Action: call MCP connector (alarms/calendar/etc.) to schedule the expected behavior
  - Annotate as “self-learned” and notify the user for transparency
- Monitor feedback: cancelling or editing the auto-action increases “pain” and decays the rule; successful runs reinforce it.
- Feed rule summaries into the system prompt + MCP connectors so LLM conversations stay aware of learned habits.
