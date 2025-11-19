# KV-1 Living System Roadmap

Goal: evolve KV-1 into a continuously running, self-learning agent. Work will be delivered in smaller, verifiable chunks.

## Stage 1 – Infrastructure & Safety
1. **LLM Pipeline Hardening**
   - Implement retry/backoff, logging, and quota handling for Ollama.
   - Add `execute=False` dry-run mode (already present) + ability to stub for tests.
2. **Safe Web Research Layer**
   - Build a dedicated `WebResearcher` with domain allowlists, caching, rate-limits, robots.txt respect.
   - Provide hooks for future plugin sources (Khan, Wikipedia dumps).
3. **Telemetry & Persistence**
   - Persist STM/LTM metadata, surprise episode logs, and surf history.
   - Add instrumentation (timings, successes, failures).

## Stage 2 – Autonomy Loops
1. **Background Scheduler**
   - ✅ Implemented (`AutonomyScheduler`) – runs self-learning, curiosity, nightly reflection, genesis, and evaluation cycles.
2. **Curiosity Queue**
   - ✅ Implemented priority queue; evaluation failures enqueue new queries.
3. **Evaluation Harness**
   - ✅ Implemented baseline algebra/calculus/thermo tasks with keyword grading, trauma feedback, and curiosity integration.

## Stage 3 – Self-Discovery Expansion
1. **Domain Curriculum**
   - Structured progression (algebra → calculus → thermo → other sciences).
   - Map each domain to resource packs (docs/videos/API).
2. **Habit & Memory Synthesis**
   - Detect recurring user commands and promote routines.
   - Link with traumas, energy levels, and proactive triggers.
3. **Reflection-to-Action Loop**
   - Summarize daily learning, set next-day goals, and log to genesis tracker.

## Stage 4 – Safety & Deployment (later)
- Sandboxed browser / headless environment.
- On-device resource constraints (battery, CPU monitoring).
- Interfaces for human oversight (approve/reject new automations).

---

Next actionable work (Stage 1.1):
- Add retryable Ollama client with error logging.
- Introduce structured logging for `ThreeStageLearner` episodes.
- Expose `kv1.log_event(...)` for telemetry.
