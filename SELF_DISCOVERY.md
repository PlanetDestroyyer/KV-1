# Self-Discovery Learning System

True autonomous goal-driven learning - the AI discovers knowledge through need, not scripted curriculum.

## Philosophy

**Traditional curriculum (curriculum_orchestrator.py):**
- Follows pre-defined path: word → sentence → paragraph → ...
- No autonomy - just executes a script
- Learns concepts in order, whether needed or not
- Like forcing a child to memorize a textbook in order

**Self-Discovery (self_discovery_orchestrator.py):**
- Starts with a goal: "solve 2x + 5 = 15"
- Tries to achieve it
- Discovers what's missing through failure
- Recursively learns prerequisites
- Like a human exploring and discovering

## How It Works

### 1. Goal Pursuit Loop

```
1. Attempt goal with current knowledge
2. Fail → Identify missing concepts
3. Learn each missing concept (recursively)
4. Retry goal
5. Repeat until success or max attempts
```

### 2. Recursive Learning

When learning a concept, the system:
1. Checks if already in LTM (skip if known)
2. Searches web for information
3. Asks LLM to extract definition and prerequisites
4. Recursively learns prerequisites first
5. Stores in persistent LTM

### 3. Persistent Memory

- **Storage**: JSON file (`ltm_memory.json`)
- **Load on startup**: Reuses previous knowledge
- **Save after each concept**: Never forgets
- **Accumulative**: Grows over time like human memory

### 4. Learning Journal

Tracks the discovery path:
- What was learned
- Why it was needed
- What prerequisites it required
- Depth of recursion

## Usage

### Basic Usage

```bash
python run_self_discovery.py "solve 2x + 5 = 15"
```

### Example Learning Session

**First run (blank memory):**
```bash
$ python run_self_discovery.py "solve 2x + 5 = 15"

[i] No existing LTM found, starting fresh

[Attempt 1] Trying to achieve goal
Goal: solve 2x + 5 = 15
Known concepts: 0

SUCCESS: no
ANSWER: cannot complete
MISSING: variable, equation, solving equations, subtraction, division

[L] Discovering: variable
  [i] Searching web...
  [+] Retrieved 423 chars from web
  [i] Definition: A symbol representing an unknown value...
  [OK] Stored in LTM

[L] Discovering: equation
  [i] Already in LTM, skipping

... (learns remaining concepts)

[Attempt 2] Trying to achieve goal
SUCCESS: yes
ANSWER: x = 5

[OK] GOAL ACHIEVED!
Attempts: 2
Concepts learned: 5

[#] LEARNING JOURNAL
1. Learned: variable
   Definition: A symbol representing an unknown value
   Needed for: solve 2x + 5 = 15

2. Learned: equation
   Definition: Mathematical statement of equality
   Needed for: solve 2x + 5 = 15

... (shows full discovery path)
```

**Second run (reusing memory):**
```bash
$ python run_self_discovery.py "solve 3x - 7 = 20"

[+] Loaded 5 concepts from LTM

[Attempt 1] Trying to achieve goal
Known concepts: 5

SUCCESS: yes
ANSWER: x = 9

[OK] GOAL ACHIEVED!
No new learning needed - used existing knowledge!
```

## Example Goals

### Beginner (Language & Numbers)
```bash
python run_self_discovery.py "Count from 1 to 10"
python run_self_discovery.py "What are the letters from A to E"
python run_self_discovery.py "Add 5 and 3"
```

### Intermediate (Algebra)
```bash
python run_self_discovery.py "Solve 2x + 5 = 15"
python run_self_discovery.py "Calculate 25% of 80"
python run_self_discovery.py "What is 3 squared"
python run_self_discovery.py "Solve for y: 4y - 12 = 0"
```

### Advanced (Math & Science)
```bash
python run_self_discovery.py "Solve x^2 - 5x + 6 = 0"
python run_self_discovery.py "Calculate area of circle with radius 5"
python run_self_discovery.py "What is escape velocity"
python run_self_discovery.py "Explain why ice floats"
python run_self_discovery.py "Calculate kinetic energy of 5kg object at 10m/s"
```

## Advanced Usage

### Custom LTM File
```bash
# Use different memory files for different domains
python run_self_discovery.py "solve x + 3 = 7" --ltm math_memory.json
python run_self_discovery.py "what is photosynthesis" --ltm biology_memory.json
```

### Reset Memory
```bash
# Start fresh, delete LTM
python run_self_discovery.py "solve 2x = 10" --reset
```

### Inspect LTM
```bash
# View what the AI has learned
python -c "import json; print(json.dumps(json.load(open('ltm_memory.json')), indent=2))"
```

## Architecture

### Components

1. **SelfDiscoveryOrchestrator**
   - Main autonomous learning loop
   - Goal pursuit and retry logic
   - Coordinates LLM, web researcher, and LTM

2. **PersistentLTM**
   - JSON-based long-term memory
   - Load/save from disk
   - Concept storage and retrieval

3. **LearningEntry**
   - Concept name
   - Definition
   - Timestamp
   - Why it was learned
   - Source (web/primitive/inference)

4. **GoalAttempt**
   - Success status
   - Result or error
   - Missing concepts identified

### Learning Flow

```
┌──────────────┐
│   Set Goal   │
└──────┬───────┘
       │
       v
┌──────────────┐    Success    ┌──────────┐
│ Attempt Goal ├──────────────>│   Done   │
└──────┬───────┘                └──────────┘
       │
       │ Fail
       v
┌────────────────────┐
│ Identify Missing   │
│    Concepts        │
└─────────┬──────────┘
          │
          v
    ┌─────────────┐
    │   For each  │
    │   missing   │
    │   concept   │
    └──────┬──────┘
           │
           v
    ┌──────────────┐
    │ In LTM?      │
    └──┬───────┬───┘
       │ No    │ Yes
       v       └──> Skip
    ┌──────────────┐
    │ Search Web   │
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │ Extract Def  │
    │ & Prereqs    │
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │ Learn Prereqs│ (Recursive)
    │ Recursively  │
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │  Store LTM   │
    └──────────────┘
```

## Comparison: Curriculum vs Self-Discovery

| Feature | Curriculum | Self-Discovery |
|---------|-----------|----------------|
| Learning trigger | Pre-defined order | Goal-driven need |
| Autonomy | None (scripted) | High (discovers path) |
| Memory persistence | Session-only | Disk-persisted JSON |
| Knowledge reuse | No | Yes (loads from file) |
| Learning path | Fixed: A→B→C | Emergent: discovers prerequisites |
| Like human learning | No | Yes |
| Motivation | None | Goal achievement |

## Benefits

✅ **True autonomy** - Discovers what it needs to know
✅ **Goal-driven** - Learns with purpose
✅ **Persistent memory** - Never forgets, builds over time
✅ **Efficient** - Only learns what's needed
✅ **Natural emergence** - Discovers A-Z, 0-9 only if required
✅ **Learning history** - Shows journey like human civilization
✅ **Reusable knowledge** - Second goals use first goal's learning

## Future Enhancements

- **Multi-goal sessions**: Pursue sequence of related goals
- **Knowledge graph**: Visualize concept dependencies
- **Spaced repetition**: Review and strengthen weak concepts
- **Self-testing**: Verify retained knowledge over time
- **Knowledge export**: Share learned concepts between instances
- **Meta-learning**: Learn how to learn more efficiently

## Files

- `self_discovery_orchestrator.py` - Main orchestrator class
- `run_self_discovery.py` - CLI runner
- `ltm_memory.json` - Persistent long-term memory (auto-generated)
- `SELF_DISCOVERY.md` - This documentation

## Related Systems

- `curriculum_orchestrator.py` - Old curriculum-based learning (pre-scripted)
- `core/three_stage_learner.py` - Biological learning model (surprise, rehearsal, consolidation)
- `core/web_researcher.py` - Web content retrieval
- `hsokv/` - HSOKV memory system (frozen embeddings)
