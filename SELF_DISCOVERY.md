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

### 2. Goal-Aware Domain Detection

The system automatically detects the goal's domain:
- **Mathematics**: solve, equation, calculate, algebra, quadratic, etc.
- **Science**: energy, force, physics, chemistry, atom, etc.
- **Programming**: code, function, algorithm, data structure, etc.
- **Language**: word, sentence, grammar, alphabet, etc.
- **Literature**: book, novel, story, author, etc.

This ensures focused learning - no plant biology when solving math equations!

### 3. Recursive Learning with Relevance Filtering

When learning a concept, the system:
1. Checks if already in LTM (skip if known)
2. Validates concept is relevant to goal domain
3. Searches web with domain-aware query ("mathematical root" vs "plant root")
4. Asks LLM to extract definition and domain-specific prerequisites
5. Filters prerequisites by relevance (skips cross-domain concepts)
6. Recursively learns relevant prerequisites first
7. Stores in persistent LTM

### 4. Persistent Memory

- **Storage**: JSON file (`ltm_memory.json`)
- **Load on startup**: Reuses previous knowledge
- **Save after each concept**: Never forgets
- **Accumulative**: Grows over time like human memory

### 5. Learning Journal

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

## Goal-Aware Learning in Action

### The Focus Problem (SOLVED)

**Problem**: Without domain awareness, the system would learn 400+ irrelevant concepts:
- Goal: "solve x^2 - 5x + 6 = 0"
- System learns "roots of equation"
- Wikipedia returns article about plant roots
- Follows chain: roots(plant) → eukaryotes → cells → biology → chemistry → bonds(financial!) → interest rates → ...
- Result: 420 concepts learned, including plant biology, finance, networking, music, philosophy
- Outcome: FAILED to solve equation, exhausted web quota

**Solution**: Goal-aware domain detection and relevance filtering

### How It Works Now

```bash
$ python run_self_discovery.py "solve x^2 - 5x + 6 = 0"

[i] Detected goal domain: mathematics
[i] Goal keywords: solve, quadratic, equation

[Attempt 1] Trying to achieve goal
...
MISSING: quadratic equation, roots, factoring

[L] Discovering: quadratic equation
    Query: mathematical quadratic equation  # Domain-aware!
    [+] Retrieved content from Wikipedia (mathematics)
    [i] Relevant prerequisites: equation, variable, polynomial

[L] Discovering: roots
    Query: mathematical root of equation  # Not "plant root"!
    [+] Retrieved content from Math StackExchange
    [i] Relevant prerequisites: equation, solution
    [i] Filtered 3 irrelevant prerequisites  # Skipped plant biology!

[OK] GOAL ACHIEVED!
Learned: 8 relevant concepts (was 420 before!)
```

### Domain Detection Examples

| Goal | Detected Domain | Keywords | Skips |
|------|----------------|----------|-------|
| "solve 2x + 5 = 15" | mathematics | solve, equation | plant biology, finance |
| "explain photosynthesis" | science | photosynthesis | algebra, programming |
| "write a for loop in Python" | programming | write, loop, python | literature, math |
| "what is a noun" | language | noun | equations, variables |
| "summarize Hamlet" | literature | summarize, hamlet | physics, chemistry |

### Relevance Filtering

**Mathematics domain** accepts:
- equation, variable, algebra, solve, calculate, root (math), function (math)

**Mathematics domain** rejects:
- plant, eukaryote, photosynthesis (biology)
- bond (finance), interest rate, investment (finance)
- TCP/IP, network, protocol (networking)
- drum, percussion, instrument (music)

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
✅ **Domain-aware** - Stays focused on goal domain, filters irrelevant concepts
✅ **Persistent memory** - Never forgets, builds over time
✅ **Efficient** - Only learns what's needed and relevant
✅ **Natural emergence** - Discovers A-Z, 0-9 only if required
✅ **Learning history** - Shows journey like human civilization
✅ **Reusable knowledge** - Second goals use first goal's learning
✅ **Smart search** - Context-aware queries prevent ambiguity ("mathematical root" not "plant root")

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
