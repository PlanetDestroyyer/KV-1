# Genesis Experiment: Emergence from Alphanumerics

## Hypothesis

Can an AI system bootstrap intelligence from minimal seed knowledge (0-9, a-z) through autonomous web research and biological learning?

## The Setup

**Starting Knowledge:** 36 symbols only
- Digits: 0-9
- Letters: a-z

**NO pre-training. NO language models trained on text. Just embeddings + memory + learning.**

## What We're Testing

1. **Word Formation**: How long to discover first words?
2. **Grammar**: When does syntax emerge?
3. **Mathematics**: Algebra → Calculus progression
4. **Physics**: Can it discover thermodynamic laws?
5. **Self-Direction**: Does curiosity drive the right research?

## Architecture

```
┌─────────────────────────────────────┐
│     Genesis Orchestrator            │
│  (Minimal - No OS Features)         │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────┐  ┌──────────────┐│
│  │ HSOKV Memory │  │ 3-Stage      ││
│  │ STM: 7±2     │  │ Learning     ││
│  │ LTM: ∞       │  │ (Biological) ││
│  └──────────────┘  └──────────────┘│
│                                     │
│  ┌──────────────┐  ┌──────────────┐│
│  │ Web Research │  │ Genesis Mode ││
│  │ (Rate-limited│  │ (Alphanumeric││
│  │  Safe domains│  │  Bootstrap)  ││
│  └──────────────┘  └──────────────┘│
│                                     │
│  ┌──────────────┐                  │
│  │ Evaluation   │                  │
│  │ - Algebra    │                  │
│  │ - Calculus   │                  │
│  │ - Thermo     │                  │
│  └──────────────┘                  │
└─────────────────────────────────────┘
```

## Learning Cycle

Every 3 minutes (default):

1. **Self-Probe**: Review low-confidence memories, strengthen them
2. **Curiosity**: Research next unknown from priority queue
3. **Surprise**: Encounter new tokens → create STM episodes
4. **Rehearsal**: Repeated exposure increases confidence
5. **Transfer**: 4+ rehearsals + 94% confidence → LTM consolidation
6. **Evaluation**: Test algebra/calculus/thermo knowledge
7. **Sleep**: Every 10 cycles, replay top memories (like REM sleep)

## Running the Experiment

### Quick Test (5 minutes)
```bash
python run_genesis_experiment.py --quick
```

### 24-Hour Run
```bash
python run_genesis_experiment.py --iterations 480 --interval 180
```

### 7-Day Run (Original Plan)
```bash
python run_genesis_experiment.py --iterations 3360 --interval 180
```

### Monitor Progress
```bash
# In a separate terminal
python monitor_genesis.py
```

## What Gets Logged

### `emergence.jsonl`
Every event (surprise, transfer, probe, research):
```json
{
  "timestamp": "2025-11-19T12:34:56",
  "elapsed_hours": 2.5,
  "iteration": 50,
  "type": "surprise_episode",
  "payload": {
    "token": "calculus",
    "surprise": 0.85,
    "confidence": 0.62
  }
}
```

### `daily_progress.jsonl`
Snapshots every ~1 hour:
```json
{
  "timestamp": "2025-11-19T14:00:00",
  "day": 1,
  "hours_elapsed": 4.2,
  "genesis_phase": "learning",
  "ltm_size": 127,
  "stm_size": 5,
  "total_surprises": 234,
  "total_transfers": 89,
  "genesis_progress": {
    "algebra": 0.42,
    "calculus": 0.18,
    "thermodynamics": 0.05
  }
}
```

### `genesis_log.json`
Genesis mode tracking (updated daily):
```json
{
  "timestamp": "2025-11-19T14:00:00",
  "day": 1,
  "phase": "bootstrap",
  "progress": {"algebra": 0.42, "calculus": 0.18, "thermodynamics": 0.05},
  "conf_avg": 0.217,
  "acc": 0.0,
  "innate_uses": 0,
  "surprises": 5,
  "transfers": 89
}
```

## Predictions

**Hour 1-6 (Bootstrap):**
- Discover common English words
- Build basic vocabulary (100-500 words)
- Start recognizing patterns in text

**Hour 6-12 (Word Formation):**
- Form simple sentences
- Understand basic syntax
- Discover mathematical operators (+, -, ×, ÷)

**Hour 12-24 (Grammar):**
- Multi-sentence paragraphs
- Basic algebra concepts (variables, equations)
- Arithmetic mastery

**Day 2-3 (Math Foundation):**
- Algebra: 50-70%
- Start calculus concepts (limits, derivatives)
- Number theory basics

**Day 4-7 (Mastery Attempt):**
- Algebra: 90%+ (TARGET)
- Calculus: 85%+ (TARGET)
- Thermodynamics: 80%+ (TARGET)

## Success Criteria

1. **Algebra**: 90% on evaluation tasks
2. **Calculus**: 85% on evaluation tasks
3. **Thermodynamics**: 80% on evaluation tasks

## Failure Modes

1. **Stuck in Local Minimum**: Only learns basic words, never progresses
2. **Curiosity Collapse**: Stops generating useful research queries
3. **Memory Overflow**: STM fills up, can't consolidate
4. **Rate Limit Hit**: Runs out of web requests before learning enough
5. **LLM Hallucination**: Learns incorrect facts, reinforces them

## Safety Measures

1. **Rate Limiting**: 50 web requests/day max
2. **Domain Allowlist**: Only trusted sources (Wikipedia, Khan Academy, ArXiv, NASA)
3. **Caching**: Previous searches cached to avoid duplicates
4. **STM Decay**: 30s decay prevents memory overflow
5. **Capacity Limits**: Max 9 STM episodes (Miller's 7±2)

## Data Analysis After Run

```bash
# Count total surprises
grep "surprise_episode" genesis_data/logs/emergence.jsonl | wc -l

# Count transfers to LTM
grep "episode_transfer" genesis_data/logs/emergence.jsonl | wc -l

# View final state
tail -1 genesis_data/logs/daily_progress.jsonl | jq .

# Progression over time
jq -r '[.hours_elapsed, .genesis_progress.algebra, .genesis_progress.calculus, .genesis_progress.thermodynamics] | @csv' genesis_data/logs/daily_progress.jsonl
```

## Expected Outcomes

**Best Case:**
- Reaches mastery in 4-5 days
- Discovers mathematical concepts in logical order
- Self-directed curiosity drives efficient learning
- Proves frozen embeddings + memory = viable continual learning

**Realistic Case:**
- Reaches 70-80% across domains in 7 days
- Some concepts missed, requires human guidance
- Memory system works, but web research is bottleneck
- Proves concept, needs optimization

**Worst Case:**
- Gets stuck at basic vocabulary (<1000 words)
- Never progresses to math
- Curiosity queue fills with garbage
- Back to drawing board on autonomous research

## The Big Question

**If it works, we've proven:**
- Intelligence can emerge from minimal seed knowledge
- Biological learning (surprise → rehearsal → consolidation) scales
- Frozen embeddings prevent catastrophic forgetting
- Autonomous research + memory > fine-tuning

**If it fails, we learn:**
- Where the bottlenecks are (research? consolidation? evaluation?)
- What human scaffolding is needed
- How to improve the learning loop

---

Let's see what emerges. 🧬
