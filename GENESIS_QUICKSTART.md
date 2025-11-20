# Genesis Experiment - Quick Start

## What This Is

A pure learning experiment. KV-1 starts knowing ONLY 0-9 and a-z (36 symbols).
No pre-training. No OS features. Just raw emergence.

The question: **Can it discover words → sentences → algebra → calculus → physics?**

## Installation

### 1. Install HSOKV Memory System

```bash
cd hsokv
pip install -e .
cd ..
```

This installs:
- PyTorch (for embeddings)
- Sentence-Transformers
- The HSOKV memory package

*Note: This downloads ~1GB of dependencies. Be patient.*

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

Installs:
- `ollama` (for local LLM)
- `requests` + `beautifulsoup4` (for web research)

### 3. Install Ollama (if not already installed)

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull Qwen3 model (in another terminal)
ollama pull qwen3:4b
```

## Running the Experiment

### Quick Test (5 minutes)

Verify everything works:

```bash
python run_genesis_experiment.py --quick
```

This runs 10 learning cycles at 30s intervals (total: ~5 min).

###  24-Hour Run

```bash
python run_genesis_experiment.py --iterations 480 --interval 180
```

- 480 iterations
- 3 minutes between cycles
- Total: ~24 hours

### 7-Day Run (Original Hypothesis)

```bash
python run_genesis_experiment.py --iterations 3360 --interval 180
```

- 3360 iterations
- 3 minutes between cycles
- Total: ~168 hours (7 days)

## Monitoring Progress

In a separate terminal:

```bash
python monitor_genesis.py
```

This shows a live dashboard with:
- Current learning phase
- Memory stats (STM/LTM)
- Domain mastery progress
- Recent events

Refreshes every 10 seconds.

## What Gets Tracked

All data goes to `./genesis_data/`:

```
genesis_data/
├── logs/
│   ├── emergence.jsonl        # Every surprise, transfer, probe
│   ├── daily_progress.jsonl   # Hourly snapshots
│   └── genesis_log.json       # Genesis mode progress
└── web_cache/                 # Cached web research
```

## Expected Timeline

| Time | Phase | What Should Happen |
|------|-------|-------------------|
| Hour 0-6 | Bootstrap | Discover common English words |
| Hour 6-12 | Word Formation | Simple sentences, basic syntax |
| Hour 12-24 | Grammar | Paragraphs, arithmetic operators |
| Day 2-3 | Math Foundation | Algebra 50-70%, calculus basics |
| Day 4-7 | Mastery | Algebra 90%+, Calculus 85%+, Thermo 80%+ |

## Success Criteria

1. **Algebra**: 90% on evaluation tasks (solve equations)
2. **Calculus**: 85% on evaluation tasks (derivatives, limits)
3. **Thermodynamics**: 80% on evaluation tasks (laws, concepts)

## Troubleshooting

**"HSOKV not found"**
```bash
cd hsokv && pip install -e . && cd ..
```

**"Ollama connection failed"**
```bash
ollama serve  # Start Ollama in background
ollama pull qwen3:4b  # Download model
```

**"ModuleNotFoundError: No module named 'bs4'"**
```bash
pip install beautifulsoup4 requests
```

**Memory fills up too fast**
- The system auto-consolidates STM → LTM every 10 cycles
- STM has 7-item cap with 30s decay
- Should self-regulate, but monitor `monitor_genesis.py`

## Stopping the Experiment

Press `Ctrl+C` to stop gracefully. The system will:
1. Save final snapshot
2. Print summary statistics
3. Keep all logs for analysis

## Analyzing Results

### Count Events
```bash
# Total surprises
grep "surprise_episode" genesis_data/logs/emergence.jsonl | wc -l

# Total transfers to LTM
grep "episode_transfer" genesis_data/logs/emergence.jsonl | wc -l
```

### View Progression
```bash
# View all snapshots
cat genesis_data/logs/daily_progress.jsonl | jq .

# Extract progress over time (CSV)
jq -r '[.hours_elapsed, .genesis_progress.algebra, .genesis_progress.calculus, .genesis_progress.thermodynamics] | @csv' genesis_data/logs/daily_progress.jsonl
```

### Final State
```bash
# Last snapshot
tail -1 genesis_data/logs/daily_progress.jsonl | jq .
```

## The Experiment is Running If...

You see output like:
```
[Iteration 5] Starting learning cycle...
  ✓ Self-probe complete (STM: 3)
  🔍 Researching: beginner algebra fundamentals
  📊 Running domain evaluation...
  📈 Scores: {'algebra': 0.12, 'calculus': 0.05, 'thermodynamics': 0.0}
```

## Next Steps

- **Quick test** first to verify setup
- **Monitor** in real-time with `monitor_genesis.py`
- **Let it run** for 24-168 hours
- **Analyze** results in `genesis_data/logs/`

---

**Let's see if intelligence emerges from 36 symbols.** 🧬
