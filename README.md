# KV-1 🧠

**An autonomous learning system for goal-driven knowledge acquisition with persistent memory.**

KV-1 is a research system that learns concepts on-demand by: attempting a goal → identifying knowledge gaps → searching the web → learning prerequisites recursively → storing in persistent memory → retrying until success.

---

## 🎯 Core Concept

**Problem**: Most AI systems have static knowledge. They can't learn new concepts during operation.

**Solution**: KV-1 implements autonomous learning with:
- Goal-driven knowledge acquisition (learns only what's needed)
- Recursive prerequisite learning (up to 7 levels deep)
- Persistent neurosymbolic memory (text + 384-D tensors + symbolic formulas)
- Multi-source web research (Wikipedia, ArXiv, StackExchange, etc.)
- Optional 3-stage biological learning (surprise → rehearsal → consolidation)

---

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│  Self-Discovery Orchestrator                            │
│  • Goal pursuit loop                                    │
│  • Concept discovery with recursive learning            │
│  • Parallel batch processing (up to 10 concepts)        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Hybrid Memory System                                   │
│  • STM: 50 slots, O(1) lookup (0.001ms)               │
│  • LTM: GPU semantic search (1-5ms)                    │
│  • Disk: Persistent JSON storage                       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Knowledge Acquisition Pipeline                         │
│  • Web Researcher: 9 sources (Wikipedia, ArXiv, etc)   │
│  • LLM Bridge: Ollama/Gemini support                   │
│  • MathConnect: Symbolic math reasoning (SymPy)        │
│  • Validator: Optional multi-source verification       │
└─────────────────────────────────────────────────────────┘
```

### Memory Architecture

```
User Query: "prime numbers"
     ↓
STM Check (O(1) lookup)
     ↓ miss
LTM Search (GPU cosine similarity)
     ↓ found (similarity: 0.89)
Promote to STM (consolidation)
     ↓
Next query → STM hit (instant)
```

**Performance:**
- STM hit: ~0.001ms
- LTM semantic search: ~1-5ms
- Disk persistence: ~50-200ms (atomic write)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/PlanetDestroyyer/KV-1
cd KV-1

# Install HSOKV memory system
cd hsokv && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt

# Choose LLM provider:

# Option 1: Ollama (local, free)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# Option 2: Gemini (cloud, API key required)
export GEMINI_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Learn and solve algebra
python run_self_discovery.py "solve 2x + 5 = 15"

# Learn calculus concepts
python run_self_discovery.py "find the derivative of x^3 + 2x^2"

# Learn number theory
python run_self_discovery.py "express 50 as sum of two primes"
```

### Configuration Options

```bash
# Quality modes
python run_self_discovery.py "solve x² = 16" \
  --validate              # Enable multi-source validation (slower, higher quality)
  --no-rehearsal          # Disable 3-stage learning (faster, lower quality)
  --target-confidence 0.75 # Mastery threshold (0.65-0.90)

# Memory management
python run_self_discovery.py "what is a prime" \
  --ltm my_memory.json    # Custom memory file
  --reset                 # Start with blank memory

# LLM provider
python run_self_discovery.py "solve x² = 16" \
  --provider gemini \
  --model gemini-2.0-flash-exp \
  --api-key YOUR_KEY
```

**Quality vs Speed:**

| Mode | Command | Validation | 3-Stage | Speed | Quality |
|------|---------|-----------|---------|-------|---------|
| **Fast** | `--no-rehearsal` | OFF | OFF | ⚡⚡⚡ | ⭐⭐ |
| **Balanced** (default) | - | OFF | ON | ⚡⚡ | ⭐⭐⭐⭐ |
| **Quality** | `--validate` | ON | ON | ⚡ | ⭐⭐⭐⭐⭐ |

---

## 📋 How It Works

### Learning Flow

```
1. User Goal: "solve x² - 5x + 6 = 0"
   ↓
2. Attempt with current knowledge (LTM: 0 concepts)
   → Result: FAIL
   → Missing: ["quadratic formula", "factoring"]
   ↓
3. For each missing concept (parallel processing):
   a) Check LTM (semantic search, threshold=0.85)
   b) Try 5 web search query variations
   c) Extract: definition, prerequisites, examples (LLM)
   d) Recursively learn prerequisites (max depth: 7)
   e) [Optional] 3-Stage Learning:
      • Test understanding (0.0-1.0)
      • Rehearse until confidence >= 0.70
      • Store with final confidence
   f) [Optional] Multi-source validation (3 sources)
   g) Store in memory:
      • Text definition
      • 384-D tensor embedding (GPU)
      • Symbolic formulas (SymPy)
      • Examples/procedures
   h) Save to disk (ltm_memory.json)
   ↓
4. Retry goal with new knowledge
   → Result: SUCCESS
   → Answer: x = 2, x = 3
```

### 3-Stage Learning (Biological Inspiration)

```
STAGE 1: Surprise Episode
→ Test: "Explain this concept in your own words"
→ Confidence: 0.60 (partial understanding)

STAGE 2: Rehearsal Loop (up to 4 rounds)
→ Practice: "Solve a problem using this concept"
→ Confidence: 0.60 → 0.75 → 0.85
→ Stop when: confidence >= 0.70 (acceptable)
           OR confidence >= 0.75 (confirmed)
           OR max rounds reached

STAGE 3: Cortical Transfer
→ Store with final confidence
→ Mark as mastered (0.70+) or needs reinforcement (<0.70)
```

**Confidence Thresholds:**
- 0.65: Acceptable (minimum to store)
- 0.70: Good understanding (default target)
- 0.75+: Excellent/confirmed mastery

---

## 🧮 Mathematical Reasoning (MathConnect)

KV-1 stores mathematical concepts as symbolic equations (SymPy), not just text.

**Example:**

```python
# Input: "Pythagorean theorem: a squared plus b squared equals c squared"

# Parsed to:
Eq(a**2 + b**2, c**2)  # SymPy symbolic equation

# Stored with:
• Text: "In a right triangle, a² + b² = c²"
• Tensor: [0.123, -0.456, ..., 0.789] (384-D)
• Formula: "a**2 + b**2 = c**2"
• Examples: ["3² + 4² = 5²", ...]

# Used for:
• Connection finding (relates to distance formula, trig identities)
• Symbolic manipulation (substitution, solving)
• Theorem composition (derive new results)
```

---

## 🧠 Memory System

### Hybrid Memory (STM + LTM + Disk)

**Short-Term Memory (STM):**
- Capacity: 50 slots (configurable)
- Decay: 5 minutes without access
- Lookup: O(1) direct match
- Speed: ~0.001ms

**Long-Term Memory (LTM):**
- Capacity: Unlimited
- Storage: GPU tensor matrix (384-D embeddings)
- Lookup: Cosine similarity search
- Speed: ~1-5ms

**Disk Persistence:**
- Format: JSON (ltm_memory.json)
- Write: Atomic (temp → rename)
- Load: Automatic on startup

**Data Flow:**
```
learn("prime numbers", definition)
  → Store in LTM (tensor + text)
  → Store in STM (fast lookup)
  → Save to disk (persistent)

recall("primes")
  → Check STM (miss)
  → Search LTM (found: "prime numbers", similarity=0.92)
  → Promote to STM
  → Next recall("primes") → STM hit (instant)
```

---

## ⚙️ Configuration

### System Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_depth` | 7 | 1-15 | Max recursive prerequisite learning depth |
| `stm_capacity` | 50 | 7-100 | Short-term memory slots |
| `target_confidence` | 0.70 | 0.65-0.90 | Mastery threshold for 3-stage learning |
| `daily_cap` | Unlimited | Any | Web requests per day |
| `max_parallel_concepts` | 10 | 1-50 | Concepts learned simultaneously |

### Environment Variables

```bash
# LLM Configuration
export OLLAMA_HOST="http://localhost:11434"
export GEMINI_API_KEY="your-api-key"

# Memory
export LTM_PATH="./ltm_memory.json"
```

---

## 🎓 Learning Curriculum

260 questions organized in 6 phases for systematic knowledge building:

1. **Foundational Math** (35 questions): Arithmetic → Algebra → Trig → Complex numbers
2. **Calculus & Analysis** (50 questions): Limits → Derivatives → Integrals → Series
3. **Advanced Math** (30 questions): Linear algebra → Abstract algebra → Discrete math
4. **Number Theory** (35 questions): Primes → Diophantine → Riemann zeta
5. **Complex Analysis** (25 questions): Analytic functions → Residues → Continuation
6. **Toward Riemann** (25 questions): Hypothesis understanding → Critical line → Zero distribution

See [LEARNING_CURRICULUM.md](LEARNING_CURRICULUM.md) for full list.

**Run curriculum:**
```bash
# Run all phases
python run_curriculum.py --phase all

# Run specific phase
python run_curriculum.py --phase 1

# Resume from checkpoint
python run_curriculum.py --resume
```

---

## 📊 System Statistics

**Typical Performance:**
- Concept learning time: 15-30 seconds (balanced mode)
- Memory per concept: ~1-2KB
- 1000 concepts: ~1-2MB disk space
- STM hit rate: >80% for recent queries
- LTM search accuracy: ~90% (similarity >= 0.85)

**Benchmark Results:**
- 18/19 hard problems solved (95% success rate)
- Includes: Collatz sequence, Chinese Remainder, Prime factorization
- See benchmarks/ for comparison scripts

---

## 🐛 Known Limitations

1. **Domain Specificity:** Optimized for mathematics; general knowledge works but less effective
2. **LLM Dependency:** Quality limited by underlying LLM (Ollama/Gemini)
3. **Web Content Quality:** Depends on finding good explanations online
4. **No Visual Learning:** Text-only (no images, diagrams, videos)
5. **Loop Detection:** Can get stuck if concepts are too abstract/poorly defined
6. **Security:** Uses `exec()` for math parsing (sandboxing needed for production)

---

## 📁 Project Structure

```
KV-1/
├── self_discovery_orchestrator.py  # Main learning loop (1934 lines)
├── run_self_discovery.py           # CLI interface
├── run_curriculum.py               # Curriculum runner
│
├── core/                           # Core modules (~9.4K lines)
│   ├── llm.py                      # LLM bridge (Ollama/Gemini)
│   ├── hybrid_memory.py            # STM + LTM + Disk
│   ├── neurosymbolic_gpu.py        # GPU tensor operations
│   ├── web_researcher.py           # 9-source web scraper
│   ├── knowledge_validator.py      # Multi-source validation
│   ├── math_connect.py             # Symbolic math (SymPy)
│   ├── meta_learner.py             # Learning strategy adaptation
│   ├── transfer_learning.py        # Cross-domain transfer
│   └── ...                         # Other AGI modules
│
├── hsokv/                          # HSOKV memory library
│   ├── dual_memory.py              # STM + LTM implementation
│   ├── memory.py                   # Base memory classes
│   └── embedders.py                # Sentence embeddings
│
├── benchmarks/                     # Performance comparisons
├── tools/                          # Utilities
├── LEARNING_CURRICULUM.md          # 260 questions
├── HOW_TO_RUN.md                   # Quick start guide
└── INSTALLATION.md                 # Setup instructions
```

---

## 🤝 Contributing

This is a research project. Contributions welcome for:
- Improving learning algorithms
- Adding new knowledge domains
- Enhancing memory efficiency
- Fixing bugs (see GitHub issues)

---

## 📄 License

See LICENSE file for details.

---

## 🔗 Links

- **Repository**: https://github.com/PlanetDestroyyer/KV-1
- **HSOKV Memory**: https://github.com/PlanetDestroyyer/hsokv
- **Issues**: https://github.com/PlanetDestroyyer/KV-1/issues

---

## ⚠️ Important Notes

**This is a research system, not production-ready:**
- No sandboxing (uses `exec()` for math parsing)
- Limited error handling
- Optimized for mathematical domains
- Requires significant disk space for large knowledge bases

**Best Use Cases:**
- Research in autonomous learning
- Educational tool for learning mathematics
- Knowledge base construction
- Curriculum-free learning experiments

---

**Built by [@PlanetDestroyyer](https://github.com/PlanetDestroyyer)**

*A research exploration in autonomous knowledge acquisition and persistent learning.*
