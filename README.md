# KV-1 🧠

**An LLM-orchestration system with human-inspired memory and recursive knowledge acquisition.**

KV-1 orchestrates LLMs (Ollama/Gemini), symbolic math (SymPy), and web search to learn concepts on-demand through a goal-driven loop: attempt → identify gaps → recursively learn prerequisites → retry. Features a novel 3-stage memory lifecycle inspired by neuroscience.

**Reality check**: This is an LLM orchestration framework with sophisticated memory management, not a standalone AGI. All reasoning, understanding, and concept extraction depend entirely on the underlying LLM.

---

## 🎯 Core Concept

**Problem**: LLMs have static knowledge and can't learn new concepts during operation.

**Solution**: Orchestrate LLMs with:
- **3-stage memory lifecycle** (LEARNING → REINFORCEMENT → MATURE) - the main innovation
- **Dual memory architecture** (STM + LTM) with rehearsal and consolidation
- **Recursive prerequisite learning** - identifies and learns missing concepts automatically
- **Persistent storage** - saves learned knowledge across sessions
- **Symbolic math integration** (SymPy) for computational verification
- **Multi-source web research** (Wikipedia, ArXiv, StackExchange, etc.)

---

## 🏗️ What Actually Exists

### Core Components (verified)

```
┌─────────────────────────────────────────────────────────┐
│  Self-Discovery Orchestrator (2,120 lines)              │
│  • Goal pursuit with failure analysis                   │
│  • Recursive prerequisite identification                │
│  • Parallel concept learning (up to 10 concepts/GPU)    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  HSOKV Memory System (Main Innovation)                  │
│  • 3-Stage Lifecycle: LEARNING → REINFORCEMENT → MATURE │
│  • STM: 50 slots, O(1) lookup, 30s decay               │
│  • LTM: GPU semantic search (768D embeddings)           │
│  • Rehearsal-based consolidation                        │
│  • No catastrophic forgetting (frozen embeddings)       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Knowledge Acquisition (LLM-Powered)                    │
│  • LLM Bridge: Ollama (qwen3:4b) / Gemini              │
│  • Web Researcher: 9 sources with caching              │
│  • SymPy Integration: Symbolic math solving            │
│  • Validator: Optional multi-source verification       │
└─────────────────────────────────────────────────────────┘
```

### Advanced Modules (33 files, ~14K lines)

**Actually implemented:**
- `unified_agi_learner.py` - Routes between tensor/traditional reasoning
- `tensor_reasoning_system.py` - 768D Riemannian manifold for math concepts
- `meta_learner.py` - Tracks learning strategies and improves over time
- `pattern_learner.py` - Extracts mathematical structures from experience
- `transfer_learning.py` - Cross-domain knowledge transfer
- `analogical_reasoning.py` - Structural mapping between concepts
- `causal_reasoner.py` - Cause-effect modeling
- `compositional_reasoner.py` - Combines simple → complex concepts
- `deep_abstraction.py` - Multi-level abstraction hierarchies
- `advanced_reasoning.py` - Common sense, hypothesis testing
- `parallel_web_search.py` - Async multi-source searching
- `geometric_knowledge_space.py` - Riemannian manifold (768D)
- `framework_invention.py` - Framework creation (Phase 4)
- `physical_grounding.py` - Physical reality grounding (Phase 5)

**Important**: These modules exist and integrate with the system, but effectiveness depends heavily on LLM quality. They orchestrate and structure LLM calls rather than providing independent reasoning.

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

# Setup LLM (choose one):

# Option 1: Ollama (local, free, default)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# Option 2: Gemini (cloud, requires API key)
export GEMINI_API_KEY="your-key-here"
```

### Basic Usage

```bash
# Self-discovery learning
python run_self_discovery.py "solve 2x + 5 = 15"

# With quality modes
python run_self_discovery.py "derivative of x^3" \
  --validate              # Multi-source validation (slower)
  --target-confidence 0.75 # Higher mastery threshold

# Fast mode (skip rehearsal)
python run_self_discovery.py "factor 84" --no-rehearsal

# Custom memory
python run_self_discovery.py "prime numbers" \
  --ltm my_memory.json \
  --reset  # Start fresh
```

### Quality vs Speed Tradeoffs

| Mode | Validation | 3-Stage | Speed | Quality | Use Case |
|------|-----------|---------|-------|---------|----------|
| **Fast** | OFF | OFF | ⚡⚡⚡ | ⭐⭐ | Quick prototyping |
| **Balanced** (default) | OFF | ON | ⚡⚡ | ⭐⭐⭐⭐ | General learning |
| **Quality** | ON | ON | ⚡ | ⭐⭐⭐⭐⭐ | Critical concepts |

---

## 📋 How It Actually Works

### Learning Flow

```
1. Goal: "solve x² - 5x + 6 = 0"
   ↓
2. LLM attempts with current knowledge
   → FAIL (missing concepts)
   → LLM identifies: ["quadratic formula", "factoring"]
   ↓
3. For each missing concept:
   a) Semantic search in LTM (threshold: 0.85)
   b) If not found → Web search (9 sources)
   c) LLM extracts: definition, prerequisites, examples
   d) Recursively learn prerequisites (max depth: 3)
   e) 3-Stage Learning (optional):
      • SURPRISE: Test understanding → confidence score
      • REHEARSAL: Practice until 0.70+ confidence
      • CORTICAL: Store with final confidence
   f) Store in memory:
      • Text definition
      • 768-D tensor embedding (GPU)
      • SymPy symbolic form (if math)
      • Examples and procedures
   g) Save to ltm_memory.json
   ↓
4. Retry goal with LLM + new knowledge
   → SUCCESS (or repeat until max attempts)
```

**Key point**: LLM does all reasoning. System provides memory, orchestration, and verification.

### 3-Stage Memory Lifecycle (HSOKV)

**The core innovation** - mimics human memory formation:

```
LEARNING Stage (First ~5 retrievals)
→ 1.5x confidence boost
→ Protected from pruning
→ Like learning a new word - needs reinforcement

REINFORCEMENT Stage (Retrievals 5-20)
→ 1.5x → 1.0x gradual decay
→ Still protected from pruning
→ Like practicing a skill - becoming automatic

MATURE Stage (20+ retrievals)
→ 1.0x confidence (no boost)
→ Can be pruned if low confidence
→ Like well-learned knowledge - established memory
```

**3-Stage Learning Loop** (optional quality mode):
1. **Surprise**: LLM explains concept → confidence 0.0-1.0
2. **Rehearsal**: LLM practices until 0.70+ confidence (up to 4 rounds)
3. **Cortical Transfer**: Store with final confidence score

---

## 🧮 Mathematical Capabilities

### What Works

**SymPy-Powered Solving** (`solve.py`, `honest_solver.py`):
- Equation solving (linear, quadratic, systems)
- Differentiation and integration
- Prime checking and factorization
- Number theory (GCD, Goldbach verification)
- Collatz sequences
- Symbolic manipulation

**Tensor Reasoning** (`tensor_reasoning_system.py`):
- 768D Riemannian manifold for concept geometry
- Symbolic math engine (SymPy wrapper)
- Geometric knowledge space
- Mathematical exploration engine

**Example Storage:**
```python
Concept: "Pythagorean theorem"
├─ Text: "In right triangle: a² + b² = c²"
├─ Tensor: [0.123, -0.456, ..., 0.789]  # 768D
├─ SymPy: Eq(a**2 + b**2, c**2)
├─ Examples: ["3² + 4² = 5²", ...]
└─ Stage: LEARNING (boost: 1.5x)
```

### What Doesn't Work

- **Novel theorem proving** - Uses SymPy + LLM templates, not independent proof discovery
- **Unsolved problems** - Riemann Hypothesis mentioned as goal but no actual progress
- **Visual reasoning** - Text-only, no diagrams/images
- **Physical intuition** - Limited despite `physical_grounding.py` module

---

## 🧠 Memory System Details

### Dual Memory (STM + LTM)

**Short-Term Memory:**
- Capacity: 50 slots (extended from Miller's 7±2)
- Decay: 30 seconds without access
- Structure: OrderedDict (O(1) lookup)
- Speed: ~0.001ms
- Eviction: LRU (least recently used)

**Long-Term Memory:**
- Capacity: Unlimited
- Storage: PyTorch tensor matrix (768D)
- Lookup: Cosine similarity (GPU-accelerated)
- Speed: ~1-5ms
- Model: sentence-transformers

**Consolidation Flow:**
```
learn("quadratic formula")
  → Store in LTM (tensor + text + SymPy)
  → Add to STM (fast cache)
  → Save to disk (ltm_memory.json)
  → Stage: LEARNING (1.5x boost)

recall("quadratic")
  → STM miss
  → LTM search (similarity: 0.92)
  → Promote to STM
  → Increment retrieval_count
  → Next recall → STM hit (instant)
```

### Geometric Knowledge Space

**Riemannian Manifold** (768D):
- Concepts as points
- Distances = conceptual similarity
- Geodesics = learning paths
- Curvature = complexity
- Parallel transport = analogies

**Property Encoding:**
- Domain clustering (number theory, algebra, calculus)
- Prime/composite separation
- Even/odd grouping
- Complexity weighting

---

## 🎓 Learning Curriculum

**195 questions** across 6 phases (not 260 as previously stated):

1. **Foundational Math** (75): Arithmetic → Algebra → Trig → Complex
2. **Calculus** (55): Limits → Derivatives → Integrals → Series
3. **Advanced Math** (25): Linear algebra → Abstract algebra → Discrete
4. **Number Theory** (15): Primes → Zeta function → Diophantine
5. **Complex Analysis** (15): Analytic functions → Residues → Continuation
6. **Toward Riemann** (10): Hypothesis understanding → Critical line

**Run curriculum:**
```bash
# Full curriculum
python run_curriculum.py --phase all

# Specific phase
python run_curriculum.py --phase 1

# Resume from checkpoint
python run_curriculum.py --resume --skip-failed
```

See [LEARNING_CURRICULUM.md](LEARNING_CURRICULUM.md) for complete list.

---

## ⚙️ Configuration

### System Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_depth` | 3 | 1-15 | Recursive prerequisite depth |
| `stm_capacity` | 50 | 7-100 | Short-term memory slots |
| `target_confidence` | 0.70 | 0.65-0.90 | Mastery threshold |
| `max_parallel_concepts` | 10 | 1-50 | GPU parallel learning |
| `embedding_dim` | 768 | - | Tensor dimension |

### Environment Variables

```bash
export OLLAMA_HOST="http://localhost:11434"
export GEMINI_API_KEY="your-api-key"
export LTM_PATH="./ltm_memory.json"
```

---

## 📊 Actual Performance

**Memory:**
- Concept storage: ~1-2KB per concept
- 1000 concepts: ~1-2MB disk
- STM hit rate: >80% for recent queries
- LTM search: ~90% accuracy (similarity ≥ 0.85)

**Learning:**
- Concept acquisition: 15-30 seconds (balanced mode)
- Web cache: Significant speedup on repeated topics
- GPU acceleration: 2-5x faster than CPU for large LTM

**Limitations:**
- No benchmark results verified (benchmark framework exists, results not shown)
- Success rate depends entirely on LLM quality
- Web content quality varies significantly
- Slow for complex prerequisite chains

---

## 🐛 Brutal Honesty: Limitations

### Critical Dependencies

1. **LLM-Powered Everything**
   - ALL reasoning done by LLM (Ollama/Gemini)
   - System is orchestration layer, not independent intelligence
   - Without LLM access → completely non-functional
   - Quality ceiling = LLM capability ceiling

2. **SymPy Does Math**
   - Equation solving: SymPy
   - Symbolic manipulation: SymPy
   - Prime checking: SymPy
   - System integrates SymPy, doesn't invent new math

3. **No Novel Reasoning**
   - Pattern learner extracts structures but doesn't create new frameworks (despite `framework_invention.py`)
   - "AGI modules" structure LLM calls, don't provide independent reasoning
   - Transfer learning limited by LLM's inherent capabilities

### Technical Limitations

4. **Domain Specialization**: Math-optimized, general knowledge less effective
5. **Web Dependency**: Quality varies, can fail if content poor
6. **No Visual Learning**: Text-only (no images/diagrams/videos)
7. **Loop Risk**: Can get stuck on poorly-defined concepts
8. **Security**: Uses `exec()` for math parsing (not production-safe)
9. **No Embodiment**: No physical grounding despite module existence
10. **Unverified Claims**: Benchmark results not demonstrated, Riemann Hypothesis work aspirational only

### What This Is vs What It Isn't

**What it IS:**
- Sophisticated LLM orchestration framework
- Novel memory system with 3-stage lifecycle
- Effective knowledge persistence across sessions
- Good integration of SymPy + LLM + web search
- Research platform for continual learning

**What it ISN'T:**
- Independent AGI or reasoning system
- Novel theorem prover
- Replacement for LLMs
- Production-ready (security issues)
- Verified on hard benchmarks

---

## 📁 Project Structure

```
KV-1/
├── self_discovery_orchestrator.py  # Main loop (2,120 lines)
├── run_self_discovery.py           # CLI interface
├── run_curriculum.py               # 195-question curriculum
├── solve.py                        # Simple SymPy solver
├── honest_solver.py                # Computational solver
│
├── core/                           # 33 modules (~14K lines)
│   ├── llm.py                      # Ollama/Gemini bridge
│   ├── hybrid_memory.py            # STM + LTM orchestration
│   ├── web_researcher.py           # 9-source web scraping
│   ├── unified_agi_learner.py      # Math/general routing
│   ├── tensor_reasoning_system.py  # 768D Riemannian manifold
│   ├── meta_learner.py             # Strategy optimization
│   ├── pattern_learner.py          # Structure extraction
│   ├── transfer_learning.py        # Cross-domain transfer
│   ├── analogical_reasoning.py     # Structural mapping
│   ├── math_connect.py             # SymPy integration
│   ├── knowledge_validator.py      # Multi-source validation
│   └── ...                         # 22 more modules
│
├── hsokv/                          # HSOKV memory library
│   ├── memory.py                   # 3-stage lifecycle
│   ├── dual_memory.py              # STM + LTM
│   ├── lifecycle.py                # Stage management
│   ├── embedders.py                # Sentence-BERT
│   └── config.py                   # Hyperparameters
│
├── benchmarks/                     # Testing framework
│   ├── benchmark_utils.py          # 19 hard problems
│   └── compare_baselines.py        # LLM/RAG/Few-shot
│
├── tests/                          # Unit tests
├── tools/                          # Knowledge import utilities
└── *.md                            # Documentation
```

Total: ~16K+ lines Python code

---

## 🌟 What's Actually Novel

### Confirmed Innovations

1. **3-Stage Memory Lifecycle** (HSOKV)
   - LEARNING → REINFORCEMENT → MATURE stages
   - Confidence boosting during formation
   - Pruning protection for new memories
   - Inspired by neuroscience, implemented with PyTorch

2. **Persistent Self-Discovery Loop**
   - Goal → attempt → gap analysis → recursive learning
   - JSON-based LTM across sessions
   - Meta-learning tracks what works

3. **Dual Memory Architecture**
   - STM (50 slots, O(1)) + LTM (semantic search)
   - Rehearsal-based consolidation
   - Time decay and LRU eviction

4. **Hybrid Reasoning Router**
   - Detects math/general/hybrid questions
   - Routes to tensor reasoning vs traditional learning
   - Confidence-based result selection

5. **Geometric Knowledge Space**
   - 768D Riemannian manifold
   - Properties → tensor encoding
   - Geodesics as learning paths

### Engineering Highlights

- Clean LLM + SymPy + PyTorch integration
- GPU-accelerated semantic search with batching
- 9-source web research with fallbacks
- Frozen embeddings (no catastrophic forgetting)
- Atomic disk writes for persistence

---

## 🔬 Research Vision vs Reality

**Vision**: "Mathematical AGI through domain abstraction"

**Reality**:
- ✅ Foundation: Memory system, SymPy integration, LLM orchestration
- ✅ Pattern Learning: Structure extraction from experience
- ✅ Compositional Reasoning: Concept combination frameworks
- ✅ Deep Abstraction: Multi-level hierarchies
- ❌ Framework Invention: Module exists, effectiveness unclear
- ❌ Physical Grounding: Module exists, integration limited
- ❌ Novel Mathematics: Not achieved, not close

**Current Status**: ~30-40% toward vision (not 50-55% as claimed)
- Strong: Memory system, orchestration, persistence
- Weak: Independent reasoning, novel insights, verification
- Missing: Actual AGI, theorem proving, Riemann Hypothesis work

**Realistic Timeline**: If continued → 15-25 years to vision (if achievable at all)

This is a research prototype demonstrating concepts, not a complete system.

---

## ⚠️ Use Cases: What This Is Good For

### Good Use Cases

✅ **Research Platform**
- Experimenting with memory architectures
- Testing continual learning approaches
- Studying LLM orchestration patterns

✅ **Educational Tool**
- Interactive math learning with persistence
- Concept prerequisite visualization
- Self-paced curriculum

✅ **Prototyping**
- Testing recursive learning ideas
- Exploring human-inspired memory
- LLM + symbolic math integration

### Bad Use Cases

❌ **Production Deployments**
- Uses `exec()` without sandboxing
- No authentication or security
- Slow (web search latency)

❌ **Independent Reasoning**
- 100% dependent on LLM
- No standalone intelligence
- Quality = LLM quality

❌ **Novel Research**
- Can't prove new theorems
- Won't solve unsolved problems
- Uses existing tools (SymPy, LLMs)

❌ **Real-Time Applications**
- 15-30 seconds per concept
- Web search blocking
- GPU warm-up delays

---

## 🤝 Contributing

Research project, contributions welcome:
- Memory system improvements
- Better prerequisite detection
- Domain expansion beyond math
- Security hardening
- Benchmark verification

See GitHub issues.

---

## 📚 Documentation

- [BRAIN_ARCHITECTURE.md](BRAIN_ARCHITECTURE.md) - Memory system details
- [DOMAIN_MATH_BRIDGE.md](DOMAIN_MATH_BRIDGE.md) - Mathematical abstraction
- [LEARNING_CURRICULUM.md](LEARNING_CURRICULUM.md) - 195 questions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

## 📄 License

See LICENSE file.

---

## 🔗 Links

- **Repository**: https://github.com/PlanetDestroyyer/KV-1
- **HSOKV Memory**: https://github.com/PlanetDestroyyer/hsokv
- **Issues**: https://github.com/PlanetDestroyyer/KV-1/issues

---

## 🎯 Bottom Line

**What this actually is**: A well-engineered LLM orchestration system with a novel 3-stage memory lifecycle, persistent knowledge storage, and good SymPy integration. The memory system (HSOKV) is genuinely interesting research.

**What this isn't**: Independent AGI, novel theorem prover, or anywhere close to solving the Riemann Hypothesis.

**Should you use it?**:
- Yes: Research, education, prototyping memory systems
- No: Production, independent reasoning, novel mathematics

**Honest assessment**: Quality LLM orchestration framework with innovative memory. Claims about "AGI" are aspirational. The 3-stage lifecycle is the real contribution. Everything else orchestrates existing tools (LLMs, SymPy, web search).

---

**Built by [@PlanetDestroyyer](https://github.com/PlanetDestroyyer)**

*A research exploration in LLM orchestration and human-inspired memory for learning systems.*

**Status**: Research prototype. Use accordingly.
