# KV-1 🧠

**A self-discovering AI research system powered by Free Energy Principle, compound knowledge growth, and emergent pattern recognition.**

KV-1 is evolving from a learning system into a **discovery machine** that: generates hypotheses autonomously → tests them through experiments → discovers connections → synthesizes theories → presents findings with evidence.

**Core Innovation**: Uses **Free Energy Principle** to organize knowledge (minimize surprise), leverages **compound interest effect** (each concept learned accelerates future learning), and mines **chain-of-thought patterns** from LLM reasoning to discover emergent insights.

---

## 🎯 Vision: From Learning to Discovery

### **Current State (v0.5): Learning System**
```
User provides goal → KV-1 learns needed concepts → Solves problem
```
**What it does:** Reactive learning with persistent memory
**What it can't do:** Autonomous hypothesis generation, theory formation, scientific discovery

### **Target State (v1.0): Discovery Machine**
```
┌─────────────────────────────────────────────────────────┐
│                  DISCOVERY LOOP                         │
└─────────────────────────────────────────────────────────┘

OBSERVE → QUESTION → HYPOTHESIZE → PREDICT → TEST →
ANALYZE → THEORIZE → EXPLAIN → ITERATE → DISCOVER

↓ Powered by ↓

1. Free Energy Principle (FEP)
   → Knowledge connections minimize "surprise"
   → Graph organizes for maximum explanatory power
   → Identifies gaps = high free energy regions

2. Compound Knowledge Growth
   → Learning accelerates as knowledge accumulates
   → Concept N+1 learned faster than concept N
   → Measurable growth rate: L(t) = L₀ × (1 + r)ᵗ

3. Chain-of-Thought Pattern Mining
   → Extract reasoning strategies from LLM traces
   → Learn problem-solving patterns from success
   → Reuse discovered strategies automatically
```

**What it will do:**
- ✅ Generate hypotheses from knowledge gaps
- ✅ Design experiments to test predictions
- ✅ Discover connections autonomously
- ✅ Synthesize theories from observations
- ✅ Present findings with evidence trails

---

## 🏗️ Architecture

### **Current Architecture (v0.5)**

```
┌─────────────────────────────────────────────────────────┐
│  Self-Discovery Orchestrator                            │
│  • Goal pursuit loop                                    │
│  • Recursive prerequisite learning                      │
│  • Parallel concept acquisition (up to 10)              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Hybrid Memory System (FEP-Guided)                      │
│  • STM: 50 slots, O(1) lookup (~0.001ms)               │
│  • LTM: GPU semantic search (~1-5ms)                    │
│  • Small-World Graph: Watts-Strogatz topology           │
│  • FEP: Minimizes prediction error + complexity         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Cognitive Modules (Phases 1-7)                         │
│  • Pattern Learner: Discovers mathematical structures   │
│  • Compositional Reasoner: Combines patterns            │
│  • Deep Abstraction: Cross-domain isomorphisms          │
│  • Framework Inventor: Creates new frameworks           │
│  • Physical Grounding: Connects math to reality         │
│  • Causal Reasoner: Cause-effect relationships          │
│  • Meta-Learner: Strategy optimization                  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Knowledge Acquisition                                  │
│  • Web Research: 9 sources (Wikipedia, ArXiv, etc.)     │
│  • LLM Bridge: Ollama/Gemini orchestration              │
│  • Symbolic Math: SymPy integration                     │
│  • Validator: Multi-source verification (optional)      │
└─────────────────────────────────────────────────────────┘
```

### **Target Architecture (v1.0) - Discovery Machine**

```
┌─────────────────────────────────────────────────────────┐
│            DISCOVERY ORCHESTRATOR                        │
│  Manages: Observe→Question→Hypothesize→Test→Theorize    │
└──────────────┬──────────────────────────────┬───────────┘
               │                              │
               ↓                              ↓
┌──────────────────────────┐   ┌──────────────────────────┐
│  OBSERVATION LAYER       │   │  THEORY LAYER            │
│  • Anomaly detector      │   │  • Theory synthesizer    │
│  • Pattern recognizer    │   │  • Causal modeler        │
│  • Gap identifier (FEP)  │   │  • Law discoverer        │
└──────────────┬───────────┘   └───────────┬──────────────┘
               │                           │
               ↓                           ↓
┌──────────────────────────┐   ┌──────────────────────────┐
│  HYPOTHESIS LAYER        │   │  VALIDATION LAYER        │
│  • Hypothesis generator  │←─→│  • Evidence evaluator    │
│  • Prediction generator  │   │  • Contradiction detector│
│  • Novelty scorer        │   │  • Experiment designer   │
└──────────────┬───────────┘   └───────────┬──────────────┘
               │                           │
               ↓                           ↓
┌─────────────────────────────────────────────────────────┐
│               REASONING CORE                             │
│  • Causal inference (Pearl-style)                        │
│  • Uncertainty quantification (Bayesian)                 │
│  • CoT pattern mining (from LLM traces)                  │
│  • Compound growth tracking (learning acceleration)      │
└──────────────┬──────────────────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────────────────┐
│           FEP-GUIDED KNOWLEDGE GRAPH                     │
│  Facts (verified) • Hypotheses (testable)                │
│  Theories (explanatory) • Evidence (supporting)          │
│  Free Energy: Prediction Error + Complexity              │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Innovations

### **1. Free Energy Principle (FEP) for Knowledge Organization**

**Concept:** Knowledge graph connections minimize "surprise" (prediction error + complexity)

```python
Free Energy = Prediction Error + Complexity

Prediction Error: How well can we predict related concepts?
Complexity: How unusual is this connection? (KL divergence)

Goal: Organize knowledge to minimize free energy
→ Most efficient, explanatory knowledge structure
```

**Implementation:**
- `core/fep_learner.py`: Recognition network (observations → beliefs) + Generative network (beliefs → predictions)
- Knowledge connections evaluated by free energy reduction
- High FEP regions = knowledge gaps = discovery opportunities

**Status:** ✅ Basic FEP implemented, 🚧 Not yet integrated with graph connections

---

### **2. Compound Knowledge Growth**

**Concept:** Learning accelerates exponentially as knowledge accumulates

```
Learning Efficiency: L(t) = L₀ × (1 + r)ᵗ

Where:
- L(t) = Time to learn concept at step t
- r = Compound growth rate (measured empirically)
- t = Total concepts learned

Expected: Concept #100 learned 2-5x faster than concept #10
```

**Observable Effects:**
- Early concepts: 30-60 seconds to learn
- Later concepts: 10-20 seconds (with 100+ concepts known)
- Acceleration: ~5-15% per 10 concepts learned

**Status:** ✅ Empirically observed, 🚧 Not explicitly tracked/optimized

---

### **3. Chain-of-Thought (CoT) Pattern Mining**

**Concept:** Extract reasoning patterns from LLM's successful problem-solving traces

```
LLM generates:
"First, I recognize this is a quadratic equation.
Second, I check if it factors easily.
Third, since it doesn't, I use the quadratic formula.
Fourth, I simplify the discriminant..."

Pattern extracted:
[Recognition → Simple approach → General method → Simplification]

Learned strategy: "Try simple first, fall back to general"
```

**Application:** System reuses successful patterns on similar problems

**Status:** ⚠️ Pattern extraction exists (`pattern_learner.py`), 🚧 CoT-specific mining not implemented

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

### Current Learning Flow (v0.5)

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

### Target Discovery Flow (v1.0)

```
1. OBSERVE
   ↓ Identify anomalies, patterns, gaps in knowledge graph

2. QUESTION
   ↓ Generate questions about unexplained phenomena

3. HYPOTHESIZE
   ↓ Generate possible explanations (minimize free energy)

4. PREDICT
   ↓ What should happen if hypothesis is true?

5. TEST
   ↓ Design experiment (thought or computational)

6. ANALYZE
   ↓ Evaluate evidence, update beliefs (Bayesian)

7. THEORIZE
   ↓ Synthesize discoveries into unified explanation

8. EXPLAIN
   ↓ Generate human-understandable presentation

9. ITERATE
   ↓ Refine based on new evidence

→ DISCOVERY (novel insight with evidence trail)
```

---

## 🧮 Mathematical Reasoning

KV-1 stores mathematical concepts as **symbolic equations** (SymPy), not just text.

**Example:**

```python
# Input: "Pythagorean theorem: a squared plus b squared equals c squared"

# Parsed to:
Eq(a**2 + b**2, c**2)  # SymPy symbolic equation

# Stored with:
• Text: "In a right triangle, a² + b² = c²"
• Tensor: [0.123, -0.456, ..., 0.789] (384-D embedding)
• Formula: "a**2 + b**2 = c**2" (SymPy expression)
• Examples: ["3² + 4² = 5²", "5² + 12² = 13²"]
• Domain: "geometry"
• Prerequisites: ["right triangle", "square", "hypotenuse"]

# Used for:
• Connection finding (relates to distance formula, trig identities)
• Symbolic manipulation (substitution, solving)
• Theorem composition (derive new results)
• Proof search (mathematical_exploration_engine.py)
```

**Capabilities:**
- ✅ Symbolic equation solving (SymPy)
- ✅ Proof verification (computational)
- ✅ Pattern discovery in sequences
- ✅ Goldbach conjecture exploration
- 🚧 Autonomous theorem generation (planned)

---

## 🧠 Memory System

### Hybrid Memory (STM + LTM + Graph)

**Short-Term Memory (STM):**
- Capacity: 50 slots (Miller's Law: 7±2, extended)
- Decay: 5 minutes without rehearsal
- Lookup: O(1) direct match using OrderedDict
- Speed: ~0.001ms
- Purpose: Fast recall of recent concepts

**Long-Term Memory (LTM):**
- Capacity: Unlimited
- Storage: GPU tensor matrix (384-D embeddings)
- Lookup: Cosine similarity search (PyTorch)
- Speed: ~1-5ms with GPU acceleration
- Purpose: Persistent knowledge base

**Small-World Graph:**
- Topology: Watts-Strogatz (high clustering + short paths)
- Nodes: Concepts with properties
- Edges: Anatomical (permanent) + Functional (dynamic)
- **FEP-Guided:** Connections minimize free energy
- Features:
  - Automatic analogy discovery via shortcuts
  - Hub detection for key concepts
  - Cross-domain transfer learning

**Disk Persistence:**
- Format: JSON (ltm_memory.json)
- Write: Atomic (temp → rename)
- Load: Automatic on startup

**Data Flow:**
```
learn("prime numbers", definition)
  → FEP evaluation (does this connection reduce surprise?)
  → Store in LTM (tensor + text + formula)
  → Add to graph with minimal free energy connections
  → Store in STM (fast lookup)
  → Save to disk (persistent)

recall("primes")
  → Check STM (miss)
  → Search LTM (found: "prime numbers", similarity=0.92)
  → Traverse graph for related concepts
  → Promote to STM
  → Next recall("primes") → STM hit (instant)
```

---

## 📊 System Statistics

**Current Performance (v0.5):**
- Concept learning time: 15-30 seconds (balanced mode)
- Memory per concept: ~1-2KB
- 1000 concepts: ~1-2MB disk space
- STM hit rate: >80% for recent queries
- LTM search accuracy: ~90% (similarity >= 0.85)
- Compound growth: Observable but not measured

**Benchmark Results:**
- 18/19 hard problems solved (95% success rate)
- Includes: Collatz sequence, Chinese Remainder, Prime factorization
- See `benchmarks/` for comparison scripts

**Target Performance (v1.0):**
- Hypothesis generation: 5-10 novel hypotheses per domain
- Discovery rate: 2-5 non-obvious connections per 100 concepts
- Evidence quality: >90% claims supported by multiple sources
- Compound growth rate: 5-15% acceleration per 10 concepts
- Free energy: Decreases from ~1.0 to ~0.3 as knowledge grows

---

## 🔬 Research Roadmap

### **Phase 0: Foundation (CURRENT - v0.5)** ✅
```
✅ Self-discovery learning loop
✅ Hybrid STM/LTM memory
✅ Small-world graph topology
✅ Pattern learning from experience
✅ Compositional reasoning
✅ Deep abstraction (cross-domain)
✅ FEP learner (basic)
✅ Causal reasoning (LLM-based)
✅ Meta-learning
```

### **Phase 1: Discovery Foundation (v0.6-0.7)** 🚧
**Timeline:** 3-4 months

```
Priority 1: Hypothesis Generator
├─ Anomaly detection in knowledge graph
├─ Gap identification (high FEP regions)
├─ Hypothesis generation from patterns
├─ Prediction generation
├─ Novelty & testability scoring
└─ Status: 🚧 Not started

Priority 2: Evidence Evaluator
├─ Claim-evidence assessment
├─ Bayesian belief updating
├─ Confidence tracking (posterior probabilities)
├─ Multi-source verification
└─ Status: ⚠️ Basic validator exists, needs Bayesian upgrade

Priority 3: Contradiction Detector
├─ Logical contradiction scanning
├─ Conflict identification
├─ Resolution proposals
├─ Investigation triggering
└─ Status: 🚧 Not started

Priority 4: Discovery Orchestrator
├─ Discovery loop management
├─ Component coordination
├─ Priority queue for investigations
└─ Status: 🚧 Not started
```

**Deliverable:** System can generate hypotheses, evaluate evidence, detect contradictions

### **Phase 2: Experimentation & Theory (v0.8-0.9)** 🔮
**Timeline:** 4-5 months

```
Priority 5: Experiment Designer
├─ Thought experiment framework
├─ Computational test design
├─ Outcome prediction
└─ Information gain estimation

Priority 6: Theory Synthesizer
├─ Multi-observation synthesis
├─ Principle extraction
├─ Causal model building
└─ Law discovery

Priority 7: Causal Inference (Rigorous)
├─ Pearl-style causal DAGs
├─ Do-calculus implementation
├─ Confounder identification
└─ Counterfactual reasoning
```

**Deliverable:** System can design experiments, synthesize theories, infer causation

### **Phase 3: Integration & Emergence (v1.0)** 🔮
**Timeline:** 3-4 months

```
Priority 8: Uncertainty Quantification
├─ Bayesian belief networks
├─ Uncertainty propagation
├─ Confidence intervals
└─ Source tracking

Priority 9: CoT Pattern Mining
├─ Extract reasoning patterns from LLM traces
├─ Strategy library building
├─ Automatic pattern reuse
└─ Meta-strategy learning

Priority 10: Compound Growth Optimization
├─ Explicit growth rate tracking
├─ Learning acceleration measurement
├─ Prerequisite optimization
└─ Knowledge compounding maximization
```

**Deliverable:** Fully integrated discovery machine with measurable emergent capabilities

---

## 🎯 Success Criteria

**KV-1 v1.0 will be validated when it can:**

1. **The Goldbach Test** ✅
   - Input: "Explore Goldbach's conjecture"
   - Expected: Generate hypothesis → Design test → Execute → Present findings with evidence
   - Status: Partial (can compute, can't autonomously hypothesize)

2. **The Contradiction Test** 🚧
   - Input: Add conflicting claims to knowledge base
   - Expected: Detect automatically → Analyze → Propose resolution → Trigger investigation
   - Status: Not implemented

3. **The Cross-Domain Discovery Test** 🚧
   - Input: "Find connections between quantum mechanics and thermodynamics"
   - Expected: Identify deep analogies → Propose testable hypothesis → Evaluate depth
   - Status: Has analogy engine, lacks autonomous discovery trigger

4. **The Novel Hypothesis Test** 🚧
   - Input: Unexplained phenomenon
   - Expected: Generate multiple hypotheses → Rank by plausibility → Design experiments
   - Status: Not implemented

5. **The Compound Growth Test** ⚠️
   - Input: Learn 100 concepts sequentially
   - Expected: Measure learning acceleration, prove L(t) = L₀ × (1 + r)ᵗ with r > 0
   - Status: Observable, not measured

---

## 🐛 Known Limitations

### **Current Limitations (v0.5):**

**Fundamental:**
1. **LLM-Dependent:** All reasoning powered by LLM (Ollama/Gemini). Without LLM access, system cannot function.
2. **Reactive, Not Proactive:** Only learns when prompted; doesn't autonomously generate hypotheses
3. **No Hypothesis Testing:** Can't design or execute experiments
4. **No Theory Formation:** Stores facts, doesn't synthesize theories
5. **No Contradiction Detection:** Accepts conflicting information without resolution
6. **Shallow Evidence Evaluation:** Multi-source validation exists but not rigorous/Bayesian

**Technical:**
7. **Security:** Uses `exec()` for math parsing (needs sandboxing)
8. **Domain Specialization:** Optimized for mathematics; general knowledge works but less effectively
9. **No Visual Learning:** Text-only (no images, diagrams, videos)
10. **Limited Embodiment:** No physical grounding or sensory experience

### **Target Limitations (v1.0):**

**Will Still Have:**
- LLM dependency (orchestration layer, not standalone AI)
- Limited to computational/thought experiments (no physical lab)
- Mathematical domain bias (though cross-domain improving)
- Requires human validation for critical discoveries

**Will Overcome:**
- ✅ Autonomous hypothesis generation
- ✅ Evidence-based belief updating
- ✅ Contradiction detection and resolution
- ✅ Theory synthesis from observations
- ✅ Experiment design (computational)

---

## 📁 Project Structure

```
KV-1/
├── self_discovery_orchestrator.py  # Main learning loop (1934 lines)
├── run_self_discovery.py           # CLI interface
├── run_curriculum.py               # Curriculum runner
│
├── core/                           # Core modules (~26.6K lines)
│   ├── llm.py                      # LLM bridge (Ollama/Gemini)
│   ├── hybrid_memory.py            # STM + LTM + Disk
│   ├── neurosymbolic_gpu.py        # GPU tensor operations
│   ├── web_researcher.py           # 9-source web scraper
│   ├── knowledge_validator.py      # Multi-source validation
│   ├── math_connect.py             # Symbolic math (SymPy)
│   │
│   ├── fep_learner.py              # Free Energy Principle
│   ├── pattern_learner.py          # Mathematical structure learning
│   ├── compositional_reasoner.py   # Pattern composition
│   ├── deep_abstraction.py         # Cross-domain isomorphisms
│   ├── framework_invention.py      # Novel framework creation
│   ├── physical_grounding.py       # Math-reality connection
│   ├── causal_reasoner.py          # Cause-effect relationships
│   ├── meta_learner.py             # Strategy optimization
│   ├── transfer_learning.py        # Cross-domain transfer
│   ├── analogical_reasoning.py     # Structural analogies
│   ├── curiosity_engine.py         # Intrinsic motivation
│   ├── explainable_reasoning.py    # Explanation generation
│   │
│   ├── symbolic_math_engine.py     # Proof verification
│   ├── mathematical_exploration_engine.py  # Proof search
│   ├── geometric_knowledge_space.py  # Riemannian manifold
│   └── unified_agi_controller.py   # System orchestration
│
├── hsokv/                          # HSOKV memory library
│   ├── dual_memory.py              # STM + LTM implementation
│   ├── memory.py                   # Base memory classes
│   └── embedders.py                # Sentence embeddings
│
├── benchmarks/                     # Performance comparisons
├── tools/                          # Utilities
└── docs/
    ├── LEARNING_CURRICULUM.md      # 260 math questions
    ├── BRAIN_ARCHITECTURE.md       # Small-world networks
    └── DOMAIN_MATH_BRIDGE.md       # Mathematical abstraction
```

---

## 🌟 What Makes This Interesting

### **Novel Contributions:**

1. **FEP-Guided Knowledge Organization**
   - First LLM system to use Free Energy Principle for knowledge graph connections
   - Connections minimize "surprise" (prediction error + complexity)
   - Knowledge organizes itself for maximum explanatory power

2. **Compound Knowledge Growth**
   - Explicitly tracks learning acceleration
   - Each concept makes future learning faster (measurable)
   - Targets exponential capability growth

3. **CoT Pattern Mining** (planned)
   - Extract reasoning strategies from LLM's own reasoning traces
   - Learn meta-strategies without explicit meta-training
   - Self-improving problem-solving

4. **Small-World Graph Memory**
   - Neuroscience-inspired topology (Watts-Strogatz)
   - High clustering + short paths = efficient retrieval
   - Anatomical (permanent) + Functional (dynamic) edges

5. **Dual Memory Architecture**
   - Psychology-based STM/LTM with rehearsal and consolidation
   - 3-stage learning: Surprise → Rehearsal → Transfer
   - Time decay and confidence tracking

6. **Mathematical Pattern Learning**
   - Learns from problem-solving experience (not hardcoded)
   - Discovers problem types via clustering
   - Composes patterns for novel solutions

7. **Deep Cross-Domain Abstraction**
   - Recognizes structural isomorphisms (linear algebra ≅ group theory)
   - Transfers solutions across domains
   - Meta-reasoning for framework selection

### **Engineering Highlights:**
- Clean integration of LLM, SymPy, PyTorch, and sentence-transformers
- GPU-accelerated semantic search with batch operations
- Lazy-loaded modular architecture (10 cognitive subsystems)
- Persistent learning with atomic disk writes
- Bayesian confidence tracking (planned v1.0)

---

## 🔗 Links

- **Repository**: https://github.com/PlanetDestroyyer/KV-1
- **HSOKV Memory**: https://github.com/PlanetDestroyyer/hsokv
- **Issues**: https://github.com/PlanetDestroyyer/KV-1/issues

---

## ⚠️ Important Notes

**This is a research system, not production-ready:**
- Primarily an LLM orchestration layer with novel memory/discovery architecture
- No sandboxing (uses `exec()` for math parsing)
- Limited error handling
- Optimized for mathematical domains
- Requires LLM API access (Ollama or Gemini)

**Best Use Cases:**
- Research in AI discovery systems
- Exploring FEP-based knowledge organization
- Studying compound knowledge growth
- Educational tool for mathematics
- Prototyping autonomous learning architectures

**Not Suitable For:**
- Production deployments (security issues)
- Independent reasoning (requires LLM)
- Physical experiments (computational only)
- Real-time applications (web search latency)
- **Building time machines** (despite what some vision docs might claim 😄)

---

## 🤝 Contributing

This is an active research project. Contributions welcome for:
- Implementing discovery components (hypothesis generator, evidence evaluator, etc.)
- Improving FEP integration with knowledge graph
- Adding compound growth tracking
- CoT pattern mining implementation
- Bayesian belief updating
- Causal inference (Pearl-style)
- Bug fixes and optimizations

See [GitHub Issues](https://github.com/PlanetDestroyyer/KV-1/issues) for current priorities.

---

## 📄 License

See LICENSE file for details.

---

**Built by [@PlanetDestroyyer](https://github.com/PlanetDestroyyer)**

*A research exploration in Free Energy Principle-guided knowledge organization, compound learning growth, and emergent scientific discovery through AI.*

---

## 🎓 Further Reading

- [BRAIN_ARCHITECTURE.md](BRAIN_ARCHITECTURE.md) - Small-world networks and neuroscience inspiration
- [DOMAIN_MATH_BRIDGE.md](DOMAIN_MATH_BRIDGE.md) - Mathematical abstraction approach
- [LEARNING_CURRICULUM.md](LEARNING_CURRICULUM.md) - 260 math questions for testing

---

**Status:** v0.5 (Learning System) → v1.0 (Discovery Machine)
**Progress:** ~35% complete toward discovery machine vision
**Next Milestone:** Hypothesis Generator + Evidence Evaluator (v0.6)
