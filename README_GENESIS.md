# 🧬 Genesis Experiment: Self-Discovery from Alphanumerics

## What's Happening Here

**The Setup:**
- System memory starts with ONLY: `a-z` and `0-9` (36 symbols)
- Gemma 4B LLM is told via system prompt: "You know nothing except these 36 symbols"
- LLM has web search capability
- 3-stage biological learning mechanism (surprise → rehearsal → consolidation)

**The Process:**

```
1. LLM (constrained): "I only know a-z and 0-9. What should I learn first?"
2. LLM decides: "I should search for 'what is a'"
3. System searches web → gets result
4. 3-Stage Learning extracts new words from result
5. New words → Surprise episodes in STM (working memory)
6. LLM uses these words repeatedly (rehearsal)
7. After 4+ uses + 94% confidence → Transfer to LTM (permanent memory)
8. LTM grows: 36 symbols → 100 words → 1000 concepts → algebra → calculus
```

**The Experiment:**
Can the system self-discover:
- Basic words (first hour)
- Grammar and syntax (hours 6-12)
- Mathematical operators (hours 12-24)
- Algebra (days 2-3)
- Calculus (days 4-7)
- Thermodynamics (days 4-7)

## How It Works

### System Prompt (Constrains LLM)
```
YOU KNOW ONLY: a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9

CONSTRAINTS:
- DO NOT use knowledge from pre-training
- If you don't know something, you MUST search the web first
- Build knowledge incrementally

STRATEGY:
1. Start with simple searches like "what is a"
2. Learn common words first
3. Use learned words to search for more complex concepts
```

### Self-Directed Learning Loop

Every 3 minutes:

1. **Decide** what to learn next (LLM self-reflection)
2. **Search** web for that concept
3. **Extract** new words from results (surprise detection)
4. **Store** in STM (working memory, 7±2 capacity)
5. **Rehearse** by using concepts repeatedly
6. **Consolidate** to LTM after 4+ uses (permanent storage)
7. **Evaluate** progress on algebra/calculus/thermodynamics
8. **Sleep** every 10 cycles (batch consolidation)

### Memory Architecture

```
STM (Short-Term Memory)
├─ Capacity: 7±2 items (Miller's magic number)
├─ Decay: 30 seconds without rehearsal
└─ Purpose: Learn new concepts

        ↓ (4+ uses + 94% confidence)

LTM (Long-Term Memory)
├─ Capacity: Unlimited
├─ Decay: Never (frozen embeddings)
└─ Purpose: Permanent knowledge base
```

## Running It

### Quick Test (5 min)
```bash
python run_genesis_experiment.py --quick
```

### 24-Hour Run
```bash
python run_genesis_experiment.py --iterations 480 --interval 180
```

### Monitor Progress
```bash
python monitor_genesis.py
```

## What You'll See

```
============================================================
[Iteration 1] Self-Learning Cycle
============================================================
🤔 Deciding what to learn next...
💡 Decision: I should search for basic words like "what" and "is"...
🔍 Searching based on decision: 'i should search'
🧠 Reviewing working memory...
  ✓ Working memory: 3 active concepts

📚 Currently Learning:
  - 'search' (conf: 65%, uses: 2/4)
  - 'should' (conf: 70%, uses: 1/4)
  - 'basic' (conf: 62%, uses: 0/4)

============================================================
```

After a few hours:
```
📸 HOURLY SNAPSHOT
============================================================
⏱️  Hours elapsed: 4.2
🧠 Phase: LEARNING
📚 LTM size: 247 learned concepts
🔄 STM size: 5/7
⚡ Total surprises: 1,834
✅ Total transfers: 189
🌐 Web requests today: 38

🎯 Domain Progress:
  algebra         [████████░░░░░░░░░░░░] 42.3%
  calculus        [███░░░░░░░░░░░░░░░░░] 15.7%
  thermodynamics  [█░░░░░░░░░░░░░░░░░░░] 3.2%
============================================================
```

## The Key Insight

**Traditional AI:**
- Pre-trained on billions of tokens
- Knows everything already
- Can't learn new things without retraining

**Genesis System:**
- Starts knowing NOTHING (except symbols)
- Learns through web search
- Memory grows incrementally
- Never forgets (frozen embeddings)
- Self-directed (decides what to learn)

**This proves:** Intelligence can emerge from minimal seed knowledge + autonomous learning + memory.

## Success Criteria

After 7 days:
- ✅ Algebra: 90%+
- ✅ Calculus: 85%+
- ✅ Thermodynamics: 80%+

If successful, we've demonstrated:
1. Self-directed learning works
2. 3-stage biological learning scales
3. Frozen embeddings prevent forgetting
4. Web research can replace pre-training

---

**Current Status:** Ready to run. Waiting for Ollama + HSOKV installation.

**Start with:** `python run_genesis_experiment.py --quick`
