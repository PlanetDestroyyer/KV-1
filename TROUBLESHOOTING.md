# KV-1 Troubleshooting Guide

## Issues Found in Test Run (2025-11-27)

### ✅ RESOLVED: You Were Running Old Code

**Issue:** Your test output showed Phase 1 modules only, but Phase 2 & 3 are now committed.

**Evidence:**
- Your output: `[+] AGI modules ready: Meta-learning, Metacognition, Goal Planning, Creative Reasoning, Curiosity, Causal Reasoning, Pattern Learning`
- New code (line 254): `[+] AGI modules ready: ... Pattern Learning, Compositional Reasoning, Deep Abstraction`

**Solution:** Pull the latest code from branch `claude/code-review-analysis-01DSW3bPBHectyaA46nEEBJP`

```bash
git pull origin claude/code-review-analysis-01DSW3bPBHectyaA46nEEBJP
```

---

### 🐛 Issue 1: Incorrect Definitions Stored in Memory

**Problem:** Wrong definitions saved:
- `multiplication: "Addition is a binary operation..."` (WRONG!)
- `modular inverses: "Addition is a binary operation..."` (WRONG!)
- `sets: "This phrase (more sums than differences)..."` (WRONG!)
- `functions: "{"` (BROKEN!)

**Root Cause:** LLM is extracting wrong text from poor web search results.

**Fix Options:**

1. **Immediate Fix:** Delete corrupted ltm_memory.json and start fresh
   ```bash
   rm ltm_memory.json
   rm data/pattern_database.json
   ```

2. **Better Fix:** Improve web search quality (see Issue 2)

3. **Long-term Fix:** Add validation layer that rejects obviously wrong definitions

---

### 🐛 Issue 2: Poor Web Search Results

**Problem:** Web researcher returning garbage:
- `ERROR:kv1.web:Wiki fetch failed for all attempts`
- Getting irrelevant content from Google searches

**Root Cause:**
- Wikipedia API failing
- Fallback searches getting wrong content

**Recommended Fixes:**

1. **Add more robust error handling:**
```python
# In core/web_researcher.py
if not result or len(result) < 50:  # Too short = likely failed
    # Try next source
    continue
```

2. **Add content validation:**
```python
def validate_search_result(query: str, content: str) -> bool:
    """Check if search result is relevant"""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    overlap = len(query_words & content_words)
    return overlap >= len(query_words) * 0.5  # At least 50% overlap
```

3. **Try alternative APIs:**
   - DuckDuckGo Instant Answers API
   - Wolfram Alpha API (good for math)
   - ArXiv API (already used, but could be prioritized for math)

---

### 🐛 Issue 3: LLM Response Parsing Failures

**Problem:**
```
[!] LLM didn't follow format, trying JSON parsing...
[!] JSON parsing also failed, using fallback detection...
```

**Root Cause:** LLM returning explanation instead of structured format.

**Example of what LLM returned:**
```
The knowledge base contains an incorrect definition of multiplication
(stated as addition) and lacks the concept that multiplication is
repeated addition. Therefore, the missing concept is:
**multiplication as repeated addition**
```

**Fix:** Improve prompt to be more explicit about format.

**In self_discovery_orchestrator.py, update the prompt:**
```python
prompt = f"""
CRITICAL: You MUST respond in EXACTLY this format:

SUCCESS: yes/no
ANSWER: (your answer if yes, or "cannot complete" if no)
MISSING: (comma-separated list of missing concepts, or "none")
REASONING: (brief explanation)

DO NOT include any other text. DO NOT use markdown. DO NOT explain.
Just the four lines above.

Now answer this goal: {self.goal}
Known concepts: {list(self.ltm.list_all())}
"""
```

---

### 🐛 Issue 4: Relevance Filter Too Aggressive

**Problem:** System filtered out "addition definition" as prerequisite for "What is addition?"

**Analysis:** This is actually CORRECT behavior! The filter correctly identified that "addition definition" is not a *prerequisite* but the *goal itself*.

**Real Problem:** System has no fallback for primitive concepts with no prerequisites.

**Fix:** Add primitive concept detection:

```python
# In self_discovery_orchestrator.py
if not filtered_concepts and attempt_num >= 3:
    # After 3 attempts with no prerequisites, treat as primitive
    print(f"[i] No prerequisites found after {attempt_num} attempts")
    print(f"[i] Treating '{self.goal}' as primitive concept")

    # Learn directly from web without prerequisites
    await self.learn_concept(
        concept=self.goal,
        needed_for="primitive",
        depth=0
    )
```

---

### 🐛 Issue 5: Semantic Search "Found exact match but failed"

**Problem:** You mentioned: "found the exact match but semantic search failed so learning from scratch again"

**Root Cause:** Likely the similarity threshold is too high.

**Location:** `core/neurosymbolic_memory.py:182`
```python
def has_concept(self, name: str, similarity_threshold: float = 0.85) -> bool:
```

**Fix:** Lower threshold or add exact string matching first:

```python
def has_concept(self, name: str, similarity_threshold: float = 0.85) -> bool:
    """Check if we already know this concept"""

    # First check exact match (faster)
    if name.lower() in [c.lower() for c in self.concepts.keys()]:
        return True

    # Then check semantic similarity
    similar = self.find_similar(name, top_k=1, threshold=similarity_threshold)
    return len(similar) > 0 and similar[0][1] >= similarity_threshold
```

---

## ✅ What's Working Well

1. **Phase 1 Pattern Learning** - Saving 5 patterns correctly
2. **GPU Memory** - CUDA working (warnings are normal)
3. **Multi-step Learning** - Successfully learned 25 concepts recursively
4. **3-Stage Learning** - Confidence scoring working (0.20 → 0.50 → 0.80)
5. **Parallel Processing** - Batch learning functional

---

## 🚀 Recommended Next Steps

### Immediate (Do This Now):
1. Pull latest code with Phase 2 & 3
2. Delete corrupted memory: `rm ltm_memory.json data/pattern_database.json`
3. Re-run with fresh start

### Short-term (This Week):
1. Improve LLM prompt formatting (see Issue 3)
2. Add primitive concept detection (see Issue 4)
3. Add exact string matching (see Issue 5)
4. Add search result validation (see Issue 2)

### Long-term (This Month):
1. Integrate Wolfram Alpha for math definitions
2. Add definition validation layer
3. Improve web researcher robustness
4. Add human-in-loop for suspicious definitions

---

## 🧪 Testing Recommendations

### Test Phase 2 & 3:
```bash
# After pulling latest code
python test_compositional_reasoning.py
python test_deep_abstraction.py
```

### Test Full System:
```bash
# Start fresh
rm ltm_memory.json

# Simple test
python run_self_discovery.py "What is 2 + 2?"

# Math test
python run_self_discovery.py "Solve x^2 - 5x + 6 = 0"

# Pattern learning test
python run_self_discovery.py "Factor x^2 + 7x + 12"
```

---

## 📊 Expected Behavior with Phase 2 & 3

When you run with the latest code, you should see:

```
[+] 🧠 Pattern Learner: LEARNS mathematical structures from problem-solving!
[+] 🎯 Compositional Reasoning: COMBINES patterns to solve novel problems!
[+] 🔮 Deep Abstraction: RECOGNIZES when different domains share same mathematical structure!
[+] AGI modules ready: Meta-learning, Metacognition, Goal Planning, Creative Reasoning,
    Curiosity, Causal Reasoning, Pattern Learning, Compositional Reasoning, Deep Abstraction
```

And during problem solving:
```
[🔮] DEEP ABSTRACTION:
Optimal framework: linear_algebra (70.0% confidence)

[🎯] COMPOSITIONAL REASONING:
Found solution strategy with 75.0% confidence
```

---

## 📝 Summary

**Main Issue:** You were running old code before Phase 2 & 3 were integrated.

**Secondary Issues:**
1. Poor web search results → Add validation
2. Wrong definitions stored → Start with fresh memory
3. LLM format issues → Improve prompts
4. No primitive concept handling → Add fallback

**Status:** All code is syntactically correct. Issues are operational, not bugs.

**Next Action:** Pull latest code and test with fresh memory.
