# Testing Guide - Problem-Solving Engine

## Quick Start

### Test on Small Sample (Fast - 5 questions)
```bash
python test_on_curriculum.py --max 5
```

### Test on First 50 Questions
```bash
python test_on_curriculum.py --max 50
```

### Test on All 272 Questions (Full Curriculum)
```bash
# Verbose mode (shows each question)
python test_on_curriculum.py

# Quiet mode (faster, shows progress every 10 questions)
python test_on_curriculum.py --quiet
```

### Test Specific Range
```bash
# Test questions 100-150
python test_on_curriculum.py --start 100 --max 50
```

### Test Without Saving Results
```bash
python test_on_curriculum.py --max 20 --no-save
```

## What the Test Does

1. **Loads Curriculum**: Parses all 272 questions from `LEARNING_CURRICULUM.md`
2. **Initializes System**: Sets up all 8 core components + LLM
3. **Solves Questions**: Runs problem solver on each question
4. **Tracks Learning**: Monitors compound growth and pattern development
5. **Saves Results**: Outputs JSON file with detailed results

## Expected Output

### During Test
```
================================================================================
                PROBLEM-SOLVING ENGINE - CURRICULUM TEST
================================================================================

[1/5] Loading curriculum...
  ✓ Loaded 272 questions from curriculum
  Testing on 50 questions (#1 to #50)

[2/5] Initializing LLM Bridge...
  ✓ LLM ready (Qwen3:4b via Ollama)

[3/5] Initializing Problem-Solving Engine...
  ✓ Vector Store (FAISS)
  ✓ Knowledge Graph (FEP-guided)
  ✓ Memory System
  ✓ CoT Pattern Miner
  ✓ Bayesian Reasoner
  ✓ Compound Growth Tracker
  ✓ Meta-Cognitive Monitor

[4/5] Testing on Questions...
================================================================================

Question 1/50 (Curriculum #1)
Q: What is addition and how does it work?
Domain: arithmetic
--------------------------------------------------------------------------------
[Solving process...]
✓ Solved in 2.3s (confidence: 45.0%)

Question 2/50 (Curriculum #2)
Q: What is multiplication and how does it relate to addition?
Domain: arithmetic
--------------------------------------------------------------------------------
[Solving process...]
✓ Solved in 1.8s (confidence: 62.0%)

[...]
```

### Final Statistics
```
================================================================================
[5/5] FINAL RESULTS & STATISTICS
================================================================================

📊 TEST SUMMARY:
  Questions attempted: 50
  Questions solved: 50
  Success rate: 100.0%
  Total time: 2.5 minutes
  Average time per question: 3.0s

🧠 LEARNING PROGRESS:
  Problems solved: 50
  Patterns learned: 8
  Memories stored: 35
  Knowledge concepts: 12
  Average solve time: 3.0s
  Speedup factor: 2.1x
  ✅ LEARNING ACCELERATION DETECTED! 2.1x faster!

🚀 COMPOUND GROWTH:
  Growth rate: 0.0234
  Acceleration: 18.5%
  Learning speedup: 2.1x

📚 PERFORMANCE BY DOMAIN:
  arithmetic          : 15/15 (100.0%) - avg 2.1s
  algebra             : 12/12 (100.0%) - avg 2.8s
  geometry            : 10/10 (100.0%) - avg 3.2s
  calculus            : 8/8 (100.0%) - avg 4.1s
  number_theory       : 5/5 (100.0%) - avg 3.5s

💾 Results saved to: curriculum_test_results_20250106_143022.json
```

## Results File Format

The JSON output file contains:
```json
{
  "metadata": {
    "timestamp": "20250106_143022",
    "total_questions": 50,
    "success_rate": 1.0,
    "total_time_seconds": 150.5,
    "start_question": 1,
    "end_question": 50
  },
  "summary": {
    "problems_solved": 50,
    "patterns_learned": 8,
    "speedup_factor": 2.1,
    "compound_growth": {...}
  },
  "domain_stats": {
    "arithmetic": {
      "attempted": 15,
      "solved": 15,
      "total_time": 31.5
    },
    ...
  },
  "results": [
    {
      "question_number": 1,
      "question": "What is addition and how does it work?",
      "domain": "arithmetic",
      "difficulty": 0.12,
      "solution": "Addition is...",
      "confidence": 0.45,
      "time_taken": 2.3,
      "patterns_used": 0,
      "success": true
    },
    ...
  ]
}
```

## Performance Expectations

### With LLM (Ollama Running)
- **Speed**: 2-5 seconds per question (depends on LLM response time)
- **Quality**: High-quality answers with reasoning
- **Learning**: Strong pattern extraction from LLM reasoning

### Without LLM (Fallback Mode)
- **Speed**: 0.5-2 seconds per question (much faster)
- **Quality**: Pattern-based answers (simpler but functional)
- **Learning**: Pattern reuse and memory adaptation

### Compound Growth Effect
After solving ~20-30 questions, you should see:
- Speedup factor > 1.5x (50% faster)
- More patterns learned and reused
- Higher confidence on familiar domains
- Faster solving times for similar problems

## Test Scenarios

### Scenario 1: Quick Sanity Check (5 questions)
```bash
python test_on_curriculum.py --max 5
# Expected: ~15 seconds, basic functionality verification
```

### Scenario 2: Learning Curve Demo (30 questions)
```bash
python test_on_curriculum.py --max 30
# Expected: ~2 minutes, shows compound growth starting
```

### Scenario 3: Full Curriculum (272 questions)
```bash
# With LLM
python test_on_curriculum.py --quiet
# Expected: 15-20 minutes, strong compound growth

# Without LLM (faster)
python test_on_curriculum.py --quiet
# Expected: 5-8 minutes, demonstrates pattern learning
```

### Scenario 4: Advanced Topics Only (questions 200-272)
```bash
python test_on_curriculum.py --start 200
# Expected: Tests on hardest questions (Riemann Hypothesis level)
```

## Troubleshooting

### If Ollama Not Running
- System will use pattern-based fallback
- Tests will run faster but with simpler answers
- Learning still works (patterns, memory, compound growth)

### If Tests Run Slowly
- Use `--quiet` flag for less output
- Start Ollama may be slow to respond (normal)
- First few questions are slower (building knowledge base)

### If Getting Errors
- Check dependencies: `pip install numpy faiss-cpu networkx ollama`
- Check Ollama is accessible: `ollama list`
- Try with `--max 5` first to isolate issues

## Interpreting Results

### Success Rate
- **90-100%**: Excellent - system is working well
- **70-90%**: Good - most questions solved
- **<70%**: Check for errors in specific domains

### Speedup Factor
- **>2.0x**: Strong compound growth - learning is accelerating
- **1.5-2.0x**: Moderate growth - system is improving
- **<1.5x**: Weak growth - may need more questions to show effect

### Pattern Learning
- **8+ patterns**: Good pattern extraction
- **3-7 patterns**: Basic patterns identified
- **<3 patterns**: May need more diverse problems

### Domain Performance
- Compare success rates across domains
- Identify strengths (high success rate)
- Identify weaknesses (low success rate)
- Track improvement over time

## Next Steps After Testing

1. **Analyze Results**: Look at the JSON file for detailed insights
2. **Identify Patterns**: See which reasoning patterns emerged
3. **Domain Analysis**: Check which mathematical areas are strongest
4. **Compound Growth**: Verify learning acceleration is happening
5. **Iterate**: Test again with improvements to measure progress

---

**Start with a small test, then scale up!**
```bash
# Quick test first
python test_on_curriculum.py --max 10

# Then full test if it works
python test_on_curriculum.py --quiet
```
