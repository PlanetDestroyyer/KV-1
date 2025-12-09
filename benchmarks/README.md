# KV-1 Benchmark Suite

Comprehensive comparison of KV-1 against baseline methods.

## Methods Compared

### 1. **KV-1 + Learning** (Your System)
- Autonomous learning through failure
- Web-powered knowledge discovery
- Persistent long-term memory
- Validates concepts before storing

### 2. **LLM Alone**
- Direct queries to LLM
- No learning, no retrieval, no examples
- Baseline performance

### 3. **LLM + RAG**
- Retrieves web content per problem
- No persistent memory between problems
- Common industry approach

### 4. **LLM + Few-Shot**
- Provides example problems in prompt
- Traditional few-shot learning
- No web access

---

## Quick Start

### Using Gemini (Recommended for benchmarks)

```bash
# Set API key
export GEMINI_API_KEY="your-key-here"

# Quick test (first 3 problems)
python benchmarks/compare_baselines.py --provider gemini --model gemini-2.5-flash --quick

# Full benchmark (all 8 problems)
python benchmarks/compare_baselines.py --provider gemini --model gemini-2.5-flash
```

### Using Ollama (Local, Free)

```bash
# Make sure Ollama is running
ollama serve

# Pull model
ollama pull qwen3:4b

# Run benchmark
python benchmarks/compare_baselines.py --provider ollama --model qwen3:4b --quick
```

---

## Usage Options

### Test specific methods only

```bash
# Test only LLM alone and Few-shot (fastest)
python benchmarks/compare_baselines.py --methods llm-alone few-shot --quick

# Test only KV-1 vs LLM alone
python benchmarks/compare_baselines.py --methods kv1 llm-alone --quick

# All methods
python benchmarks/compare_baselines.py --methods llm-alone few-shot rag kv1
```

### Custom configuration

```bash
# Use different model
python benchmarks/compare_baselines.py --provider gemini --model gemini-1.5-pro

# Save results with timestamp
python benchmarks/compare_baselines.py --provider gemini --quick
# Creates: BENCHMARK_RESULTS_20250120_143022.txt
```

---

## Expected Runtime

With Gemini (15 RPM limit):

| Mode | Problems | Approx Time |
|------|----------|-------------|
| Quick (3 problems) | 3 × 4 methods | ~5 minutes |
| Full (8 problems) | 8 × 4 methods | ~15 minutes |
| KV-1 only | 8 problems | ~10 minutes |

---

## Output Format

```
======================================================================
KV-1 BENCHMARK COMPARISON RESULTS
======================================================================

Method                    Accuracy     Time (avg)   Score
----------------------------------------------------------------------
KV-1 + Learning           8/8 (100.0%)   45.2s        100.0/100
LLM + Few-shot            6/8 ( 75.0%)    5.3s         75.0/100
LLM + RAG                 5/8 ( 62.5%)    8.1s         62.5/100
LLM Alone                 3/8 ( 37.5%)    3.2s         37.5/100
======================================================================
```

Results are saved to `BENCHMARK_RESULTS_[timestamp].txt` with detailed breakdown.

---

## API Costs (Gemini)

Gemini 1.5-flash is **FREE** with generous limits:
- 15 requests/minute
- 1 million tokens/day
- $0.00 cost

Full benchmark (~200 API calls) = **$0.00** 🎉

---

## Troubleshooting

### "LLM not configured properly"
- **Gemini**: Set `GEMINI_API_KEY` environment variable
- **Ollama**: Make sure `ollama serve` is running

### Rate limit errors
- Use `--quick` to test on fewer problems
- Add delays between problems (modify code)
- Use Ollama for unlimited local testing

### "Module not found"
```bash
pip install google-generativeai requests beautifulsoup4 ollama
```

---

## Adding New Baselines

Create a new file `benchmarks/your_baseline.py`:

```python
from benchmarks.benchmark_utils import BenchmarkProblem, BenchmarkResult

class YourBaseline:
    def solve_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        # Your implementation
        pass

    def run_benchmark(self, problems):
        results = []
        for problem in problems:
            result = self.solve_problem(problem)
            results.append(result)
        return results
```

Then add to `compare_baselines.py`.
