# Benchmark Setup Guide

## Prerequisites

The benchmark system supports two LLM providers:

### 1. **Gemini** (Recommended for benchmarks)
- Free API with generous limits (15 RPM, 1M tokens/day)
- No local setup needed
- Best for final benchmark results

### 2. **Ollama** (Recommended for development)
- Completely free, no rate limits
- Requires local installation
- Best for testing and iteration

---

## Setup Steps

### Option A: Gemini Setup (Cloud)

```bash
# 1. Get API key from https://makersuite.google.com/app/apikey

# 2. Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# 3. Install dependencies
pip install google-generativeai requests beautifulsoup4

# 4. Test it works
python -c "from core.llm import LLMBridge; llm = LLMBridge(provider='gemini'); print(llm.describe())"

# 5. Run quick benchmark
python benchmarks/compare_baselines.py --provider gemini --model gemini-2.5-flash --quick
```

### Option B: Ollama Setup (Local)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama service
ollama serve

# 3. In another terminal, pull model
ollama pull qwen3:4b

# 4. Install dependencies
pip install ollama requests beautifulsoup4

# 5. Test it works
python -c "from core.llm import LLMBridge; llm = LLMBridge(provider='ollama', default_model='qwen3:4b'); print(llm.describe())"

# 6. Run quick benchmark
python benchmarks/compare_baselines.py --provider ollama --model qwen3:4b --quick
```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError` or import errors:

```bash
# Clean install in virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
pip install google-generativeai  # For Gemini support
```

### Gemini API Errors

- **"API key not provided"**: Set `GEMINI_API_KEY` environment variable
- **"Rate limit exceeded"**: Wait 60 seconds or use `--quick` mode
- **"Model not found"**: Check available models at https://ai.google.dev/models/gemini

### Ollama Errors

- **"Failed to connect"**: Make sure `ollama serve` is running
- **"Model not found"**: Run `ollama pull qwen3:4b`
- **Connection refused**: Check Ollama is running on http://localhost:11434

---

## Running Your First Benchmark

Start with the quickest test:

```bash
# Test only LLM-alone (fastest, no learning)
python benchmarks/compare_baselines.py --methods llm-alone --quick

# Expected output:
# [LLM Alone] Problem 1/3: Find the value of x where x^x = 256...
#   ✓ 2.3s
# ...
# Results: 1/3 correct (33.3%)
```

Then try with learning:

```bash
# Test KV-1 vs LLM-alone
python benchmarks/compare_baselines.py --methods kv1 llm-alone --quick
```

Finally, run full comparison:

```bash
# All methods, all problems (~15 minutes with Gemini)
python benchmarks/compare_baselines.py --provider gemini --model gemini-2.5-flash
```

---

## Quick Reference

| Command | What it does | Time |
|---------|-------------|------|
| `--quick` | Test on first 3 problems | ~5 min |
| `--methods llm-alone` | Only test LLM baseline | ~1 min |
| `--methods kv1` | Only test KV-1 learning | ~10 min |
| (default) | All methods, all problems | ~15 min |

---

## Expected Results

Based on initial testing, you should see:

```
KV-1 + Learning          ~90-95%  (slow but accurate)
LLM + Few-shot           ~60-75%  (fast, decent)
LLM + RAG                ~50-65%  (moderate)
LLM Alone                ~30-45%  (fast but weak)
```

The exact results will depend on your model choice!

---

## Next Steps After Setup

1. **Quick test**: `python benchmarks/compare_baselines.py --quick`
2. **Review results**: Check `BENCHMARK_RESULTS_*.txt`
3. **Iterate**: Adjust KV-1 parameters to improve
4. **Full benchmark**: Run without `--quick` for paper results
