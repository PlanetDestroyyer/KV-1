# KV-1 AGI System - Installation Guide

## Quick Start

### Install All Dependencies
```bash
pip install -r requirements.txt
```
**Installs**: All necessary packages
- Tensor reasoning ✅
- Web research ✅
- LLM integration ✅
- Testing & development tools ✅
- Visualization & benchmarking ✅
- Advanced ML libraries ✅

---

## Installation Steps

### 1. Prerequisites
```bash
# Python 3.9+ required
python --version

# Git (for cloning)
git --version
```

### 2. Clone Repository
```bash
git clone https://github.com/PlanetDestroyyer/KV-1.git
cd KV-1
```

### 3. Install Dependencies

#### For CPU-only systems:
```bash
# Install PyTorch for CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

#### For GPU systems (CUDA 11.8):
```bash
# Install PyTorch for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

#### For GPU systems (CUDA 12.1+):
```bash
# Install PyTorch for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Install HSOKV (Memory System)
HSOKV is already included as a submodule at `./hsokv/`. No additional installation needed.

If you want to install it separately:
```bash
cd hsokv
pip install -e .
cd ..
```

### 5. Set Up LLM (Ollama)

#### Install Ollama:
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.com/download
```

#### Pull recommended model:
```bash
# Primary model (7B, best quality)
ollama pull qwen2.5:7b-instruct

# Alternative lightweight model (4B, faster)
ollama pull qwen2.5:4b-instruct
```

### 6. Verify Installation

```bash
# Test tensor reasoning system
python -c "
from core.tensor_reasoning_system import TensorReasoningSystem
system = TensorReasoningSystem()
print('✅ Tensor reasoning system loaded!')
stats = system.get_stats()
print(f'✅ {stats[\"primitives\"][\"axioms\"]} axioms, {stats[\"primitives\"][\"operations\"]} operations')
"

# Test unified AGI learner
python -c "
from core.unified_agi_learner import UnifiedAGILearner
print('✅ Unified AGI learner loaded!')
"

# Run test problem
python run_self_discovery.py "solve x + 3 = 7"
```

---

## Package Details

### All Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | GPU-accelerated tensors |
| numpy | ≥1.24.0 | Numerical computing |
| sympy | ≥1.12 | Symbolic mathematics |
| sentence-transformers | ≥2.2.0 | Semantic embeddings |
| beautifulsoup4 | ≥4.12.0 | Web scraping |
| requests | ≥2.31.0 | HTTP requests |
| ollama | ≥0.3.0 | Local LLM integration |
| pytest | ≥7.0.0 | Testing framework |
| matplotlib | ≥3.5.0 | Visualization |
| ... and more | | See requirements.txt for full list |

---

## Troubleshooting

### PyTorch Installation Issues
If PyTorch doesn't install correctly:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall for your system
# Visit: https://pytorch.org/get-started/locally/
# Select your OS, CUDA version, and get the correct command
```

### CUDA Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, either:
# 1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
# 2. Use CPU version (slower but works)
```

### Ollama Connection Issues
```bash
# Check Ollama is running
ollama list

# Start Ollama service
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Memory Issues
If you run out of memory:
1. Use minimal install instead of full
2. Use smaller LLM model (qwen2.5:4b instead of 7b)
3. Reduce tensor dimension in configs (384 instead of 768)

---

## System Requirements

### Minimum:
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 10 GB free
- **Python**: 3.9+
- **OS**: Linux, macOS, or Windows

### Recommended:
- **CPU**: 8+ cores
- **RAM**: 16 GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for tensor operations)
- **Storage**: 20 GB free
- **Python**: 3.10+
- **OS**: Linux (best compatibility)

---

## Next Steps

After installation:

1. **Run your first query**:
   ```bash
   python run_self_discovery.py "what is machine learning?"
   ```

2. **Test mathematical reasoning**:
   ```bash
   python run_self_discovery.py "prove that prime numbers are infinite"
   ```

3. **Explore tensor reasoning**:
   ```python
   from core.tensor_reasoning_system import TensorReasoningSystem
   import asyncio

   async def test():
       system = TensorReasoningSystem()
       result = await system.solve("prove Goldbach conjecture for n=10")
       print(result.explanation)

   asyncio.run(test())
   ```

4. **Check documentation**:
   - `README.md` - System overview
   - `TENSOR_REASONING_README.md` - Tensor system details
   - `docs/` - Full documentation

---

## Support

- **Issues**: https://github.com/PlanetDestroyyer/KV-1/issues
- **Discussions**: https://github.com/PlanetDestroyyer/KV-1/discussions

---

## License

See LICENSE file for details.
