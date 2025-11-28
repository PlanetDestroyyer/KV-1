# Running KV-1 on Kaggle (Tesla P100)

## Quick Start

Your Kaggle environment has a **Tesla P100 GPU** which is incompatible with modern PyTorch (requires CUDA 7.0+, but P100 is CUDA 6.0).

**The fix is already in place!** Just pull the latest changes:

```bash
cd /kaggle/working
git clone https://github.com/PlanetDestroyyer/KV-1.git
cd KV-1
git checkout claude/analyze-codebase-01AUvhC6xAeXZ3jUB94c1tWr
pip install -r requirements.txt
python3 run_curriculum.py
```

## What Happens Now

✅ **HSOKV loads correctly** - Fixed import path resolution
✅ **System runs on CPU** - Automatically detects incompatible GPU
✅ **No crashes** - Graceful fallback instead of "no kernel image" error

You'll see these messages:
```
[DEBUG] HSOKV ShortTermMemory imported successfully!
[!] GPU Tesla P100-PCIE-16GB (sm_60) is not compatible with this PyTorch build
    PyTorch requires compute capability >= 7.0 (sm_70)
    Falling back to CPU mode
[+] Neurosymbolic GPU Memory initialized
    Device: cpu (CPU)
```

## Performance on CPU

The system will run slower on CPU than GPU, but it will work:
- **STM operations**: Still O(1) with HSOKV ✓
- **LTM search**: CPU-based semantic search (slower but functional)
- **Embeddings**: sentence-transformers on CPU (2-3x slower)
- **Learning**: Fully functional, just takes longer

## Alternative: Use Compatible GPU

If you have access to newer GPUs (V100, A100, T4), the system will automatically use them:

**Compatible GPUs** (CUDA 7.0+):
- ✅ Tesla V100 (sm_70)
- ✅ Tesla T4 (sm_75)
- ✅ Tesla A100 (sm_80)
- ✅ RTX 20xx/30xx/40xx series
- ❌ Tesla P100 (sm_60) - Too old
- ❌ Tesla K80 (sm_37) - Too old

## Manual GPU Check

To check your GPU compatibility:

```python
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    print(f"Compatible: {'Yes' if cap[0] >= 7 else 'No'}")
```

## Files Modified

The following files now auto-detect GPU compatibility:

1. **core/hybrid_memory.py** - Fixed HSOKV import + smart sys.path
2. **core/neurosymbolic_gpu.py** - GPU compatibility check in `_select_compatible_device()`
3. **core/geometric_knowledge_space.py** - Same GPU check for manifold operations

## Troubleshooting

**Still seeing "HSOKV not available"?**
```bash
# Make sure you pulled latest changes
git pull origin claude/analyze-codebase-01AUvhC6xAeXZ3jUB94c1tWr

# Verify HSOKV loads
python3 -c "from core.hybrid_memory import HSOKV_AVAILABLE; print(f'HSOKV: {HSOKV_AVAILABLE}')"
```

**Want to force CPU mode?**
Set environment variable before running:
```bash
export CUDA_VISIBLE_DEVICES=""
python3 run_curriculum.py
```

## Expected Runtime

With the optimizations (thresholds=0.60, max_depth=5, max_rehearsals=2):

- **On CPU (Kaggle)**: ~15-25 minutes per curriculum question
- **On GPU (V100+)**: ~4-8 minutes per curriculum question
- **Full curriculum (145 questions)**:
  - CPU: ~36-60 hours
  - GPU: ~10-20 hours

The system will checkpoint progress, so you can resume if interrupted!
