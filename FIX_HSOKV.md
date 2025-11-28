# HSOKV Not Loading - Fix Instructions

## Problem
You're seeing: `[!] HSOKV not available, using simple dict for STM`

## Root Cause
The issue is that torch and dependencies are either:
1. Not installed in your current environment
2. Installed as CPU-only version (you need GPU version)
3. Git changes not pulled

## Quick Fix (FOR GPU SYSTEMS)

Run this single command in your KV-1 directory:

```bash
bash install_deps_gpu.sh
```

This script will:
- Pull latest git changes with the HSOKV fix
- Uninstall CPU-only torch
- Install GPU-enabled torch with CUDA support
- Install all dependencies from requirements.txt
- Verify everything works

## Alternative: Manual Fix

If the script doesn't work, run these commands manually:

```bash
# 1. Pull latest changes
git pull origin claude/analyze-codebase-01AUvhC6xAeXZ3jUB94c1tWr

# 2. Uninstall old torch
pip uninstall torch -y

# 3. Install GPU torch (for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install numpy sentence-transformers transformers beautifulsoup4 lxml sympy scipy requests ollama pydantic python-dotenv colorlog aiohttp pytest scikit-learn

# 5. Test it works
python3 -c "
import sys
sys.path.insert(0, '.')
from core.hybrid_memory import HSOKV_AVAILABLE
print(f'HSOKV_AVAILABLE: {HSOKV_AVAILABLE}')
"
```

## For Different CUDA Versions

- **CUDA 11.8**: `--index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `--index-url https://download.pytorch.org/whl/cu121`
- **CPU only**: `--index-url https://download.pytorch.org/whl/cpu`

## Verification

After installation, you should see:
```
[DEBUG] HSOKV ShortTermMemory imported successfully!
[+] Using Hybrid Memory (STM + LTM + GPU Tensors)
```

Instead of:
```
[!] HSOKV not available, using simple dict for STM
```

## What Changed

We fixed two issues:
1. **sys.path configuration** in `core/hybrid_memory.py` - now correctly points to project root
2. **Missing dependencies** - torch and related packages needed for HSOKV

## Still Not Working?

Run the diagnostic script:
```bash
bash fix_hsokv.sh
```

This will show exactly what's failing and help debug the issue.
