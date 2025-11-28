#!/bin/bash
# KV-1 GPU Dependencies Installation Script
# This script installs all required dependencies for GPU systems

echo "=============================================="
echo "KV-1 GPU Dependencies Installation"
echo "=============================================="
echo ""

# Pull latest git changes
echo "[1/5] Pulling latest git changes..."
git pull origin claude/analyze-codebase-01AUvhC6xAeXZ3jUB94c1tWr
echo ""

# Uninstall CPU-only torch if present
echo "[2/5] Removing CPU-only torch (if present)..."
pip uninstall torch -y 2>/dev/null || echo "  (torch not installed, skipping)"
echo ""

# Install GPU version of torch
echo "[3/5] Installing PyTorch with CUDA support..."
echo "  This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
echo ""

# Install other dependencies
echo "[4/5] Installing remaining dependencies..."
pip install -r requirements.txt --no-cache-dir -q
echo "  ✓ Dependencies installed"
echo ""

# Verify installation
echo "[5/5] Verifying installation..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

print("\n  Testing torch CUDA support...")
try:
    import torch
    print(f"  ✓ torch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ✗ torch error: {e}")

print("\n  Testing HSOKV import...")
try:
    from hsokv.hsokv.dual_memory import ShortTermMemory
    print("  ✓ HSOKV ShortTermMemory imported successfully!")
except Exception as e:
    print(f"  ✗ HSOKV import failed: {e}")

print("\n  Testing HybridMemory...")
try:
    from core.hybrid_memory import HSOKV_AVAILABLE
    print(f"  HSOKV_AVAILABLE: {HSOKV_AVAILABLE}")
    if HSOKV_AVAILABLE:
        print("  ✓✓✓ SUCCESS! HSOKV is enabled!")
    else:
        print("  ✗✗✗ WARNING! HSOKV is still disabled!")
except Exception as e:
    print(f"  ✗ HybridMemory error: {e}")
EOF

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "You can now run KV-1 with:"
echo "  python3 run_curriculum.py"
echo ""
