#!/bin/bash
echo "==============================================="
echo "HSOKV Diagnostic and Fix Script"
echo "==============================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python3 --version
echo ""

# Check if torch is installed
echo "[2/6] Checking if torch is installed..."
python3 -c "import torch; print(f'  ✓ torch version: {torch.__version__}')" 2>&1
TORCH_STATUS=$?
echo ""

# Check CUDA availability
echo "[3/6] Checking CUDA/GPU availability..."
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1
echo ""

# If torch is not installed, install it
if [ $TORCH_STATUS -ne 0 ]; then
    echo "[4/6] torch is NOT installed. Installing dependencies..."
    echo "  Detecting GPU..."

    # Check if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✓ NVIDIA GPU detected! Installing GPU version of torch..."
        pip install torch numpy sentence-transformers transformers --no-cache-dir -q
    else
        echo "  No NVIDIA GPU detected. Installing CPU version..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir -q
        pip install numpy sentence-transformers transformers --no-cache-dir -q
    fi
    echo "  ✓ Dependencies installed"
else
    echo "[4/6] torch is already installed ✓"
fi
echo ""

# Test HSOKV import
echo "[5/6] Testing HSOKV import..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
try:
    from hsokv.hsokv.dual_memory import ShortTermMemory
    print("  ✓ HSOKV ShortTermMemory imported successfully!")
    print(f"    Class: {ShortTermMemory}")
except Exception as e:
    print(f"  ✗ HSOKV import failed: {e}")
    import traceback
    traceback.print_exc()
EOF
echo ""

# Test HybridMemory
echo "[6/6] Testing HybridMemory with HSOKV..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from core.hybrid_memory import HSOKV_AVAILABLE
print(f"  HSOKV_AVAILABLE: {HSOKV_AVAILABLE}")
if HSOKV_AVAILABLE:
    print("  ✓✓✓ SUCCESS! HSOKV is now working!")
else:
    print("  ✗✗✗ FAILED! HSOKV is still not available")
    print("  Please run this script again or check the errors above")
EOF
echo ""

echo "==============================================="
echo "Diagnostic complete!"
echo "==============================================="
