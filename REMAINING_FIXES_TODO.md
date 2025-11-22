# Remaining Critical Fixes TODO

## Status: Fixed 3 CRITICAL bugs, 57 remain

### ✅ FIXED (Just Now):
1. ✅ ModuleNotFoundError in core/__init__.py
2. ✅ run_curriculum.py argument passing crash
3. ✅ Invalid os.system() command

---

## 🚨 CRITICAL SECURITY (Must Fix ASAP):

### 1. math_connect.py Line 173: Arbitrary Code Execution via exec()
**Severity**: CRITICAL 🔴
```python
# CURRENT (DANGEROUS):
exec(f"from sympy import *\n{code}", {}, local_vars)  # Can execute ANY Python code!

# FIX: Remove this entirely or use AST whitelisting
# For now, DISABLE this feature by commenting out the exec() call
```

### 2. math_connect.py Line 148: Code Execution via sympify()
**Severity**: HIGH 🟠
```python
# CURRENT (RISKY):
sympify(equation_text)  # Can evaluate arbitrary expressions

# FIX:
sympify(equation_text, evaluate=False)  # Safer, doesn't execute
```

---

## 🔥 HIGH PRIORITY CRASHES:

### 3. hybrid_memory.py Line 388: Tensor Device Mismatch
```python
# CURRENT:
tensor.cpu().numpy()  # Assumes on GPU

# FIX:
tensor.detach().cpu().numpy()  # Works for both GPU/CPU + handles gradients
```

### 4. hybrid_memory.py Lines 434-439: O(n²) Tensor Concatenation
```python
# CURRENT (SLOW):
for c in data.values():
    if self.ltm.concept_matrix is None:
        self.ltm.concept_matrix = torch.tensor(c["tensor"])
    else:
        self.ltm.concept_matrix = torch.cat([self.ltm.concept_matrix, ...])

# FIX (100x FASTER):
tensors = [torch.tensor(c["tensor"]) for c in data.values() if c.get("tensor")]
if tensors:
    self.ltm.concept_matrix = torch.stack(tensors)
```

### 5. web_researcher.py Line 44: Permanent Rate Limiting
```python
# CURRENT (BROKEN):
self.requests_today = 0  # Never resets!

# FIX: Add date-based reset
from datetime import datetime, date

def __init__(self, ...):
    self.requests_today = 0
    self.last_reset_date = date.today()

def fetch(self, ...):
    # Reset counter if new day
    if date.today() != self.last_reset_date:
        self.requests_today = 0
        self.last_reset_date = date.today()
```

### 6. self_discovery_orchestrator.py Line 1004: NoneType Error
```python
# CURRENT:
if self.using_mathconnect and self._is_mathematical_concept(...):
    success = self.math_connect.learn_theorem_from_text(...)  # Crashes if None!

# FIX:
if self.using_mathconnect and self.math_connect and self._is_mathematical_concept(...):
    success = self.math_connect.learn_theorem_from_text(...)
```

### 7. hybrid_memory.py Line 208: Division by Zero
```python
# CURRENT:
avg_time = sum(recent_times) / len(recent_times)  # Could be empty list!

# FIX:
avg_time = sum(recent_times) / len(recent_times) if recent_times else 0.0
```

---

## ⚠️ MEDIUM PRIORITY:

### 8. Fragile LLM Response Parsing
- Add default values: `text = ""` before parsing
- Check for None before string operations

### 9. Missing Error Handling
- Wrap file operations in try/except
- Add validation for LLM responses

### 10. Type Annotations
- Fix `List[str] = None` → `Optional[List[str]] = None`

---

## Performance Optimizations:

### 11. Parallel Web Fetching
```python
# Current: Sequential (slow)
for source in sources:
    result = fetch(source)

# Better: Parallel (fast)
import asyncio
results = await asyncio.gather(*[fetch(s) for s in sources])
```

### 12. Import Optimization
- Move `import re`, `import numpy` to top-level
- Avoid repeated imports in functions

---

## Testing Priority:

1. Test curriculum runner: `python run_curriculum.py --phase 1 --max-attempts 1`
2. Test self-discovery: `python run_self_discovery.py "solve 2x + 5 = 15"`
3. Check for crashes with empty LTM
4. Test GPU/CPU fallback

---

## Estimated Impact:

- **Security**: Removing exec() prevents remote code execution
- **Performance**: Fixing tensor concatenation = 100x faster load times
- **Stability**: Fixing rate limit reset prevents permanent blocking
- **User Experience**: No more mysterious crashes

---

**Next Session**: Implement these fixes systematically, test thoroughly, commit incrementally.
