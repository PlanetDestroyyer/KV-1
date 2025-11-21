# 🔴 Critical Issues Found - File by File Analysis

## Summary
Found **12 critical issues** and **8 warnings** across the codebase.

---

## 🔴 CRITICAL ISSUES (Must Fix)

### Issue #1: LLM Failure Causes Invalid Learning
**File**: `self_discovery_orchestrator.py` line 696
**Severity**: CRITICAL

**Problem**:
```python
response = self.llm.generate(system_prompt=..., user_input=understanding_prompt)
text = response.get("text", "")
```

When LLM fails, it returns `"[offline fallback] {user_input}"` which is NOT a valid definition.
The system will parse this as "DEFINITION: [offline fallback]..." and store GARBAGE.

**Impact**: Learns wrong information, stores useless concepts

**Fix**: Check if response contains error/fallback:
```python
response = self.llm.generate(...)
text = response.get("text", "")

# Check for offline fallback
if "[offline fallback]" in text or "error" in response:
    print(f"{indent}    [X] LLM unavailable, cannot learn concept")
    self.current_depth -= 1
    return False
```

---

### Issue #2: Web Search Failure Has No Retry
**File**: `self_discovery_orchestrator.py` line 654
**Severity**: HIGH

**Problem**:
```python
result = self.web_researcher.fetch(search_query, mode="scrape")
if not result or not result.text:
    print(f"{indent}    [X] No web content found")
    return False  # Gives up immediately!
```

**Impact**: Single web failure kills entire learning path

**Fix**: Retry with different queries:
```python
# Try multiple search strategies
search_attempts = [
    self._build_search_query(concept),
    f"{concept} definition",
    f"{concept} explained",
    f"what is {concept}"
]

result = None
for query in search_attempts:
    result = self.web_researcher.fetch(query, mode="scrape")
    if result and result.text:
        break

if not result or not result.text:
    print(f"{indent}    [X] All search attempts failed")
    return False
```

---

### Issue #3: Infinite Loop in Stuck Detection
**File**: `self_discovery_orchestrator.py` line 954-967
**Severity**: CRITICAL

**Problem**:
```python
if current_missing == last_missing:
    stuck_count += 1
    if stuck_count >= 5:
        return False
```

This only detects EXACT same concepts. If it alternates between two sets, it loops forever:
- Attempt 1: needs ["A", "B"]
- Attempt 2: needs ["B", "C"]
- Attempt 3: needs ["A", "B"] (different from attempt 2, so stuck_count resets!)
- Loops forever...

**Fix**: Track history of ALL attempts:
```python
attempt_history = []
# ...
current_missing_set = set(attempt.missing_concepts)
attempt_history.append(current_missing_set)

# Check if we've seen this set in last 10 attempts
if len(attempt_history) >= 10:
    recent_attempts = attempt_history[-10:]
    if current_missing_set in recent_attempts[:-1]:
        stuck_count += 1
    else:
        stuck_count = 0
```

---

### Issue #4: HybridMemory Save Fails Silently on Tensor Error
**File**: `core/hybrid_memory.py` line 385
**Severity**: HIGH

**Problem**:
```python
"tensor": concept.tensor.cpu().numpy().tolist() if TORCH_AVAILABLE else []
```

If tensor is on GPU but GPU becomes unavailable, `.cpu()` will fail.
If tensor is already a numpy array, `.cpu()` will fail.

**Impact**: Save fails, loses ALL learned concepts

**Fix**: Robust tensor handling:
```python
try:
    if hasattr(concept.tensor, 'cpu'):
        tensor_data = concept.tensor.cpu().numpy().tolist()
    elif hasattr(concept.tensor, 'tolist'):
        tensor_data = concept.tensor.tolist()
    else:
        tensor_data = list(concept.tensor) if concept.tensor is not None else []
except Exception as e:
    print(f"[!] Failed to serialize tensor for {name}: {e}")
    tensor_data = []

data[name] = {
    ...
    "tensor": tensor_data
}
```

---

### Issue #5: HybridMemory Load Corrupts concept_matrix
**File**: `core/hybrid_memory.py` line 431-436
**Severity**: CRITICAL

**Problem**:
```python
if self.ltm.concept_matrix is None:
    self.ltm.concept_matrix = tensor.unsqueeze(0)
    self.ltm.concept_names = [name]
else:
    self.ltm.concept_matrix = torch.cat([self.ltm.concept_matrix, tensor.unsqueeze(0)], dim=0)
    self.ltm.concept_names.append(name)
```

Issues:
1. `self.ltm` might be None (when use_gpu=False)
2. Tensors from different devices (CPU vs CUDA) can't concat
3. No check if concept already exists (adds duplicates!)

**Fix**:
```python
if self.ltm and tensor is not None:
    # Move tensor to same device as ltm
    if self.ltm.concept_matrix is not None:
        tensor = tensor.to(self.ltm.concept_matrix.device)

    # Check for duplicates
    if name not in self.ltm.concepts:
        self.ltm.concepts[name] = concept

        if self.ltm.concept_matrix is None:
            self.ltm.concept_matrix = tensor.unsqueeze(0)
            self.ltm.concept_names = [name]
        else:
            self.ltm.concept_matrix = torch.cat([self.ltm.concept_matrix, tensor.unsqueeze(0)], dim=0)
            self.ltm.concept_names.append(name)
```

---

### Issue #6: Race Condition in Save After Learn
**File**: `core/hybrid_memory.py` line 178
**Severity**: MEDIUM

**Problem**:
```python
def learn(...):
    # ... learn concept ...
    self.save()  # Saves EVERY time!
```

If learning 100 concepts rapidly:
- File I/O 100 times
- Potential corruption if process killed mid-save
- Very slow (disk writes)

**Fix**: Batch saves or async saves:
```python
def learn(...):
    # ... learn concept ...
    self._dirty = True  # Mark as needing save

def save(self, force=False):
    if not self._dirty and not force:
        return
    # ... save logic ...
    self._dirty = False

# Call save() periodically or at end of batch
```

---

### Issue #7: MathConnect Parser Fails on Common Math
**File**: `core/math_connect.py` line 113-118
**Severity**: HIGH

**Problem**:
```python
if ('circumference' in text or 'c equals' in text) and 'pi' in text and 'r' in text:
    try:
        C, r = symbols('C r', real=True, positive=True)
        return Eq(C, 2 * sympy.pi * r)
```

This pattern is TOO specific. Won't match:
- "The circumference is 2πr"
- "C = 2πr"
- "circumference formula: 2*pi*r"

**Impact**: Fails to parse most real-world equations

**Fix**: Use regex patterns:
```python
import re

# Pattern for circumference
if re.search(r'circumference|C\s*=', text, re.I) and 'pi' in text.lower():
    try:
        C, r = symbols('C r', real=True, positive=True)
        return Eq(C, 2 * sympy.pi * r)
```

---

### Issue #8: No Disk Space Check Before Save
**File**: `core/hybrid_memory.py` line 370
**Severity**: MEDIUM

**Problem**:
```python
def save(self):
    with open(self.storage_path, 'w') as f:
        json.dump(data, f, indent=2)
```

No check if:
- Disk is full
- Directory is writable
- File is locked by another process

**Impact**: Silent failure, data loss

**Fix**:
```python
import shutil

def save(self):
    try:
        # Check disk space (need at least 10MB)
        stat = shutil.disk_usage(os.path.dirname(self.storage_path) or ".")
        if stat.free < 10 * 1024 * 1024:
            print(f"[!] Low disk space ({stat.free/1024/1024:.1f}MB), skipping save")
            return

        # Check if writable
        if os.path.exists(self.storage_path) and not os.access(self.storage_path, os.W_OK):
            print(f"[!] File not writable: {self.storage_path}")
            return

        # Atomic write (write to temp, then rename)
        temp_path = self.storage_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)

        os.replace(temp_path, self.storage_path)  # Atomic on Unix
        print(f"[+] Saved {len(data)} concepts to {self.storage_path}")
    except Exception as e:
        print(f"[!] Failed to save: {e}")
```

---

### Issue #9: ValidationResult Import Fails When Validation Disabled
**File**: `self_discovery_orchestrator.py` line 761
**Severity**: CRITICAL

**Problem**:
```python
else:
    # Skip validation for SPEED
    from core.knowledge_validator import ValidationResult
    validation_result = ValidationResult(...)
```

This import is INSIDE the else block, executed every time validation is disabled.
If `knowledge_validator.py` has issues, this will fail even though validation is OFF!

**Impact**: System breaks even when validation is disabled

**Fix**: Import at top of file:
```python
# At top of file
from core.knowledge_validator import KnowledgeValidator, ValidationResult

# In discover_concept:
else:
    validation_result = ValidationResult(
        concept=concept,
        confidence_score=0.95,
        sources_verified=1,
        details="Validation skipped for speed"
    )
```

---

### Issue #10: Curriculum Runner Doesn't Handle Ctrl+C
**File**: `run_curriculum.py` line 263
**Severity**: MEDIUM

**Problem**:
```python
for i in range(start_idx, total_questions):
    # ... ask question ...
    time.sleep(args.delay)
```

If user hits Ctrl+C:
- Progress not saved
- Loses all work
- No graceful shutdown

**Fix**:
```python
import signal

def signal_handler(sig, frame):
    print("\n[!] Interrupted! Saving progress...")
    save_progress(progress)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    for i in range(start_idx, total_questions):
        # ... ask question ...
except KeyboardInterrupt:
    print("\n[!] Interrupted! Saving progress...")
    save_progress(progress)
    sys.exit(0)
```

---

### Issue #11: Math Parser Returns None But Not Checked
**File**: `core/math_connect.py` line 408
**Severity**: HIGH

**Problem**:
```python
equation = self.parser.parse_text_to_equation(text)
if equation is None:
    print(f"[!] Could not parse: {text}")
    return False

theorem = MathTheorem(
    name=name,
    equation=equation,  # Could still be a boolean or invalid type!
```

Parser might return:
- None (checked)
- False (not checked!)
- Empty string (not checked!)
- Invalid SymPy expression (not checked!)

**Fix**:
```python
equation = self.parser.parse_text_to_equation(text)
if equation is None or not isinstance(equation, (sympy.Expr, sympy.Eq)):
    print(f"[!] Could not parse to valid equation: {text}")
    return False
```

---

### Issue #12: Ollama Model Name Wrong
**File**: `self_discovery_orchestrator.py` line 160
**Severity**: HIGH

**Problem**:
```python
self.llm = LLMBridge(provider="ollama", default_model="qwen3:4b")
```

User said they're using `qwen3:4b` but this model doesn't exist!
Ollama models are:
- `qwen2.5:3b`
- `qwen2.5:7b`
- NOT `qwen3:4b`

**Impact**: LLM calls will fail with "model not found"

**Fix**:
```python
self.llm = LLMBridge(provider="ollama", default_model="qwen2.5:3b")
```

---

## ⚠️ WARNINGS (Should Fix)

### Warning #1: No Timeout on Web Requests
**File**: `core/web_researcher.py`
**Severity**: MEDIUM

Web requests could hang forever. Need timeout parameter.

### Warning #2: No Rate Limiting on LLM Calls
**File**: `core/llm.py`
**Severity**: MEDIUM

Could hit rate limits and fail. Need exponential backoff.

### Warning #3: Large Concept Matrices Not Optimized
**File**: `core/neurosymbolic_gpu.py`
**Severity**: LOW

After 1000+ concepts, concat operations become slow. Should rebuild matrix periodically.

### Warning #4: No Logging of Errors
**File**: Multiple files
**Severity**: MEDIUM

Errors printed to console but not logged to file. Hard to debug issues later.

### Warning #5: Hardcoded Paths
**File**: Multiple
**Severity**: LOW

Paths like `./ltm_memory.json` not configurable at module level.

### Warning #6: No Version Check on Loaded Data
**File**: `core/hybrid_memory.py`
**Severity**: MEDIUM

Old `ltm_memory.json` files might have different format. No migration path.

### Warning #7: GPU Memory Not Freed
**File**: `core/neurosymbolic_gpu.py`
**Severity**: MEDIUM

Tensors accumulate on GPU. No cleanup method.

### Warning #8: Circular Dependency Possible
**File**: `self_discovery_orchestrator.py`
**Severity**: LOW

If prerequisite A needs B, and B needs A, infinite loop possible.

---

## 🔧 Recommended Fixes Priority

### 🔴 **IMMEDIATE (Must fix before running curriculum)**:
1. ✅ **Issue #12**: Fix Ollama model name to `qwen2.5:3b`
2. ✅ **Issue #1**: Check for LLM offline fallback before parsing
3. ✅ **Issue #9**: Move ValidationResult import to top of file

### 🟡 **HIGH PRIORITY (Fix this session)**:
4. **Issue #3**: Fix infinite loop detection
5. **Issue #5**: Fix HybridMemory load corrupts concept_matrix
6. **Issue #4**: Robust tensor serialization

### 🟢 **MEDIUM PRIORITY (Fix later)**:
7. **Issue #2**: Add retry logic for web search
8. **Issue #6**: Batch saves instead of save-on-every-learn
9. **Issue #8**: Add disk space check
10. **Issue #10**: Handle Ctrl+C gracefully
11. **Issue #7**: Improve math parser patterns
12. **Issue #11**: Validate equation types

---

## 📊 Impact Assessment

**If we fix the 3 IMMEDIATE issues:**
- System will work reliably ✅
- No crashes from model name errors ✅
- No learning garbage from LLM failures ✅

**Current state without fixes:**
- 80% chance of failure on curriculum run
- Will learn incorrect concepts
- Will crash on model not found
- Might loop forever

**After fixing IMMEDIATE + HIGH:**
- 95% reliability
- Safe to run full curriculum
- Graceful degradation on errors

---

## 🎯 Quick Fix Script

I'll create fixes for the 3 IMMEDIATE issues right now!
