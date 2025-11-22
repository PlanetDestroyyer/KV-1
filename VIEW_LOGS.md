# How to View KV-1 Logs

## 📝 Automatic Logging

**ALL output is now automatically saved to log files!** Every time you run KV-1, a timestamped log file is created.

---

## 📂 Log File Locations

All logs are saved in the `./logs/` directory:

```
KV-1/
└── logs/
    ├── self_discovery_2025-11-22_14-30-45.log
    ├── self_discovery_2025-11-22_15-42-12.log
    ├── curriculum_full_curriculum_2025-11-22_16-00-00.log
    ├── curriculum_phase1_2025-11-22_17-15-30.log
    └── ...
```

---

## 🔍 Viewing Logs

### View the latest log:
```bash
# Linux/Mac
tail -f logs/*.log | tail -n 100

# Or just the most recent:
ls -t logs/*.log | head -1 | xargs cat
```

### View specific log:
```bash
cat logs/self_discovery_2025-11-22_14-30-45.log

# Or with less (scrollable):
less logs/self_discovery_2025-11-22_14-30-45.log
```

### Search logs:
```bash
# Find all mentions of "quadratic formula"
grep -r "quadratic formula" logs/

# Find errors:
grep -r "\[X\]" logs/
grep -r "Error" logs/
```

### View logs in real-time (while running):
```bash
# In one terminal, run KV-1:
python run_self_discovery.py "solve x² = 16"

# In another terminal, watch the log:
tail -f logs/self_discovery_*.log
```

---

## 📊 What's Logged?

**Everything that appears on screen is also saved to the log file:**
- ✅ All learning attempts and results
- ✅ Web search queries and results
- ✅ LLM prompts and responses
- ✅ Concept definitions learned
- ✅ Prerequisites discovered
- ✅ Success/failure messages
- ✅ Error messages and stack traces
- ✅ Memory operations (STM/LTM)
- ✅ MathConnect operations
- ✅ Validation results
- ✅ Complete learning journey

---

## 📋 Log File Format

Each log file includes:

```
================================================================================
KV-1 Learning Session Started
Timestamp: 2025-11-22T14:30:45.123456
Log file: ./logs/self_discovery_2025-11-22_14-30-45.log
================================================================================

[+] All output being saved to: ./logs/self_discovery_2025-11-22_14-30-45.log

============================================================
SELF-DISCOVERY LEARNING SYSTEM
============================================================
Goal: solve x² = 16
LTM file: ./ltm_memory.json
Max attempts: UNLIMITED (will run until success)
Hybrid Memory: ENABLED (STM + LTM + GPU)
Validation: DISABLED (fast mode)
3-Stage Learning: ENABLED (target confidence: 0.85)
============================================================

[+] Loaded 42 concepts from LTM
...
[Attempt 1] Trying goal with 42 concepts in LTM...
...

================================================================================
Session Ended: 2025-11-22T14:35:12.987654
Log saved to: ./logs/self_discovery_2025-11-22_14-30-45.log
================================================================================
```

---

## 🗂️ Log File Types

### Self-Discovery Logs
**Pattern**: `self_discovery_YYYY-MM-DD_HH-MM-SS.log`
- Created when running `run_self_discovery.py`
- Contains single goal learning session

### Curriculum Logs
**Pattern**: `curriculum_phaseX_YYYY-MM-DD_HH-MM-SS.log`
- Created when running `run_curriculum.py`
- Contains multiple questions from curriculum
- Phase name included in filename

---

## 💡 Tips

### Keep logs organized:
```bash
# Clean up old logs (older than 30 days)
find logs/ -name "*.log" -mtime +30 -delete

# Archive logs by month:
mkdir -p logs/archive/2025-11
mv logs/*2025-11*.log logs/archive/2025-11/
```

### Analyze learning patterns:
```bash
# How many concepts learned per session?
grep "Concepts learned:" logs/*.log

# Success rate:
grep -c "\[OK\] GOAL ACHIEVED" logs/*.log
grep -c "\[X\]" logs/*.log

# Most common missing concepts:
grep "Missing concepts:" logs/*.log | sort | uniq -c | sort -rn
```

### Share logs for debugging:
```bash
# Compress and share specific log:
tar -czf my-issue.tar.gz logs/self_discovery_2025-11-22_14-30-45.log
```

---

## ⚙️ Disable Logging (if needed)

To disable file logging (console only):

```python
from core.logger import setup_logging

# Disable logging
setup_logging(enabled=False)
```

Or modify the code to skip `setup_logging()` call entirely.

---

## 🐛 Debugging with Logs

When reporting issues, **always include the log file**!

1. Reproduce the issue
2. Find the log file in `./logs/`
3. Attach it to your bug report

Example:
```bash
# Find the most recent log
ls -t logs/*.log | head -1

# View it
cat logs/self_discovery_2025-11-22_14-30-45.log

# Attach to GitHub issue or email
```

---

## 📈 Log Analysis Scripts

### Count learning sessions:
```bash
ls logs/*.log | wc -l
```

### Total learning time:
```bash
# Extract session start/end times from logs
grep "Session Started\|Session Ended" logs/*.log
```

### Most learned concepts:
```bash
# Extract all learned concepts
grep "Learned:" logs/*.log | awk '{print $NF}' | sort | uniq -c | sort -rn
```

---

**All output is automatically logged - you never lose your learning history!** 🎉
