# KV-1 Merge Guide

Apply the changes below directly inside the existing KV-1 checkout.

## 1. Add New Files

```bash
cp core/three_stage_learner.py /path/to/KV-1/core/three_stage_learner.py
cp core/genesis_mode.py /path/to/KV-1/core/genesis_mode.py
cp test_genesis.py /path/to/KV-1/test_genesis.py
cp merge_guide.md /path/to/KV-1/merge_guide.md
```

## 2. Apply Diffs to Existing Files

```bash
patch -p1 < diff_core_orchestrator.patch
patch -p1 < diff_readme.patch
```

## 3. Run Tests

```bash
python test_genesis.py
```

## 4. Commit

```bash
git add core/three_stage_learner.py core/genesis_mode.py test_genesis.py core/orchestrator.py README.md merge_guide.md
git commit -m "Integrate three-stage learner and genesis mode"
```
