# 📚 Knowledge Sources for KV-1

**Complete guide to feeding mathematical/AI/ML knowledge to KV-1**

---

## 🎯 Quick Start

### Option 1: Use the Knowledge Importer Script

```bash
# Make executable
chmod +x tools/knowledge_importer.py

# Import AI/ML papers from arXiv
python tools/knowledge_importer.py --source arxiv --query "transformer attention" --max 10

# Import mathematical formulas from MathWorld
python tools/knowledge_importer.py --source mathworld --topic Calculus

# Import Wikipedia math articles
python tools/knowledge_importer.py --source wikipedia --category "Theorems" --max 20

# Import number sequences from OEIS
python tools/knowledge_importer.py --source oeis --query "prime numbers"
```

### Option 2: Feed URLs Directly to KV-1

```bash
# Learn from a specific Wikipedia page
python run_self_discovery.py "learn all theorems from https://en.wikipedia.org/wiki/Calculus"

# Learn from arXiv paper
python run_self_discovery.py "understand the Transformer architecture from arXiv"
```

---

## 📊 Best Knowledge Sources

### 🔴 **HIGHEST PRIORITY - Start Here**

#### 1. **Wolfram MathWorld** (FREE)
**URL**: https://mathworld.wolfram.com

**What to import**:
- All major theorem pages
- Formula collections
- Proof outlines

**Best topics for KV-1**:
- [Calculus](https://mathworld.wolfram.com/Calculus.html)
- [Number Theory](https://mathworld.wolfram.com/NumberTheory.html)
- [Linear Algebra](https://mathworld.wolfram.com/LinearAlgebra.html)
- [Complex Analysis](https://mathworld.wolfram.com/ComplexAnalysis.html)

**How to use**:
```bash
# Visit each topic page
# Copy formulas and theorems
# Feed to KV-1:

python run_self_discovery.py "learn the Fundamental Theorem of Calculus from MathWorld"
```

---

#### 2. **arXiv.org** (FREE - Daily Updates)
**URL**: https://arxiv.org

**Best categories for KV-1**:
- `cs.AI` - Artificial Intelligence: https://arxiv.org/list/cs.AI/recent
- `cs.LG` - Machine Learning: https://arxiv.org/list/cs.LG/recent
- `cs.CL` - NLP: https://arxiv.org/list/cs.CL/recent
- `math.NT` - Number Theory: https://arxiv.org/list/math.NT/recent
- `stat.ML` - Statistics/ML: https://arxiv.org/list/stat.ML/recent

**API Access**:
```python
# Use the knowledge_importer.py script
python tools/knowledge_importer.py --source arxiv --category "cs.AI" --query "attention mechanism" --max 20

# Or manually via API:
# http://export.arxiv.org/api/query?search_query=all:neural+networks&max_results=10
```

**Recent Breakthrough Papers** (Feed these to KV-1):
- Transformer: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805
- GPT-3: https://arxiv.org/abs/2005.14165
- AlphaGo: https://arxiv.org/abs/1712.01815

---

#### 3. **Wikipedia Math Portal** (FREE)
**URL**: https://en.wikipedia.org/wiki/Portal:Mathematics

**Best pages to import**:
- [List of Mathematical Theorems](https://en.wikipedia.org/wiki/List_of_theorems)
- [List of Mathematical Proofs](https://en.wikipedia.org/wiki/Category:Article_proofs)
- [List of Equations](https://en.wikipedia.org/wiki/List_of_equations)
- [Mathematical Formulas](https://en.wikipedia.org/wiki/Category:Mathematical_identities)

**Categories to scrape**:
```bash
python tools/knowledge_importer.py --source wikipedia --category "Theorems" --max 100
python tools/knowledge_importer.py --source wikipedia --category "Mathematical_proofs" --max 100
python tools/knowledge_importer.py --source wikipedia --category "Calculus" --max 50
```

---

### 🟡 **HIGH VALUE - Import Next**

#### 4. **NIST Digital Library of Mathematical Functions**
**URL**: https://dlmf.nist.gov

**What**: Authoritative reference for special functions
- Bessel functions
- Elliptic integrals
- Gamma functions
- Hypergeometric functions

**Use**: Perfect for advanced mathematical reasoning

---

#### 5. **ProofWiki** (FREE)
**URL**: https://proofwiki.org

**What**: 20,000+ mathematical proofs
**Why**: Teaches KV-1 HOW to prove theorems, not just WHAT they are

**Example pages**:
- https://proofwiki.org/wiki/Fundamental_Theorem_of_Calculus
- https://proofwiki.org/wiki/Pythagorean_Theorem
- https://proofwiki.org/wiki/Prime_Number_Theorem

---

#### 6. **Papers With Code** (FREE)
**URL**: https://paperswithcode.com

**What**: 50,000+ ML papers + code implementations
**Best for**: Latest ML/AI with reproducible results

**Example searches**:
- Attention mechanisms
- Reinforcement learning
- Computer vision
- NLP transformers

**Use**:
```bash
# Visit paperswithcode.com
# Find paper + code
# Feed paper abstract to KV-1
# KV-1 learns the concept
# You run the code to verify
```

---

#### 7. **OEIS** (Online Encyclopedia of Integer Sequences)
**URL**: https://oeis.org

**What**: Every known integer sequence
**Use**: Number theory, pattern recognition

**Examples**:
```bash
python tools/knowledge_importer.py --source oeis --query "Fibonacci"
python tools/knowledge_importer.py --source oeis --query "prime numbers"
python tools/knowledge_importer.py --source oeis --query "perfect squares"
```

---

### 🟢 **GOOD TO HAVE - Import Later**

#### 8. **Semantic Scholar** (FREE API)
**URL**: https://www.semanticscholar.org
**API**: https://api.semanticscholar.org

**What**: 200M+ papers with AI-powered search
**Use**: Finding related papers via citation graphs

---

#### 9. **Distill.pub** (FREE)
**URL**: https://distill.pub

**What**: Best visual ML explanations
**Articles** (Read these!):
- [Attention & Augmented RNNs](https://distill.pub/2016/augmented-rnns/)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)

---

#### 10. **MIT OpenCourseWare** (FREE)
**URL**: https://ocw.mit.edu

**Best courses to import**:
- [Linear Algebra (18.06)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)
- [Introduction to Algorithms](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/)

---

## 🛠️ How to Import Knowledge into KV-1

### Method 1: Automated Import (Recommended)

```bash
# 1. Use knowledge importer to fetch sources
python tools/knowledge_importer.py --source arxiv --query "neural networks" --max 20

# 2. This creates: imported_knowledge/arxiv_papers.json

# 3. Feed to KV-1 via curriculum:
# Create custom curriculum from imported knowledge
# Then run: python run_curriculum.py --phase custom
```

---

### Method 2: Manual Import via Goals

```bash
# Just ask KV-1 to learn from a source!

python run_self_discovery.py "learn the Fundamental Theorem of Calculus"
python run_self_discovery.py "understand the Transformer architecture"
python run_self_discovery.py "explain the Riemann Hypothesis"
```

KV-1 will:
1. Search the web for these concepts
2. Find Wikipedia, MathWorld, arXiv, etc.
3. Extract formulas and definitions
4. Store them symbolically in MathConnect
5. Build knowledge graph automatically

---

### Method 3: Bulk Import via Curriculum

**Create a custom curriculum file**:

```python
# custom_curriculum.py
CUSTOM_CURRICULUM = [
    # From MathWorld
    "What is the Fundamental Theorem of Calculus?",
    "What is L'Hôpital's rule?",
    "What is Taylor series expansion?",

    # From arXiv
    "Explain the Transformer attention mechanism",
    "What is backpropagation in neural networks?",
    "How does reinforcement learning work?",

    # From Wikipedia
    "What is the Riemann Hypothesis?",
    "Explain Gödel's Incompleteness Theorems",
    "What is the P vs NP problem?"
]
```

Then run it:
```bash
python run_curriculum.py --custom custom_curriculum.py
```

---

## 📥 Recommended Import Order

### Week 1: Foundations
1. **MathWorld Calculus** → All basic theorems
2. **Wikipedia Theorems** → 100 most important
3. **ProofWiki** → Basic proofs

### Week 2: Advanced Math
1. **MathWorld Linear Algebra**
2. **MathWorld Complex Analysis**
3. **NIST Special Functions**

### Week 3: AI/ML Basics
1. **arXiv: Attention mechanism papers**
2. **Papers With Code: Top 100 ML papers**
3. **Distill.pub articles**

### Week 4: Cutting Edge
1. **arXiv: Recent papers (last 30 days)**
2. **Semantic Scholar: Citation networks**
3. **Custom research areas**

---

## 🔥 Pro Tips

### 1. **Start with Formulas, Not Papers**
- Papers have context/discussion
- **Formulas are pure knowledge**
- Feed MathWorld first, papers second

### 2. **Use MathConnect for Symbolic Learning**
When KV-1 learns a formula, it stores it symbolically:
```python
# Not this: "The quadratic formula is negative b plus or minus..."
# But this: Eq(x, (-b + sqrt(b**2 - 4*a*c)) / (2*a))
```

This lets KV-1 **manipulate and derive** new formulas!

### 3. **Prioritize Worked Examples**
- Theorems with step-by-step proofs
- Papers with mathematical derivations
- Wikipedia pages with "Proof" sections

### 4. **Build Knowledge Graphs**
After importing 100+ formulas, KV-1 can:
- Find connections between theorems
- Derive new relationships
- Answer "How does X relate to Y?"

---

## 💾 Storage Format

All imported knowledge gets stored in:
```
ltm_memory.json          # Persistent LTM (text + tensors)
math_theorems.json       # MathConnect symbolic storage
imported_knowledge/      # Raw imports from sources
  ├── arxiv_papers.json
  ├── wikipedia_articles.json
  ├── oeis_sequences.json
  └── mathworld_topics.json
```

---

## 🎯 Goal: Build Toward Riemann Hypothesis

### Recommended Learning Path:

1. **Basic Calculus** (MathWorld)
   - Limits, derivatives, integrals
   - Fundamental theorem

2. **Complex Analysis** (MathWorld + Wikipedia)
   - Complex numbers, functions
   - Analytic continuation

3. **Number Theory** (OEIS + arXiv)
   - Prime numbers, factorization
   - Diophantine equations

4. **Riemann Zeta Function** (Wikipedia + NIST)
   - Definition, properties
   - Functional equation

5. **Zeros of ζ(s)** (arXiv papers)
   - Critical line
   - Non-trivial zeros
   - **Riemann Hypothesis**: All non-trivial zeros have Re(s) = 1/2

---

## 📞 Support

- **arXiv API**: https://arxiv.org/help/api/
- **Wikipedia API**: https://www.mediawiki.org/wiki/API:Main_page
- **OEIS API**: https://oeis.org/wiki/JSON_Format
- **Semantic Scholar API**: https://api.semanticscholar.org/

---

## ⚡ Quick Commands

```bash
# Import 100 most important theorems
python tools/knowledge_importer.py --source wikipedia --category "Theorems" --max 100

# Import latest AI papers
python tools/knowledge_importer.py --source arxiv --category "cs.AI" --query "transformer" --max 50

# Import number theory
python tools/knowledge_importer.py --source mathworld --topic "NumberTheory"

# Import famous sequences
python tools/knowledge_importer.py --source oeis --query "Fibonacci prime perfect"

# Feed to KV-1 and let it learn!
python run_curriculum.py --phase all
```

**The more you feed KV-1, the smarter it gets! 🚀**
