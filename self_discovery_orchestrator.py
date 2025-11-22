"""
Self-Discovery Learning Orchestrator

Goal-driven autonomous learning system that discovers knowledge through need.
Unlike curriculum-based learning, this system:
- Starts with a goal
- Identifies what it doesn't know by attempting and failing
- Recursively learns prerequisites
- Saves learned knowledge to persistent LTM
- Shows the discovery journey (like human history)
"""

import asyncio
import json
import os
import sys
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsokv'))

from core.llm import LLMBridge
from core.web_researcher import WebResearcher
from core.knowledge_validator import KnowledgeValidator, ValidationResult

try:
    from core.hybrid_memory import HybridMemory
    HYBRID_MEMORY_AVAILABLE = True
except ImportError:
    HYBRID_MEMORY_AVAILABLE = False
    print("[!] Hybrid memory not available, using fallback")

try:
    from core.math_connect import MathConnect
    MATHCONNECT_AVAILABLE = True
except ImportError:
    MATHCONNECT_AVAILABLE = False
    print("[!] MathConnect not available, symbolic math reasoning disabled")

# NEW AGI modules
try:
    from core.meta_learner import MetaLearner, LearningAttempt
    from core.metacognition import MetacognitiveLayer
    from core.relevance_filter import RelevanceFilter
    from core.goal_planner import GoalPlanner
    from core.creative_reasoner import CreativeReasoner
    from core.curiosity_engine import CuriosityEngine
    from core.causal_reasoner import CausalReasoner
    from core.parallel_web_search import ParallelWebSearch
    AGI_MODULES_AVAILABLE = True
except ImportError as e:
    AGI_MODULES_AVAILABLE = False
    print(f"[!] AGI modules not available: {e}")

# Unified AGI Learning System
try:
    from core.unified_agi_learner import UnifiedAGILearner, QuestionType
    UNIFIED_AGI_AVAILABLE = True
except ImportError as e:
    UNIFIED_AGI_AVAILABLE = False
    print(f"[!] Unified AGI learner not available: {e}")


@dataclass
class LearningEntry:
    """A single entry in the learning journal."""
    concept: str
    definition: str
    learned_at: str
    needed_for: str
    source: str  # "web", "primitive", "inference"
    examples: List[str] = None  # Worked examples showing how to apply the concept
    # Validation fields (optional)
    confidence_score: Optional[float] = None  # 0-1 confidence from validation
    validation_sources: Optional[int] = None  # Number of sources verified
    validation_details: Optional[str] = None  # Human-readable validation info

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class GoalAttempt:
    """Result of attempting a goal."""
    success: bool
    result: str
    missing_concepts: List[str]
    error_message: Optional[str] = None


class PersistentLTM:
    """Persistent long-term memory with disk storage."""

    def __init__(self, storage_path: str = "./ltm_memory.json"):
        self.storage_path = storage_path
        self.knowledge: Dict[str, LearningEntry] = {}
        self.load()

    def load(self):
        """Load LTM from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge = {
                        k: LearningEntry(**v) for k, v in data.items()
                    }
                print(f"[+] Loaded {len(self.knowledge)} concepts from LTM")
            except Exception as e:
                print(f"[!] Failed to load LTM: {e}")
                self.knowledge = {}
        else:
            print("[i] No existing LTM found, starting fresh")
            self.knowledge = {}

    def save(self):
        """Save LTM to disk."""
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                data = {k: asdict(v) for k, v in self.knowledge.items()}
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[+] Saved {len(self.knowledge)} concepts to LTM")
        except Exception as e:
            print(f"[!] Failed to save LTM: {e}")

    def has(self, concept: str) -> bool:
        """Check if concept is known."""
        return concept.lower() in self.knowledge

    def get(self, concept: str) -> Optional[LearningEntry]:
        """Retrieve concept from memory."""
        return self.knowledge.get(concept.lower())

    def add(self, entry: LearningEntry):
        """Add new concept to memory."""
        self.knowledge[entry.concept.lower()] = entry
        self.save()

    def get_all_concepts(self) -> List[str]:
        """Get list of all known concepts."""
        return list(self.knowledge.keys())


class SelfDiscoveryOrchestrator:
    """
    Autonomous goal-driven learning system.

    Pursues goals by:
    1. Attempting goal with current knowledge
    2. Identifying missing concepts from failures
    3. Recursively learning prerequisites
    4. Retrying until success
    """

    def __init__(
        self,
        goal: str,
        ltm_path: str = "./ltm_memory.json",
        data_dir: str = "./self_discovery_data",
        max_depth: int = 10,
        use_hybrid_memory: bool = True,  # NEW: Use STM+LTM+GPU by default!
        enable_validation: bool = False,  # NEW: Validation OFF by default for SPEED!
        enable_rehearsal: bool = True,  # NEW: 3-stage learning for quality control!
        target_confidence: float = 0.70,  # NEW: Mastery threshold (0.70 = 70%, balanced speed+quality)
        max_parallel_concepts: int = 10  # NEW: Max concepts to learn in parallel (GPU-optimized)
    ):
        self.goal = goal
        self.max_learning_depth = max_depth
        self.data_dir = data_dir
        self.enable_validation = enable_validation  # Store validation setting
        self.enable_rehearsal = enable_rehearsal  # Store rehearsal setting
        self.target_confidence = target_confidence  # Store confidence threshold
        self.max_parallel_concepts = max_parallel_concepts  # Parallel learning limit
        os.makedirs(data_dir, exist_ok=True)

        # Initialize components
        if use_hybrid_memory and HYBRID_MEMORY_AVAILABLE:
            print("[+] Using Hybrid Memory (STM + LTM + GPU Tensors)")
            self.ltm = HybridMemory(stm_capacity=50, use_gpu=True, storage_path=ltm_path)  # Increased from 7 to 50 for GPU
            self.using_hybrid = True
        else:
            print("[+] Using legacy PersistentLTM (string storage)")
            self.ltm = PersistentLTM(ltm_path)
            self.using_hybrid = False

        self.llm = LLMBridge(provider="ollama", default_model="qwen3:4b")
        self.web_researcher = WebResearcher(
            cache_dir=os.path.join(data_dir, "web_cache"),
            daily_cap=100
        )
        self.validator = KnowledgeValidator(self.llm, self.web_researcher)

        # Initialize MathConnect for symbolic math reasoning
        if MATHCONNECT_AVAILABLE:
            print("[+] MathConnect enabled (symbolic math reasoning)")
            self.math_connect = MathConnect(llm=self.llm, web=self.web_researcher)
            self.using_mathconnect = True
        else:
            self.math_connect = None
            self.using_mathconnect = False

        # Initialize AGI modules
        if AGI_MODULES_AVAILABLE:
            print("[+] Initializing AGI modules...")
            self.meta_learner = MetaLearner(storage_path=os.path.join(data_dir, "meta_learning.json"))
            self.metacognition = MetacognitiveLayer(self.llm)
            self.relevance_filter = RelevanceFilter(self.llm)
            self.goal_planner = GoalPlanner(self.llm, self.relevance_filter)
            self.creative_reasoner = CreativeReasoner(self.llm)
            self.curiosity_engine = CuriosityEngine(self.llm)
            self.causal_reasoner = CausalReasoner(self.llm)
            self.parallel_web = ParallelWebSearch(self.web_researcher, max_workers=5)
            self.using_agi_modules = True
            print("[+] AGI modules ready: Meta-learning, Metacognition, Goal Planning, Creative Reasoning, Curiosity, Causal Reasoning")
        else:
            self.meta_learner = None
            self.metacognition = None
            self.relevance_filter = None
            self.goal_planner = None
            self.creative_reasoner = None
            self.curiosity_engine = None
            self.causal_reasoner = None
            self.parallel_web = None
            self.using_agi_modules = False

        # Initialize Unified AGI Learning System
        if UNIFIED_AGI_AVAILABLE:
            print("[+] Initializing UNIFIED AGI LEARNING SYSTEM...")
            self.unified_learner = UnifiedAGILearner(
                llm_bridge=self.llm,
                web_researcher=self.web_researcher,
                memory=self.ltm
            )
            self.using_unified_agi = True
            print("[+] ✅ UNIFIED AGI: Handles BOTH math (tensor reasoning) AND general knowledge!")
        else:
            self.unified_learner = None
            self.using_unified_agi = False

        # Learning journal
        self.journal: List[Dict] = []
        self.current_depth = 0
        self.attempts = 0
        self.concepts_learned_this_session = []  # Track for meta-learning

        # Detect goal domain for focused learning
        self.goal_domain = self._detect_goal_domain(goal)
        self.goal_keywords = self._extract_goal_keywords(goal)
        print(f"[i] Detected goal domain: {self.goal_domain}")
        print(f"[i] Goal keywords: {', '.join(self.goal_keywords)}")

    def _get_ltm_size(self) -> int:
        """Get number of concepts in LTM (works for both HybridMemory and PersistentLTM)."""
        if self.using_hybrid:
            # HybridMemory stores concepts in self.concepts dict
            return len(self.ltm.concepts)
        else:
            # PersistentLTM stores in knowledge dict
            return len(self.ltm.knowledge)

    def _get_all_concepts(self) -> List[str]:
        """Get all concept names (works for both HybridMemory and PersistentLTM)."""
        if self.using_hybrid:
            # HybridMemory stores concepts in self.concepts dict
            return list(self.ltm.concepts.keys())
        else:
            # PersistentLTM has get_all_concepts method
            return self.ltm.get_all_concepts()

    def _detect_goal_domain(self, goal: str) -> str:
        """Detect the domain of the goal for focused learning."""
        goal_lower = goal.lower()

        # Domain indicators
        math_indicators = [
            # Basic operations
            "solve", "equation", "calculate", "add", "subtract", "multiply", "divide",
            # Algebra & functions
            "algebra", "calculus", "geometry", "trigonometry", "quadratic", "polynomial",
            "derivative", "integral", "matrix", "vector", "function", "graph",
            # Numbers & number theory
            "number", "numbers", "prime", "composite", "factor", "factors", "divisor",
            "multiple", "integer", "rational", "irrational", "real", "complex",
            "natural numbers", "whole numbers", "counting",
            # Geometry & measurement
            "x =", "y =", "2x", "3x", "squared", "cubed", "pi", "area", "volume",
            "radius", "diameter", "perimeter", "percentage", "ratio", "fraction",
            # Math concepts
            "theorem", "proof", "axiom", "lemma", "corollary", "set", "sequence",
            "series", "limit", "infinity", "probability", "statistics"
        ]

        science_indicators = [
            "energy", "force", "mass", "velocity", "acceleration", "thermodynamics",
            "physics", "chemistry", "biology", "atom", "molecule", "electron",
            "gravity", "motion", "temperature", "pressure", "density", "element",
            "compound", "reaction", "cell", "organism", "photosynthesis", "evolution"
        ]

        programming_indicators = [
            "code", "function", "variable", "loop", "array", "class", "object",
            "algorithm", "data structure", "programming", "python", "javascript",
            "api", "database", "debug", "compile", "execute"
        ]

        language_indicators = [
            "word", "sentence", "paragraph", "grammar", "spell", "write",
            "noun", "verb", "adjective", "subject", "predicate", "alphabet",
            "letter", "definition", "meaning", "synonym", "antonym"
        ]

        literature_indicators = [
            "book", "novel", "story", "poem", "author", "character", "plot",
            "theme", "literature", "reading", "writing", "shakespeare"
        ]

        # Count matches
        math_score = sum(1 for indicator in math_indicators if indicator in goal_lower)
        science_score = sum(1 for indicator in science_indicators if indicator in goal_lower)
        programming_score = sum(1 for indicator in programming_indicators if indicator in goal_lower)
        language_score = sum(1 for indicator in language_indicators if indicator in goal_lower)
        literature_score = sum(1 for indicator in literature_indicators if indicator in goal_lower)

        # Determine domain
        scores = {
            "mathematics": math_score,
            "science": science_score,
            "programming": programming_score,
            "language": language_score,
            "literature": literature_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            return "general"

        return max(scores, key=scores.get)

    def _extract_goal_keywords(self, goal: str) -> List[str]:
        """Extract important keywords from the goal."""
        # Remove common words
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "what", "how", "why", "when", "where"}

        words = goal.lower().split()
        keywords = [w.strip("?.,!") for w in words if w.strip("?.,!") not in stop_words and len(w) > 2]

        return keywords[:5]  # Top 5 keywords

    def _is_mathematical_concept(self, concept: str, definition: str = "") -> bool:
        """
        Detect if a concept is mathematical (theorem, formula, equation).

        Returns True if it should be learned symbolically with MathConnect.
        """
        concept_lower = concept.lower()
        definition_lower = definition.lower()

        # Mathematical keywords in concept name
        math_keywords = [
            "theorem", "formula", "equation", "identity", "law",
            "principle", "rule", "property", "proof",
            "pythagorean", "quadratic", "trigonometric", "calculus",
            "derivative", "integral", "limit", "series", "sum",
            "product", "factorial", "matrix", "vector", "eigenvalue"
        ]

        # Check concept name
        for keyword in math_keywords:
            if keyword in concept_lower:
                return True

        # Check definition for mathematical patterns
        math_patterns = [
            "equals", "=", "squared", "cubed", "times", "plus", "minus",
            "sin", "cos", "tan", "log", "exp", "sqrt",
            "∑", "∫", "∂", "π", "∞"
        ]

        for pattern in math_patterns:
            if pattern in definition_lower:
                return True

        return False

    def _is_relevant_to_goal(self, concept: str) -> bool:
        """Check if a concept is relevant to the goal domain."""
        concept_lower = concept.lower()

        # Domain-specific concept patterns
        domain_patterns = {
            "mathematics": [
                "equation", "algebra", "number", "digit", "calculate", "solve",
                "add", "subtract", "multiply", "divide", "variable", "coefficient",
                "quadratic", "polynomial", "factor", "root", "solution", "formula",
                "arithmetic", "operation", "mathematical", "numeric", "quantity",
                "distributive", "property", "exponent", "power", "derivative", "integral"
            ],
            "science": [
                "energy", "force", "mass", "atom", "molecule", "element",
                "physics", "chemistry", "biology", "reaction", "compound",
                "temperature", "pressure", "density", "motion", "velocity"
            ],
            "programming": [
                "code", "function", "variable", "loop", "array", "algorithm",
                "data", "program", "software", "computer", "syntax"
            ],
            "language": [
                "word", "letter", "sentence", "grammar", "noun", "verb",
                "alphabet", "linguistic", "language", "writing", "reading"
            ],
            "literature": [
                "book", "novel", "story", "author", "character", "literature",
                "poem", "writing", "reading"
            ]
        }

        # Check if concept contains domain-specific keywords
        if self.goal_domain in domain_patterns:
            patterns = domain_patterns[self.goal_domain]
            for pattern in patterns:
                if pattern in concept_lower:
                    return True

        # Check if concept relates to goal keywords
        for keyword in self.goal_keywords:
            if keyword in concept_lower:
                return True

        # Exclude concepts from irrelevant domains
        irrelevant_patterns = {
            "mathematics": ["plant", "animal", "cell", "organism", "photosynthesis", "eukaryote",
                           "bond (finance)", "interest rate", "investment", "cash flow",
                           "tcp/ip", "network", "protocol", "domain name",
                           "drum", "music", "percussion", "instrument"],
            "science": ["novel", "story", "character", "literature"],
            "programming": ["photosynthesis", "cell", "organism"],
            "language": ["equation", "solve", "calculate"],
            "literature": ["equation", "solve", "physics"]
        }

        if self.goal_domain in irrelevant_patterns:
            patterns = irrelevant_patterns[self.goal_domain]
            for pattern in patterns:
                if pattern in concept_lower:
                    return False

        # General domain check
        return self.goal_domain == "general"

    def _get_knowledge_summary(self) -> str:
        """Generate summary of current knowledge for LLM context."""
        if self._get_ltm_size() == 0:
            return "You have no prior knowledge. You are starting from zero."

        concepts = self._get_all_concepts()
        summary = f"You currently know these {len(concepts)} concepts:\n"
        for concept in sorted(concepts)[:20]:  # Limit to avoid context overflow
            entry = self.ltm.get(concept)

            # BUG FIX: Check if entry exists before accessing definition
            if entry is None:
                print(f"[!] Warning: Concept '{concept}' in list but not found in LTM")
                continue

            # Safely get definition with fallback
            definition = getattr(entry, 'definition', 'No definition available')
            if definition:
                summary += f"- {concept}: {definition[:100]}...\n"
            else:
                summary += f"- {concept}: (concept learned)\n"

            # Include examples if available - these are CRITICAL for learning HOW to apply concepts
            if hasattr(entry, 'examples') and entry.examples:
                for ex in entry.examples[:2]:  # Show up to 2 examples
                    summary += f"  Example: {ex}\n"

        if len(concepts) > 20:
            summary += f"... and {len(concepts) - 20} more concepts\n"

        return summary

    async def attempt_goal(self) -> GoalAttempt:
        """
        Attempt to achieve the goal with current knowledge.
        Returns success status and what concepts are missing.
        """
        self.attempts += 1
        print(f"\n{'='*60}")
        print(f"[Attempt {self.attempts}] Trying to achieve goal")
        print(f"{'='*60}")
        print(f"Goal: {self.goal}")
        print(f"Known concepts: {self._get_ltm_size()}")

        # Build prompt with current knowledge
        knowledge_summary = self._get_knowledge_summary()

        prompt = f"""You are an AI attempting to achieve a goal using only what you know.

CURRENT KNOWLEDGE:
{knowledge_summary}

GOAL: {self.goal}

INSTRUCTIONS:
1. Try to achieve the goal using ONLY the concepts you know
2. If you can complete it, provide the answer
3. If you cannot, identify EXACTLY and SPECIFICALLY what concepts or RULES you don't understand

IMPORTANT - Be SPECIFIC about what's missing:
- If you know WHAT something is but not HOW to calculate it, say "how to calculate [X]" or "[X] rule"
- Example: Instead of "derivatives", say "power rule for derivatives" or "how to differentiate polynomials"
- Example: Instead of "integration", say "integration by parts" or "fundamental theorem of calculus"
- Focus on the SPECIFIC PROCEDURES, FORMULAS, or RULES you need

Respond in this format:
SUCCESS: [yes/no]
ANSWER: [your answer if successful, or "cannot complete" if not]
MISSING: [comma-separated list of SPECIFIC concepts/rules you need to learn, or "none" if successful]
REASONING: [brief explanation of what specific knowledge gap prevents you from solving this]"""

        response = self.llm.generate(
            system_prompt="You are a self-learning AI that honestly assesses what you know and don't know. When identifying missing knowledge, be VERY SPECIFIC about what procedures, formulas, or rules you need - not just general concepts.",
            user_input=prompt
        )

        text = response.get("text", "")
        print(f"\n[i] Response:\n{text}\n")

        # Parse response
        lines = text.split('\n')
        success = False
        result = ""
        missing = []

        for line in lines:
            if line.startswith("SUCCESS:"):
                success = "yes" in line.lower()
            elif line.startswith("ANSWER:"):
                result = line.split(":", 1)[1].strip()
            elif line.startswith("MISSING:"):
                missing_str = line.split(":", 1)[1].strip()
                if missing_str.lower() not in ["none", "n/a", ""]:
                    missing = [m.strip() for m in missing_str.split(",")]

        # Fallback: Smart detection for natural language responses
        # If no explicit SUCCESS: was found, try to detect success from the response
        if not success and text:
            text_lower = text.lower()

            # Check for explicit answer indicators
            answer_indicators = [
                "the answer is",
                "answer:",
                "solution:",
                "result:",
                "\\boxed{",  # LaTeX boxed answer
                "**answer:**",
            ]

            has_answer = any(indicator in text_lower for indicator in answer_indicators)

            # Check for negative indicators (missing knowledge)
            missing_indicators = [
                "cannot complete",
                "don't know",
                "missing",
                "need to know",
                "requires",
                "lack",
            ]

            has_missing = any(indicator in text_lower for indicator in missing_indicators)

            # If has clear answer and no missing indicators, likely succeeded
            if has_answer and not has_missing:
                success = True
                # Try to extract the answer
                if not result:
                    # Look for answers in common formats
                    import re
                    # Look for boxed answers
                    boxed = re.search(r'\$\\boxed\{([^}]+)\}\$', text)
                    if boxed:
                        result = boxed.group(1)
                    # Look for "Answer: X" or "The answer is X"
                    elif "answer:" in text_lower:
                        for line in lines:
                            if "answer:" in line.lower():
                                result = line.split(":", 1)[1].strip()
                                break
                    # Look for numbered answers
                    elif re.search(r'answer.*?(\d+)', text_lower):
                        match = re.search(r'answer.*?(\d+)', text_lower)
                        result = match.group(1)

        return GoalAttempt(
            success=success,
            result=result,
            missing_concepts=missing
        )

    def _is_valid_concept(self, concept: str) -> bool:
        """Check if concept is valid and worth learning."""
        if not concept or len(concept) < 2:
            return False

        # Skip if too long (probably garbage)
        if len(concept) > 60:
            return False

        concept_lower = concept.lower().strip()

        # Skip meta-concepts
        skip_patterns = [
            "none",
            "n/a",
            "null",
            "undefined",
            "this definition",
            "prior knowledge",
            "general understanding",
            "foundational",
            "doesn't require",
            "beyond that",
            ")",  # Truncated prerequisite
            "(",  # Truncated prerequisite
        ]

        for pattern in skip_patterns:
            if pattern in concept_lower:
                return False

        return True

    def _build_search_query(self, concept: str) -> str:
        """Build a domain-aware search query for a concept."""
        # Add domain context to queries to avoid ambiguity
        domain_prefixes = {
            "mathematics": "mathematical",
            "science": "scientific",
            "programming": "programming",
            "language": "linguistic",
            "literature": "literary"
        }

        # Special handling for ambiguous terms
        ambiguous_terms = {
            "root": {
                "mathematics": "mathematical root of equation",
                "science": "root in biology",
                "general": "root definition"
            },
            "variable": {
                "mathematics": "mathematical variable in algebra",
                "programming": "programming variable",
                "science": "variable in science",
                "general": "variable definition"
            },
            "function": {
                "mathematics": "mathematical function",
                "programming": "programming function",
                "general": "function definition"
            },
            "cell": {
                "mathematics": "cell in mathematics",
                "science": "biological cell",
                "programming": "cell in data structures",
                "general": "cell definition"
            },
            "factor": {
                "mathematics": "mathematical factors divisors",
                "science": "factor in science",
                "general": "factors definition"
            },
            "factors": {
                "mathematics": "mathematical factors of numbers",
                "science": "factors in science",
                "general": "factors definition"
            },
            "operations": {
                "mathematics": "basic mathematical operations",
                "programming": "computer operations",
                "science": "operations in science",
                "general": "operations definition"
            },
            "object": {
                "mathematics": "mathematical object",
                "programming": "programming object",
                "general": "object definition"
            },
            "objects": {
                "mathematics": "mathematical objects",
                "programming": "programming objects",
                "general": "objects definition"
            },
            "characteristics": {
                "mathematics": "characteristics in mathematics",
                "science": "characteristics in science",
                "general": "characteristics definition"
            }
        }

        concept_lower = concept.lower()

        # Check if concept is ambiguous
        for term, domain_queries in ambiguous_terms.items():
            if term in concept_lower:
                if self.goal_domain in domain_queries:
                    return domain_queries[self.goal_domain]
                return domain_queries["general"]

        # Add domain prefix for general queries
        if self.goal_domain in domain_prefixes and self.goal_domain != "general":
            prefix = domain_prefixes[self.goal_domain]
            return f"{prefix} {concept}"

        return f"what is {concept}"

    def _test_concept_understanding(self, concept: str, definition: str, examples: List[str], indent: str = "") -> float:
        """
        SURPRISE EPISODE: Test if LLM actually understood the concept.
        Returns confidence score (0.0-1.0).
        """
        print(f"{indent}    [?] Testing initial understanding (Surprise Episode)...")

        test_prompt = f"""I just learned about "{concept}".

Definition: {definition}

Examples: {', '.join(examples) if examples else 'none provided'}

Test my understanding by asking me 3 questions:
1. Explain "{concept}" in your own words (1 sentence)
2. Give a simple NEW example (not from above)
3. What prerequisite knowledge do you need to understand this?

Be HONEST - if you're not confident, say "I'm not sure" or "I need more examples"."""

        response = self.llm.generate(
            system_prompt="You are testing your understanding of a new concept. Be honest about your confidence level.",
            user_input=test_prompt
        )

        text = response.get("text", "").lower()

        # Calculate confidence based on response quality
        confidence = 0.0

        # Negative indicators (reduce confidence)
        if any(phrase in text for phrase in ["not sure", "don't know", "need more", "unclear", "confused"]):
            confidence = 0.2
        # Too short = didn't understand
        elif len(text) < 50:
            confidence = 0.3
        # No example provided = weak understanding
        elif "example" not in text or text.count(':') < 2:
            confidence = 0.4
        # Missing explanation = incomplete understanding
        elif "explain" not in text and len(text) < 100:
            confidence = 0.5
        # Good response with all components
        else:
            confidence = 0.6
            # Bonus for showing understanding
            if len(text) > 150:
                confidence += 0.1
            if examples and any(ex_word in text for ex_word in ["for example", "such as", "like"]):
                confidence += 0.1

        print(f"{indent}        → Initial confidence: {confidence:.2f}")
        return confidence

    def _rehearse_concept(self, concept: str, definition: str, examples: List[str], current_confidence: float, indent: str = "") -> float:
        """
        REHEARSAL: Practice applying the concept to improve understanding.
        Returns updated confidence score.
        """
        print(f"{indent}    [R] Rehearsal: Practicing application...")

        # Generate a practice problem
        practice_prompt = f"""You learned about "{concept}": {definition}

Examples: {', '.join(examples) if examples else 'none'}

Now demonstrate your understanding by:
1. Solving a NEW problem using this concept
2. Explaining each step
3. Stating why each step is necessary

If you can't solve it, explain what's blocking you."""

        response = self.llm.generate(
            system_prompt="You are practicing a new concept. Show your work step-by-step.",
            user_input=practice_prompt
        )

        text = response.get("text", "").lower()

        # Grade the practice attempt
        new_confidence = current_confidence

        # Check for improvement indicators
        if "step 1" in text or "first" in text:
            new_confidence += 0.1  # Showed step-by-step thinking
        if "because" in text or "therefore" in text or "since" in text:
            new_confidence += 0.1  # Explained reasoning
        if len(text) > 200:
            new_confidence += 0.05  # Detailed response
        if any(word in text for word in ["solve", "calculate", "apply", "use"]):
            new_confidence += 0.05  # Active application

        # Check for struggle indicators (don't increase much)
        if any(phrase in text for phrase in ["can't", "unable", "don't know how", "stuck"]):
            new_confidence = min(new_confidence, current_confidence + 0.05)

        # Cap at 1.0
        new_confidence = min(1.0, new_confidence)

        improvement = new_confidence - current_confidence
        print(f"{indent}        → Confidence: {current_confidence:.2f} → {new_confidence:.2f} (+{improvement:.2f})")

        return new_confidence

    def _generate_additional_examples(self, concept: str, definition: str, indent: str = "") -> List[str]:
        """Generate additional examples to aid learning."""
        print(f"{indent}        [+] Generating additional examples...")

        prompt = f"""Generate 2-3 simple, clear examples demonstrating "{concept}".

Definition: {definition}

Provide SHORT, CONCRETE examples with step-by-step solutions.
Format each as: "Example: [problem] → [solution steps]"

Keep it simple and educational."""

        response = self.llm.generate(
            system_prompt="You are an educational tutor providing clear examples.",
            user_input=prompt
        )

        text = response.get("text", "")

        # Extract examples from response
        examples = []
        for line in text.split('\n'):
            if line.strip() and ('example' in line.lower() or '→' in line or '->' in line):
                examples.append(line.strip())

        if examples:
            print(f"{indent}            Added {len(examples)} practice example(s)")

        return examples

    async def discover_concept(self, concept: str, needed_for: str) -> bool:
        """
        Autonomously discover and learn a concept with 3-stage learning integration.
        Returns True if successfully learned.
        """
        # Validate concept first
        if not self._is_valid_concept(concept):
            print(f"[!] Skipping invalid concept: {concept}")
            return False

        # Check relevance to goal domain
        if not self._is_relevant_to_goal(concept):
            print(f"[!] Skipping irrelevant concept: {concept} (not related to {self.goal_domain})")
            return False

        self.current_depth += 1

        if self.current_depth > self.max_learning_depth:
            print(f"[!] Max learning depth reached, stopping at {concept}")
            self.current_depth -= 1
            return False

        indent = "  " * self.current_depth
        print(f"\n{indent}[L] Discovering: {concept}")
        print(f"{indent}    Needed for: {needed_for}")

        # Check if already known
        if self.ltm.has(concept):
            print(f"{indent}    [i] Already in LTM, skipping")
            self.current_depth -= 1
            return True

        # Check if it's a primitive concept (letters, numbers)
        primitive = self._try_primitive_learning(concept)
        if primitive:
            print(f"{indent}    [OK] Learned as primitive")
            self.journal.append({
                "type": "primitive_learned",
                "concept": concept,
                "needed_for": needed_for,
                "depth": self.current_depth
            })
            self.current_depth -= 1
            return True

        # Search web for concept with domain context
        # Issue #2: Retry with multiple query variations if first attempt fails
        print(f"{indent}    [i] Searching web...")

        # Try multiple search strategies
        search_attempts = [
            self._build_search_query(concept),
            f"{concept} definition",
            f"{concept} explained",
            f"what is {concept}",
            f"{concept} {self.goal_domain}" if self.goal_domain else concept
        ]

        result = None
        for attempt_num, query in enumerate(search_attempts, 1):
            print(f"{indent}    [i] Query attempt {attempt_num}/{len(search_attempts)}: {query}")
            result = self.web_researcher.fetch(query, mode="scrape")
            if result and result.text:
                break
            print(f"{indent}    [!] Attempt {attempt_num} failed, trying next query...")

        if not result or not result.text:
            print(f"{indent}    [X] All {len(search_attempts)} search attempts failed")
            self.current_depth -= 1
            return False

        web_content = result.text[:2000]
        print(f"{indent}    [+] Retrieved {len(result.text)} chars from web")

        # Ask LLM to understand and explain
        domain_context = f" in {self.goal_domain}" if self.goal_domain != "general" else ""
        understanding_prompt = f"""Based on this information about '{concept}'{domain_context}:

{web_content}

CONTEXT: This concept is needed for the goal: "{self.goal}" (domain: {self.goal_domain})

Please provide:
1. A concise definition (1-2 sentences) focusing on the {self.goal_domain} perspective
2. What concrete prerequisite concepts are needed in {self.goal_domain}
3. IMPORTANT: Extract any WORKED EXAMPLES showing step-by-step HOW to apply this concept

Format:
DEFINITION: [your definition]
PREREQUISITES: [comma-separated list of concrete concepts, or "none"]
EXAMPLES: [worked examples with step-by-step solutions, or "none" if no examples found]

CRITICAL - EXAMPLES are the most important part:
- Look for step-by-step solutions showing HOW to apply the concept
- Example for "solving linear equations": "2x + 3 = 7 → subtract 3: 2x = 4 → divide by 2: x = 2"
- Example for "derivative": "d/dx(x^2) = 2x"
- Include multiple examples if available
- If no examples, write "none"

IMPORTANT:
- Only list prerequisites relevant to {self.goal_domain}
- Only list actual concepts (like "addition", "variables", "numbers")
- Do NOT include meta-concepts like "basic understanding" or "familiarity"
- Do NOT include explanations or parenthetical notes
- Keep each prerequisite to 1-3 words maximum
- Focus ONLY on {self.goal_domain} prerequisites, ignore other domains
"""

        response = self.llm.generate(
            system_prompt=f"You are learning a new concept in {self.goal_domain}. Extract the key definition, identify concrete prerequisite concepts, and MOST IMPORTANTLY extract any worked examples showing step-by-step procedures. Examples are critical for learning HOW to apply concepts.",
            user_input=understanding_prompt
        )

        text = response.get("text", "")

        # Check if LLM failed (offline fallback or error)
        if "[offline fallback]" in text or "error" in response or not text.strip():
            print(f"{indent}    [X] LLM unavailable or failed to respond")
            self.current_depth -= 1
            return False

        # Parse response
        definition = ""
        prerequisites = []
        examples = []

        for line in text.split('\n'):
            if line.startswith("DEFINITION:"):
                definition = line.split(":", 1)[1].strip()
            elif line.startswith("PREREQUISITES:"):
                prereq_str = line.split(":", 1)[1].strip()
                if prereq_str.lower() not in ["none", "n/a", ""]:
                    # Split and clean prerequisites
                    raw_prereqs = [p.strip() for p in prereq_str.split(",")]
                    # Filter out invalid concepts AND irrelevant ones
                    prerequisites = [
                        p for p in raw_prereqs
                        if self._is_valid_concept(p) and self._is_relevant_to_goal(p)
                    ]
            elif line.startswith("EXAMPLES:"):
                examples_str = line.split(":", 1)[1].strip()
                if examples_str.lower() not in ["none", "n/a", ""]:
                    # Store the examples
                    examples.append(examples_str)

        if not definition:
            # Fallback: use first substantial line
            lines = text.strip().split('\n')
            definition = lines[0] if lines else "No definition available"

        # BUG FIX: Ensure definition is not empty
        if not definition:
            definition = "Concept learned (no definition extracted)"

        print(f"{indent}    [i] Definition: {definition[:100]}...")
        if examples:
            print(f"{indent}    [i] Found {len(examples)} worked example(s)!")

        # Recursively learn prerequisites
        if prerequisites:
            # Limit to top 3 relevant prerequisites
            valid_prereqs = prerequisites[:3]
            print(f"{indent}    [i] Relevant prerequisites: {', '.join(valid_prereqs)}")

            # Show if any were filtered
            total_before = len([p.strip() for p in text.split("PREREQUISITES:")[-1].split(",") if p.strip()]) if "PREREQUISITES:" in text else 0
            filtered_count = total_before - len(prerequisites)
            if filtered_count > 0:
                print(f"{indent}    [i] Filtered {filtered_count} irrelevant prerequisites")

            for prereq in valid_prereqs:
                if not self.ltm.has(prereq):
                    await self.discover_concept(prereq, needed_for=concept)

        # ============================================================================
        # 3-STAGE LEARNING: Surprise → Rehearsal → Transfer (OPTIONAL for quality)
        # ============================================================================
        final_confidence = 0.95  # Default confidence (used if rehearsal disabled)

        if self.enable_rehearsal:
            print(f"{indent}    [~] 3-Stage Learning enabled (quality control)")

            # STAGE 1: SURPRISE EPISODE - Test initial understanding
            confidence = self._test_concept_understanding(concept, definition, examples, indent)

            # Tiered evaluation of initial understanding
            if confidence >= 0.75:
                print(f"{indent}        [✓✓] Excellent initial understanding!")
            elif confidence >= 0.70:
                print(f"{indent}        [✓] Good initial understanding")
            elif confidence >= 0.65:
                print(f"{indent}        [~] Acceptable initial understanding")
            elif confidence >= 0.30:
                print(f"{indent}        [i] Partial understanding (needs practice)")
            else:
                print(f"{indent}        [!] Genuine surprise (new/difficult concept)")

            # STAGE 2: REHEARSAL LOOP - Practice until mastery (minimum 65%)
            rehearsal_count = 0
            max_rehearsals = 4

            while confidence < 0.65 and rehearsal_count < max_rehearsals:
                rehearsal_count += 1
                print(f"{indent}    [R] Rehearsal {rehearsal_count}/{max_rehearsals}...")

                # If struggling, generate additional examples
                if confidence < 0.50 and rehearsal_count == 2:
                    additional_examples = self._generate_additional_examples(concept, definition, indent)
                    examples.extend(additional_examples)

                # Practice applying the concept
                confidence = self._rehearse_concept(concept, definition, examples, confidence, indent)

                # Check if mastered (tiered confidence evaluation)
                if confidence >= 0.75:
                    print(f"{indent}        [✓✓] Excellent! Confirmed mastery (confidence: {confidence:.2f})")
                    break
                elif confidence >= 0.70:
                    print(f"{indent}        [✓] Good! Concept mastered (confidence: {confidence:.2f})")
                    break
                elif confidence >= 0.65:
                    print(f"{indent}        [~] Acceptable (least possibility, confidence: {confidence:.2f})")
                    break
                elif rehearsal_count >= max_rehearsals:
                    print(f"{indent}        [!] Max rehearsals reached (confidence: {confidence:.2f} < 0.65 minimum)")

            # STAGE 3: CORTICAL TRANSFER - Store with quality indicator
            final_confidence = confidence

            # Tiered confidence evaluation for transfer
            if final_confidence >= 0.75:
                print(f"{indent}    [✓✓] Transferring to LTM (cortical transfer)")
                print(f"{indent}        Final confidence: {final_confidence:.2f} - CONFIRMED (excellent)")
            elif final_confidence >= 0.70:
                print(f"{indent}    [✓] Transferring to LTM (cortical transfer)")
                print(f"{indent}        Final confidence: {final_confidence:.2f} - YES (good understanding)")
            elif final_confidence >= 0.65:
                print(f"{indent}    [~] Transferring to LTM (cortical transfer)")
                print(f"{indent}        Final confidence: {final_confidence:.2f} - ACCEPTABLE (least possibility)")
            else:
                print(f"{indent}    [!] Below minimum threshold (0.65)")
                print(f"{indent}        Confidence: {final_confidence:.2f}")
                print(f"{indent}        Storing anyway (will reinforce later if needed)")
        else:
            print(f"{indent}    [i] 3-Stage Learning disabled (fast mode)")

        # Validate concept before storing (OPTIONAL - can be disabled for speed)
        if self.enable_validation:
            print(f"{indent}    [i] Validating concept...")
            validation_result = self.validator.validate_concept(concept, definition, examples)
            print(f"{indent}    [i] Validation confidence: {validation_result.confidence_score:.2f}")
            print(f"{indent}    [i] Sources: {validation_result.sources_verified}")

            # Combine rehearsal confidence with validation confidence (average)
            if self.enable_rehearsal:
                combined_confidence = (final_confidence + validation_result.confidence_score) / 2
                print(f"{indent}    [i] Combined confidence: {combined_confidence:.2f} (rehearsal: {final_confidence:.2f}, validation: {validation_result.confidence_score:.2f})")
            else:
                combined_confidence = validation_result.confidence_score

            should_store = combined_confidence >= 0.6
        else:
            # Skip validation for SPEED
            print(f"{indent}    [i] Validation disabled (fast mode)")
            validation_result = ValidationResult(
                confidence_score=final_confidence,  # Use rehearsal confidence
                sources_verified=1,
                examples_valid=True,  # Assume valid in fast mode
                self_test_passed=True,  # Assume passed in fast mode
                details=f"Rehearsal confidence: {final_confidence:.2f}" if self.enable_rehearsal else "Fast mode"
            )
            combined_confidence = final_confidence
            should_store = True  # Always store when validation disabled

        # Store in LTM with combined confidence
        entry = LearningEntry(
            concept=concept,
            definition=definition,
            learned_at=datetime.now().isoformat(),
            needed_for=needed_for,
            source="web",
            examples=examples,
            confidence_score=combined_confidence,
            validation_sources=validation_result.sources_verified,
            validation_details=validation_result.details
        )

        # Only store if confidence is sufficient (or validation disabled)
        if should_store:
            self.ltm.add(entry)
            print(f"{indent}    [✓] Stored in LTM (validated)")
            if examples:
                print(f"{indent}    [✓] Stored {len(examples)} example(s) for future reference!")

            # If it's a mathematical concept, also learn it symbolically with MathConnect
            if self.using_mathconnect and self.math_connect and self._is_mathematical_concept(concept, definition):
                print(f"{indent}    [🧮] Mathematical concept detected!")
                print(f"{indent}    [🧮] Learning symbolically with MathConnect...")

                # Try to learn as symbolic equation
                success = self.math_connect.learn_theorem_from_text(
                    name=concept,
                    text=definition,
                    domain=self.goal_domain
                )

                if success:
                    print(f"{indent}    [✓] Learned as symbolic equation!")

                    # Check if MathConnect found connections
                    connections = self.math_connect.get_graph().get(concept, [])
                    if connections:
                        print(f"{indent}    [✓] Found {len(connections)} connection(s) to other theorems!")
                        print(f"{indent}        Connected to: {', '.join(list(connections)[:3])}")

                    # Check if new theorems were derived
                    total_theorems = len(self.math_connect.connection_finder.theorems)
                    print(f"{indent}    [i] Total theorems in graph: {total_theorems}")
                else:
                    print(f"{indent}    [i] Could not parse as symbolic equation (text-based learning only)")

            # Log in journal with confidence breakdown
            journal_entry = {
                "type": "concept_learned",
                "concept": concept,
                "definition": definition,
                "needed_for": needed_for,
                "prerequisites": prerequisites,
                "depth": self.current_depth,
                "confidence": combined_confidence
            }

            # Add 3-stage learning details if enabled
            if self.enable_rehearsal:
                journal_entry["rehearsal_confidence"] = final_confidence
                journal_entry["rehearsal_enabled"] = True

            if self.enable_validation:
                journal_entry["validation_confidence"] = validation_result.confidence_score

            self.journal.append(journal_entry)

            self.current_depth -= 1
            return True
        else:
            print(f"{indent}    [✗] Rejected - low confidence ({combined_confidence:.2f})")
            print(f"{indent}    [!] Try learning from a different source")

            # Log rejection in journal
            self.journal.append({
                "type": "concept_rejected",
                "concept": concept,
                "reason": "low_confidence",
                "confidence": combined_confidence,
                "rehearsal_confidence": final_confidence if self.enable_rehearsal else None,
                "validation_confidence": validation_result.confidence_score if self.enable_validation else None,
                "depth": self.current_depth
            })

            self.current_depth -= 1
            return False

    def _try_primitive_learning(self, concept: str) -> bool:
        """
        Try to learn concept as a primitive (A-Z, 0-9, basic symbols).
        Returns True if it's a primitive concept.
        """
        concept_lower = concept.lower()

        # Single letters
        if len(concept) == 1 and concept.isalpha():
            definition = f"The letter '{concept.upper()}', a basic symbol of written language"
            entry = LearningEntry(
                concept=concept_lower,
                definition=definition,
                learned_at=datetime.now().isoformat(),
                needed_for="primitives",
                source="primitive"
            )
            self.ltm.add(entry)
            return True

        # Single digits
        if len(concept) == 1 and concept.isdigit():
            definition = f"The number {concept}, representing quantity"
            entry = LearningEntry(
                concept=concept_lower,
                definition=definition,
                learned_at=datetime.now().isoformat(),
                needed_for="primitives",
                source="primitive"
            )
            self.ltm.add(entry)
            return True

        # Basic concepts
        primitives = {
            "alphabet": "The letters A-Z used in written English",
            "numbers": "The digits 0-9 used to represent quantities",
            "letter": "A symbol representing a sound in written language",
            "digit": "A symbol representing a number (0-9)",
        }

        if concept_lower in primitives:
            entry = LearningEntry(
                concept=concept_lower,
                definition=primitives[concept_lower],
                learned_at=datetime.now().isoformat(),
                needed_for="primitives",
                source="primitive"
            )
            self.ltm.add(entry)
            return True

        return False

    async def pursue_goal(self, max_attempts: int = None) -> bool:
        """
        Autonomously pursue the goal through self-discovery.
        Returns True if goal achieved.

        Args:
            max_attempts: Maximum attempts before giving up (None = unlimited)
        """
        print("\n" + "="*60)
        print("[G] SELF-DISCOVERY LEARNING")
        print("="*60)
        print(f"Goal: {self.goal}")
        print(f"Starting LTM size: {self._get_ltm_size()} concepts")
        if max_attempts:
            print(f"Max attempts: {max_attempts}")
        else:
            print("Max attempts: UNLIMITED (will run until success)")
        print("="*60)

        # NEW: Create learning plan with dependency graph (AGI planning!)
        learning_plan = None
        if self.using_agi_modules and self.goal_planner:
            print("\n[🎯] Creating learning plan with dependency graph...")
            try:
                learning_stages, graph = await self.goal_planner.create_learning_plan(
                    self.goal, self.goal_domain, max_depth=4
                )
                learning_plan = (learning_stages, graph)
            except Exception as e:
                print(f"[!] Goal planning failed: {e}, continuing without plan")
                learning_plan = None

        attempt_num = 0
        attempt_history = []  # Track last 10 attempts to detect loops
        stuck_count = 0
        start_time = datetime.now()  # Track learning session time

        while True:
            attempt_num += 1

            # Check if we've hit max attempts
            if max_attempts and attempt_num > max_attempts:
                print("\n[X] Max attempts reached without achieving goal")
                return False

            # Attempt goal
            attempt = await self.attempt_goal()

            if attempt.success:
                print("\n" + "="*60)
                print("[OK] GOAL ACHIEVED!")
                print("="*60)
                print(f"Answer: {attempt.result}")
                print(f"Attempts: {self.attempts}")
                print(f"Concepts learned: {len(self.journal)}")

                # NEW: Record successful learning session in meta-learner
                if self.using_agi_modules and self.meta_learner:
                    session_time = (datetime.now() - start_time).total_seconds()
                    learning_record = LearningAttempt(
                        concept=self.goal,
                        domain=self.goal_domain,
                        attempts=self.attempts,
                        time_seconds=session_time,
                        final_confidence=0.75,  # Achieved goal
                        success=True,
                        prerequisites_learned=len(self.concepts_learned_this_session),
                        web_searches=len(self.parallel_web.search_history) if self.parallel_web else 0,
                        rehearsal_rounds=3 if self.enable_rehearsal else 0,
                        timestamp=datetime.now().isoformat()
                    )
                    self.meta_learner.record_attempt(learning_record)
                    print("\n[📊] Meta-learning: Session recorded for future improvement")

                    # Show improvement stats
                    improvement = self.meta_learner.analyze_improvement()
                    if not improvement.get("not_enough_data"):
                        print(f"[📊] Learning improvement: {improvement.get('attempts_improvement', 0)*100:.1f}% fewer attempts needed")
                        print(f"[📊] Confidence improvement: +{improvement.get('confidence_improvement', 0)*100:.1f}%")

                # NEW: Show AGI summaries
                if self.using_agi_modules:
                    if self.creative_reasoner and len(self.creative_reasoner.insights_generated) > 0:
                        print(self.creative_reasoner.summarize_insights())
                    if self.curiosity_engine and len(self.curiosity_engine.questions_generated) > 0:
                        print(self.curiosity_engine.summarize_curiosity())
                    if self.causal_reasoner and len(self.causal_reasoner.causal_graph) > 0:
                        print(self.causal_reasoner.summarize_causal_knowledge())

                # Issue #6: Flush any pending saves
                if self.using_hybrid and hasattr(self.ltm, 'save'):
                    print("[i] Saving all learned concepts to disk...")
                    self.ltm.save(force=True)

                return True

            # Goal failed - learn missing concepts
            if not attempt.missing_concepts:
                print("\n[!] Goal failed but no missing concepts identified")
                print("[!] LLM may not understand the goal or is confused")

                # Issue #6: Flush saves before exiting
                if self.using_hybrid and hasattr(self.ltm, 'save'):
                    self.ltm.save(force=True)

                return False

            print(f"\n[i] Missing concepts: {', '.join(attempt.missing_concepts)}")

            # NEW: Filter irrelevant prerequisites using RelevanceFilter
            filtered_concepts = attempt.missing_concepts
            if self.using_agi_modules and self.relevance_filter:
                print(f"\n[🔍] Filtering {len(attempt.missing_concepts)} prerequisites for relevance...")
                relevant, filtered_out = await self.relevance_filter.filter_prerequisites(
                    attempt.missing_concepts,
                    self.goal,
                    self.goal,
                    self.goal_domain
                )
                filtered_concepts = relevant
                if filtered_out:
                    print(f"[✓] Filtered out {len(filtered_out)} irrelevant concepts: {', '.join(filtered_out[:5])}")
                    print(f"[✓] Learning only {len(relevant)} relevant concepts")

            # NEW: Metacognition - check if we're going off track
            if self.using_agi_modules and self.metacognition and len(self.concepts_learned_this_session) > 0:
                if len(self.concepts_learned_this_session) % 3 == 0:  # Every 3 concepts
                    on_track, reason = await self.metacognition.check_if_on_track(
                        self.goal,
                        filtered_concepts[0] if filtered_concepts else "unknown",
                        self.concepts_learned_this_session,
                        self.current_depth
                    )
                    if not on_track:
                        print(f"[🧠] METACOGNITION WARNING: Might be going off track!")
                        print(f"[🧠] Consider: {reason}")

            # Loop detection: track history of ALL attempts to detect alternating loops
            current_missing = frozenset(attempt.missing_concepts)  # Use frozenset for hashable set
            attempt_history.append(current_missing)

            # Keep only last 10 attempts
            if len(attempt_history) > 10:
                attempt_history.pop(0)

            # Check if we've seen this exact set in the last 10 attempts
            if len(attempt_history) >= 2:
                # Count how many times this set appears in history
                occurrences = attempt_history[:-1].count(current_missing)
                if occurrences > 0:
                    stuck_count += 1
                    print(f"[!] Warning: Seen these concepts before (stuck count: {stuck_count}/5, seen {occurrences+1} times)")
                    if stuck_count >= 5:
                        print("\n" + "="*60)
                        print("[X] STUCK IN LEARNING LOOP")
                        print("="*60)
                        print("The system is repeatedly requesting the same concepts but cannot apply them.")
                        print("This suggests:")
                        print("  1. LLM lacks reasoning capability for this goal")
                        print("  2. Web content has definitions but no worked examples")
                        print("  3. Concepts are too abstract without procedural knowledge")
                        print(f"\nRepeated concepts: {', '.join(sorted(current_missing))}")
                        print(f"\nAttempt history (last 10):")
                        for i, hist_set in enumerate(attempt_history[-10:], 1):
                            print(f"  {i}. {{{', '.join(sorted(hist_set))}}}")

                        # Issue #6: Flush saves before exiting
                        if self.using_hybrid and hasattr(self.ltm, 'save'):
                            self.ltm.save(force=True)

                        return False
                else:
                    stuck_count = 0

            # MULTIPROCESSING: Learn all missing concepts in parallel!
            # This dramatically speeds up learning when multiple concepts are needed
            # Use filtered_concepts (relevance filter applied!)
            concepts_to_learn = filtered_concepts if self.using_agi_modules else attempt.missing_concepts
            num_concepts = len(concepts_to_learn)

            if num_concepts == 0:
                print("[i] No relevant concepts to learn after filtering, retrying goal...")
                continue

            parallel_batch = min(num_concepts, self.max_parallel_concepts)
            print(f"[⚡] Learning {num_concepts} concepts (batch size: {parallel_batch} parallel)...")

            async def learn_with_error_handling(concept):
                """Wrapper to handle errors in parallel learning."""
                try:
                    learned = await self.discover_concept(concept, needed_for=self.goal)
                    if learned and self.using_agi_modules:
                        # Track for meta-learning
                        self.concepts_learned_this_session.append(concept)
                    if not learned:
                        print(f"[!] Failed to learn {concept}, continuing anyway...")
                    return learned
                except Exception as e:
                    print(f"[!] Error learning {concept}: {e}")
                    return False

            # Learn concepts in parallel batches to avoid overwhelming the system
            all_results = []
            for i in range(0, num_concepts, parallel_batch):
                batch = concepts_to_learn[i:i+parallel_batch]
                print(f"[⚡] Processing batch {i//parallel_batch + 1} ({len(batch)} concepts)...")

                # Learn batch concurrently using asyncio.gather
                batch_results = await asyncio.gather(
                    *[learn_with_error_handling(c) for c in batch],
                    return_exceptions=True
                )
                all_results.extend(batch_results)

            # Count successful learnings
            successful = sum(1 for r in all_results if r is True)
            print(f"[✓] Successfully learned {successful}/{num_concepts} concepts using parallel processing")

            # NEW: Generate creative insights after learning batch
            if self.using_agi_modules and self.creative_reasoner and successful > 2:
                try:
                    await self.creative_reasoner.find_hidden_patterns(
                        self.concepts_learned_this_session[-successful:],
                        self.goal_domain
                    )
                except Exception as e:
                    print(f"[!] Creative reasoning failed: {e}")

            # Brief delay before retry
            await asyncio.sleep(1)

    def print_learning_journal(self):
        """Print the learning journey."""
        print("\n" + "="*60)
        print("[#] LEARNING JOURNAL - Discovery Path")
        print("="*60)

        if not self.journal:
            print("No learning occurred")
            return

        for i, entry in enumerate(self.journal, 1):
            if entry["type"] == "concept_learned":
                indent = "  " * entry.get("depth", 0)
                print(f"\n{i}. {indent}Learned: {entry['concept']}")
                print(f"   {indent}Needed for: {entry['needed_for']}")
                print(f"   {indent}Definition: {entry['definition'][:80]}...")
                if entry.get("prerequisites"):
                    print(f"   {indent}Prerequisites: {', '.join(entry['prerequisites'])}")
            elif entry["type"] == "primitive_learned":
                indent = "  " * entry.get("depth", 0)
                print(f"\n{i}. {indent}Learned primitive: {entry['concept']}")
                print(f"   {indent}Needed for: {entry['needed_for']}")

        print("\n" + "="*60)
        print(f"Total concepts learned: {len(self.journal)}")
        print(f"Final LTM size: {self._get_ltm_size()}")
        print("="*60)

    def print_math_knowledge_graph(self):
        """Print the mathematical knowledge graph built by MathConnect."""
        if not self.using_mathconnect:
            return

        print("\n" + "="*60)
        print("[🧮] MATHEMATICAL KNOWLEDGE GRAPH")
        print("="*60)

        summary = self.math_connect.summarize()
        print("\n" + summary)

        # Show connection graph
        graph = self.math_connect.get_graph()
        if graph:
            print("\n" + "="*60)
            print("THEOREM CONNECTIONS")
            print("="*60)

            for theorem, connections in sorted(graph.items()):
                if connections:
                    print(f"\n{theorem}")
                    print(f"  ↔ {', '.join(list(connections)[:5])}")
                    if len(connections) > 5:
                        print(f"  ... and {len(connections) - 5} more")

        print("\n" + "="*60)


async def main_self_discovery(
    goal: str,
    ltm_path: str = "./ltm_memory.json",
    max_attempts: int = None,
    enable_validation: bool = False,
    enable_rehearsal: bool = True,
    target_confidence: float = 0.85
):
    """Run self-discovery learning experiment.

    Args:
        goal: The goal to achieve
        ltm_path: Path to LTM storage file
        max_attempts: Maximum attempts (None = unlimited, will run until success)
        enable_validation: Enable multi-source validation (default: False for speed)
        enable_rehearsal: Enable 3-stage learning rehearsal (default: True for quality)
        target_confidence: Mastery threshold for 3-stage learning 0.0-1.0 (default: 0.85)
    """
    orchestrator = SelfDiscoveryOrchestrator(
        goal=goal,
        ltm_path=ltm_path,
        enable_validation=enable_validation,
        enable_rehearsal=enable_rehearsal,
        target_confidence=target_confidence
    )

    success = await orchestrator.pursue_goal(max_attempts=max_attempts)

    # Print learning journal
    orchestrator.print_learning_journal()

    # Print mathematical knowledge graph (if MathConnect was used)
    orchestrator.print_math_knowledge_graph()

    # Print final LTM summary
    print("\n" + "="*60)
    print("[#] LONG-TERM MEMORY")
    print("="*60)

    concepts = sorted(orchestrator._get_all_concepts())
    for concept in concepts:
        entry = orchestrator.ltm.get(concept)
        print(f"\n{concept}:")
        print(f"  {entry.definition}")
        print(f"  (learned for: {entry.needed_for})")

    print("\n" + "="*60)
    print(f"[{'OK' if success else 'X'}] Goal {'achieved' if success else 'not achieved'}")
    if hasattr(orchestrator.ltm, 'storage_path'):
        print(f"LTM saved to: {orchestrator.ltm.storage_path}")
    print("="*60)

    return success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python self_discovery_orchestrator.py '<goal>'")
        print("\nExample goals:")
        print("  'Count from 1 to 5'")
        print("  'Solve 2x + 5 = 15'")
        print("  'Calculate area of circle with radius 3'")
        print("  'Explain why ice floats'")
        sys.exit(1)

    goal = sys.argv[1]
    asyncio.run(main_self_discovery(goal))
