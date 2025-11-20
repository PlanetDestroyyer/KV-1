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
from core.knowledge_validator import KnowledgeValidator


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
        max_depth: int = 10
    ):
        self.goal = goal
        self.max_learning_depth = max_depth
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Initialize components
        self.ltm = PersistentLTM(ltm_path)
        self.llm = LLMBridge(provider="ollama", default_model="qwen3:4b")
        self.web_researcher = WebResearcher(
            cache_dir=os.path.join(data_dir, "web_cache"),
            daily_cap=100
        )
        self.validator = KnowledgeValidator(self.llm, self.web_researcher)

        # Learning journal
        self.journal: List[Dict] = []
        self.current_depth = 0
        self.attempts = 0

        # Detect goal domain for focused learning
        self.goal_domain = self._detect_goal_domain(goal)
        self.goal_keywords = self._extract_goal_keywords(goal)
        print(f"[i] Detected goal domain: {self.goal_domain}")
        print(f"[i] Goal keywords: {', '.join(self.goal_keywords)}")

    def _detect_goal_domain(self, goal: str) -> str:
        """Detect the domain of the goal for focused learning."""
        goal_lower = goal.lower()

        # Domain indicators
        math_indicators = [
            "solve", "equation", "calculate", "add", "subtract", "multiply", "divide",
            "algebra", "calculus", "geometry", "trigonometry", "quadratic", "polynomial",
            "derivative", "integral", "matrix", "vector", "function", "graph",
            "x =", "y =", "2x", "3x", "squared", "cubed", "pi", "area", "volume",
            "radius", "diameter", "perimeter", "percentage", "ratio", "fraction"
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
        if not self.ltm.knowledge:
            return "You have no prior knowledge. You are starting from zero."

        concepts = self.ltm.get_all_concepts()
        summary = f"You currently know these {len(concepts)} concepts:\n"
        for concept in sorted(concepts)[:20]:  # Limit to avoid context overflow
            entry = self.ltm.get(concept)
            summary += f"- {concept}: {entry.definition[:100]}...\n"
            # Include examples if available - these are CRITICAL for learning HOW to apply concepts
            if entry.examples:
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
        print(f"Known concepts: {len(self.ltm.knowledge)}")

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

    async def discover_concept(self, concept: str, needed_for: str) -> bool:
        """
        Autonomously discover and learn a concept.
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
        print(f"{indent}    [i] Searching web...")
        search_query = self._build_search_query(concept)
        print(f"{indent}    [i] Query: {search_query}")
        result = self.web_researcher.fetch(search_query, mode="scrape")

        if not result or not result.text:
            print(f"{indent}    [X] No web content found")
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
            definition = text.strip().split('\n')[0]

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

        # Validate concept before storing
        print(f"{indent}    [i] Validating concept...")
        validation_result = self.validator.validate_concept(concept, definition, examples)

        print(f"{indent}    [i] Confidence: {validation_result.confidence_score:.2f}")
        print(f"{indent}    [i] Sources: {validation_result.sources_verified}")

        # Store in LTM with validation info
        entry = LearningEntry(
            concept=concept,
            definition=definition,
            learned_at=datetime.now().isoformat(),
            needed_for=needed_for,
            source="web",
            examples=examples,
            confidence_score=validation_result.confidence_score,
            validation_sources=validation_result.sources_verified,
            validation_details=validation_result.details
        )

        # Only store if confidence is sufficient
        if self.validator.should_store(validation_result, threshold=0.6):
            self.ltm.add(entry)
            print(f"{indent}    [✓] Stored in LTM (validated)")
            if examples:
                print(f"{indent}    [✓] Stored {len(examples)} example(s) for future reference!")

            # Log in journal
            self.journal.append({
                "type": "concept_learned",
                "concept": concept,
                "definition": definition,
                "needed_for": needed_for,
                "prerequisites": prerequisites,
                "depth": self.current_depth,
                "confidence": validation_result.confidence_score
            })

            self.current_depth -= 1
            return True
        else:
            print(f"{indent}    [✗] Rejected - low confidence ({validation_result.confidence_score:.2f})")
            print(f"{indent}    [!] Try learning from a different source")

            # Log rejection in journal
            self.journal.append({
                "type": "concept_rejected",
                "concept": concept,
                "reason": "low_validation_confidence",
                "confidence": validation_result.confidence_score,
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
        print(f"Starting LTM size: {len(self.ltm.knowledge)} concepts")
        if max_attempts:
            print(f"Max attempts: {max_attempts}")
        else:
            print("Max attempts: UNLIMITED (will run until success)")
        print("="*60)

        attempt_num = 0
        last_missing = set()
        stuck_count = 0

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
                return True

            # Goal failed - learn missing concepts
            if not attempt.missing_concepts:
                print("\n[!] Goal failed but no missing concepts identified")
                print("[!] LLM may not understand the goal or is confused")
                return False

            print(f"\n[i] Missing concepts: {', '.join(attempt.missing_concepts)}")

            # Loop detection: check if we're stuck requesting the same concepts
            current_missing = set(attempt.missing_concepts)
            if current_missing == last_missing:
                stuck_count += 1
                print(f"[!] Warning: Requesting same concepts again (stuck count: {stuck_count}/5)")
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
                    return False
            else:
                stuck_count = 0
                last_missing = current_missing

            # Learn each missing concept
            for concept in attempt.missing_concepts:
                learned = await self.discover_concept(concept, needed_for=self.goal)
                if not learned:
                    print(f"[!] Failed to learn {concept}, continuing anyway...")

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
        print(f"Final LTM size: {len(self.ltm.knowledge)}")
        print("="*60)


async def main_self_discovery(goal: str, ltm_path: str = "./ltm_memory.json", max_attempts: int = None):
    """Run self-discovery learning experiment.

    Args:
        goal: The goal to achieve
        ltm_path: Path to LTM storage file
        max_attempts: Maximum attempts (None = unlimited, will run until success)
    """
    orchestrator = SelfDiscoveryOrchestrator(
        goal=goal,
        ltm_path=ltm_path
    )

    success = await orchestrator.pursue_goal(max_attempts=max_attempts)

    # Print learning journal
    orchestrator.print_learning_journal()

    # Print final LTM summary
    print("\n" + "="*60)
    print("[#] LONG-TERM MEMORY")
    print("="*60)

    concepts = sorted(orchestrator.ltm.get_all_concepts())
    for concept in concepts:
        entry = orchestrator.ltm.get(concept)
        print(f"\n{concept}:")
        print(f"  {entry.definition}")
        print(f"  (learned for: {entry.needed_for})")

    print("\n" + "="*60)
    print(f"[{'OK' if success else 'X'}] Goal {'achieved' if success else 'not achieved'}")
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
