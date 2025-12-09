"""
Unified AGI Learning System

Intelligently routes between tensor reasoning and traditional learning.

For MATHEMATICAL questions:
  → Uses tensor reasoning (symbolic + geometric + exploration)

For GENERAL questions:
  → Uses traditional learning (web search + LLM + memory)

For HYBRID questions:
  → Uses BOTH approaches!

This makes KV-1 a TRUE AGI that handles ANY domain!
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Import both systems
try:
    from core.tensor_reasoning_system import TensorReasoningSystem, ReasoningResult
    TENSOR_AVAILABLE = True
except ImportError:
    TENSOR_AVAILABLE = False
    print("[!] Tensor reasoning not available")


class QuestionType(Enum):
    """Type of question"""
    MATHEMATICAL = "mathematical"
    GENERAL = "general"
    HYBRID = "hybrid"


@dataclass
class UnifiedResult:
    """Result from unified learning"""
    success: bool
    answer: str
    method: str  # "tensor", "traditional", "hybrid"
    question_type: QuestionType
    confidence: float
    details: Dict[str, Any]


class UnifiedAGILearner:
    """
    Complete AGI learning system.

    Handles BOTH mathematical reasoning AND general knowledge!
    Intelligently routes to the right system.
    """

    def __init__(self, llm_bridge, web_researcher, memory):
        """
        Initialize with both reasoning systems.

        Args:
            llm_bridge: For general learning
            web_researcher: For web search
            memory: For knowledge storage
        """
        print("\n[🚀] Initializing UNIFIED AGI LEARNING SYSTEM...")

        self.llm = llm_bridge
        self.web = web_researcher
        self.memory = memory

        # Initialize tensor reasoning if available
        if TENSOR_AVAILABLE:
            print("[✓] Tensor Reasoning System available")
            self.tensor_system = TensorReasoningSystem(dimension=768)
            self.has_tensor = True
        else:
            print("[!] Tensor Reasoning System not available (using traditional only)")
            self.tensor_system = None
            self.has_tensor = False

        print("[✅] UNIFIED AGI SYSTEM READY!")
        print("    - Tensor reasoning: ENABLED" if self.has_tensor else "    - Tensor reasoning: DISABLED")
        print("    - Traditional learning: ENABLED")
        print("    - Hybrid mode: ENABLED" if self.has_tensor else "    - Hybrid mode: DISABLED")

    def detect_question_type(self, question: str) -> QuestionType:
        """
        Detect if question is mathematical, general, or hybrid.

        Uses pattern matching to classify.
        """
        question_lower = question.lower()

        # Mathematical indicators
        math_keywords = [
            # Proof-related
            "prove", "proof", "theorem", "lemma", "conjecture",
            # Math operations
            "solve", "equation", "calculate", "compute", "evaluate",
            # Number theory
            "prime", "factor", "divisor", "gcd", "lcm", "composite",
            # Algebra
            "quadratic", "polynomial", "derivative", "integral", "limit",
            # Geometry
            "triangle", "circle", "angle", "area", "volume",
            # Logic
            "if and only if", "iff", "implies", "forall", "exists",
            # Symbols
            "=", "+", "-", "*", "/", "^", "√", "∑", "∫"
        ]

        # General knowledge indicators
        general_keywords = [
            "what is", "who is", "when did", "where is", "how does",
            "full form", "stands for", "meaning of", "definition of",
            "explain", "describe", "tell me about", "information about"
        ]

        # Count matches
        math_score = sum(1 for kw in math_keywords if kw in question_lower)
        general_score = sum(1 for kw in general_keywords if kw in question_lower)

        # Classify
        if math_score > 0 and general_score > 0:
            return QuestionType.HYBRID
        elif math_score > 0:
            return QuestionType.MATHEMATICAL
        else:
            return QuestionType.GENERAL

    async def learn(
        self,
        question: str,
        force_method: Optional[str] = None
    ) -> UnifiedResult:
        """
        Unified learning interface.

        Automatically routes to tensor reasoning or traditional learning!

        Args:
            question: Question to answer
            force_method: Optional force "tensor" or "traditional"

        Returns:
            UnifiedResult with answer
        """
        print(f"\n{'='*70}")
        print(f"[🧠] UNIFIED AGI LEARNER")
        print(f"{'='*70}")
        print(f"Question: {question}")

        # Detect question type
        if force_method:
            print(f"Method: FORCED → {force_method}")
            qtype = QuestionType.MATHEMATICAL if force_method == "tensor" else QuestionType.GENERAL
        else:
            qtype = self.detect_question_type(question)
            print(f"Detected type: {qtype.value}")

        print(f"{'='*70}\n")

        # Route to appropriate system
        if force_method == "tensor" or (qtype == QuestionType.MATHEMATICAL and self.has_tensor):
            # Use tensor reasoning
            print("[🔬] Routing to TENSOR REASONING SYSTEM...")
            return await self._learn_tensor(question)

        elif force_method == "traditional" or qtype == QuestionType.GENERAL:
            # Use traditional learning
            print("[📚] Routing to TRADITIONAL LEARNING...")
            return await self._learn_traditional(question)

        elif qtype == QuestionType.HYBRID and self.has_tensor:
            # Use both!
            print("[🔀] HYBRID MODE: Using BOTH systems...")
            return await self._learn_hybrid(question)

        else:
            # Fallback to traditional
            print("[📚] Fallback to TRADITIONAL LEARNING...")
            return await self._learn_traditional(question)

    async def _learn_tensor(self, question: str) -> UnifiedResult:
        """Use tensor reasoning system"""
        try:
            result = await self.tensor_system.solve(question, method='auto', timeout=60)

            return UnifiedResult(
                success=result.success,
                answer=result.answer,
                method="tensor",
                question_type=QuestionType.MATHEMATICAL,
                confidence=result.confidence,
                details={
                    'proof_steps': result.proof_steps,
                    'reasoning_method': result.method,
                    'exploration_stats': result.exploration_stats
                }
            )
        except Exception as e:
            print(f"[!] Tensor reasoning failed: {e}")
            print("[!] Falling back to traditional learning...")
            return await self._learn_traditional(question)

    async def _learn_traditional(self, question: str) -> UnifiedResult:
        """Use traditional web search + LLM learning"""

        # Search web
        print(f"[🌐] Searching web for: {question}")
        search_result = self.web.fetch(question)

        if not search_result:
            return UnifiedResult(
                success=False,
                answer="No information found",
                method="traditional",
                question_type=QuestionType.GENERAL,
                confidence=0.0,
                details={'error': 'Web search failed'}
            )

        # Use LLM to extract answer
        print(f"[🤖] Extracting answer using LLM...")

        prompt = f"""Based on this information:

{search_result.text[:2000]}

Answer the question: {question}

Provide a clear, concise answer (2-3 sentences).
"""

        response = self.llm.generate(
            system_prompt="You extract accurate answers from web content.",
            user_input=prompt
        )

        answer = response.get('text', 'Could not extract answer')

        # Store in memory
        if self.memory:
            try:
                # Try to store
                if hasattr(self.memory, 'store'):
                    self.memory.store(question, answer, confidence=0.8)
            except:
                pass

        return UnifiedResult(
            success=True,
            answer=answer,
            method="traditional",
            question_type=QuestionType.GENERAL,
            confidence=0.8,
            details={'source': 'web', 'length': len(answer)}
        )

    async def _learn_hybrid(self, question: str) -> UnifiedResult:
        """Use BOTH tensor reasoning AND traditional learning"""

        print("[🔀] Using BOTH systems in parallel...")

        # Run both in parallel
        tensor_task = self._learn_tensor(question)
        traditional_task = self._learn_traditional(question)

        results = await asyncio.gather(tensor_task, traditional_task, return_exceptions=True)

        tensor_result = results[0] if not isinstance(results[0], Exception) else None
        trad_result = results[1] if not isinstance(results[1], Exception) else None

        # Combine results
        if tensor_result and tensor_result.success:
            # Tensor reasoning succeeded
            print("[✓] Tensor reasoning succeeded, using that answer")
            combined_answer = f"Mathematical proof: {tensor_result.answer}"

            if trad_result and trad_result.success:
                # Add context from traditional
                combined_answer += f"\n\nAdditional context: {trad_result.answer}"

            return UnifiedResult(
                success=True,
                answer=combined_answer,
                method="hybrid",
                question_type=QuestionType.HYBRID,
                confidence=max(tensor_result.confidence, 0.8),
                details={
                    'tensor': tensor_result.details,
                    'traditional': trad_result.details if trad_result else {}
                }
            )

        elif trad_result and trad_result.success:
            # Only traditional succeeded
            print("[!] Tensor reasoning failed, using traditional answer")
            return UnifiedResult(
                success=True,
                answer=trad_result.answer,
                method="hybrid_fallback",
                question_type=QuestionType.HYBRID,
                confidence=trad_result.confidence,
                details={'traditional': trad_result.details}
            )

        else:
            # Both failed
            return UnifiedResult(
                success=False,
                answer="Could not answer question with either method",
                method="hybrid_failed",
                question_type=QuestionType.HYBRID,
                confidence=0.0,
                details={'error': 'Both methods failed'}
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both systems"""
        stats = {
            'has_tensor_reasoning': self.has_tensor,
            'traditional_learning': True
        }

        if self.has_tensor and self.tensor_system:
            stats['tensor'] = self.tensor_system.get_stats()

        return stats


# Quick test
async def test_unified_system():
    """Test the unified system with both math and general questions"""

    # Mock components for testing
    class MockLLM:
        def generate(self, system_prompt, user_input):
            return {'text': 'Artificial Intelligence is the simulation of human intelligence by machines.'}

    class MockWeb:
        def search(self, query):
            return "AI stands for Artificial Intelligence. It is a branch of computer science..."

    class MockMemory:
        def store(self, key, value, confidence):
            print(f"[Stored] {key}: {value[:50]}...")

    # Initialize
    system = UnifiedAGILearner(MockLLM(), MockWeb(), MockMemory())

    # Test mathematical question
    print("\n" + "="*70)
    print("TEST 1: Mathematical Question")
    print("="*70)
    result = await system.learn("What are prime numbers?")
    print(f"\nAnswer: {result.answer}")
    print(f"Method: {result.method}")
    print(f"Success: {result.success}")

    # Test general question
    print("\n" + "="*70)
    print("TEST 2: General Question")
    print("="*70)
    result = await system.learn("What is the full form of AI?")
    print(f"\nAnswer: {result.answer}")
    print(f"Method: {result.method}")
    print(f"Success: {result.success}")

    # Stats
    print("\n" + "="*70)
    print("SYSTEM STATS")
    print("="*70)
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_unified_system())
