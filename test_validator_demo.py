#!/usr/bin/env python3
"""
Quick demo of knowledge validation system.
Shows how concepts are validated before storage.
"""

from core.knowledge_validator import KnowledgeValidator, ValidationResult
from core.llm import LLMBridge
from core.web_researcher import WebResearcher
import os

def demo_validation():
    """Demonstrate validation on a sample concept."""
    print("=" * 60)
    print("KV-1 Knowledge Validation Demo")
    print("=" * 60)

    # Initialize components (will use offline mode if Ollama not available)
    llm = LLMBridge(provider="ollama", default_model="qwen3:4b")
    web = WebResearcher(cache_dir="./demo_cache", daily_cap=10)
    validator = KnowledgeValidator(llm, web)

    # Test concepts
    test_cases = [
        {
            "concept": "prime numbers",
            "definition": "Natural numbers greater than 1 that have no positive divisors other than 1 and themselves",
            "examples": [
                "2, 3, 5, 7, 11 are prime numbers",
                "To check if n is prime, test divisibility up to √n"
            ]
        },
        {
            "concept": "photosynthesis",
            "definition": "Process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen",
            "examples": [
                "6CO2 + 6H2O + light energy → C6H12O6 + 6O2"
            ]
        },
        {
            "concept": "made-up-concept-xyz",
            "definition": "This is a fake concept that should fail validation",
            "examples": []
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test {i}: {test['concept']}")
        print(f"{'─' * 60}")
        print(f"Definition: {test['definition'][:80]}...")
        print(f"Examples: {len(test['examples'])} provided")

        # Validate
        print("\n[Validating...]")
        result = validator.validate_concept(
            test['concept'],
            test['definition'],
            test['examples']
        )

        # Show results
        print(f"\n✓ Confidence Score: {result.confidence_score:.2f}")
        print(f"✓ Sources Verified: {result.sources_verified}")
        print(f"✓ Examples Valid: {result.examples_valid}")
        print(f"✓ Self-Test Passed: {result.self_test_passed}")

        # Decision
        should_store = validator.should_store(result, threshold=0.6)
        if should_store:
            print(f"\n✅ APPROVED - Would store in LTM")
        else:
            print(f"\n❌ REJECTED - Low confidence, need better sources")

        print(f"\nDetails:\n{result.details}")

    print(f"\n{'=' * 60}")
    print("Demo Complete!")
    print(f"{'=' * 60}")

    # Cleanup
    if os.path.exists("./demo_cache"):
        import shutil
        shutil.rmtree("./demo_cache")

if __name__ == "__main__":
    demo_validation()
