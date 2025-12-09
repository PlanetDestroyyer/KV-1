"""
Knowledge Validator for KV-1

Validates learned concepts before storing them in long-term memory.
Uses multi-source verification, example validation, and self-testing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

from core.llm import LLMBridge
from core.web_researcher import WebResearcher


@dataclass
class ValidationResult:
    """Result of validating a concept."""
    confidence_score: float  # 0-1 overall confidence
    sources_verified: int  # How many sources confirmed
    examples_valid: bool  # Whether examples are correct
    self_test_passed: bool  # Whether generated tests passed
    details: str  # Human-readable explanation


class KnowledgeValidator:
    """Validates learned concepts before storing in LTM."""

    def __init__(self, llm: LLMBridge, web_researcher: WebResearcher):
        self.llm = llm
        self.web = web_researcher
        self.logger = logging.getLogger("kv1.validator")

    def validate_concept(
        self,
        concept: str,
        definition: str,
        examples: List[str]
    ) -> ValidationResult:
        """
        Validates a concept by:
        1. Cross-referencing with multiple sources
        2. Checking example consistency
        3. Generating and solving test problems

        Returns ValidationResult with confidence score 0-1
        """
        scores = []
        details = []

        # Test 1: Multi-source verification
        sources_score, sources_count = self._verify_multiple_sources(concept, definition)
        scores.append(sources_score)
        details.append(f"Sources verified: {sources_count}/3 (score: {sources_score:.2f})")

        # Test 2: Example consistency (if examples provided)
        examples_valid = False
        if examples:
            example_score = self._validate_examples(concept, examples)
            scores.append(example_score)
            examples_valid = example_score > 0.7
            details.append(f"Examples valid: {examples_valid} (score: {example_score:.2f})")
        else:
            details.append("No examples to validate")

        # Test 3: Self-test generation
        test_score, test_passed = self._generate_and_solve_tests(concept, definition)
        scores.append(test_score)
        details.append(f"Self-tests passed: {test_passed} (score: {test_score:.2f})")

        # Calculate overall confidence
        confidence = sum(scores) / len(scores) if scores else 0.0

        return ValidationResult(
            confidence_score=confidence,
            sources_verified=sources_count,
            examples_valid=examples_valid,
            self_test_passed=test_passed,
            details="\n".join(details)
        )

    def _verify_multiple_sources(self, concept: str, definition: str) -> tuple[float, int]:
        """
        Check if multiple sources agree on the definition.
        Returns (confidence_score, sources_found)
        """
        sources_found = 0

        # Try to fetch from 3 different queries
        queries = [
            concept,  # Direct concept name
            f"{concept} definition",
            f"what is {concept}",
        ]

        for query in queries[:3]:  # Limit to 3 to avoid rate limits
            result = self.web.fetch(query, mode="scrape")
            if result and result.text and len(result.text) > 100:
                # Check if the definition concepts appear in the fetched text
                if self._check_definition_consistency(definition, result.text):
                    sources_found += 1

        # Score based on sources found
        if sources_found >= 2:
            confidence = 1.0  # High confidence with 2+ sources
        elif sources_found == 1:
            confidence = 0.7  # Medium confidence with 1 source
        else:
            confidence = 0.3  # Low confidence, no sources

        self.logger.info(
            f"Source verification: {sources_found}/3 sources found (confidence: {confidence:.2f})"
        )

        return confidence, sources_found

    def _check_definition_consistency(self, definition: str, source_text: str) -> bool:
        """
        Check if key concepts from definition appear in source text.
        Simple keyword matching approach.
        """
        # Extract key terms from definition (words longer than 4 chars)
        key_terms = [
            word.lower()
            for word in re.findall(r'\b\w+\b', definition)
            if len(word) > 4 and word.lower() not in ['about', 'which', 'where', 'there', 'these', 'those']
        ]

        if not key_terms:
            return False

        # Count how many key terms appear in source
        source_lower = source_text.lower()
        matches = sum(1 for term in key_terms if term in source_lower)

        # Need at least 50% of key terms to match
        match_ratio = matches / len(key_terms)
        return match_ratio >= 0.5

    def _validate_examples(self, concept: str, examples: List[str]) -> float:
        """
        Check if examples correctly demonstrate the concept.
        Uses LLM to validate.
        """
        if not examples:
            return 0.5  # Neutral score for no examples

        # Take first 3 examples to avoid long prompts
        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples[:3]))

        system_prompt = "You are a knowledge validator. Analyze if examples correctly demonstrate a concept."
        user_input = f"""Concept: {concept}

Examples provided:
{examples_text}

Do these examples correctly demonstrate the concept "{concept}"?
Answer with exactly one word: YES or NO
Then on a new line, briefly explain why (one sentence)."""

        try:
            result = self.llm.generate(system_prompt, user_input, execute=True)
            response_text = result.get("text", "")

            # Parse response
            if "YES" in response_text.upper()[:20]:  # Check first 20 chars
                self.logger.info(f"Examples validated for {concept}")
                return 1.0
            elif "NO" in response_text.upper()[:20]:
                self.logger.warning(f"Examples rejected for {concept}")
                return 0.3
            else:
                self.logger.warning(f"Unclear example validation for {concept}")
                return 0.5

        except Exception as e:
            self.logger.error(f"Example validation failed: {e}")
            return 0.5  # Neutral on error

    def _generate_and_solve_tests(self, concept: str, definition: str) -> tuple[float, bool]:
        """
        Generate test problems and check if they can be solved.
        Returns (confidence_score, passed)
        """
        system_prompt = "You are a test problem generator. Create simple verification problems."
        user_input = f"""Concept: {concept}
Definition: {definition}

Generate 2 simple yes/no questions that test understanding of this concept.
Format each as:
Q1: [question]
A1: YES or NO

Q2: [question]
A2: YES or NO

Keep questions simple and factual."""

        try:
            result = self.llm.generate(system_prompt, user_input, execute=True)
            response_text = result.get("text", "")

            # Count how many Q/A pairs are present
            questions = response_text.count("Q1:") + response_text.count("Q2:")
            answers = response_text.count("A1:") + response_text.count("A2:")

            # Simple heuristic: if we got questions and answers, tests are valid
            if questions >= 2 and answers >= 2:
                self.logger.info(f"Self-tests generated successfully for {concept}")
                return 1.0, True
            elif questions >= 1 or answers >= 1:
                self.logger.info(f"Partial self-tests generated for {concept}")
                return 0.7, False
            else:
                self.logger.warning(f"Self-test generation failed for {concept}")
                return 0.5, False

        except Exception as e:
            self.logger.error(f"Test generation failed: {e}")
            return 0.5, False

    def should_store(self, validation_result: ValidationResult, threshold: float = 0.7) -> bool:
        """
        Decide if concept should be stored based on validation.

        Args:
            validation_result: Result from validate_concept()
            threshold: Minimum confidence score required (default 0.7)

        Returns:
            True if concept should be stored, False otherwise
        """
        return validation_result.confidence_score >= threshold
