"""
Meta-Cognitive Monitor

SELF-AWARENESS FOR AGI!

Key Innovation: THE SYSTEM KNOWS WHAT IT KNOWS (AND DOESN'T KNOW)!

Meta-Cognition:
- Self-awareness of capabilities
- Confidence calibration
- Know your limitations
- Monitor your own reasoning
- Detect when you're uncertain
- Request help when needed

This is what makes AGI TRUSTWORTHY and RELIABLE!

Without metacognition, overconfidence leads to errors.
With metacognition, AGI knows when to say "I don't know"!
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from collections import defaultdict


class CapabilityLevel(Enum):
    """Levels of capability."""
    NONE = 0.0           # Cannot do this
    NOVICE = 0.3         # Basic ability
    INTERMEDIATE = 0.6   # Competent
    ADVANCED = 0.8       # Highly capable
    EXPERT = 0.95        # Mastery


class ConfidenceLevel(Enum):
    """Confidence levels."""
    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class CapabilityAssessment:
    """Assessment of a specific capability."""
    capability_name: str
    level: CapabilityLevel
    confidence: float = 0.5  # How confident in this assessment
    evidence: List[str] = field(default_factory=list)  # Evidence for this level
    last_tested: Optional[str] = None
    assessed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningTrace:
    """A trace of reasoning with confidence."""
    id: str
    task: str
    reasoning_steps: List[str]

    # Confidence tracking
    step_confidences: List[float]  # Confidence in each step
    overall_confidence: float

    # Outcome
    result: Any
    actual_correctness: Optional[bool] = None  # Was result actually correct?

    # Calibration
    confidence_error: Optional[float] = None  # |confidence - correctness|

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetaCognitiveMonitor:
    """
    Monitors and calibrates system's self-awareness.

    META-COGNITION ENGINE!

    Capabilities:
    1. Self-Assessment - Know your capabilities
    2. Confidence Calibration - Accurate confidence estimates
    3. Uncertainty Detection - Detect when uncertain
    4. Limitation Awareness - Know what you can't do
    5. Request Assistance - Ask for help when needed

    This prevents:
    - Overconfidence errors
    - Unsafe actions in uncertainty
    - Poor decision-making

    This enables:
    - Trustworthy AI
    - Accurate uncertainty quantification
    - Appropriate help-seeking
    """

    def __init__(self):
        # Capability assessments
        self.capabilities: Dict[str, CapabilityAssessment] = {}

        # Reasoning traces (for calibration)
        self.reasoning_traces: List[ReasoningTrace] = []
        self.trace_count = 0

        # Calibration metrics
        self.calibration_error = 0.0  # How well-calibrated are we?

        # Uncertainty thresholds
        self.uncertainty_threshold = 0.6  # Below this = uncertain
        self.request_help_threshold = 0.4  # Below this = request help

        # Domain expertise
        self.domain_expertise: Dict[str, float] = defaultdict(float)

        print("[Meta-Cognitive Monitor] Initialized")
        print("  Self-awareness active!")
        print("  System will monitor its own reasoning")

    def assess_capability(self, capability_name: str, evidence: List[str] = None) -> CapabilityAssessment:
        """
        Assess a specific capability.

        SELF-ASSESSMENT!

        Args:
            capability_name: Name of capability
            evidence: Evidence of capability level

        Returns:
            CapabilityAssessment
        """
        evidence = evidence or []

        # Determine capability level based on evidence
        # In real implementation, would use more sophisticated assessment

        if len(evidence) == 0:
            level = CapabilityLevel.NONE
            confidence = 0.9  # Confident we can't do this (no evidence)

        elif len(evidence) < 3:
            level = CapabilityLevel.NOVICE
            confidence = 0.6

        elif len(evidence) < 10:
            level = CapabilityLevel.INTERMEDIATE
            confidence = 0.7

        elif len(evidence) < 30:
            level = CapabilityLevel.ADVANCED
            confidence = 0.8

        else:
            level = CapabilityLevel.EXPERT
            confidence = 0.9

        assessment = CapabilityAssessment(
            capability_name=capability_name,
            level=level,
            confidence=confidence,
            evidence=evidence[:5]  # Store top 5
        )

        self.capabilities[capability_name] = assessment

        return assessment

    def monitor_reasoning(self, task: str, reasoning_steps: List[str],
                         step_confidences: List[float], result: Any) -> ReasoningTrace:
        """
        Monitor a reasoning process.

        META-COGNITIVE MONITORING!

        This tracks confidence at each step and overall.

        Args:
            task: What task we're reasoning about
            reasoning_steps: Steps of reasoning
            step_confidences: Confidence in each step (0-1)
            result: Final result

        Returns:
            ReasoningTrace
        """
        # Compute overall confidence
        # Weakest link: overall confidence = min of step confidences
        overall_confidence = min(step_confidences) if step_confidences else 0.5

        # Alternatively: average confidence
        # overall_confidence = np.mean(step_confidences) if step_confidences else 0.5

        trace = ReasoningTrace(
            id=f"trace_{self.trace_count}",
            task=task,
            reasoning_steps=reasoning_steps,
            step_confidences=step_confidences,
            overall_confidence=overall_confidence,
            result=result
        )

        self.reasoning_traces.append(trace)
        self.trace_count += 1

        # Check if uncertain
        if overall_confidence < self.uncertainty_threshold:
            print(f"[Meta-Cog] ⚠️  UNCERTAINTY DETECTED in task: {task}")
            print(f"  Confidence: {overall_confidence:.2%}")

            if overall_confidence < self.request_help_threshold:
                print(f"  🆘 REQUESTING ASSISTANCE - confidence too low!")

        return trace

    def update_calibration(self, trace_id: str, actual_correctness: bool):
        """
        Update calibration with actual outcome.

        CALIBRATION LEARNING!

        When we find out if we were actually right, we can calibrate.

        Args:
            trace_id: ID of reasoning trace
            actual_correctness: Was the result actually correct?
        """
        # Find trace
        trace = next((t for t in self.reasoning_traces if t.id == trace_id), None)

        if not trace:
            return

        trace.actual_correctness = actual_correctness

        # Calibration error
        # If confident and wrong: large error
        # If uncertain and wrong: small error
        predicted_correctness = trace.overall_confidence
        actual = 1.0 if actual_correctness else 0.0

        trace.confidence_error = abs(predicted_correctness - actual)

        # Update overall calibration
        errors = [t.confidence_error for t in self.reasoning_traces if t.confidence_error is not None]
        if errors:
            self.calibration_error = np.mean(errors)

        print(f"[Meta-Cog] Calibration updated:")
        print(f"  Predicted confidence: {predicted_correctness:.2%}")
        print(f"  Actual correctness: {actual}")
        print(f"  Error: {trace.confidence_error:.2%}")
        print(f"  Overall calibration error: {self.calibration_error:.2%}")

    def is_confident(self, task: str, required_confidence: float = 0.7) -> bool:
        """
        Check if confident enough to proceed with task.

        CONFIDENCE CHECK!

        Args:
            task: Task to check
            required_confidence: Minimum required confidence

        Returns:
            True if confident enough
        """
        # Check if we have capability for this task type
        # Simple heuristic: check domain expertise

        # Extract domain from task (simple keyword matching)
        domain = "general"
        for keyword in ['math', 'physics', 'biology', 'chemistry']:
            if keyword in task.lower():
                domain = keyword
                break

        expertise = self.domain_expertise.get(domain, 0.5)

        if expertise < required_confidence:
            print(f"[Meta-Cog] ⚠️  Not confident enough in {domain}")
            print(f"  Expertise: {expertise:.2%}, Required: {required_confidence:.2%}")
            return False

        return True

    def should_request_help(self, confidence: float) -> bool:
        """
        Determine if should request help.

        HELP-SEEKING BEHAVIOR!

        This is key for safe AI - knowing when to ask for help!

        Args:
            confidence: Current confidence level

        Returns:
            True if should request help
        """
        return confidence < self.request_help_threshold

    def get_confidence_explanation(self, trace_id: str) -> str:
        """
        Generate explanation of confidence level.

        EXPLAINABLE CONFIDENCE!

        Args:
            trace_id: Reasoning trace ID

        Returns:
            Explanation string
        """
        trace = next((t for t in self.reasoning_traces if t.id == trace_id), None)

        if not trace:
            return "Trace not found"

        explanation = f"Confidence Analysis for: {trace.task}\n\n"

        explanation += "Step-by-step confidence:\n"
        for i, (step, conf) in enumerate(zip(trace.reasoning_steps, trace.step_confidences), 1):
            explanation += f"  {i}. {step[:60]}...\n"
            explanation += f"     Confidence: {conf:.2%}\n"

        explanation += f"\nOverall confidence: {trace.overall_confidence:.2%}\n"

        # Explain why confident/uncertain
        if trace.overall_confidence >= 0.8:
            explanation += "\n✓ HIGH CONFIDENCE - All reasoning steps are solid"
        elif trace.overall_confidence >= 0.6:
            explanation += "\n⚠️  MODERATE CONFIDENCE - Some uncertainty in reasoning"
        else:
            explanation += "\n❌ LOW CONFIDENCE - Significant uncertainty present"

        # Weakest link
        min_conf_idx = trace.step_confidences.index(min(trace.step_confidences))
        explanation += f"\n\nWeakest link: Step {min_conf_idx + 1} ({trace.step_confidences[min_conf_idx]:.2%} confidence)"

        return explanation

    def update_domain_expertise(self, domain: str, success: bool, learning_rate: float = 0.1):
        """
        Update expertise in a domain based on performance.

        LEARNING FROM EXPERIENCE!

        Args:
            domain: Domain name
            success: Was task successful?
            learning_rate: How fast to update
        """
        current = self.domain_expertise.get(domain, 0.5)

        # Update expertise
        target = 1.0 if success else 0.0
        updated = current + learning_rate * (target - current)

        self.domain_expertise[domain] = updated

        print(f"[Meta-Cog] Updated {domain} expertise: {current:.2%} → {updated:.2%}")

    def get_statistics(self) -> Dict:
        """Get meta-cognitive statistics."""
        # Confidence calibration
        well_calibrated = sum(
            1 for t in self.reasoning_traces
            if t.confidence_error is not None and t.confidence_error < 0.2
        )

        total_calibrated = sum(
            1 for t in self.reasoning_traces
            if t.confidence_error is not None
        )

        # Average confidence
        avg_confidence = np.mean([t.overall_confidence for t in self.reasoning_traces]) if self.reasoning_traces else 0

        # Capability levels
        capability_distribution = defaultdict(int)
        for cap in self.capabilities.values():
            capability_distribution[cap.level.name] += 1

        return {
            'status': 'active',
            'total_capabilities': len(self.capabilities),
            'capability_distribution': dict(capability_distribution),
            'total_reasoning_traces': len(self.reasoning_traces),
            'calibration_error': self.calibration_error,
            'well_calibrated_ratio': well_calibrated / total_calibrated if total_calibrated > 0 else 0,
            'avg_confidence': avg_confidence,
            'domain_expertise': dict(self.domain_expertise)
        }

    def demonstrate_metacognition(self):
        """Demonstrate meta-cognitive monitoring."""
        print("\n" + "="*70)
        print("META-COGNITIVE MONITOR - Demonstration")
        print("="*70)

        stats = self.get_statistics()

        print(f"\n🧠 SELF-AWARENESS:")
        print(f"  Tracked capabilities: {stats['total_capabilities']}")
        print(f"  Reasoning traces: {stats['total_reasoning_traces']}")

        if stats['capability_distribution']:
            print(f"\n📊 CAPABILITY DISTRIBUTION:")
            for level, count in stats['capability_distribution'].items():
                print(f"  {level}: {count}")

        print(f"\n🎯 CALIBRATION:")
        print(f"  Calibration error: {stats['calibration_error']:.2%}")
        print(f"  Well-calibrated: {stats['well_calibrated_ratio']:.1%}")
        print(f"  Average confidence: {stats['avg_confidence']:.2%}")

        if stats['domain_expertise']:
            print(f"\n🔬 DOMAIN EXPERTISE:")
            for domain, expertise in stats['domain_expertise'].items():
                print(f"  {domain}: {expertise:.1%}")

        print(f"\n💡 META-COGNITIVE CAPABILITIES:")
        print("  ✓ Self-assessment of capabilities")
        print("  ✓ Confidence calibration")
        print("  ✓ Uncertainty detection")
        print("  ✓ Help-seeking when uncertain")
        print("  ✓ Confidence explanation")
        print("  ✓ Domain expertise tracking")

        print("\n🎯 KEY INSIGHT:")
        print("  System KNOWS WHAT IT KNOWS and KNOWS WHAT IT DOESN'T KNOW")
        print("  This makes AGI TRUSTWORTHY and RELIABLE!")

        print("\n" + "="*70)


# Demo
if __name__ == "__main__":
    print("Meta-Cognitive Monitor")
    print("Self-awareness and confidence calibration for AGI!")
    print()

    # Create monitor
    monitor = MetaCognitiveMonitor()

    # Assess some capabilities
    monitor.assess_capability("prime_number_theory", evidence=["proven PNT", "analyzed gaps", "verified conjectures"])
    monitor.assess_capability("quantum_mechanics", evidence=["basic understanding"])

    # Monitor reasoning
    trace = monitor.monitor_reasoning(
        task="Prove Goldbach conjecture",
        reasoning_steps=[
            "Consider even number n > 2",
            "Search for prime pairs summing to n",
            "Use sieve methods to find primes",
            "Verify for all cases up to 10^18"
        ],
        step_confidences=[0.9, 0.8, 0.9, 0.7],
        result="Verified computationally up to 10^18"
    )

    # Demonstrate
    monitor.demonstrate_metacognition()
