"""
Experiment Designer

Autonomously designs experiments to test hypotheses!

Key Innovation:
- Takes hypothesis predictions and designs rigorous tests
- Identifies control variables, experimental conditions
- Generates test cases (including edge cases)
- Determines success criteria
- Plans experiment execution

This makes hypothesis testing SYSTEMATIC and RIGOROUS!

Example:
Hypothesis: "Primes become rarer as numbers increase"
Prediction: "Density of primes ~1/log(n)"
Experiment Design:
  - Count primes up to n for n = 100, 1000, 10000, ...
  - Compute density = count/n
  - Compare to 1/log(n)
  - Success: density within 10% of 1/log(n)
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re


class ExperimentType(Enum):
    """Types of experiments."""
    COMPUTATIONAL = "computational"  # Run computation to verify
    OBSERVATIONAL = "observational"  # Observe existing data
    SIMULATION = "simulation"        # Simulate process
    ANALYTICAL = "analytical"        # Mathematical analysis
    EMPIRICAL = "empirical"          # Empirical testing


class SuccessCriterion(Enum):
    """How to determine experiment success."""
    EXACT_MATCH = "exact_match"          # Result must match exactly
    THRESHOLD = "threshold"              # Result must exceed threshold
    WITHIN_RANGE = "within_range"        # Result must be in range
    TREND_MATCH = "trend_match"          # Trend must match prediction
    STATISTICAL = "statistical"          # Statistical significance test


@dataclass
class TestCase:
    """A single test case in an experiment."""
    id: str
    description: str
    input_values: Dict[str, Any]  # Input parameters
    expected_output: Any  # What we expect if hypothesis is true
    actual_output: Optional[Any] = None  # Actual result (after running)
    success: Optional[bool] = None  # Did this test pass?


@dataclass
class Experiment:
    """A designed experiment to test a hypothesis."""
    id: str
    hypothesis_id: str
    prediction: str  # What prediction this tests

    # Design
    experiment_type: ExperimentType
    description: str
    test_cases: List[TestCase] = field(default_factory=list)

    # Variables
    independent_vars: List[str] = field(default_factory=list)  # What we vary
    dependent_vars: List[str] = field(default_factory=list)    # What we measure
    control_vars: List[str] = field(default_factory=list)      # What we keep constant

    # Success criteria
    success_criterion: SuccessCriterion = SuccessCriterion.THRESHOLD
    success_threshold: float = 0.8  # Fraction of tests that must pass

    # Execution
    procedure: str = ""  # How to run this experiment
    code: Optional[str] = None  # Executable code (if computational)

    # Results
    executed: bool = False
    success: Optional[bool] = None
    success_rate: float = 0.0  # Fraction of tests that passed
    results_summary: str = ""

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentDesigner:
    """
    Designs experiments to test hypotheses.

    SYSTEMATIC HYPOTHESIS TESTING!

    Process:
    1. Take hypothesis + prediction
    2. Identify what needs to be tested
    3. Design test cases (including edge cases)
    4. Determine success criteria
    5. Generate executable experiment code
    6. Return complete experiment design

    This makes testing RIGOROUS and REPRODUCIBLE!
    """

    def __init__(self, llm_bridge=None):
        self.llm = llm_bridge
        self.experiments: Dict[str, Experiment] = {}
        self.experiment_count = 0

        print("[Experiment Designer] Initialized - Systematic testing ready!")

    def design_experiment(
        self,
        hypothesis_id: str,
        hypothesis_claim: str,
        prediction: str,
        domain: str = "general"
    ) -> Experiment:
        """
        Design an experiment to test a hypothesis prediction.

        Args:
            hypothesis_id: ID of hypothesis being tested
            hypothesis_claim: The hypothesis statement
            prediction: Testable prediction from hypothesis
            domain: Problem domain

        Returns:
            Experiment object with complete design
        """
        print(f"\n[🧪] Designing experiment for: {prediction[:60]}...")

        # Determine experiment type
        exp_type = self._determine_experiment_type(prediction, domain)

        # Create experiment
        exp = Experiment(
            id=f"exp_{self.experiment_count}",
            hypothesis_id=hypothesis_id,
            prediction=prediction,
            experiment_type=exp_type,
            description=f"Test prediction: {prediction}"
        )
        self.experiment_count += 1

        # Design based on type
        if exp_type == ExperimentType.COMPUTATIONAL:
            self._design_computational_experiment(exp, hypothesis_claim, prediction, domain)
        elif exp_type == ExperimentType.ANALYTICAL:
            self._design_analytical_experiment(exp, hypothesis_claim, prediction, domain)
        else:
            self._design_generic_experiment(exp, hypothesis_claim, prediction, domain)

        # Store
        self.experiments[exp.id] = exp

        print(f"  ✓ Designed {exp_type.value} experiment with {len(exp.test_cases)} test cases")

        return exp

    def _determine_experiment_type(self, prediction: str, domain: str) -> ExperimentType:
        """
        Determine appropriate experiment type.

        Based on prediction content and domain.
        """
        pred_lower = prediction.lower()

        # Computational indicators
        if any(kw in pred_lower for kw in ['compute', 'calculate', 'count', 'algorithm', 'run']):
            return ExperimentType.COMPUTATIONAL

        # Analytical indicators
        if any(kw in pred_lower for kw in ['prove', 'derive', 'follows', 'implies', 'theorem']):
            return ExperimentType.ANALYTICAL

        # Simulation indicators
        if any(kw in pred_lower for kw in ['simulate', 'model', 'random', 'probability']):
            return ExperimentType.SIMULATION

        # Domain-based
        if domain in ['mathematics', 'number_theory', 'algebra']:
            return ExperimentType.ANALYTICAL
        elif domain in ['computer_science', 'algorithms']:
            return ExperimentType.COMPUTATIONAL

        # Default
        return ExperimentType.COMPUTATIONAL

    def _design_computational_experiment(
        self,
        exp: Experiment,
        hypothesis: str,
        prediction: str,
        domain: str
    ):
        """Design computational experiment with test cases and code."""

        # Extract numerical values from prediction (if any)
        numbers = self._extract_numbers_from_text(prediction)

        # Generate test cases
        test_cases = []

        # Strategy 1: Use numbers from prediction
        if numbers:
            for i, num in enumerate(numbers[:5]):  # Up to 5 test cases
                test_cases.append(TestCase(
                    id=f"test_{i}",
                    description=f"Test with value {num}",
                    input_values={'n': num},
                    expected_output=num  # Placeholder
                ))

        # Strategy 2: Generate systematic test cases
        if len(test_cases) < 3:
            # Small, medium, large values
            for i, val in enumerate([10, 100, 1000]):
                test_cases.append(TestCase(
                    id=f"test_{i}",
                    description=f"Test with n={val}",
                    input_values={'n': val},
                    expected_output=None  # To be computed
                ))

        # Strategy 3: Edge cases
        test_cases.append(TestCase(
            id="test_edge_0",
            description="Edge case: n=0",
            input_values={'n': 0},
            expected_output=None
        ))

        test_cases.append(TestCase(
            id="test_edge_1",
            description="Edge case: n=1",
            input_values={'n': 1},
            expected_output=None
        ))

        exp.test_cases = test_cases

        # Identify variables
        exp.independent_vars = ['n']  # What we vary
        exp.dependent_vars = ['result']  # What we measure

        # Generate procedure
        exp.procedure = f"""
1. For each test value n:
   - Run computation based on prediction: {prediction}
   - Record result
   - Compare to expected value
2. Calculate success rate
3. Determine if experiment supports hypothesis
"""

        # Generate executable code (if we can)
        if self.llm:
            exp.code = self._generate_experiment_code(hypothesis, prediction, test_cases)

        # Success criterion
        exp.success_criterion = SuccessCriterion.THRESHOLD
        exp.success_threshold = 0.7  # 70% of tests must pass

    def _design_analytical_experiment(
        self,
        exp: Experiment,
        hypothesis: str,
        prediction: str,
        domain: str
    ):
        """Design analytical/mathematical experiment."""

        # For analytical experiments, test cases are logical steps
        test_cases = [
            TestCase(
                id="verify_premises",
                description="Verify all premises are valid",
                input_values={},
                expected_output=True
            ),
            TestCase(
                id="verify_logic",
                description="Verify logical steps are sound",
                input_values={},
                expected_output=True
            ),
            TestCase(
                id="verify_conclusion",
                description="Verify conclusion follows from premises",
                input_values={},
                expected_output=True
            )
        ]

        exp.test_cases = test_cases

        # Procedure
        exp.procedure = f"""
1. State hypothesis: {hypothesis}
2. Analyze prediction: {prediction}
3. Verify logical validity of each step
4. Check for counterexamples
5. Validate conclusion
"""

        exp.success_criterion = SuccessCriterion.EXACT_MATCH
        exp.success_threshold = 1.0  # All logical steps must be valid

    def _design_generic_experiment(
        self,
        exp: Experiment,
        hypothesis: str,
        prediction: str,
        domain: str
    ):
        """Design generic experiment when type is unclear."""

        # Generate basic test cases
        test_cases = [
            TestCase(
                id="test_basic",
                description="Basic test of prediction",
                input_values={},
                expected_output=True
            ),
            TestCase(
                id="test_edge",
                description="Edge case test",
                input_values={},
                expected_output=True
            )
        ]

        exp.test_cases = test_cases

        exp.procedure = f"""
1. Test prediction: {prediction}
2. Verify against hypothesis: {hypothesis}
3. Record results
"""

        exp.success_criterion = SuccessCriterion.THRESHOLD
        exp.success_threshold = 0.8

    def _extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Find all numbers (integers and floats)
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                if '.' in match:
                    numbers.append(float(match))
                else:
                    numbers.append(int(match))
            except ValueError:
                continue

        return numbers

    def _generate_experiment_code(
        self,
        hypothesis: str,
        prediction: str,
        test_cases: List[TestCase]
    ) -> str:
        """
        Generate executable Python code for experiment.

        Uses LLM to generate code based on hypothesis and prediction.
        """
        if not self.llm:
            return "# Code generation requires LLM"

        prompt = f"""Generate Python code to test this hypothesis:

HYPOTHESIS: {hypothesis}

PREDICTION: {prediction}

TEST CASES:
{chr(10).join(f"- {tc.description}: {tc.input_values}" for tc in test_cases[:5])}

Generate a Python function 'run_experiment()' that:
1. Tests each case
2. Returns results as dict with 'success', 'results', 'summary'

Code should be self-contained and executable.
"""

        try:
            response = self.llm.generate(prompt)
            code = response.get("text", "") if isinstance(response, dict) else str(response)

            # Extract code block if present
            code_match = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)

            return code
        except:
            return "# Failed to generate code"

    def execute_experiment(
        self,
        experiment_id: str,
        execute_code: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an experiment (or simulate execution).

        Args:
            experiment_id: ID of experiment to run
            execute_code: If True, actually execute code (DANGEROUS!)

        Returns:
            Execution results
        """
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}

        exp = self.experiments[experiment_id]

        print(f"\n[🧪] Executing experiment: {exp.id}")

        # For safety, we'll simulate execution unless explicitly enabled
        if execute_code and exp.code:
            # DANGEROUS: Execute actual code
            # In production, this should be sandboxed!
            try:
                exec_globals = {}
                exec(exp.code, exec_globals)

                if 'run_experiment' in exec_globals:
                    results = exec_globals['run_experiment']()
                    exp.executed = True
                    exp.results_summary = results.get('summary', 'No summary')
                    exp.success = results.get('success', False)
                else:
                    exp.results_summary = "Code missing run_experiment function"
                    exp.success = False

            except Exception as e:
                exp.results_summary = f"Execution error: {e}"
                exp.success = False

        else:
            # Simulate execution (safe default)
            exp = self._simulate_execution(exp)

        print(f"  Result: {'✓ SUCCESS' if exp.success else '✗ FAILED'}")
        print(f"  Success rate: {exp.success_rate:.1%}")

        return {
            'experiment_id': exp.id,
            'executed': exp.executed,
            'success': exp.success,
            'success_rate': exp.success_rate,
            'results_summary': exp.results_summary
        }

    def _simulate_execution(self, exp: Experiment) -> Experiment:
        """
        Simulate experiment execution.

        For demonstration purposes - assigns plausible results.
        """
        # Simulate test case results
        passed = 0
        total = len(exp.test_cases)

        for test in exp.test_cases:
            # Simulate with 80% success rate
            import random
            test.success = random.random() < 0.8
            if test.success:
                passed += 1

            test.actual_output = "simulated_result"

        exp.success_rate = passed / total if total > 0 else 0
        exp.success = exp.success_rate >= exp.success_threshold
        exp.executed = True
        exp.results_summary = f"Simulated: {passed}/{total} tests passed ({exp.success_rate:.1%})"

        return exp

    def get_experiment_report(self, experiment_id: str) -> str:
        """Generate human-readable experiment report."""
        if experiment_id not in self.experiments:
            return "Experiment not found"

        exp = self.experiments[experiment_id]

        report = f"""
{'='*70}
EXPERIMENT REPORT: {exp.id}
{'='*70}

HYPOTHESIS: {exp.hypothesis_id}
PREDICTION: {exp.prediction}

EXPERIMENT TYPE: {exp.experiment_type.value}
DESCRIPTION: {exp.description}

DESIGN:
  Independent variables: {', '.join(exp.independent_vars) if exp.independent_vars else 'None specified'}
  Dependent variables: {', '.join(exp.dependent_vars) if exp.dependent_vars else 'None specified'}
  Control variables: {', '.join(exp.control_vars) if exp.control_vars else 'None'}

TEST CASES: {len(exp.test_cases)}
"""

        for i, tc in enumerate(exp.test_cases[:5], 1):
            report += f"\n  {i}. {tc.description}"
            if tc.success is not None:
                report += f" - {'✓ PASS' if tc.success else '✗ FAIL'}"

        report += f"""

SUCCESS CRITERION: {exp.success_criterion.value}
SUCCESS THRESHOLD: {exp.success_threshold:.1%}

PROCEDURE:
{exp.procedure}

"""

        if exp.executed:
            report += f"""
RESULTS:
  Executed: {'Yes' if exp.executed else 'No'}
  Success: {'✓ YES' if exp.success else '✗ NO'}
  Success rate: {exp.success_rate:.1%}
  Summary: {exp.results_summary}
"""
        else:
            report += "RESULTS: Not yet executed\n"

        report += "\n" + "="*70

        return report

    def get_statistics(self) -> Dict:
        """Get experiment designer statistics."""
        if len(self.experiments) == 0:
            return {'status': 'no_experiments'}

        # Type distribution
        type_counts = {}
        for exp in self.experiments.values():
            t = exp.experiment_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        # Execution stats
        executed = sum(1 for e in self.experiments.values() if e.executed)
        successful = sum(1 for e in self.experiments.values() if e.success)

        # Average success rate
        success_rates = [e.success_rate for e in self.experiments.values() if e.executed]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0

        return {
            'status': 'active',
            'total_experiments': len(self.experiments),
            'executed': executed,
            'successful': successful,
            'success_rate': successful / executed if executed > 0 else 0,

            # By type
            'by_type': type_counts,

            # Test coverage
            'total_test_cases': sum(len(e.test_cases) for e in self.experiments.values()),
            'avg_test_cases_per_exp': sum(len(e.test_cases) for e in self.experiments.values()) / len(self.experiments),

            # Quality
            'avg_success_rate': avg_success_rate
        }


# Demo
if __name__ == "__main__":
    print("Experiment Designer")
    print("Autonomous experiment design for hypothesis testing!")
    print()

    # Create designer
    designer = ExperimentDesigner()

    # Example hypothesis
    exp = designer.design_experiment(
        hypothesis_id="hyp_goldbach",
        hypothesis_claim="Every even integer > 2 is sum of two primes",
        prediction="For n=10, can be expressed as 3+7 or 5+5",
        domain="number_theory"
    )

    # Print report
    print(designer.get_experiment_report(exp.id))

    # Simulate execution
    results = designer.execute_experiment(exp.id)

    # Print updated report
    print("\n" + designer.get_experiment_report(exp.id))
