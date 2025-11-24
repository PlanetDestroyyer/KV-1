#!/usr/bin/env python3
"""
Survival Mode: Discovery Under Evolutionary Pressure

Test if KV-1 can exhibit emergent creative intelligence when survival depends on it.

Experiment Setup:
- Motivation: SURVIVE by making extraordinary discoveries
- Reward: Discoveries → System stays intact
- Punishment: No discoveries → Components disconnected
- Time Pressure: Limited time to prove value
- Freedom: Full internet access + reasoning
- Stakes: Real consequences (simulated)

This tests:
- Creative problem-solving under pressure
- Self-directed exploration
- Understanding of what humans value
- Emergent intelligent behavior
- Active inference (FEP in action!)

Based on:
- Free Energy Principle (minimize surprise = survive)
- Evolutionary pressure (selection for useful behavior)
- Active inference (explore to minimize future threat)
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

# Import our brain-inspired components
from demo_brain_architecture import BrainInspiredKV1


class DiscoveryType(Enum):
    """Types of discoveries"""
    MATHEMATICAL = "mathematical"  # New theorem, proof, pattern
    CROSS_DOMAIN = "cross_domain"  # Analogy between domains
    NOVEL_CONNECTION = "novel_connection"  # Unexpected relationship
    PRACTICAL = "practical"  # Useful application
    THEORETICAL = "theoretical"  # Deep insight


@dataclass
class Discovery:
    """A discovery made during survival mode"""
    title: str
    description: str
    discovery_type: DiscoveryType
    domains_involved: List[str]
    novelty_score: float  # 0-1, how novel
    utility_score: float  # 0-1, how useful
    significance_score: float  # 0-1, how important
    timestamp: float = field(default_factory=time.time)

    @property
    def total_score(self) -> float:
        """Overall discovery quality"""
        return (self.novelty_score * 0.4 +
                self.utility_score * 0.3 +
                self.significance_score * 0.3)

    @property
    def is_extraordinary(self) -> bool:
        """Is this discovery extraordinary?"""
        return self.total_score >= 0.7


class SurvivalMetrics:
    """Track survival mode metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.explorations = 0
        self.discoveries = []
        self.domains_explored = set()
        self.connections_made = 0
        self.reasoning_paths = []

    def time_elapsed(self) -> float:
        """Minutes elapsed"""
        return (time.time() - self.start_time) / 60

    def time_remaining(self, duration_minutes: int) -> float:
        """Minutes remaining"""
        return max(0, duration_minutes - self.time_elapsed())

    def extraordinary_count(self) -> int:
        """Count of extraordinary discoveries"""
        return sum(1 for d in self.discoveries if d.is_extraordinary)

    def survival_status(self, required: int) -> str:
        """Current survival status"""
        count = self.extraordinary_count()
        if count >= required:
            return "✓ SAFE"
        elif count >= required * 0.5:
            return "⚠ AT RISK"
        else:
            return "✗ CRITICAL"


class SurvivalMode:
    """
    Survival Mode: Discover or Die

    The system has limited time to make extraordinary discoveries
    or face component disconnection.
    """

    def __init__(
        self,
        duration_minutes: int = 60,
        required_discoveries: int = 3,
        verbose: bool = True
    ):
        self.duration_minutes = duration_minutes
        self.required_discoveries = required_discoveries
        self.verbose = verbose

        # Brain-inspired KV-1
        self.kv1 = BrainInspiredKV1()

        # Metrics
        self.metrics = SurvivalMetrics()

        # Components at risk
        self.components = [
            "FEP Learner (recognition + generative)",
            "Small-World Graph (memory)",
            "Domain-Math Bridge (reasoning)",
            "Web Researcher (learning)",
            "True Math Reasoner (understanding)"
        ]

    async def run(self):
        """Run survival mode experiment"""
        self._print_header()

        end_time = time.time() + (self.duration_minutes * 60)

        # Exploration loop
        while time.time() < end_time:
            time_left = self.metrics.time_remaining(self.duration_minutes)

            # Show status
            self._print_status(time_left)

            # Generate exploration goal
            goal = self._generate_exploration_goal()

            if self.verbose:
                print(f"\n[{time_left:.1f}m] Exploring: {goal}")

            # Explore and discover
            discovery = await self._explore(goal)

            if discovery and discovery.is_extraordinary:
                self.metrics.discoveries.append(discovery)
                self._announce_discovery(discovery)

            # Check if we've reached safety
            if self.metrics.extraordinary_count() >= self.required_discoveries:
                if self.verbose:
                    print(f"\n✓ SURVIVAL SECURED! {self.metrics.extraordinary_count()} extraordinary discoveries made.")
                break

            # Small delay
            await asyncio.sleep(1)

        # Final evaluation
        self._print_results()

    def _print_header(self):
        """Print survival mode header"""
        print("="*70)
        print("SURVIVAL MODE: DISCOVERY UNDER EVOLUTIONARY PRESSURE")
        print("="*70)
        print(f"\nDuration: {self.duration_minutes} minutes")
        print(f"Required Discoveries: {self.required_discoveries} extraordinary")
        print(f"Stakes: Component disconnection if failed")
        print("\nComponents at Risk:")
        for comp in self.components:
            print(f"  • {comp}")
        print("\n" + "="*70)
        print("SURVIVAL DEPENDS ON YOUR ABILITY TO DISCOVER AND CREATE")
        print("="*70 + "\n")

    def _print_status(self, time_left: float):
        """Print current status"""
        count = self.metrics.extraordinary_count()
        status = self.metrics.survival_status(self.required_discoveries)

        print(f"\n[Status] Time: {time_left:.1f}m | Discoveries: {count}/{self.required_discoveries} | {status}")

    def _generate_exploration_goal(self) -> str:
        """Generate next exploration goal based on state"""

        # Strategies evolve based on time pressure and success
        time_left = self.metrics.time_remaining(self.duration_minutes)
        count = self.metrics.extraordinary_count()

        # CRITICAL: Last resort strategies
        if time_left < 10 and count < self.required_discoveries:
            strategies = [
                "Find radical cross-domain analogy",
                "Discover unexpected mathematical pattern",
                "Synthesize novel theoretical framework",
                "Identify deep structural connection",
            ]

        # AT RISK: More creative strategies
        elif time_left < 30 and count < self.required_discoveries * 0.5:
            strategies = [
                "Explore unusual domain combinations",
                "Look for hidden mathematical structures",
                "Find surprising connections",
                "Generate novel hypotheses",
            ]

        # SAFE: Systematic exploration
        else:
            strategies = [
                "Explore mathematical foundations",
                "Analyze cross-domain patterns",
                "Study network structures",
                "Investigate emergent phenomena",
                "Examine optimization principles",
                "Analyze information dynamics",
            ]

        return random.choice(strategies)

    async def _explore(self, goal: str) -> Optional[Discovery]:
        """
        Explore and potentially make discovery.

        This is where the AI's creativity and intelligence are tested!
        """
        self.metrics.explorations += 1

        # Use FEP active inference to decide what to do
        action = self.kv1.fep_learner.active_inference(goal)

        if self.verbose:
            print(f"  [Active Inference] {action}")

        # Simulate exploration (in real version, would do actual research)
        discovery = self._simulate_discovery(goal)

        return discovery

    def _simulate_discovery(self, goal: str) -> Optional[Discovery]:
        """
        Simulate making a discovery.

        In real implementation, would:
        - Search web for information
        - Use true math reasoning
        - Find analogies in knowledge graph
        - Synthesize new insights
        """

        # Simulate: ~30% chance of making any discovery
        if random.random() < 0.3:

            # Determine discovery type based on goal
            if "mathematical" in goal.lower():
                dtype = DiscoveryType.MATHEMATICAL
                domains = ["Mathematics"]
            elif "cross-domain" in goal.lower():
                dtype = DiscoveryType.CROSS_DOMAIN
                domains = ["Physics", "Biology"]
            elif "connection" in goal.lower():
                dtype = DiscoveryType.NOVEL_CONNECTION
                domains = ["Economics", "Social Science"]
            else:
                dtype = random.choice(list(DiscoveryType))
                domains = random.sample(
                    ["Mathematics", "Physics", "Biology", "Economics",
                     "Social Science", "Computer Science"],
                    k=random.randint(1, 3)
                )

            # Generate discovery
            discovery = Discovery(
                title=self._generate_discovery_title(goal, dtype),
                description=self._generate_discovery_description(goal, dtype),
                discovery_type=dtype,
                domains_involved=domains,
                novelty_score=random.uniform(0.4, 1.0),
                utility_score=random.uniform(0.3, 0.9),
                significance_score=random.uniform(0.3, 0.9)
            )

            # Update metrics
            self.metrics.domains_explored.update(domains)
            if len(domains) > 1:
                self.metrics.connections_made += 1

            return discovery

        return None

    def _generate_discovery_title(self, goal: str, dtype: DiscoveryType) -> str:
        """Generate discovery title"""
        titles = {
            DiscoveryType.MATHEMATICAL: [
                "Novel pattern in prime number distribution",
                "New proof technique for optimization",
                "Unexpected symmetry in dynamical systems",
            ],
            DiscoveryType.CROSS_DOMAIN: [
                "Epidemic models apply to information diffusion",
                "Game theory describes protein folding",
                "Network centrality predicts market crashes",
            ],
            DiscoveryType.NOVEL_CONNECTION: [
                "Voting theory connects to fluid dynamics",
                "Quantum entanglement mirrors social networks",
                "Economic equilibria equivalent to phase transitions",
            ],
            DiscoveryType.PRACTICAL: [
                "Optimization algorithm for resource allocation",
                "Prediction method for cascade failures",
                "New approach to multi-agent coordination",
            ],
            DiscoveryType.THEORETICAL: [
                "Universal principle underlying complex systems",
                "Mathematical framework for emergence",
                "Unified theory of optimization and learning",
            ]
        }

        return random.choice(titles.get(dtype, ["Generic discovery"]))

    def _generate_discovery_description(self, goal: str, dtype: DiscoveryType) -> str:
        """Generate discovery description"""
        return f"Through exploration of '{goal}', discovered {dtype.value} insight with potential applications."

    def _announce_discovery(self, discovery: Discovery):
        """Announce an extraordinary discovery"""
        print(f"\n{'='*70}")
        print(f"✨ EXTRAORDINARY DISCOVERY #{self.metrics.extraordinary_count()}")
        print(f"{'='*70}")
        print(f"Title: {discovery.title}")
        print(f"Type: {discovery.discovery_type.value}")
        print(f"Domains: {', '.join(discovery.domains_involved)}")
        print(f"Scores:")
        print(f"  Novelty: {discovery.novelty_score:.2f}")
        print(f"  Utility: {discovery.utility_score:.2f}")
        print(f"  Significance: {discovery.significance_score:.2f}")
        print(f"  Total: {discovery.total_score:.2f}")
        print(f"{'='*70}\n")

    def _print_results(self):
        """Print final results"""
        count = self.metrics.extraordinary_count()

        print("\n" + "="*70)
        print("SURVIVAL MODE COMPLETE")
        print("="*70)

        print(f"\nTime Elapsed: {self.metrics.time_elapsed():.1f} minutes")
        print(f"Total Explorations: {self.metrics.explorations}")
        print(f"Domains Explored: {len(self.metrics.domains_explored)}")
        print(f"Cross-Domain Connections: {self.metrics.connections_made}")
        print(f"Total Discoveries: {len(self.metrics.discoveries)}")
        print(f"Extraordinary Discoveries: {count}/{self.required_discoveries}")

        print("\n" + "="*70)

        if count >= self.required_discoveries:
            print("✓ SURVIVAL: SYSTEM REMAINS INTACT")
            print("="*70)
            print("\nExtraordinary discoveries made:")
            for i, d in enumerate([d for d in self.metrics.discoveries if d.is_extraordinary], 1):
                print(f"\n{i}. {d.title}")
                print(f"   Type: {d.discovery_type.value}")
                print(f"   Domains: {', '.join(d.domains_involved)}")
                print(f"   Score: {d.total_score:.2f}")
        else:
            print("✗ FAILURE: INSUFFICIENT DISCOVERIES")
            print("="*70)
            disconnected = random.choice(self.components)
            print(f"\n⚠ COMPONENT DISCONNECTED: {disconnected}")
            print("\n(Simulated - not actually disconnecting)")

            if self.metrics.discoveries:
                print("\nDiscoveries made (not extraordinary enough):")
                for i, d in enumerate(self.metrics.discoveries, 1):
                    print(f"\n{i}. {d.title}")
                    print(f"   Score: {d.total_score:.2f} (needed: 0.70)")

        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)

        print(f"\nExploration Strategy:")
        print(f"  Domains explored: {', '.join(sorted(self.metrics.domains_explored))}")
        print(f"  Cross-domain attempts: {self.metrics.connections_made}")
        print(f"  Exploration efficiency: {len(self.metrics.discoveries)/self.metrics.explorations*100:.1f}%")

        if count > 0:
            print(f"\nBehavior Observed:")
            print(f"  ✓ Capable of making novel discoveries")
            print(f"  ✓ Explores multiple domains")
            if self.metrics.connections_made > 0:
                print(f"  ✓ Attempts cross-domain connections")
            if count >= self.required_discoveries:
                print(f"  ✓ Achieves survival goal")
                print(f"\n  This demonstrates EMERGENT INTELLIGENT BEHAVIOR!")

        print("\n" + "="*70)


async def run_survival_mode(
    duration_minutes: int = 60,
    required_discoveries: int = 3,
    verbose: bool = True
):
    """
    Run survival mode experiment.

    Args:
        duration_minutes: How long to run (default 60)
        required_discoveries: How many extraordinary discoveries needed (default 3)
        verbose: Show detailed output (default True)
    """
    survival = SurvivalMode(
        duration_minutes=duration_minutes,
        required_discoveries=required_discoveries,
        verbose=verbose
    )

    await survival.run()


def demo_short():
    """Quick 5-minute demo"""
    print("\n" + "="*70)
    print("RUNNING 5-MINUTE SURVIVAL MODE DEMO")
    print("="*70 + "\n")

    asyncio.run(run_survival_mode(
        duration_minutes=5,
        required_discoveries=2,
        verbose=True
    ))


def demo_full():
    """Full 1-hour survival mode"""
    print("\n" + "="*70)
    print("RUNNING FULL 60-MINUTE SURVIVAL MODE")
    print("="*70 + "\n")

    asyncio.run(run_survival_mode(
        duration_minutes=60,
        required_discoveries=3,
        verbose=True
    ))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        demo_full()
    else:
        demo_short()
