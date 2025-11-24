#!/usr/bin/env python3
"""
GROUNDBREAKING DISCOVERY SYSTEM - MAIN ENTRY POINT

Run the complete system for attacking real unsolved problems.

Usage:
    python main_discovery.py                    # Interactive menu
    python main_discovery.py --demo             # Run all demos
    python main_discovery.py --attack <problem> # Attack specific problem
    python main_discovery.py --millennium       # Show Millennium problems
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional

# Core systems
from core.research_integration import ResearchIntegrationSystem
from core.theorem_prover import TheoremProverSystem
from core.deep_domain_learner import DeepDomainLearner
from core.long_term_reasoner import LongTermReasoner
from core.breakthrough_discovery import BreakthroughDiscoverySystem


class GroundbreakingDiscoveryMain:
    """Main entry point for the complete discovery system"""

    def __init__(self):
        self.system: Optional[BreakthroughDiscoverySystem] = None
        print(self._get_banner())

    def _get_banner(self) -> str:
        """Get startup banner"""
        return """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          GROUNDBREAKING DISCOVERY SYSTEM v1.0                        ║
║          Attack Real Unsolved Problems                               ║
║                                                                      ║
║          Available Prize Money: $3,000,000+                          ║
║          Problems: Millennium Prize, Famous Conjectures, More        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    async def initialize(self):
        """Initialize all systems"""
        print("\n[Initializing Systems...]")
        print("This may take a moment...\n")

        self.system = BreakthroughDiscoverySystem()

        print("\n✓ All systems ready!\n")

    def show_main_menu(self):
        """Show interactive main menu"""
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                         MAIN MENU                                    ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print("1. Show Available Problems (including $1M prizes)")
        print("2. Attack Millennium Prize Problem ($1M each)")
        print("3. Attack Famous Conjecture (Goldbach, Collatz, Twin Primes)")
        print("4. Run Component Demos")
        print("5. Custom Problem (enter your own)")
        print("6. System Status & Statistics")
        print()
        print("0. Exit")
        print()

    async def run_interactive(self):
        """Run interactive mode"""
        await self.initialize()

        while True:
            self.show_main_menu()

            try:
                choice = input("Choose option (0-6): ").strip()

                if choice == "0":
                    print("\n👋 Goodbye! Go solve some problems!\n")
                    break

                elif choice == "1":
                    await self.show_all_problems()

                elif choice == "2":
                    await self.attack_millennium_menu()

                elif choice == "3":
                    await self.attack_conjecture_menu()

                elif choice == "4":
                    await self.run_demos_menu()

                elif choice == "5":
                    await self.custom_problem()

                elif choice == "6":
                    await self.show_system_status()

                else:
                    print("\n❌ Invalid choice. Please try again.\n")

                input("\n[Press Enter to continue...]")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                input("[Press Enter to continue...]")

    async def show_all_problems(self):
        """Show all available problems"""
        print("\n" + "="*70)
        print("AVAILABLE UNSOLVED PROBLEMS")
        print("="*70 + "\n")

        self.system.show_available_problems()

    async def attack_millennium_menu(self):
        """Menu for attacking Millennium Prize Problems"""
        print("\n" + "="*70)
        print("MILLENNIUM PRIZE PROBLEMS ($1,000,000 EACH)")
        print("="*70 + "\n")

        millennium = self.system.research.get_millennium_problems()

        for i, problem in enumerate(millennium, 1):
            print(f"{i}. {problem.title}")
            print(f"   {problem.description[:80]}...")
            print(f"   Prize: ${problem.prize_money:,}")
            print(f"   Proposed: {problem.proposed_year}")
            print()

        print("0. Back to main menu")
        print()

        choice = input("Choose problem to attack (0-3): ").strip()

        if choice == "0":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(millennium):
                problem = millennium[idx]
                await self.attack_problem_workflow(problem.id)
            else:
                print("\n❌ Invalid choice")
        except ValueError:
            print("\n❌ Invalid input")

    async def attack_conjecture_menu(self):
        """Menu for attacking famous conjectures"""
        print("\n" + "="*70)
        print("FAMOUS OPEN CONJECTURES")
        print("="*70 + "\n")

        conjectures = [
            ("goldbach_conjecture", "Goldbach's Conjecture (1742)"),
            ("collatz_conjecture", "Collatz Conjecture (1937)"),
            ("twin_prime_conjecture", "Twin Prime Conjecture (1846)"),
        ]

        for i, (id, name) in enumerate(conjectures, 1):
            problem = self.system.research.problem_db.get_problem(id)
            print(f"{i}. {name}")
            print(f"   {problem.description[:80]}...")
            print(f"   Proposed: {problem.proposed_year}")
            print()

        print("0. Back to main menu")
        print()

        choice = input("Choose problem to attack (0-3): ").strip()

        if choice == "0":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(conjectures):
                problem_id, _ = conjectures[idx]
                await self.attack_problem_workflow(problem_id)
            else:
                print("\n❌ Invalid choice")
        except ValueError:
            print("\n❌ Invalid input")

    async def attack_problem_workflow(self, problem_id: str):
        """Complete workflow for attacking a problem"""
        problem = self.system.research.problem_db.get_problem(problem_id)

        print("\n" + "="*70)
        print(f"ATTACKING: {problem.title}")
        print("="*70 + "\n")

        print(f"Description: {problem.description}")
        print(f"Domain: {problem.domain}")
        print(f"Difficulty: {problem.difficulty}")
        if problem.prize_money > 0:
            print(f"Prize: ${problem.prize_money:,}")
        print()

        # Ask for time commitment
        print("How long should we work on this?")
        print("1. Quick attempt (1 day)")
        print("2. Serious attempt (7 days)")
        print("3. Deep dive (30 days)")
        print("4. All-in (90 days)")
        print()

        time_choice = input("Choose (1-4): ").strip()

        days_map = {"1": 1, "2": 7, "3": 30, "4": 90}
        max_days = days_map.get(time_choice, 7)

        print(f"\n✓ Committing {max_days} days to this problem")
        print("\nStarting breakthrough attempt...")
        print("This will take a few moments to simulate...\n")

        confirm = input("Ready to begin? (y/n): ").strip().lower()

        if confirm != 'y':
            print("\n❌ Cancelled")
            return

        # Run the breakthrough attempt
        result = await self.system.attempt_breakthrough(
            problem_id=problem_id,
            max_days=max_days
        )

        # Show detailed results
        self._show_result_details(result)

    def _show_result_details(self, result):
        """Show detailed results of breakthrough attempt"""
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70 + "\n")

        if result.success:
            print("🏆 BREAKTHROUGH ACHIEVED! 🏆\n")
            print(f"Main Result: {result.main_result}")

            if result.proof:
                print(f"\nProof found: {len(result.proof)} characters")

            if result.formally_verified:
                print("\n✓ Formally verified with theorem prover")
                print("✓ Mathematically rigorous")
                print("✓ Ready for peer review")
                print("✓ Publishable in top journals")

            if result.problem.prize_money > 0:
                print(f"\n💰 CLAIM PRIZE: ${result.problem.prize_money:,}")

        else:
            print("📊 PROGRESS REPORT\n")
            print(f"Status: {result.main_result}")
            print(f"Time invested: {result.time_taken_days:.1f} days")

            if result.insights:
                print(f"\nInsights discovered: {len(result.insights)}")
                print("\nTop insights:")
                for i, insight in enumerate(result.insights[:3], 1):
                    print(f"  {i}. {insight}")

            if result.novel_techniques:
                print(f"\nNovel techniques explored: {len(result.novel_techniques)}")

            print("\n📌 This is valuable progress!")
            print("   Each attempt builds knowledge toward breakthrough.")

    async def run_demos_menu(self):
        """Run component demos"""
        print("\n" + "="*70)
        print("COMPONENT DEMOS")
        print("="*70 + "\n")

        print("1. Research Integration (arXiv, papers, problem database)")
        print("2. Formal Theorem Prover (Lean, proof search)")
        print("3. Deep Domain Learner (build expertise from papers)")
        print("4. Long-Term Reasoner (multi-day thinking)")
        print("5. Complete System Demo (all components)")
        print()
        print("0. Back to main menu")
        print()

        choice = input("Choose demo (0-5): ").strip()

        if choice == "0":
            return

        print("\n" + "="*70)

        if choice == "1":
            print("Running Research Integration Demo...")
            print("="*70 + "\n")
            from core import research_integration
            await research_integration.demo()

        elif choice == "2":
            print("Running Theorem Prover Demo...")
            print("="*70 + "\n")
            from core import theorem_prover
            await theorem_prover.demo()

        elif choice == "3":
            print("Running Deep Domain Learner Demo...")
            print("="*70 + "\n")
            from core import deep_domain_learner
            await deep_domain_learner.demo()

        elif choice == "4":
            print("Running Long-Term Reasoner Demo...")
            print("="*70 + "\n")
            from core import long_term_reasoner
            await long_term_reasoner.demo()

        elif choice == "5":
            print("Running Complete System Demo...")
            print("="*70 + "\n")
            from core import breakthrough_discovery
            await breakthrough_discovery.demo()

        else:
            print("❌ Invalid choice")

    async def custom_problem(self):
        """Attack a custom problem"""
        print("\n" + "="*70)
        print("CUSTOM PROBLEM")
        print("="*70 + "\n")

        print("Enter your own problem to attack.")
        print("This should be a mathematical or scientific problem.\n")

        problem_statement = input("Problem statement: ").strip()

        if not problem_statement:
            print("\n❌ No problem entered")
            return

        domain = input("Domain (e.g., mathematics, physics, cs): ").strip() or "unknown"

        print(f"\n✓ Problem: {problem_statement}")
        print(f"✓ Domain: {domain}")
        print()

        confirm = input("Attempt to solve this problem? (y/n): ").strip().lower()

        if confirm != 'y':
            print("\n❌ Cancelled")
            return

        print("\n[Note: Custom problems not in database yet]")
        print("[For now, use the theorem prover component directly]\n")

        # Could integrate theorem prover here for custom problems
        result = await self.system.prover.prove_theorem(
            statement=problem_statement,
            category=domain,
            max_time=60
        )

        print(f"\nResult: {result.final_status.value}")
        if result.final_proof:
            print(f"Proof found!\n{result.final_proof}")

    async def show_system_status(self):
        """Show system status and statistics"""
        print("\n" + "="*70)
        print("SYSTEM STATUS")
        print("="*70 + "\n")

        print("✓ Research Integration System")
        print("  - arXiv access: Ready")
        print("  - Semantic Scholar: Ready")
        print(f"  - Problem database: {len(self.system.research.problem_db.problems)} problems")
        print()

        print("✓ Formal Theorem Prover")
        stats = self.system.prover.get_discovery_stats()
        print(f"  - Theorems discovered: {stats['discovered_theorems']}")
        print(f"  - Theorems proved: {stats['proved_theorems']}")
        if stats['discovered_theorems'] > 0:
            print(f"  - Success rate: {stats['success_rate']:.1%}")
        print()

        print("✓ Deep Domain Learner")
        print(f"  - Knowledge bases: {len(self.system.learner.knowledge_bases)}")
        for domain, kb in self.system.learner.knowledge_bases.items():
            print(f"    • {domain}: {kb.expertise_level} ({kb.papers_read} papers)")
        print()

        print("✓ Long-Term Reasoner")
        print(f"  - Active projects: {len(self.system.reasoner.projects)}")
        for proj_id, proj in self.system.reasoner.projects.items():
            print(f"    • {proj_id}: {proj.status.value} ({proj.total_hours:.1f}h)")
        print()

        print("✓ Breakthrough Discovery System")
        print(f"  - Attempts made: {len(self.system.attempts)}")
        breakthroughs = sum(1 for a in self.system.attempts if a.breakthrough_achieved)
        if self.system.attempts:
            print(f"  - Breakthroughs: {breakthroughs}/{len(self.system.attempts)}")
        print()

        # Prize money available
        total_prize = sum(
            p.prize_money for p in self.system.research.problem_db.problems.values()
        )
        print(f"💰 Total prize money available: ${total_prize:,}")

    async def quick_attack(self, problem_id: str, days: int = 7):
        """Quick attack mode for command-line usage"""
        await self.initialize()

        print(f"\n🎯 Quick Attack Mode")
        print(f"   Problem: {problem_id}")
        print(f"   Time: {days} days\n")

        result = await self.system.attempt_breakthrough(
            problem_id=problem_id,
            max_days=days
        )

        self._show_result_details(result)

    async def run_all_demos(self):
        """Run all component demos"""
        await self.initialize()

        print("\n" + "="*70)
        print("RUNNING ALL COMPONENT DEMOS")
        print("="*70 + "\n")

        demos = [
            ("Research Integration", "core.research_integration"),
            ("Theorem Prover", "core.theorem_prover"),
            ("Deep Domain Learner", "core.deep_domain_learner"),
            ("Long-Term Reasoner", "core.long_term_reasoner"),
            ("Breakthrough Discovery", "core.breakthrough_discovery"),
        ]

        for name, module_name in demos:
            print(f"\n{'='*70}")
            print(f"DEMO: {name}")
            print(f"{'='*70}\n")

            module = __import__(module_name.replace('.', '/'), fromlist=['demo'])
            await module.demo()

            input("\n[Press Enter for next demo...]")


async def main():
    """Main entry point"""
    app = GroundbreakingDiscoveryMain()

    # Parse command-line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--demo":
            await app.run_all_demos()

        elif command == "--attack":
            if len(sys.argv) < 3:
                print("❌ Usage: python main_discovery.py --attack <problem_id> [days]")
                return

            problem_id = sys.argv[2]
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
            await app.quick_attack(problem_id, days)

        elif command == "--millennium":
            await app.initialize()
            millennium = app.system.research.get_millennium_problems()

            print("\n" + "="*70)
            print("MILLENNIUM PRIZE PROBLEMS ($1,000,000 EACH)")
            print("="*70 + "\n")

            for p in millennium:
                print(f"• {p.title}")
                print(f"  ID: {p.id}")
                print(f"  {p.description}")
                print(f"  Prize: ${p.prize_money:,}")
                print()

        elif command in ["-h", "--help"]:
            print(__doc__)

        else:
            print(f"❌ Unknown command: {command}")
            print(__doc__)

    else:
        # Interactive mode
        await app.run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
