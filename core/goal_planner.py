"""
Goal Planner with Dependency Graphs

Creates a dependency graph BEFORE learning, enabling:
- Optimal learning path planning
- Parallel execution of independent concepts
- Avoiding unnecessary prerequisites
- Strategic goal decomposition

This is how AGI plans instead of reacting!
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import asyncio


@dataclass
class ConceptNode:
    """Node in the dependency graph."""
    name: str
    prerequisites: List[str]
    level: int = 0  # Depth in graph (0 = root goal)
    is_learned: bool = False
    confidence: float = 0.0


class GoalPlanner:
    """
    Creates dependency graphs and optimal learning plans.

    This is the AGI "thinking before acting" component!
    """

    def __init__(self, llm_bridge, relevance_filter=None):
        self.llm = llm_bridge
        self.relevance_filter = relevance_filter
        self.graph: Dict[str, ConceptNode] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)

    async def create_learning_plan(
        self,
        goal: str,
        goal_domain: str,
        max_depth: int = 4
    ) -> Tuple[List[List[str]], Dict[str, ConceptNode]]:
        """
        Create a complete learning plan with dependency graph.

        Returns: (learning_stages, graph)
        - learning_stages: List of concept groups that can be learned in parallel
        - graph: Full dependency graph
        """
        print("\n" + "="*60)
        print("🎯 GOAL PLANNER: Creating dependency graph...")
        print("="*60)

        # Build the dependency graph
        await self._build_graph(goal, goal_domain, max_depth)

        # Topological sort to find optimal order
        learning_stages = self._topological_sort()

        # Display the plan
        self._display_plan(learning_stages)

        return learning_stages, self.graph

    async def _build_graph(
        self,
        concept: str,
        domain: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[Set[str]] = None
    ):
        """
        Recursively build dependency graph.
        """
        if visited is None:
            visited = set()

        if concept in visited or current_depth > max_depth:
            return

        visited.add(concept)

        # Get prerequisites for this concept
        prereqs = await self._get_prerequisites(concept, domain)

        # Filter irrelevant prerequisites
        if self.relevance_filter and prereqs:
            relevant, filtered = await self.relevance_filter.filter_prerequisites(
                prereqs, concept, concept, domain
            )
            prereqs = relevant

        # Create node
        node = ConceptNode(
            name=concept,
            prerequisites=prereqs,
            level=current_depth
        )
        self.graph[concept] = node

        # Build adjacency list
        for prereq in prereqs:
            self.adjacency[concept].append(prereq)
            # Recurse
            await self._build_graph(prereq, domain, max_depth, current_depth + 1, visited)

    async def _get_prerequisites(self, concept: str, domain: str) -> List[str]:
        """
        Ask LLM what prerequisites are needed.
        """
        prompt = f"""What are the ESSENTIAL prerequisites to understand "{concept}" in {domain}?

List ONLY the most fundamental concepts needed. Be minimal!

Rules:
- No more than 3-5 prerequisites
- Only include what's ABSOLUTELY necessary
- Use simple, clear concept names
- Avoid overly advanced prerequisites

Format:
PREREQUISITES:
- [concept 1]
- [concept 2]
- [concept 3]

Example for "prime numbers":
PREREQUISITES:
- natural numbers
- division
- factors
"""

        response = self.llm.generate(
            system_prompt=f"You are planning prerequisites for learning {domain}. Be minimal - only essential concepts!",
            user_input=prompt
        )

        text = response.get("text", "")

        # Parse prerequisites
        prereqs = []
        in_prereq_section = False

        for line in text.split('\n'):
            line = line.strip()
            if "PREREQUISITES:" in line.upper():
                in_prereq_section = True
                continue

            if in_prereq_section and line.startswith('-'):
                prereq = line.lstrip('- ').strip()
                if prereq and len(prereq) < 100:  # Sanity check
                    prereqs.append(prereq)

        # Limit to 5 max
        return prereqs[:5]

    def _topological_sort(self) -> List[List[str]]:
        """
        Topological sort to find optimal learning order.

        Returns stages where each stage can be learned in PARALLEL!
        """
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for node in self.graph:
            in_degree[node] = 0

        for node in self.graph:
            for prereq in self.adjacency[node]:
                in_degree[node] += 1

        # Find all nodes with no prerequisites (in-degree 0)
        queue = deque([node for node in self.graph if in_degree[node] == 0])

        stages = []

        while queue:
            # All nodes in queue can be learned in parallel!
            current_stage = list(queue)
            stages.append(current_stage)
            queue.clear()

            # Process this stage
            for node in current_stage:
                # Find all nodes that depend on this one
                for dependent in self.graph:
                    if node in self.adjacency[dependent]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

        # Reverse so we learn prerequisites first
        return list(reversed(stages))

    def _display_plan(self, stages: List[List[str]]):
        """
        Display the learning plan in a readable format.
        """
        print(f"\n[📋] Learning Plan ({len(stages)} stages):")
        print("="*60)

        for i, stage in enumerate(stages, 1):
            print(f"\nStage {i}: ({len(stage)} concepts - can learn in PARALLEL)")
            for concept in stage:
                node = self.graph[concept]
                prereq_str = f" [needs: {', '.join(node.prerequisites)}]" if node.prerequisites else " [no prerequisites]"
                print(f"  • {concept}{prereq_str}")

        print("\n" + "="*60)
        total_concepts = sum(len(stage) for stage in stages)
        print(f"[✓] Total concepts to learn: {total_concepts}")
        print(f"[✓] With parallelization: ~{len(stages)} learning rounds instead of {total_concepts}")
        print(f"[✓] Estimated speedup: {total_concepts/len(stages):.1f}x faster!\n")

    def get_next_learnable_concepts(
        self,
        already_learned: Set[str]
    ) -> List[str]:
        """
        Get concepts that can be learned NOW (all prereqs satisfied).

        Returns: List of concepts ready to learn in parallel
        """
        learnable = []

        for concept, node in self.graph.items():
            if concept in already_learned:
                continue

            # Check if all prerequisites are learned
            all_prereqs_learned = all(
                prereq in already_learned
                for prereq in node.prerequisites
            )

            if all_prereqs_learned:
                learnable.append(concept)

        return learnable

    def update_learned(self, concept: str, confidence: float):
        """
        Mark a concept as learned in the graph.
        """
        if concept in self.graph:
            self.graph[concept].is_learned = True
            self.graph[concept].confidence = confidence

    def get_progress_summary(self) -> Dict:
        """
        Get learning progress statistics.
        """
        total = len(self.graph)
        learned = sum(1 for node in self.graph.values() if node.is_learned)
        avg_confidence = sum(node.confidence for node in self.graph.values() if node.is_learned) / max(learned, 1)

        return {
            "total_concepts": total,
            "learned": learned,
            "remaining": total - learned,
            "progress_percent": (learned / total * 100) if total > 0 else 0,
            "avg_confidence": avg_confidence
        }

    def visualize_graph(self) -> str:
        """
        Create ASCII visualization of the dependency graph.
        """
        lines = ["", "Dependency Graph:", "=" * 40]

        # Group by level
        by_level = defaultdict(list)
        for concept, node in self.graph.items():
            by_level[node.level].append(concept)

        max_level = max(by_level.keys()) if by_level else 0

        for level in range(max_level + 1):
            concepts = by_level[level]
            if concepts:
                lines.append(f"\nLevel {level}:")
                for concept in concepts:
                    node = self.graph[concept]
                    status = "✓" if node.is_learned else "○"
                    prereqs = f" ← {', '.join(node.prerequisites)}" if node.prerequisites else ""
                    lines.append(f"  [{status}] {concept}{prereqs}")

        return "\n".join(lines)
