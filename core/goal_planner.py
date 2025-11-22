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
        visited: Optional[Set[str]] = None,
        max_nodes: int = 50
    ):
        """
        Recursively build dependency graph with circular dependency detection.

        Args:
            max_nodes: Maximum total nodes to prevent graph explosion (default: 50)
        """
        if visited is None:
            visited = set()

        # CRITICAL FIX: Proper depth check (use >= not >)
        if current_depth >= max_depth:
            print(f"[!] Max depth {max_depth} reached for '{concept}', stopping recursion")
            return

        # CRITICAL FIX: Prevent graph explosion
        if len(self.graph) >= max_nodes:
            print(f"[!] Max nodes ({max_nodes}) reached, stopping graph building to prevent memory explosion")
            return

        # CRITICAL FIX: Check for exact duplicates first
        if concept in visited:
            print(f"[!] Circular dependency detected: '{concept}' already in current path")
            return

        # CRITICAL FIX: Check for semantic duplicates (e.g., "sets" vs "set theory")
        concept_lower = concept.lower()
        for visited_concept in visited:
            # Check if this is semantically the same concept
            if self._are_concepts_similar(concept_lower, visited_concept.lower()):
                print(f"[!] Semantic circular dependency: '{concept}' is similar to '{visited_concept}'")
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

        # CRITICAL FIX: Detect circular references in prerequisites
        prereqs_clean = []
        for prereq in prereqs:
            prereq_lower = prereq.lower()
            # Check if prerequisite is the concept itself or very similar
            if self._are_concepts_similar(concept_lower, prereq_lower):
                print(f"[!] Self-dependency detected: '{concept}' requires '{prereq}' (skipping)")
                continue
            # Check if prerequisite is already an ancestor (would create cycle)
            is_ancestor = False
            for ancestor in visited:
                if self._are_concepts_similar(prereq_lower, ancestor.lower()):
                    print(f"[!] Ancestor dependency detected: '{prereq}' would create cycle with '{ancestor}'")
                    is_ancestor = True
                    break
            if not is_ancestor:
                prereqs_clean.append(prereq)

        # Create node
        node = ConceptNode(
            name=concept,
            prerequisites=prereqs_clean,
            level=current_depth
        )
        self.graph[concept] = node

        # Build adjacency list
        for prereq in prereqs_clean:
            self.adjacency[concept].append(prereq)
            # Recurse with updated visited set
            await self._build_graph(prereq, domain, max_depth, current_depth + 1, visited.copy(), max_nodes)

    def _are_concepts_similar(self, concept1: str, concept2: str) -> bool:
        """
        Check if two concepts are semantically similar to detect circular dependencies.

        Examples of similar concepts:
        - "sets" vs "set theory" vs "set concepts"
        - "variables" vs "variable" vs "algebraic variables"
        - "equality" vs "equation" vs "equality properties"
        """
        # Exact match
        if concept1 == concept2:
            return True

        # Remove common suffixes/prefixes
        concept1_clean = concept1.replace(" theory", "").replace(" concepts", "").replace(" properties", "")
        concept2_clean = concept2.replace(" theory", "").replace(" concepts", "").replace(" properties", "")

        # Check if one is a substring of the other (with word boundaries)
        words1 = set(concept1_clean.split())
        words2 = set(concept2_clean.split())

        # If they share most words, they're likely the same concept
        if len(words1) > 0 and len(words2) > 0:
            shared = words1 & words2
            # If 80% of words overlap, consider similar
            overlap_ratio = len(shared) / min(len(words1), len(words2))
            if overlap_ratio >= 0.8:
                return True

        # Check singular/plural forms
        import re
        singular1 = re.sub(r's$', '', concept1_clean)
        singular2 = re.sub(r's$', '', concept2_clean)
        if singular1 == singular2:
            return True

        # Check if one contains the other (for cases like "set" vs "set theory")
        if concept1_clean in concept2_clean or concept2_clean in concept1_clean:
            # But only if the contained part is significant (>3 chars)
            shorter = min(concept1_clean, concept2_clean, key=len)
            if len(shorter) > 3:
                return True

        return False

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
