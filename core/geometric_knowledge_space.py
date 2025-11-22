"""
Geometric Knowledge Space

Represents mathematical knowledge as a Riemannian manifold.
Concepts are POINTS, relations are DISTANCES, understanding is GEOMETRY!

Key ideas:
- Each concept = point in high-dimensional space
- Distance = how related concepts are
- Geodesic = learning path between concepts
- Curvature = concept complexity
- Parallel transport = analogies

This provides GEOMETRIC INTUITION to guide proof search!
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class ConceptPoint:
    """A point in the knowledge manifold"""
    name: str
    tensor: torch.Tensor
    symbolic_form: Any
    properties: Dict[str, Any]


@dataclass
class GeodesicPath:
    """A geodesic (shortest path) between concepts"""
    start: str
    end: str
    path_tensors: List[torch.Tensor]
    distance: float
    steps: List[str]


class RiemannianKnowledgeManifold:
    """
    Mathematical knowledge as a Riemannian manifold.

    Geometry guides exploration!
    """

    def __init__(self, dimension: int = 768, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.dimension = dimension
        self.device = device

        # Knowledge points
        self.concepts: Dict[str, ConceptPoint] = {}

        # Metric tensor (defines distances)
        # For now, Euclidean metric, but could be learned!
        self.metric_tensor = torch.eye(dimension, device=device)

        # Cache for computed distances
        self.distance_cache: Dict[Tuple[str, str], float] = {}

    def embed_concept(self, name: str, symbolic_form: Any, properties: Dict = None) -> torch.Tensor:
        """
        Embed a mathematical concept as a point in the manifold.

        Uses properties to determine position!
        """
        if properties is None:
            properties = {}

        # Create tensor based on properties
        tensor = self._create_tensor_from_properties(name, properties)

        # Store concept
        concept = ConceptPoint(
            name=name,
            tensor=tensor,
            symbolic_form=symbolic_form,
            properties=properties
        )

        self.concepts[name] = concept

        return tensor

    def _create_tensor_from_properties(self, name: str, properties: Dict) -> torch.Tensor:
        """
        Create tensor representation from mathematical properties.

        This is where we encode mathematical structure into geometry!
        """
        # Start with random tensor
        tensor = torch.randn(self.dimension, device=self.device)

        # Encode known properties
        # Example properties that affect position:

        if 'domain' in properties:
            # Different domains occupy different regions
            domain_encoding = {
                'number_theory': 0,
                'algebra': 1,
                'geometry': 2,
                'calculus': 3,
                'logic': 4
            }
            if properties['domain'] in domain_encoding:
                idx = domain_encoding[properties['domain']]
                tensor[idx * 100:(idx + 1) * 100] += 2.0  # Boost this region

        if 'is_prime' in properties:
            # Prime numbers cluster together
            tensor[0:100] += 1.0 if properties['is_prime'] else -1.0

        if 'is_even' in properties:
            # Even numbers cluster
            tensor[100:200] += 1.0 if properties['is_even'] else -1.0

        if 'complexity' in properties:
            # Complexity affects certain dimensions
            complexity = properties['complexity']
            tensor[200:300] *= (1 + complexity)

        # Normalize
        tensor = F.normalize(tensor, dim=0)

        return tensor

    def riemannian_distance(self, concept_a: str, concept_b: str) -> float:
        """
        Compute Riemannian distance between concepts.

        Distance = how related they are!
        Small distance = closely related
        Large distance = unrelated
        """
        # Check cache
        cache_key = (concept_a, concept_b)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        if concept_a not in self.concepts or concept_b not in self.concepts:
            return float('inf')

        tensor_a = self.concepts[concept_a].tensor
        tensor_b = self.concepts[concept_b].tensor

        # Riemannian distance using metric tensor
        diff = tensor_a - tensor_b

        # d(a,b) = sqrt(diff^T * G * diff) where G is metric tensor
        distance = torch.sqrt(diff @ self.metric_tensor @ diff).item()

        # Cache it
        self.distance_cache[cache_key] = distance

        return distance

    def find_geodesic(self, start: str, end: str, num_steps: int = 10) -> GeodesicPath:
        """
        Find geodesic (shortest path) from start to end.

        This is the LEARNING PATH!
        """
        if start not in self.concepts or end not in self.concepts:
            return None

        start_tensor = self.concepts[start].tensor
        end_tensor = self.concepts[end].tensor

        # Simple geodesic: linear interpolation
        # (In Euclidean space, geodesic is straight line)
        path_tensors = []
        steps = []

        for i in range(num_steps + 1):
            alpha = i / num_steps
            point = (1 - alpha) * start_tensor + alpha * end_tensor
            path_tensors.append(point)

            # Find nearest concept to this point
            nearest = self.find_nearest_concept(point)
            if nearest and nearest not in steps:
                steps.append(nearest)

        distance = self.riemannian_distance(start, end)

        return GeodesicPath(
            start=start,
            end=end,
            path_tensors=path_tensors,
            distance=distance,
            steps=steps
        )

    def find_nearest_concept(self, tensor: torch.Tensor, k: int = 1) -> Optional[str]:
        """
        Find nearest concept(s) to a tensor.

        Used to map arbitrary tensors back to concepts!
        """
        if not self.concepts:
            return None

        distances = []
        for name, concept in self.concepts.items():
            dist = torch.norm(tensor - concept.tensor).item()
            distances.append((dist, name))

        distances.sort()

        if k == 1:
            return distances[0][1] if distances else None
        else:
            return [name for _, name in distances[:k]]

    def compute_curvature(self, concept: str) -> float:
        """
        Compute Gaussian curvature at a concept.

        High curvature = complex concept
        Low curvature = simple concept
        """
        if concept not in self.concepts:
            return 0.0

        # Get nearby concepts
        nearby = self.get_neighbors(concept, k=5)

        if len(nearby) < 3:
            return 0.0

        # Compute variance of distances
        # High variance = curved region = complex concept
        center_tensor = self.concepts[concept].tensor
        distances = [
            torch.norm(self.concepts[n].tensor - center_tensor).item()
            for n in nearby
        ]

        curvature = np.var(distances) if distances else 0.0

        return curvature

    def get_neighbors(self, concept: str, k: int = 5) -> List[str]:
        """
        Get k nearest neighbors to a concept.
        """
        if concept not in self.concepts:
            return []

        center_tensor = self.concepts[concept].tensor

        distances = []
        for name, other_concept in self.concepts.items():
            if name == concept:
                continue
            dist = torch.norm(center_tensor - other_concept.tensor).item()
            distances.append((dist, name))

        distances.sort()
        return [name for _, name in distances[:k]]

    def parallel_transport(self, concept: str, direction_concept: str) -> Optional[str]:
        """
        Parallel transport: Find analogous concept.

        "Prime is to composite as even is to ____?"
        Uses parallel transport in manifold!
        """
        if concept not in self.concepts or direction_concept not in self.concepts:
            return None

        # Vector from concept to direction_concept
        start = self.concepts[concept].tensor
        end = self.concepts[direction_concept].tensor
        direction = end - start

        # Transport this vector to another region
        # For simplicity, apply same vector to a random concept
        # (Real parallel transport would preserve metric)

        # For now, return concept in similar direction
        result_tensor = start + direction

        return self.find_nearest_concept(result_tensor)

    def distance_to_goal(self, current_state_tensor: torch.Tensor, goal: str) -> float:
        """
        Geometric distance from current state to goal.

        Used to guide proof search!
        Small distance = getting closer to solution
        """
        if goal not in self.concepts:
            return float('inf')

        goal_tensor = self.concepts[goal].tensor

        diff = current_state_tensor - goal_tensor
        distance = torch.sqrt(diff @ self.metric_tensor @ diff).item()

        return distance

    def promising_direction(self, current: torch.Tensor, goal: str) -> torch.Tensor:
        """
        Compute most promising direction to explore.

        Returns unit vector pointing toward goal!
        """
        if goal not in self.concepts:
            return torch.zeros_like(current)

        goal_tensor = self.concepts[goal].tensor

        # Direction toward goal
        direction = goal_tensor - current

        # Normalize to unit vector
        direction = F.normalize(direction, dim=0)

        return direction

    def visualize_neighborhood(self, concept: str, k: int = 10) -> Dict[str, float]:
        """
        Get neighborhood structure around a concept.

        Returns mapping: neighbor_name -> distance
        """
        if concept not in self.concepts:
            return {}

        neighbors = self.get_neighbors(concept, k=k)

        neighborhood = {}
        for neighbor in neighbors:
            distance = self.riemannian_distance(concept, neighbor)
            neighborhood[neighbor] = distance

        return neighborhood

    def find_concept_cluster(self, center: str, radius: float) -> List[str]:
        """
        Find all concepts within radius of center.

        Useful for finding related concepts!
        """
        if center not in self.concepts:
            return []

        cluster = []
        for name in self.concepts:
            if name == center:
                continue
            distance = self.riemannian_distance(center, name)
            if distance <= radius:
                cluster.append(name)

        return cluster

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the manifold"""
        return {
            'num_concepts': len(self.concepts),
            'dimension': self.dimension,
            'device': self.device,
            'cache_size': len(self.distance_cache)
        }
