"""
Phase 8: Hierarchical Swarm Intelligence

Multi-Team Meta-Memory System for Robust Convergence and Exploration.

Architecture:
1. Agents (Local Explorers) - Gradient descent, random search, evolutionary steps
2. Team Managers (Mid-Level Controllers) - Meta-memory, strategy adaptation
3. Global Supervisor (Top-Level Coordinator) - Global memory map, team allocation

This adds REAL optimization to KV-1, not just LLM orchestration!

Key Features:
- Parallel exploration of solution/search spaces
- Emergent intelligence through swarm behavior
- Meta-memory to avoid redundant exploration
- Adaptive strategy switching based on progress
- Hierarchical communication protocols
"""

from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import random
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


class ExplorationStrategy(Enum):
    """Available exploration strategies for agents."""
    GRADIENT_DESCENT = "gradient_descent"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class AgentStatus(Enum):
    """Status of a swarm agent."""
    EXPLORING = "exploring"
    CONVERGED = "converged"
    STUCK = "stuck"
    REASSIGNED = "reassigned"
    IDLE = "idle"


@dataclass
class Position:
    """Position in search space."""
    coordinates: np.ndarray
    fitness: float = float('inf')
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "coordinates": self.coordinates.tolist(),
            "fitness": self.fitness,
            "timestamp": self.timestamp
        }


@dataclass
class ExplorationReport:
    """Report from agent to manager."""
    agent_id: str
    best_position: Position
    positions_explored: int
    gradient_estimate: Optional[np.ndarray]
    confidence: float  # 0-1
    is_stagnant: bool
    strategy_used: ExplorationStrategy
    iterations: int
    improvement_rate: float  # Fitness improvement per iteration


@dataclass
class RegionAssignment:
    """Search region assigned to an agent or team."""
    center: np.ndarray
    radius: float
    bounds: Tuple[np.ndarray, np.ndarray]  # (lower, upper)
    priority: float = 1.0


@dataclass
class TeamReport:
    """Report from team manager to supervisor."""
    team_id: str
    best_position: Position
    coverage_map: Dict[str, float]  # Region -> exploration density
    agents_status: Dict[str, AgentStatus]
    total_evaluations: int
    convergence_rate: float
    request_reinforcement: bool
    request_new_region: bool


class SwarmAgent:
    """
    Local Explorer Agent.

    Performs actual optimization using various strategies.
    Reports to team manager periodically.
    """

    def __init__(
        self,
        agent_id: str,
        dimension: int,
        strategy: ExplorationStrategy = ExplorationStrategy.GRADIENT_DESCENT,
        learning_rate: float = 0.01
    ):
        self.id = agent_id
        self.dimension = dimension
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.status = AgentStatus.IDLE

        # Current state
        self.position = np.random.randn(dimension)
        self.best_position = Position(self.position.copy(), float('inf'))
        self.velocity = np.zeros(dimension)  # For PSO

        # History
        self.positions_explored = 0
        self.fitness_history: List[float] = []
        self.stagnation_counter = 0

        # Assignment
        self.region: Optional[RegionAssignment] = None

    def explore(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        iterations: int = 100
    ) -> ExplorationReport:
        """
        Explore the search space for given iterations.

        Args:
            fitness_fn: Function to evaluate positions
            iterations: Number of exploration steps

        Returns:
            ExplorationReport with results
        """
        self.status = AgentStatus.EXPLORING
        initial_fitness = self.best_position.fitness

        for i in range(iterations):
            # Apply exploration strategy
            if self.strategy == ExplorationStrategy.GRADIENT_DESCENT:
                self._gradient_step(fitness_fn)
            elif self.strategy == ExplorationStrategy.RANDOM_SEARCH:
                self._random_step(fitness_fn)
            elif self.strategy == ExplorationStrategy.EVOLUTIONARY:
                self._evolutionary_step(fitness_fn)
            elif self.strategy == ExplorationStrategy.SIMULATED_ANNEALING:
                self._annealing_step(fitness_fn, i, iterations)
            elif self.strategy == ExplorationStrategy.PARTICLE_SWARM:
                self._pso_step(fitness_fn)

            self.positions_explored += 1

            # Check for stagnation
            if len(self.fitness_history) > 10:
                recent = self.fitness_history[-10:]
                if max(recent) - min(recent) < 1e-6:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

        # Calculate metrics
        final_fitness = self.best_position.fitness
        improvement = initial_fitness - final_fitness
        improvement_rate = improvement / iterations if iterations > 0 else 0

        # Determine if stuck
        is_stagnant = self.stagnation_counter > 20
        if is_stagnant:
            self.status = AgentStatus.STUCK
        elif improvement_rate < 1e-8:
            self.status = AgentStatus.CONVERGED
        else:
            self.status = AgentStatus.EXPLORING

        # Estimate gradient
        gradient = self._estimate_gradient(fitness_fn)

        return ExplorationReport(
            agent_id=self.id,
            best_position=self.best_position,
            positions_explored=self.positions_explored,
            gradient_estimate=gradient,
            confidence=1.0 / (1.0 + self.stagnation_counter),
            is_stagnant=is_stagnant,
            strategy_used=self.strategy,
            iterations=iterations,
            improvement_rate=improvement_rate
        )

    def _gradient_step(self, fitness_fn: Callable):
        """Gradient descent step."""
        # Numerical gradient estimation
        gradient = self._estimate_gradient(fitness_fn)
        if gradient is not None:
            # Apply gradient with momentum
            new_pos = self.position - self.learning_rate * gradient

            # Respect bounds if assigned
            if self.region:
                new_pos = np.clip(new_pos, self.region.bounds[0], self.region.bounds[1])

            fitness = fitness_fn(new_pos)
            self._update_position(new_pos, fitness)

    def _random_step(self, fitness_fn: Callable):
        """Random search step."""
        # Random perturbation
        perturbation = np.random.randn(self.dimension) * self.learning_rate * 10

        new_pos = self.position + perturbation
        if self.region:
            new_pos = np.clip(new_pos, self.region.bounds[0], self.region.bounds[1])

        fitness = fitness_fn(new_pos)
        self._update_position(new_pos, fitness)

    def _evolutionary_step(self, fitness_fn: Callable):
        """Evolutionary mutation step."""
        # Mutation
        mutation_rate = 0.1
        mutation = np.random.randn(self.dimension) * mutation_rate

        new_pos = self.best_position.coordinates + mutation
        if self.region:
            new_pos = np.clip(new_pos, self.region.bounds[0], self.region.bounds[1])

        fitness = fitness_fn(new_pos)
        self._update_position(new_pos, fitness)

    def _annealing_step(self, fitness_fn: Callable, iteration: int, max_iter: int):
        """Simulated annealing step."""
        # Temperature schedule
        temp = max(0.01, 1.0 - iteration / max_iter)

        # Random neighbor
        neighbor = self.position + np.random.randn(self.dimension) * temp
        if self.region:
            neighbor = np.clip(neighbor, self.region.bounds[0], self.region.bounds[1])

        fitness = fitness_fn(neighbor)
        current_fitness = fitness_fn(self.position)

        # Accept with probability based on temperature
        if fitness < current_fitness:
            self._update_position(neighbor, fitness)
        elif random.random() < np.exp(-(fitness - current_fitness) / temp):
            self._update_position(neighbor, fitness)

    def _pso_step(self, fitness_fn: Callable, global_best: np.ndarray = None):
        """Particle swarm optimization step."""
        if global_best is None:
            global_best = self.best_position.coordinates

        # PSO parameters
        w = 0.7  # Inertia
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social

        r1, r2 = random.random(), random.random()

        # Update velocity
        cognitive = c1 * r1 * (self.best_position.coordinates - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

        # Update position
        new_pos = self.position + self.velocity
        if self.region:
            new_pos = np.clip(new_pos, self.region.bounds[0], self.region.bounds[1])

        fitness = fitness_fn(new_pos)
        self._update_position(new_pos, fitness)

    def _estimate_gradient(self, fitness_fn: Callable, epsilon: float = 1e-5) -> Optional[np.ndarray]:
        """Estimate gradient numerically."""
        try:
            gradient = np.zeros(self.dimension)
            f0 = fitness_fn(self.position)

            for i in range(self.dimension):
                pos_plus = self.position.copy()
                pos_plus[i] += epsilon
                gradient[i] = (fitness_fn(pos_plus) - f0) / epsilon

            return gradient
        except Exception:
            return None

    def _update_position(self, new_pos: np.ndarray, fitness: float):
        """Update position and track best."""
        self.position = new_pos
        self.fitness_history.append(fitness)

        if fitness < self.best_position.fitness:
            self.best_position = Position(new_pos.copy(), fitness)
            self.stagnation_counter = 0

    def assign_region(self, region: RegionAssignment):
        """Assign a search region to this agent."""
        self.region = region
        # Initialize position within region
        self.position = region.center + np.random.randn(self.dimension) * region.radius * 0.5
        self.position = np.clip(self.position, region.bounds[0], region.bounds[1])

    def change_strategy(self, new_strategy: ExplorationStrategy):
        """Change exploration strategy."""
        self.strategy = new_strategy
        self.stagnation_counter = 0


class TeamManager:
    """
    Mid-Level Controller for a team of agents.

    Maintains meta-memory, assigns regions, adapts strategies.
    """

    def __init__(
        self,
        team_id: str,
        dimension: int,
        num_agents: int = 5
    ):
        self.id = team_id
        self.dimension = dimension

        # Create agents with diverse strategies
        strategies = list(ExplorationStrategy)
        self.agents: Dict[str, SwarmAgent] = {}
        for i in range(num_agents):
            agent_id = f"{team_id}_agent_{i}"
            strategy = strategies[i % len(strategies)]
            self.agents[agent_id] = SwarmAgent(agent_id, dimension, strategy)

        # Meta-memory
        self.explored_regions: Dict[str, float] = {}  # Region hash -> density
        self.successful_directions: List[np.ndarray] = []
        self.failed_directions: List[np.ndarray] = []

        # Team state
        self.best_position = Position(np.zeros(dimension), float('inf'))
        self.total_evaluations = 0
        self.assigned_region: Optional[RegionAssignment] = None

    def run_exploration_cycle(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        iterations_per_agent: int = 50
    ) -> TeamReport:
        """
        Run one exploration cycle with all agents.

        Args:
            fitness_fn: Fitness function to optimize
            iterations_per_agent: Steps per agent

        Returns:
            TeamReport for supervisor
        """
        reports: List[ExplorationReport] = []

        # Run agents in parallel
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(agent.explore, fitness_fn, iterations_per_agent): agent_id
                for agent_id, agent in self.agents.items()
            }

            for future in as_completed(futures):
                try:
                    report = future.result()
                    reports.append(report)
                except Exception as e:
                    print(f"[!] Agent exploration failed: {e}")

        # Process reports
        for report in reports:
            self.total_evaluations += report.positions_explored

            # Update best position
            if report.best_position.fitness < self.best_position.fitness:
                self.best_position = report.best_position

                # Record successful direction
                if report.gradient_estimate is not None:
                    self.successful_directions.append(-report.gradient_estimate)

            # Update meta-memory
            region_hash = self._hash_position(report.best_position.coordinates)
            self.explored_regions[region_hash] = self.explored_regions.get(region_hash, 0) + 1

            # Handle stuck agents
            if report.is_stagnant:
                self._reassign_stuck_agent(report.agent_id, fitness_fn)

        # Generate team report
        agents_status = {aid: agent.status for aid, agent in self.agents.items()}
        stuck_count = sum(1 for s in agents_status.values() if s == AgentStatus.STUCK)
        converged_count = sum(1 for s in agents_status.values() if s == AgentStatus.CONVERGED)

        return TeamReport(
            team_id=self.id,
            best_position=self.best_position,
            coverage_map=self.explored_regions.copy(),
            agents_status=agents_status,
            total_evaluations=self.total_evaluations,
            convergence_rate=converged_count / len(self.agents),
            request_reinforcement=stuck_count > len(self.agents) // 2,
            request_new_region=converged_count > len(self.agents) * 0.8
        )

    def _hash_position(self, pos: np.ndarray, resolution: float = 1.0) -> str:
        """Hash position to grid cell."""
        grid_pos = (pos / resolution).astype(int)
        return str(grid_pos.tolist())

    def _reassign_stuck_agent(self, agent_id: str, fitness_fn: Callable):
        """Reassign a stuck agent to new region/strategy."""
        agent = self.agents[agent_id]

        # Try a different strategy
        current_idx = list(ExplorationStrategy).index(agent.strategy)
        new_strategy = list(ExplorationStrategy)[(current_idx + 1) % len(ExplorationStrategy)]
        agent.change_strategy(new_strategy)

        # Move to unexplored region
        if self.assigned_region:
            # Find least explored direction
            if self.successful_directions:
                direction = np.mean(self.successful_directions[-5:], axis=0)
                new_center = self.best_position.coordinates + direction * self.assigned_region.radius
            else:
                new_center = self.assigned_region.center + np.random.randn(self.dimension) * self.assigned_region.radius

            new_region = RegionAssignment(
                center=new_center,
                radius=self.assigned_region.radius * 0.5,
                bounds=self.assigned_region.bounds,
                priority=1.5  # Higher priority for unexplored
            )
            agent.assign_region(new_region)

        agent.status = AgentStatus.REASSIGNED

    def receive_peer_insight(self, insight: Dict):
        """Receive insight from peer team."""
        if "successful_directions" in insight:
            self.successful_directions.extend(insight["successful_directions"][-3:])
        if "avoid_regions" in insight:
            for region in insight["avoid_regions"]:
                self.explored_regions[region] = float('inf')  # Mark as fully explored

    def get_shareable_insights(self) -> Dict:
        """Get insights to share with peers."""
        return {
            "successful_directions": self.successful_directions[-5:],
            "avoid_regions": [k for k, v in self.explored_regions.items() if v > 10],
            "best_fitness": self.best_position.fitness
        }


class GlobalSupervisor:
    """
    Top-Level Coordinator for all teams.

    Maintains global memory map, allocates teams, detects global patterns.
    """

    def __init__(
        self,
        dimension: int,
        num_teams: int = 3,
        agents_per_team: int = 5,
        search_bounds: Tuple[float, float] = (-10.0, 10.0)
    ):
        self.dimension = dimension
        self.search_bounds = search_bounds

        # Create teams
        self.teams: Dict[str, TeamManager] = {}
        for i in range(num_teams):
            team_id = f"team_{i}"
            self.teams[team_id] = TeamManager(team_id, dimension, agents_per_team)

        # Global memory
        self.global_heatmap: Dict[str, float] = {}  # Region -> total exploration
        self.best_position = Position(np.zeros(dimension), float('inf'))
        self.fitness_history: List[float] = []

        # Allocation tracking
        self.team_regions: Dict[str, RegionAssignment] = {}
        self.total_evaluations = 0
        self.cycles_completed = 0

        # Initialize team regions
        self._initialize_team_regions()

        print(f"[+] Global Supervisor: {num_teams} teams, {agents_per_team} agents each")

    def _initialize_team_regions(self):
        """Assign initial regions to teams."""
        num_teams = len(self.teams)
        bounds_range = self.search_bounds[1] - self.search_bounds[0]
        region_radius = bounds_range / (2 * np.sqrt(num_teams))

        # Distribute teams across search space
        for i, team_id in enumerate(self.teams.keys()):
            # Create diverse starting points
            angle = 2 * np.pi * i / num_teams
            center = np.zeros(self.dimension)
            center[0] = np.cos(angle) * bounds_range * 0.3
            if self.dimension > 1:
                center[1] = np.sin(angle) * bounds_range * 0.3

            region = RegionAssignment(
                center=center,
                radius=region_radius,
                bounds=(
                    np.full(self.dimension, self.search_bounds[0]),
                    np.full(self.dimension, self.search_bounds[1])
                )
            )

            self.team_regions[team_id] = region
            self.teams[team_id].assigned_region = region

            # Assign to all agents in team
            for agent in self.teams[team_id].agents.values():
                agent.assign_region(region)

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        max_cycles: int = 100,
        target_fitness: float = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run the full hierarchical swarm optimization.

        Args:
            fitness_fn: Function to minimize
            max_cycles: Maximum optimization cycles
            target_fitness: Stop when reached (optional)
            verbose: Print progress

        Returns:
            Optimization results
        """
        start_time = time.time()

        if verbose:
            print(f"[Swarm] Starting optimization with {len(self.teams)} teams")

        for cycle in range(max_cycles):
            self.cycles_completed = cycle + 1

            # Run all teams
            team_reports: List[TeamReport] = []
            for team_id, team in self.teams.items():
                report = team.run_exploration_cycle(fitness_fn, iterations_per_agent=50)
                team_reports.append(report)

            # Process team reports
            for report in team_reports:
                self.total_evaluations += report.total_evaluations

                # Update global best
                if report.best_position.fitness < self.best_position.fitness:
                    self.best_position = report.best_position
                    if verbose:
                        print(f"    [Cycle {cycle}] New best: {self.best_position.fitness:.6f}")

                # Update global heatmap
                for region, density in report.coverage_map.items():
                    self.global_heatmap[region] = self.global_heatmap.get(region, 0) + density

                # Handle team requests
                if report.request_new_region:
                    self._assign_new_region(report.team_id)
                if report.request_reinforcement:
                    self._reinforce_team(report.team_id)

            # Record fitness
            self.fitness_history.append(self.best_position.fitness)

            # Enable peer communication
            self._facilitate_peer_communication()

            # Check termination
            if target_fitness is not None and self.best_position.fitness <= target_fitness:
                if verbose:
                    print(f"[Swarm] Target fitness reached at cycle {cycle}")
                break

            # Check for global stagnation
            if len(self.fitness_history) > 10:
                recent = self.fitness_history[-10:]
                if max(recent) - min(recent) < 1e-8:
                    if verbose:
                        print(f"[Swarm] Global stagnation detected at cycle {cycle}")
                    self._global_restart()

        elapsed = time.time() - start_time

        result = {
            "best_position": self.best_position.coordinates.tolist(),
            "best_fitness": self.best_position.fitness,
            "total_evaluations": self.total_evaluations,
            "cycles_completed": self.cycles_completed,
            "elapsed_time": elapsed,
            "evaluations_per_second": self.total_evaluations / elapsed,
            "fitness_history": self.fitness_history
        }

        if verbose:
            print(f"[Swarm] Optimization complete:")
            print(f"    Best fitness: {self.best_position.fitness:.6f}")
            print(f"    Evaluations: {self.total_evaluations}")
            print(f"    Time: {elapsed:.2f}s")

        return result

    def _assign_new_region(self, team_id: str):
        """Assign team to unexplored region."""
        # Find least explored region
        all_explored = set(self.global_heatmap.keys())

        # Generate candidate regions
        best_candidate = None
        min_exploration = float('inf')

        for _ in range(10):
            candidate_center = np.random.uniform(
                self.search_bounds[0],
                self.search_bounds[1],
                self.dimension
            )
            region_hash = self._hash_position(candidate_center)

            exploration = self.global_heatmap.get(region_hash, 0)
            if exploration < min_exploration:
                min_exploration = exploration
                best_candidate = candidate_center

        if best_candidate is not None:
            current_region = self.team_regions[team_id]
            new_region = RegionAssignment(
                center=best_candidate,
                radius=current_region.radius,
                bounds=current_region.bounds
            )
            self.team_regions[team_id] = new_region
            self.teams[team_id].assigned_region = new_region

            for agent in self.teams[team_id].agents.values():
                agent.assign_region(new_region)

    def _reinforce_team(self, team_id: str):
        """Reinforce a struggling team."""
        team = self.teams[team_id]

        # Share best strategies from successful teams
        best_team = min(
            self.teams.values(),
            key=lambda t: t.best_position.fitness
        )

        if best_team.id != team_id:
            insight = best_team.get_shareable_insights()
            team.receive_peer_insight(insight)

    def _facilitate_peer_communication(self):
        """Enable communication between team managers."""
        # Sorted by performance
        sorted_teams = sorted(
            self.teams.values(),
            key=lambda t: t.best_position.fitness
        )

        # Share from best to worst
        for i, team in enumerate(sorted_teams[:-1]):
            insights = team.get_shareable_insights()
            # Share with worse-performing teams
            for worse_team in sorted_teams[i+1:]:
                worse_team.receive_peer_insight(insights)

    def _global_restart(self):
        """Perform global restart when stagnating."""
        # Keep best team, restart others
        best_team_id = min(
            self.teams.keys(),
            key=lambda tid: self.teams[tid].best_position.fitness
        )

        for team_id, team in self.teams.items():
            if team_id != best_team_id:
                # Assign to random new region
                new_center = np.random.uniform(
                    self.search_bounds[0],
                    self.search_bounds[1],
                    self.dimension
                )
                new_region = RegionAssignment(
                    center=new_center,
                    radius=self.team_regions[team_id].radius * 1.5,
                    bounds=self.team_regions[team_id].bounds
                )
                self.team_regions[team_id] = new_region
                team.assigned_region = new_region

                for agent in team.agents.values():
                    agent.assign_region(new_region)
                    # Randomize strategy
                    agent.change_strategy(random.choice(list(ExplorationStrategy)))

    def _hash_position(self, pos: np.ndarray, resolution: float = 1.0) -> str:
        """Hash position to grid cell."""
        grid_pos = (pos / resolution).astype(int)
        return str(grid_pos.tolist())

    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        team_stats = {}
        for tid, team in self.teams.items():
            team_stats[tid] = {
                "best_fitness": team.best_position.fitness,
                "evaluations": team.total_evaluations,
                "agents": {
                    aid: agent.status.value
                    for aid, agent in team.agents.items()
                }
            }

        return {
            "global_best": self.best_position.fitness,
            "total_evaluations": self.total_evaluations,
            "cycles": self.cycles_completed,
            "teams": team_stats,
            "coverage": len(self.global_heatmap)
        }


class SwarmIntelligenceEngine:
    """
    Main interface for Hierarchical Swarm Intelligence.

    Integrates with KV-1's cognitive systems for:
    - Solution space exploration
    - Framework optimization
    - Hyperparameter search
    - Multi-objective optimization
    """

    def __init__(
        self,
        storage_path: str = "./swarm_state.json"
    ):
        self.storage_path = storage_path
        self.supervisors: Dict[str, GlobalSupervisor] = {}
        self.optimization_history: List[Dict] = []

        print("[+] Swarm Intelligence Engine initialized")
        print("    Hierarchical multi-team optimization active!")

    def create_optimizer(
        self,
        name: str,
        dimension: int,
        num_teams: int = 3,
        agents_per_team: int = 5,
        search_bounds: Tuple[float, float] = (-10.0, 10.0)
    ) -> GlobalSupervisor:
        """
        Create a new swarm optimizer.

        Args:
            name: Optimizer identifier
            dimension: Search space dimension
            num_teams: Number of teams
            agents_per_team: Agents per team
            search_bounds: Search space bounds

        Returns:
            GlobalSupervisor for optimization
        """
        supervisor = GlobalSupervisor(
            dimension=dimension,
            num_teams=num_teams,
            agents_per_team=agents_per_team,
            search_bounds=search_bounds
        )

        self.supervisors[name] = supervisor
        return supervisor

    def optimize_function(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        dimension: int,
        max_evaluations: int = 10000,
        target_fitness: float = None,
        num_teams: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        Optimize a function using hierarchical swarm.

        Args:
            fitness_fn: Function to minimize
            dimension: Input dimension
            max_evaluations: Maximum function evaluations
            target_fitness: Target to reach (optional)
            num_teams: Number of teams to use
            verbose: Print progress

        Returns:
            Optimization results
        """
        # Create supervisor
        supervisor = self.create_optimizer(
            name=f"opt_{len(self.optimization_history)}",
            dimension=dimension,
            num_teams=num_teams,
            agents_per_team=5
        )

        # Estimate cycles needed
        agents_total = num_teams * 5
        evals_per_cycle = agents_total * 50
        max_cycles = max(10, max_evaluations // evals_per_cycle)

        # Run optimization
        result = supervisor.optimize(
            fitness_fn=fitness_fn,
            max_cycles=max_cycles,
            target_fitness=target_fitness,
            verbose=verbose
        )

        self.optimization_history.append(result)
        return result

    def optimize_discrete(
        self,
        options: List[Any],
        score_fn: Callable[[Any], float],
        num_samples: int = 100
    ) -> Tuple[Any, float]:
        """
        Optimize over discrete options using swarm sampling.

        Args:
            options: List of discrete options
            score_fn: Function to score options (lower is better)
            num_samples: Number of samples to evaluate

        Returns:
            (best_option, best_score)
        """
        # Evaluate samples
        scores = []
        sampled_indices = random.sample(
            range(len(options)),
            min(num_samples, len(options))
        )

        for idx in sampled_indices:
            score = score_fn(options[idx])
            scores.append((idx, score))

        # Find best
        best_idx, best_score = min(scores, key=lambda x: x[1])
        return options[best_idx], best_score

    def multi_objective_optimize(
        self,
        objectives: List[Callable[[np.ndarray], float]],
        dimension: int,
        max_cycles: int = 50
    ) -> List[Position]:
        """
        Multi-objective optimization using swarm.

        Args:
            objectives: List of objective functions
            dimension: Search space dimension
            max_cycles: Maximum cycles

        Returns:
            Pareto front positions
        """
        # Create combined objective (simple weighted sum for now)
        def combined_objective(x):
            return sum(obj(x) for obj in objectives)

        supervisor = self.create_optimizer(
            name=f"multi_obj_{len(self.optimization_history)}",
            dimension=dimension,
            num_teams=len(objectives)
        )

        # Run optimization
        result = supervisor.optimize(
            fitness_fn=combined_objective,
            max_cycles=max_cycles,
            verbose=False
        )

        # Return best positions from each team as approximate Pareto front
        pareto_front = []
        for team in supervisor.teams.values():
            pareto_front.append(team.best_position)

        return pareto_front

    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        return {
            "total_optimizations": len(self.optimization_history),
            "active_supervisors": len(self.supervisors),
            "total_evaluations": sum(
                r.get("total_evaluations", 0)
                for r in self.optimization_history
            )
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        stats = self.get_statistics()
        lines = [
            "Swarm Intelligence Engine Status:",
            f"  Total optimizations: {stats['total_optimizations']}",
            f"  Active supervisors: {stats['active_supervisors']}",
            f"  Total evaluations: {stats['total_evaluations']}"
        ]

        if self.optimization_history:
            lines.append("\nRecent optimizations:")
            for result in self.optimization_history[-3:]:
                lines.append(f"  - Best: {result.get('best_fitness', 'N/A'):.6f}")
                lines.append(f"    Evals: {result.get('total_evaluations', 'N/A')}")

        return "\n".join(lines)


# =============================================================================
# Test Functions for Benchmarking
# =============================================================================

def sphere_function(x: np.ndarray) -> float:
    """Sphere function - simple unimodal test."""
    return np.sum(x ** 2)


def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function - highly multimodal test."""
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function - valley test."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def ackley_function(x: np.ndarray) -> float:
    """Ackley function - many local minima."""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e
