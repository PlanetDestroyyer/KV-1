"""
Phase 7: Autonomous Agent System

Creates self-directed agents capable of:
1. Goal-directed behavior with autonomous planning
2. Tool use and environment interaction
3. Self-improvement through reflection
4. Multi-agent collaboration
5. Continuous learning and adaptation

Architecture:
- Agent: Autonomous reasoning entity with goals and memory
- Environment: World model the agent interacts with
- Actions: Available operations the agent can perform
- Planner: Generates action sequences to achieve goals
- Executor: Carries out planned actions
- Monitor: Tracks progress and handles failures
"""

from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
import re
import os
import uuid
import time


class AgentState(Enum):
    """Possible states of an autonomous agent."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(Enum):
    """Types of actions an agent can take."""
    THINK = "think"  # Internal reasoning
    OBSERVE = "observe"  # Gather information
    ACT = "act"  # Perform action in environment
    COMMUNICATE = "communicate"  # Inter-agent communication
    LEARN = "learn"  # Update knowledge
    PLAN = "plan"  # Create/modify plan
    DELEGATE = "delegate"  # Assign task to another agent


@dataclass
class Goal:
    """Represents an agent's goal."""

    id: str
    description: str
    priority: float  # 0-1, higher = more important
    deadline: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    status: str = "active"  # "active", "achieved", "failed", "suspended"
    progress: float = 0.0  # 0-1, how close to completion
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Action:
    """Represents an action the agent can take."""

    id: str
    action_type: ActionType
    description: str
    parameters: Dict = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    expected_effects: List[str] = field(default_factory=list)
    cost: float = 1.0  # Resource cost
    duration: float = 1.0  # Expected time


@dataclass
class ActionResult:
    """Result of executing an action."""

    action_id: str
    success: bool
    output: Any
    effects: List[str]  # What changed
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class Plan:
    """A sequence of actions to achieve a goal."""

    id: str
    goal_id: str
    actions: List[Action]
    current_step: int = 0
    status: str = "pending"  # "pending", "executing", "completed", "failed"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    estimated_cost: float = 0.0


@dataclass
class Observation:
    """An observation from the environment."""

    source: str
    content: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    relevance: float = 0.5  # How relevant to current goals


@dataclass
class AgentMemory:
    """Memory system for an agent."""

    short_term: List[Dict] = field(default_factory=list)  # Recent events
    long_term: Dict[str, Any] = field(default_factory=dict)  # Persistent knowledge
    episodic: List[Dict] = field(default_factory=list)  # Experience episodes
    working: Dict[str, Any] = field(default_factory=dict)  # Current task context

    max_short_term: int = 100

    def add_short_term(self, event: Dict):
        """Add event to short-term memory."""
        self.short_term.append({
            **event,
            "timestamp": datetime.now().isoformat()
        })
        # Trim if too long
        if len(self.short_term) > self.max_short_term:
            # Move oldest to episodic
            oldest = self.short_term.pop(0)
            self.episodic.append(oldest)

    def store_long_term(self, key: str, value: Any):
        """Store in long-term memory."""
        self.long_term[key] = {
            "value": value,
            "stored_at": datetime.now().isoformat()
        }

    def recall(self, query: str) -> List[Dict]:
        """Recall relevant memories."""
        relevant = []
        query_lower = query.lower()

        # Search short-term
        for mem in self.short_term[-20:]:  # Recent memories
            if query_lower in str(mem).lower():
                relevant.append(mem)

        # Search long-term
        for key, val in self.long_term.items():
            if query_lower in key.lower() or query_lower in str(val).lower():
                relevant.append({"key": key, **val})

        return relevant[:10]


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass


class CalculatorTool(Tool):
    """Tool for mathematical calculations."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Input: expression (string)"

    def execute(self, expression: str = "") -> Any:
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set("0123456789+-*/().^ ")
            if all(c in allowed_chars for c in expression):
                result = eval(expression.replace("^", "**"))
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": "Invalid characters in expression"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SearchTool(Tool):
    """Tool for searching knowledge base."""

    def __init__(self, knowledge_base: Dict = None):
        self.kb = knowledge_base or {}

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the knowledge base. Input: query (string)"

    def execute(self, query: str = "") -> Any:
        results = []
        query_lower = query.lower()
        for key, value in self.kb.items():
            if query_lower in key.lower() or query_lower in str(value).lower():
                results.append({"key": key, "value": value})
        return {"success": True, "results": results[:5]}


class ReasoningTool(Tool):
    """Tool for logical reasoning."""

    def __init__(self, llm_bridge=None):
        self.llm = llm_bridge

    @property
    def name(self) -> str:
        return "reason"

    @property
    def description(self) -> str:
        return "Perform logical reasoning. Input: premises (list), query (string)"

    def execute(self, premises: List[str] = None, query: str = "") -> Any:
        premises = premises or []
        if not self.llm:
            return {"success": False, "error": "LLM not available for reasoning"}

        prompt = f"""Given these premises:
{chr(10).join(f'- {p}' for p in premises)}

Logically reason about: {query}

Provide step-by-step reasoning and conclusion:"""

        try:
            result = self.llm.generate(prompt)
            return {"success": True, "reasoning": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


class AutonomousAgent:
    """
    An autonomous agent capable of goal-directed behavior.

    Features:
    - Goal management and prioritization
    - Autonomous planning and execution
    - Tool use
    - Self-reflection and improvement
    - Memory management
    """

    def __init__(
        self,
        name: str,
        llm_bridge = None,
        tools: List[Tool] = None,
        storage_path: str = None
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.llm = llm_bridge

        # State
        self.state = AgentState.IDLE
        self.goals: Dict[str, Goal] = {}
        self.active_plan: Optional[Plan] = None
        self.memory = AgentMemory()

        # Tools
        self.tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.tools[tool.name] = tool

        # Add default tools
        self.tools["calculator"] = CalculatorTool()
        self.tools["search"] = SearchTool()

        # History
        self.action_history: List[ActionResult] = []
        self.reflection_log: List[Dict] = []

        # Storage
        self.storage_path = storage_path or f"./agent_{self.id}.json"

        print(f"[+] Agent '{name}' ({self.id}): Autonomous agent initialized")
        print(f"    Tools: {list(self.tools.keys())}")

    def set_goal(
        self,
        description: str,
        priority: float = 0.5,
        deadline: Optional[str] = None
    ) -> Goal:
        """
        Set a new goal for the agent.

        Args:
            description: What to achieve
            priority: How important (0-1)
            deadline: Optional deadline

        Returns:
            Created Goal object
        """
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            description=description,
            priority=priority,
            deadline=deadline
        )

        self.goals[goal.id] = goal
        self.memory.add_short_term({
            "event": "goal_set",
            "goal_id": goal.id,
            "description": description
        })

        print(f"[Agent {self.name}] New goal: {description} (priority: {priority})")
        return goal

    def plan(self, goal_id: str) -> Optional[Plan]:
        """
        Create a plan to achieve a goal.

        Args:
            goal_id: ID of goal to plan for

        Returns:
            Plan if successful
        """
        if goal_id not in self.goals:
            print(f"[!] Goal {goal_id} not found")
            return None

        goal = self.goals[goal_id]
        self.state = AgentState.PLANNING

        print(f"[Agent {self.name}] Planning for: {goal.description}")

        if not self.llm:
            # Simple heuristic planning
            actions = self._heuristic_plan(goal)
        else:
            # LLM-based planning
            actions = self._llm_plan(goal)

        if not actions:
            print(f"[!] Failed to generate plan")
            return None

        plan = Plan(
            id=str(uuid.uuid4())[:8],
            goal_id=goal_id,
            actions=actions,
            estimated_cost=sum(a.cost for a in actions)
        )

        self.active_plan = plan
        self.memory.add_short_term({
            "event": "plan_created",
            "plan_id": plan.id,
            "goal_id": goal_id,
            "num_actions": len(actions)
        })

        print(f"[Agent {self.name}] Plan created with {len(actions)} actions")
        return plan

    def _heuristic_plan(self, goal: Goal) -> List[Action]:
        """Generate plan using heuristics."""
        actions = []

        # Observe first
        actions.append(Action(
            id=str(uuid.uuid4())[:8],
            action_type=ActionType.OBSERVE,
            description=f"Gather information about: {goal.description}",
            expected_effects=["information_gathered"]
        ))

        # Think about approach
        actions.append(Action(
            id=str(uuid.uuid4())[:8],
            action_type=ActionType.THINK,
            description=f"Reason about how to achieve: {goal.description}",
            expected_effects=["approach_determined"]
        ))

        # Execute main action
        actions.append(Action(
            id=str(uuid.uuid4())[:8],
            action_type=ActionType.ACT,
            description=f"Execute plan for: {goal.description}",
            expected_effects=["goal_progress"]
        ))

        return actions

    def _llm_plan(self, goal: Goal) -> List[Action]:
        """Generate plan using LLM."""
        prompt = f"""You are an autonomous agent creating a plan.

Goal: {goal.description}
Priority: {goal.priority}
Available tools: {', '.join(self.tools.keys())}

Create a step-by-step plan to achieve this goal.
Each step should be one of: THINK, OBSERVE, ACT, LEARN, COMMUNICATE

Respond in JSON format:
{{
    "steps": [
        {{"type": "OBSERVE", "description": "what to observe", "tool": "optional tool name"}},
        {{"type": "THINK", "description": "what to reason about"}},
        {{"type": "ACT", "description": "what action to take", "tool": "tool name"}}
    ]
}}
"""

        try:
            response = self.llm.generate(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                actions = []
                for step in data.get("steps", []):
                    action = Action(
                        id=str(uuid.uuid4())[:8],
                        action_type=ActionType[step.get("type", "ACT")],
                        description=step.get("description", ""),
                        parameters={"tool": step.get("tool")} if step.get("tool") else {},
                        expected_effects=[f"{step.get('type', 'ACT')}_completed"]
                    )
                    actions.append(action)
                return actions
        except Exception as e:
            print(f"[!] LLM planning failed: {e}")

        return self._heuristic_plan(goal)

    def execute_plan(self) -> bool:
        """
        Execute the current plan.

        Returns:
            True if plan completed successfully
        """
        if not self.active_plan:
            print("[!] No active plan to execute")
            return False

        self.state = AgentState.EXECUTING
        plan = self.active_plan

        print(f"[Agent {self.name}] Executing plan {plan.id}")

        while plan.current_step < len(plan.actions):
            action = plan.actions[plan.current_step]
            print(f"    Step {plan.current_step + 1}/{len(plan.actions)}: {action.description[:50]}...")

            # Execute action
            result = self._execute_action(action)
            self.action_history.append(result)

            if not result.success:
                print(f"    [!] Action failed: {result.errors}")
                # Try to recover
                if not self._recover_from_failure(action, result):
                    plan.status = "failed"
                    self.state = AgentState.FAILED
                    return False

            plan.current_step += 1

            # Update goal progress
            goal = self.goals.get(plan.goal_id)
            if goal:
                goal.progress = plan.current_step / len(plan.actions)

        plan.status = "completed"
        self.state = AgentState.COMPLETED

        # Mark goal as achieved
        goal = self.goals.get(plan.goal_id)
        if goal:
            goal.status = "achieved"
            goal.progress = 1.0

        print(f"[Agent {self.name}] Plan completed successfully")
        return True

    def _execute_action(self, action: Action) -> ActionResult:
        """Execute a single action."""
        start_time = time.time()
        output = None
        effects = []
        errors = []
        success = True

        try:
            if action.action_type == ActionType.THINK:
                output = self._think(action.description)
                effects.append("reasoning_completed")

            elif action.action_type == ActionType.OBSERVE:
                output = self._observe(action.description)
                effects.append("observation_recorded")

            elif action.action_type == ActionType.ACT:
                tool_name = action.parameters.get("tool")
                if tool_name and tool_name in self.tools:
                    output = self.tools[tool_name].execute(**action.parameters)
                else:
                    output = self._generic_act(action.description)
                effects.append("action_performed")

            elif action.action_type == ActionType.LEARN:
                output = self._learn(action.description)
                effects.append("knowledge_updated")

            elif action.action_type == ActionType.COMMUNICATE:
                output = self._communicate(action.description, action.parameters)
                effects.append("message_sent")

        except Exception as e:
            success = False
            errors.append(str(e))

        return ActionResult(
            action_id=action.id,
            success=success,
            output=output,
            effects=effects,
            errors=errors,
            execution_time=time.time() - start_time
        )

    def _think(self, about: str) -> str:
        """Perform internal reasoning."""
        if not self.llm:
            return f"Thought about: {about}"

        # Recall relevant memories
        memories = self.memory.recall(about)

        prompt = f"""Think step-by-step about: {about}

Relevant memories:
{json.dumps(memories[:3], default=str)}

Reasoning:"""

        return self.llm.generate(prompt).strip()

    def _observe(self, what: str) -> Dict:
        """Gather information from environment."""
        observation = Observation(
            source="environment",
            content=f"Observed: {what}",
            relevance=0.7
        )

        self.memory.add_short_term({
            "event": "observation",
            "content": observation.content
        })

        return {"observation": observation.content}

    def _generic_act(self, description: str) -> Dict:
        """Perform a generic action."""
        return {"action": description, "status": "completed"}

    def _learn(self, what: str) -> Dict:
        """Update knowledge based on experience."""
        self.memory.store_long_term(
            f"learned_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            what
        )
        return {"learned": what}

    def _communicate(self, message: str, params: Dict) -> Dict:
        """Send communication (to other agents, user, etc.)."""
        return {"message": message, "sent_to": params.get("recipient", "broadcast")}

    def _recover_from_failure(self, action: Action, result: ActionResult) -> bool:
        """Attempt to recover from a failed action."""
        print(f"[Agent {self.name}] Attempting recovery...")

        # Simple retry logic
        if len(result.errors) > 0 and "timeout" in str(result.errors[0]).lower():
            # Retry once for timeouts
            retry_result = self._execute_action(action)
            return retry_result.success

        # Try alternative approach using LLM
        if self.llm:
            prompt = f"""An action failed:
Action: {action.description}
Error: {result.errors}

Suggest an alternative approach:"""

            try:
                alternative = self.llm.generate(prompt).strip()
                print(f"    Alternative: {alternative[:100]}...")
                return True  # Assume alternative might work
            except Exception:
                pass

        return False

    def reflect(self) -> Dict:
        """
        Self-reflect on recent actions and performance.

        Returns:
            Reflection insights
        """
        self.state = AgentState.REFLECTING

        print(f"[Agent {self.name}] Reflecting on performance...")

        # Analyze recent actions
        recent_actions = self.action_history[-10:]
        success_rate = sum(1 for a in recent_actions if a.success) / max(len(recent_actions), 1)
        avg_time = sum(a.execution_time for a in recent_actions) / max(len(recent_actions), 1)

        # Goals analysis
        achieved_goals = [g for g in self.goals.values() if g.status == "achieved"]
        failed_goals = [g for g in self.goals.values() if g.status == "failed"]

        reflection = {
            "timestamp": datetime.now().isoformat(),
            "action_success_rate": success_rate,
            "avg_action_time": avg_time,
            "goals_achieved": len(achieved_goals),
            "goals_failed": len(failed_goals),
            "insights": []
        }

        # Generate insights
        if success_rate < 0.5:
            reflection["insights"].append("Low success rate - consider simpler approaches")
        if avg_time > 5.0:
            reflection["insights"].append("Slow execution - optimize action sequence")
        if len(failed_goals) > len(achieved_goals):
            reflection["insights"].append("More failures than successes - adjust goal difficulty")

        # Use LLM for deeper reflection
        if self.llm:
            prompt = f"""Reflect on this agent's performance:

Success rate: {success_rate:.2%}
Goals achieved: {len(achieved_goals)}
Goals failed: {len(failed_goals)}

Recent action results:
{json.dumps([{"desc": self.active_plan.actions[i].description if self.active_plan and i < len(self.active_plan.actions) else "N/A", "success": a.success} for i, a in enumerate(recent_actions[-5:])], indent=2)}

What could be improved? Provide 2-3 specific suggestions:"""

            try:
                llm_reflection = self.llm.generate(prompt)
                reflection["llm_insights"] = llm_reflection.strip()
            except Exception:
                pass

        self.reflection_log.append(reflection)
        self.memory.add_short_term({
            "event": "reflection",
            "success_rate": success_rate,
            "insights": reflection["insights"]
        })

        self.state = AgentState.IDLE
        return reflection

    def run_autonomous_loop(
        self,
        max_iterations: int = 10,
        goal_description: str = None
    ) -> Dict:
        """
        Run the agent autonomously until goals are achieved or max iterations.

        Args:
            max_iterations: Maximum number of plan-execute cycles
            goal_description: Optional new goal to set

        Returns:
            Summary of autonomous run
        """
        print(f"[Agent {self.name}] Starting autonomous loop...")

        if goal_description:
            self.set_goal(goal_description)

        iterations = 0
        results = {
            "iterations": 0,
            "goals_achieved": [],
            "goals_failed": [],
            "total_actions": 0
        }

        while iterations < max_iterations:
            iterations += 1
            print(f"\n--- Iteration {iterations}/{max_iterations} ---")

            # Get highest priority active goal
            active_goals = [g for g in self.goals.values() if g.status == "active"]
            if not active_goals:
                print("[Agent] All goals completed or no active goals")
                break

            active_goals.sort(key=lambda g: g.priority, reverse=True)
            current_goal = active_goals[0]

            # Plan
            plan = self.plan(current_goal.id)
            if not plan:
                current_goal.status = "failed"
                results["goals_failed"].append(current_goal.description)
                continue

            # Execute
            success = self.execute_plan()
            results["total_actions"] += len(plan.actions)

            if success:
                results["goals_achieved"].append(current_goal.description)
            else:
                results["goals_failed"].append(current_goal.description)

            # Reflect periodically
            if iterations % 3 == 0:
                self.reflect()

        results["iterations"] = iterations

        print(f"\n[Agent {self.name}] Autonomous loop completed")
        print(f"    Iterations: {iterations}")
        print(f"    Goals achieved: {len(results['goals_achieved'])}")
        print(f"    Goals failed: {len(results['goals_failed'])}")

        return results

    def save(self):
        """Save agent state to disk."""
        try:
            data = {
                "id": self.id,
                "name": self.name,
                "state": self.state.value,
                "goals": {
                    gid: {
                        "id": g.id,
                        "description": g.description,
                        "priority": g.priority,
                        "status": g.status,
                        "progress": g.progress
                    }
                    for gid, g in self.goals.items()
                },
                "memory": {
                    "short_term": self.memory.short_term[-50:],  # Last 50
                    "long_term": self.memory.long_term
                },
                "action_count": len(self.action_history),
                "reflection_count": len(self.reflection_log)
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[+] Agent state saved to {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to save agent: {e}")

    def load(self):
        """Load agent state from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.id = data.get("id", self.id)
            self.state = AgentState(data.get("state", "idle"))

            # Load goals
            for gid, g_data in data.get("goals", {}).items():
                self.goals[gid] = Goal(
                    id=g_data["id"],
                    description=g_data["description"],
                    priority=g_data["priority"],
                    status=g_data["status"],
                    progress=g_data["progress"]
                )

            # Load memory
            mem_data = data.get("memory", {})
            self.memory.short_term = mem_data.get("short_term", [])
            self.memory.long_term = mem_data.get("long_term", {})

            print(f"[+] Agent state loaded from {self.storage_path}")
        except Exception as e:
            print(f"[!] Failed to load agent: {e}")

    def get_status(self) -> Dict:
        """Get current agent status."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "active_goals": len([g for g in self.goals.values() if g.status == "active"]),
            "achieved_goals": len([g for g in self.goals.values() if g.status == "achieved"]),
            "total_actions": len(self.action_history),
            "tools_available": list(self.tools.keys()),
            "memory_size": {
                "short_term": len(self.memory.short_term),
                "long_term": len(self.memory.long_term)
            }
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        status = self.get_status()
        lines = [
            f"Autonomous Agent: {self.name} ({self.id})",
            f"  State: {status['state']}",
            f"  Active goals: {status['active_goals']}",
            f"  Achieved goals: {status['achieved_goals']}",
            f"  Total actions: {status['total_actions']}",
            f"  Tools: {', '.join(status['tools_available'])}"
        ]

        if self.reflection_log:
            last_reflection = self.reflection_log[-1]
            lines.append(f"\nLast reflection:")
            lines.append(f"  Success rate: {last_reflection.get('action_success_rate', 0):.2%}")
            if last_reflection.get("insights"):
                lines.append(f"  Insights: {last_reflection['insights'][0]}")

        return "\n".join(lines)


class MultiAgentSystem:
    """
    System for coordinating multiple autonomous agents.

    Enables:
    - Agent spawning and management
    - Inter-agent communication
    - Task delegation and collaboration
    - Collective intelligence
    """

    def __init__(self, llm_bridge = None):
        self.llm = llm_bridge
        self.agents: Dict[str, AutonomousAgent] = {}
        self.message_queue: List[Dict] = []
        self.shared_memory: Dict[str, Any] = {}

        print("[+] Multi-Agent System initialized")

    def spawn_agent(
        self,
        name: str,
        tools: List[Tool] = None
    ) -> AutonomousAgent:
        """
        Spawn a new autonomous agent.

        Args:
            name: Agent name
            tools: Optional tools for the agent

        Returns:
            Created agent
        """
        agent = AutonomousAgent(
            name=name,
            llm_bridge=self.llm,
            tools=tools
        )

        self.agents[agent.id] = agent
        print(f"[MAS] Spawned agent: {name} ({agent.id})")

        return agent

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        message_type: str = "info"
    ):
        """
        Send message between agents.

        Args:
            from_agent: Sender agent ID
            to_agent: Receiver agent ID (or "broadcast")
            message: Message content
            message_type: "info", "request", "response", "alert"
        """
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat()
        }

        self.message_queue.append(msg)

        # Deliver to agent memory
        if to_agent == "broadcast":
            for agent in self.agents.values():
                agent.memory.add_short_term({"event": "message_received", **msg})
        elif to_agent in self.agents:
            self.agents[to_agent].memory.add_short_term({"event": "message_received", **msg})

    def delegate_task(
        self,
        task: str,
        from_agent: str,
        to_agent: str,
        priority: float = 0.5
    ) -> Optional[Goal]:
        """
        Delegate a task from one agent to another.

        Args:
            task: Task description
            from_agent: Delegating agent ID
            to_agent: Receiving agent ID
            priority: Task priority

        Returns:
            Created goal for the receiving agent
        """
        if to_agent not in self.agents:
            print(f"[!] Target agent {to_agent} not found")
            return None

        target = self.agents[to_agent]
        goal = target.set_goal(task, priority)

        # Notify both agents
        self.send_message(from_agent, to_agent, f"Delegated task: {task}", "request")

        return goal

    def run_collaborative(
        self,
        main_goal: str,
        max_iterations: int = 10
    ) -> Dict:
        """
        Run all agents collaboratively toward a main goal.

        Args:
            main_goal: Shared goal description
            max_iterations: Max iterations

        Returns:
            Collaborative run results
        """
        print(f"[MAS] Starting collaborative run: {main_goal}")

        if not self.agents:
            print("[!] No agents available")
            return {"error": "no_agents"}

        # Assign goal to all agents
        for agent in self.agents.values():
            agent.set_goal(f"Contribute to: {main_goal}", priority=0.8)

        # Run agents
        results = {
            "agents": {},
            "messages_exchanged": 0
        }

        for agent in self.agents.values():
            agent_result = agent.run_autonomous_loop(max_iterations=max_iterations)
            results["agents"][agent.name] = agent_result

        results["messages_exchanged"] = len(self.message_queue)

        print(f"[MAS] Collaborative run completed")
        return results

    def get_collective_status(self) -> Dict:
        """Get status of all agents."""
        return {
            "total_agents": len(self.agents),
            "agents": {
                agent.name: agent.get_status()
                for agent in self.agents.values()
            },
            "messages_in_queue": len(self.message_queue),
            "shared_memory_keys": len(self.shared_memory)
        }

    def summarize(self) -> str:
        """Get summary of multi-agent system."""
        status = self.get_collective_status()
        lines = [
            "Multi-Agent System Status:",
            f"  Total agents: {status['total_agents']}",
            f"  Messages queued: {status['messages_in_queue']}"
        ]

        for name, agent_status in status["agents"].items():
            lines.append(f"\n  Agent '{name}':")
            lines.append(f"    State: {agent_status['state']}")
            lines.append(f"    Goals: {agent_status['active_goals']} active, {agent_status['achieved_goals']} achieved")

        return "\n".join(lines)
