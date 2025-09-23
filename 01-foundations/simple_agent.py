"""
Simple Agent Implementation
A basic example of an agent with reasoning and action capabilities
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ActionType(Enum):
    SEARCH = "search"
    CALCULATE = "calculate"
    REMEMBER = "remember"
    RESPOND = "respond"


@dataclass
class Observation:
    """Represents an observation from the environment"""
    content: str
    source: str
    timestamp: float


@dataclass
class Action:
    """Represents an action the agent can take"""
    type: ActionType
    parameters: Dict[str, Any]
    reasoning: str


class SimpleAgent:
    """
    A basic agent that follows the ReAct pattern:
    - Reason about the current situation
    - Act based on reasoning
    - Observe the results
    - Repeat until goal is achieved
    """

    def __init__(self, name: str):
        self.name = name
        self.memory: List[str] = []
        self.observations: List[Observation] = []
        self.goal: Optional[str] = None
        self.completed: bool = False

    def set_goal(self, goal: str):
        """Set the agent's current goal"""
        self.goal = goal
        self.completed = False
        self.memory.append(f"Goal set: {goal}")

    def think(self, current_situation: str) -> str:
        """
        Reasoning step - analyze current situation and decide what to do
        """
        thoughts = [
            f"Current goal: {self.goal}",
            f"Current situation: {current_situation}",
            f"Recent observations: {len(self.observations)} total",
        ]

        if self.observations:
            thoughts.append(f"Latest observation: {self.observations[-1].content}")

        # Simple reasoning logic
        if "search" in current_situation.lower() or "find" in current_situation.lower():
            reasoning = "I need to search for information"
        elif "calculate" in current_situation.lower() or "compute" in current_situation.lower():
            reasoning = "I need to perform a calculation"
        elif self.goal and "completed" in current_situation.lower():
            reasoning = "The goal appears to be completed"
            self.completed = True
        else:
            reasoning = "I need to gather more information"

        thought_process = "\\n".join(thoughts) + f"\\nReasoning: {reasoning}"
        self.memory.append(f"Thought: {reasoning}")

        return thought_process

    def act(self, reasoning: str) -> Action:
        """
        Action step - decide what action to take based on reasoning
        """
        if "search" in reasoning.lower():
            return Action(
                type=ActionType.SEARCH,
                parameters={"query": "relevant information"},
                reasoning=reasoning
            )
        elif "calculate" in reasoning.lower():
            return Action(
                type=ActionType.CALCULATE,
                parameters={"expression": "mathematical expression"},
                reasoning=reasoning
            )
        elif "completed" in reasoning.lower():
            return Action(
                type=ActionType.RESPOND,
                parameters={"message": f"Goal '{self.goal}' has been completed"},
                reasoning=reasoning
            )
        else:
            return Action(
                type=ActionType.REMEMBER,
                parameters={"information": reasoning},
                reasoning=reasoning
            )

    def observe(self, result: str, source: str = "environment") -> Observation:
        """
        Observation step - process the results of an action
        """
        import time
        observation = Observation(
            content=result,
            source=source,
            timestamp=time.time()
        )
        self.observations.append(observation)
        self.memory.append(f"Observed: {result}")
        return observation

    def run_cycle(self, current_situation: str) -> Dict[str, Any]:
        """
        Run one complete cycle of the ReAct pattern
        """
        # Think
        thoughts = self.think(current_situation)

        # Act
        action = self.act(thoughts)

        # Simulate action execution and observation
        if action.type == ActionType.SEARCH:
            result = f"Search results for: {action.parameters.get('query', 'unknown')}"
        elif action.type == ActionType.CALCULATE:
            result = f"Calculation result: {action.parameters.get('expression', 'unknown')}"
        elif action.type == ActionType.RESPOND:
            result = action.parameters.get('message', 'Response generated')
        else:
            result = f"Information remembered: {action.parameters.get('information', 'unknown')}"

        # Observe
        observation = self.observe(result, "action_result")

        return {
            "thoughts": thoughts,
            "action": action,
            "observation": observation,
            "completed": self.completed
        }

    def get_memory_summary(self) -> str:
        """Get a summary of the agent's memory"""
        return "\\n".join(self.memory[-10:])  # Last 10 entries


# Example usage and demonstration
def demonstrate_simple_agent():
    """Demonstrate the simple agent in action"""

    print("=== Simple Agent Demonstration ===\\n")

    # Create an agent
    agent = SimpleAgent("Demo Agent")

    # Set a goal
    agent.set_goal("Find information about artificial intelligence")

    # Run several cycles
    situations = [
        "I need to search for information about AI",
        "I found some search results about AI history",
        "I need to calculate the growth rate of AI research",
        "The research shows exponential growth in AI papers",
        "Goal completed successfully"
    ]

    for i, situation in enumerate(situations, 1):
        print(f"--- Cycle {i} ---")
        print(f"Situation: {situation}\\n")

        cycle_result = agent.run_cycle(situation)

        print(f"Thoughts:\\n{cycle_result['thoughts']}\\n")
        print(f"Action: {cycle_result['action'].type.value}")
        print(f"Action Reasoning: {cycle_result['action'].reasoning}\\n")
        print(f"Observation: {cycle_result['observation'].content}\\n")

        if cycle_result['completed']:
            print("🎉 Goal completed!\\n")
            break

        print("-" * 50 + "\\n")

    print("=== Agent Memory Summary ===")
    print(agent.get_memory_summary())


if __name__ == "__main__":
    demonstrate_simple_agent()