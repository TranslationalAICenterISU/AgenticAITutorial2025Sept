"""
Exercise 1: Agent Behavior Analysis
Analyze different agent behaviors and identify key characteristics

OBJECTIVE: Understand what makes AI systems "agentic" by examining different behaviors
DIFFICULTY: Beginner
TIME: 15-20 minutes
"""

def analyze_agent_behavior():
    """
    Analyze the following scenarios and determine which exhibit agentic behavior.

    For each scenario, answer:
    1. Is this agentic behavior? (Yes/No)
    2. What characteristics make it agentic or non-agentic?
    3. What type of agency is demonstrated?
    """

    scenarios = [
        {
            "id": 1,
            "description": "A chatbot that responds to user questions with pre-programmed answers",
            "behavior": "User asks 'What is Python?' → Bot responds 'Python is a programming language'",
            "analysis": {
                "is_agentic": False,
                "reasoning": "Simple stimulus-response pattern, no goal-directed behavior",
                "agency_type": "None - reactive only"
            }
        },
        {
            "id": 2,
            "description": "An AI that decides to use a calculator when asked a math question",
            "behavior": "User asks 'What is 25 * 47?' → AI thinks 'I need to calculate this' → AI uses calculator tool → AI responds with result",
            "analysis": {
                "is_agentic": True,
                "reasoning": "Shows goal-directed behavior, tool selection, and multi-step reasoning",
                "agency_type": "Goal-based agent with tool usage"
            }
        },
        {
            "id": 3,
            "description": "A system that automatically sends emails based on calendar events",
            "behavior": "Calendar event triggers → System checks attendees → System sends reminder emails",
            "analysis": {
                "is_agentic": "Partial",
                "reasoning": "Automated but lacks adaptive decision-making or learning",
                "agency_type": "Simple reflex agent"
            }
        },
        {
            "id": 4,
            "description": "An AI research assistant that breaks down complex queries into subtasks",
            "behavior": "User asks 'Research climate change impacts' → AI plans: 1) Search scientific papers 2) Analyze data 3) Summarize findings → AI executes plan → AI adapts approach based on results",
            "analysis": {
                "is_agentic": True,
                "reasoning": "Demonstrates planning, execution, adaptation, and goal pursuit",
                "agency_type": "Goal-based learning agent"
            }
        },
        {
            "id": 5,
            "description": "A translation service that converts text from one language to another",
            "behavior": "Input text in English → System translates to Spanish → Output Spanish text",
            "analysis": {
                "is_agentic": False,
                "reasoning": "Direct input-output transformation without goals or adaptation",
                "agency_type": "None - pure function"
            }
        }
    ]

    print("🔍 Agent Behavior Analysis Exercise")
    print("="*50)

    # Interactive analysis
    for scenario in scenarios:
        print(f"\\n📋 Scenario {scenario['id']}: {scenario['description']}")
        print(f"Behavior: {scenario['behavior']}")
        print()

        # Get user input
        user_agentic = input("Is this agentic behavior? (yes/no/partial): ").lower().strip()
        user_reasoning = input("Why? (your reasoning): ").strip()
        user_type = input("What type of agency (if any)?: ").strip()

        print("\\n🤖 Expert Analysis:")
        print(f"   Agentic: {scenario['analysis']['is_agentic']}")
        print(f"   Reasoning: {scenario['analysis']['reasoning']}")
        print(f"   Agency Type: {scenario['analysis']['agency_type']}")

        # Compare answers
        if str(scenario['analysis']['is_agentic']).lower() in user_agentic:
            print("   ✅ Your assessment matches the expert analysis!")
        else:
            print("   🔄 Different assessment - consider the expert reasoning")

        print("-" * 60)

    print("\\n🎯 Key Takeaways:")
    print("1. Agentic behavior involves goal-directed action")
    print("2. Agents make decisions and adapt their approach")
    print("3. Tool usage and planning indicate higher agency")
    print("4. Simple input-output transformations are not agentic")
    print("5. Learning and adaptation enhance agency")


def design_agent_architecture():
    """
    Exercise: Design your own agent architecture
    """

    print("\\n🏗️ Agent Architecture Design Exercise")
    print("="*45)

    print("\\nDesign an agent for the following task:")
    print("📧 EMAIL MANAGEMENT AGENT")
    print("Goal: Help users manage their email inbox efficiently")

    architecture_template = {
        "perception": "What inputs does the agent receive?",
        "reasoning": "How does the agent make decisions?",
        "action": "What actions can the agent take?",
        "learning": "How does the agent improve over time?",
        "memory": "What information does the agent store?",
        "tools": "What external tools does the agent use?"
    }

    user_architecture = {}

    print("\\nAnswer the following questions to design your agent:")
    for component, question in architecture_template.items():
        print(f"\\n{component.upper()}: {question}")
        user_answer = input("Your answer: ").strip()
        user_architecture[component] = user_answer

    print("\\n🏗️ YOUR AGENT ARCHITECTURE:")
    print("-" * 35)
    for component, answer in user_architecture.items():
        print(f"{component.upper()}: {answer}")

    # Expert example
    print("\\n🤖 EXPERT EXAMPLE ARCHITECTURE:")
    print("-" * 37)
    expert_example = {
        "perception": "Email content, sender info, user preferences, calendar events",
        "reasoning": "Priority scoring, category classification, response urgency assessment",
        "action": "Sort, label, archive, draft responses, schedule follow-ups",
        "learning": "User feedback on actions, pattern recognition from user behavior",
        "memory": "User preferences, common email patterns, successful action history",
        "tools": "Email API, calendar API, text classifier, template generator"
    }

    for component, example in expert_example.items():
        print(f"{component.upper()}: {example}")

    print("\\n💡 Reflection Questions:")
    print("1. How is your design similar to or different from the expert example?")
    print("2. What components are most critical for agent effectiveness?")
    print("3. How would you handle edge cases or errors?")
    print("4. What ethical considerations should this agent have?")


def react_pattern_implementation():
    """
    Exercise: Implement a simple ReAct pattern
    """

    print("\\n🔄 ReAct Pattern Implementation Exercise")
    print("="*42)

    class SimpleReActAgent:
        def __init__(self):
            self.goal = None
            self.observations = []
            self.actions_taken = []
            self.current_step = 0
            self.max_steps = 5

        def set_goal(self, goal):
            self.goal = goal
            print(f"🎯 Goal set: {goal}")

        def think(self, situation):
            """Reasoning step"""
            thought = f"Step {self.current_step + 1}: Analyzing situation: {situation}"
            print(f"💭 Thought: {thought}")

            # Simple reasoning logic
            if "calculate" in situation.lower():
                return "I need to perform a calculation"
            elif "search" in situation.lower():
                return "I need to search for information"
            elif "complete" in situation.lower():
                return "The task appears to be complete"
            else:
                return "I need to gather more information"

        def act(self, reasoning):
            """Action step"""
            if "calculation" in reasoning.lower():
                action = "Use calculator tool"
            elif "search" in reasoning.lower():
                action = "Use search tool"
            elif "complete" in reasoning.lower():
                action = "Provide final answer"
            else:
                action = "Ask for clarification"

            print(f"🎬 Action: {action}")
            self.actions_taken.append(action)
            return action

        def observe(self, action_result):
            """Observation step"""
            observation = f"Result of '{action_result}': Mock result for demonstration"
            print(f"👁️ Observation: {observation}")
            self.observations.append(observation)
            return observation

        def run_cycle(self, situation):
            """Run one complete ReAct cycle"""
            if self.current_step >= self.max_steps:
                print("🛑 Maximum steps reached")
                return False

            print(f"\\n--- ReAct Cycle {self.current_step + 1} ---")

            # Think
            reasoning = self.think(situation)

            # Act
            action = self.act(reasoning)

            # Observe
            observation = self.observe(action)

            self.current_step += 1

            # Check if goal is achieved (simple check)
            if "complete" in reasoning.lower() or "final answer" in action.lower():
                print("✅ Goal achieved!")
                return True

            return self.current_step < self.max_steps

    # Exercise implementation
    print("\\nImplement the ReAct pattern for this scenario:")
    print("🧮 Scenario: User asks 'What is the square root of 144 plus 25?'")

    agent = SimpleReActAgent()
    agent.set_goal("Calculate square root of 144 plus 25")

    # Simulate ReAct cycles
    situations = [
        "User asks for square root of 144 plus 25",
        "Need to calculate square root of 144",
        "Need to add 25 to the result",
        "Task is complete with final answer"
    ]

    for situation in situations:
        if not agent.run_cycle(situation):
            break

    print("\\n📊 Exercise Summary:")
    print(f"Actions taken: {agent.actions_taken}")
    print(f"Observations made: {len(agent.observations)}")
    print(f"Steps completed: {agent.current_step}")

    print("\\n🤔 Reflection Questions:")
    print("1. How does the ReAct pattern help with complex tasks?")
    print("2. What happens if the observation doesn't match expectations?")
    print("3. How could you improve the reasoning process?")
    print("4. When might this pattern be insufficient?")


def main():
    """Run all exercises"""

    print("🎓 FOUNDATIONS MODULE - EXERCISES")
    print("="*40)
    print("\\nWelcome to the Agentic AI Foundations exercises!")
    print("These exercises will help you understand core concepts.")
    print("\\nExercises available:")
    print("1. Agent Behavior Analysis")
    print("2. Agent Architecture Design")
    print("3. ReAct Pattern Implementation")

    while True:
        print("\\n" + "="*40)
        choice = input("\\nChoose exercise (1-3) or 'q' to quit: ").strip().lower()

        if choice == '1':
            analyze_agent_behavior()
        elif choice == '2':
            design_agent_architecture()
        elif choice == '3':
            react_pattern_implementation()
        elif choice == 'q' or choice == 'quit':
            print("\\n👋 Thanks for practicing! Continue to Module 2 when ready.")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 'q'")


if __name__ == "__main__":
    main()