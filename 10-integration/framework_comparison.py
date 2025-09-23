"""
Framework Comparison - Side-by-Side Analysis of Agentic AI Frameworks

This module provides comprehensive comparisons between different agentic AI frameworks,
helping you understand when to use each framework and their relative strengths/weaknesses.

Frameworks compared:
- OpenAI API (direct)
- Anthropic API (direct)
- LangChain
- LangGraph
- CrewAI
- DSPy
- SmolAgents
- Local Models (Ollama)

Each framework is evaluated on the same tasks to demonstrate differences.
"""

import os
import sys
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

# Import libraries (with fallbacks for missing ones)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import langchain
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    langchain = None

try:
    import langgraph
    from langgraph.graph import Graph, StateGraph
except ImportError:
    langgraph = None

try:
    import crewai
    from crewai import Agent, Task, Crew
except ImportError:
    crewai = None

try:
    import dspy
except ImportError:
    dspy = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class ComparisonResult:
    """Results from framework comparison"""
    framework: str
    task_name: str
    complexity: TaskComplexity
    execution_time: float
    success: bool
    output: str
    error_message: Optional[str] = None
    memory_usage: Optional[int] = None
    token_count: Optional[int] = None
    cost_estimate: Optional[float] = None


@dataclass
class FrameworkCapabilities:
    """Framework capability assessment"""
    name: str
    ease_of_use: int  # 1-10 scale
    flexibility: int
    performance: int
    documentation: int
    community_support: int
    production_ready: int
    learning_curve: int  # 1-10, lower is easier
    use_cases: List[str]
    pros: List[str]
    cons: List[str]


class FrameworkComparator:
    """Compares different agentic AI frameworks on standardized tasks"""

    def __init__(self):
        self.results: List[ComparisonResult] = []
        self.frameworks = self._initialize_frameworks()

    def _initialize_frameworks(self) -> Dict[str, FrameworkCapabilities]:
        """Initialize framework capability assessments"""
        return {
            "openai_direct": FrameworkCapabilities(
                name="OpenAI API (Direct)",
                ease_of_use=8,
                flexibility=9,
                performance=9,
                documentation=9,
                community_support=10,
                production_ready=10,
                learning_curve=3,
                use_cases=[
                    "Simple chatbots", "Content generation", "Code assistance",
                    "Question answering", "Text analysis", "Function calling"
                ],
                pros=[
                    "Maximum control and flexibility",
                    "Latest model access",
                    "Direct API calls - minimal overhead",
                    "Excellent documentation",
                    "Reliable and fast"
                ],
                cons=[
                    "Requires more boilerplate code",
                    "No built-in agent patterns",
                    "Manual conversation management",
                    "Limited built-in tools"
                ]
            ),
            "anthropic_direct": FrameworkCapabilities(
                name="Anthropic API (Direct)",
                ease_of_use=8,
                flexibility=9,
                performance=9,
                documentation=8,
                community_support=7,
                production_ready=9,
                learning_curve=3,
                use_cases=[
                    "Safety-critical applications", "Long-form content", "Analysis tasks",
                    "Constitutional AI", "Reasoning-heavy tasks"
                ],
                pros=[
                    "Excellent safety features",
                    "Strong reasoning capabilities",
                    "Good at following instructions",
                    "Transparent about limitations",
                    "Constitutional AI approach"
                ],
                cons=[
                    "Smaller ecosystem than OpenAI",
                    "Fewer third-party integrations",
                    "More expensive for some tasks",
                    "Limited function calling"
                ]
            ),
            "langchain": FrameworkCapabilities(
                name="LangChain",
                ease_of_use=7,
                flexibility=10,
                performance=7,
                documentation=8,
                community_support=9,
                production_ready=7,
                learning_curve=5,
                use_cases=[
                    "Rapid prototyping", "Tool integration", "Document processing",
                    "RAG systems", "Agent workflows", "Multi-modal applications"
                ],
                pros=[
                    "Huge ecosystem of tools",
                    "Rapid development",
                    "Extensive integrations",
                    "Active community",
                    "Great for experimentation"
                ],
                cons=[
                    "Can be complex for simple tasks",
                    "Sometimes unstable APIs",
                    "Performance overhead",
                    "Steep learning curve for advanced features"
                ]
            ),
            "langgraph": FrameworkCapabilities(
                name="LangGraph",
                ease_of_use=6,
                flexibility=9,
                performance=8,
                documentation=7,
                community_support=7,
                production_ready=8,
                learning_curve=7,
                use_cases=[
                    "Complex workflows", "Multi-step processes", "Stateful applications",
                    "Human-in-the-loop", "Conditional logic", "Multi-agent orchestration"
                ],
                pros=[
                    "Excellent for complex workflows",
                    "State management built-in",
                    "Visual workflow representation",
                    "Supports cycles and conditions",
                    "Human-in-the-loop patterns"
                ],
                cons=[
                    "High learning curve",
                    "Overkill for simple tasks",
                    "Complex mental model",
                    "Relatively new framework"
                ]
            ),
            "crewai": FrameworkCapabilities(
                name="CrewAI",
                ease_of_use=7,
                flexibility=8,
                performance=7,
                documentation=7,
                community_support=6,
                production_ready=6,
                learning_curve=6,
                use_cases=[
                    "Multi-agent collaboration", "Team-based tasks", "Content creation",
                    "Research projects", "Workflow automation", "Role-playing scenarios"
                ],
                pros=[
                    "Great for multi-agent scenarios",
                    "Intuitive role-based design",
                    "Good collaboration patterns",
                    "Built-in delegation",
                    "Easy to conceptualize"
                ],
                cons=[
                    "Limited to team-based scenarios",
                    "Can be resource intensive",
                    "Less mature ecosystem",
                    "May be overkill for single agents"
                ]
            ),
            "dspy": FrameworkCapabilities(
                name="DSPy",
                ease_of_use=5,
                flexibility=8,
                performance=9,
                documentation=6,
                community_support=5,
                production_ready=7,
                learning_curve=8,
                use_cases=[
                    "Performance optimization", "Research applications", "NLP tasks",
                    "Prompt optimization", "Few-shot learning", "Systematic improvement"
                ],
                pros=[
                    "Automatic optimization",
                    "Excellent performance",
                    "Principled approach",
                    "Research-grade quality",
                    "Systematic prompt engineering"
                ],
                cons=[
                    "Steep learning curve",
                    "Complex mental model",
                    "Less intuitive for beginners",
                    "Requires training data"
                ]
            ),
            "smolagents": FrameworkCapabilities(
                name="SmolAgents",
                ease_of_use=9,
                flexibility=7,
                performance=8,
                documentation=6,
                community_support=4,
                production_ready=6,
                learning_curve=4,
                use_cases=[
                    "Lightweight applications", "Resource-constrained environments",
                    "Custom agents", "Simple workflows", "Educational projects"
                ],
                pros=[
                    "Very lightweight",
                    "Easy to understand",
                    "Minimal dependencies",
                    "Fast execution",
                    "Great for learning"
                ],
                cons=[
                    "Limited built-in features",
                    "Smaller community",
                    "Less documentation",
                    "Fewer integrations"
                ]
            ),
            "local_models": FrameworkCapabilities(
                name="Local Models (Ollama)",
                ease_of_use=6,
                flexibility=8,
                performance=6,
                documentation=7,
                community_support=7,
                production_ready=7,
                learning_curve=6,
                use_cases=[
                    "Privacy-sensitive applications", "Offline scenarios",
                    "Cost optimization", "Customized models", "Research"
                ],
                pros=[
                    "Complete privacy control",
                    "No API costs",
                    "Offline capability",
                    "Customizable models",
                    "Full control over deployment"
                ],
                cons=[
                    "Requires significant hardware",
                    "Model management complexity",
                    "Generally lower performance",
                    "Setup and maintenance overhead"
                ]
            )
        }

    def compare_simple_qa_task(self) -> List[ComparisonResult]:
        """Compare frameworks on simple Q&A task"""
        question = "What is the capital of France?"
        task_name = "Simple Q&A"
        results = []

        # OpenAI Direct
        if openai and os.getenv("OPENAI_API_KEY"):
            results.append(self._test_openai_direct(question, task_name, TaskComplexity.SIMPLE))

        # Anthropic Direct
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            results.append(self._test_anthropic_direct(question, task_name, TaskComplexity.SIMPLE))

        # LangChain
        if langchain and os.getenv("OPENAI_API_KEY"):
            results.append(self._test_langchain(question, task_name, TaskComplexity.SIMPLE))

        # DSPy
        if dspy and os.getenv("OPENAI_API_KEY"):
            results.append(self._test_dspy(question, task_name, TaskComplexity.SIMPLE))

        return results

    def compare_tool_usage_task(self) -> List[ComparisonResult]:
        """Compare frameworks on tool usage task"""
        task = "Calculate 15% tip on a $67.50 bill and tell me the total amount."
        task_name = "Tool Usage (Calculator)"
        results = []

        # Test each framework's tool usage capabilities
        frameworks_to_test = ["openai_direct", "langchain"]

        for framework in frameworks_to_test:
            if framework == "openai_direct" and openai and os.getenv("OPENAI_API_KEY"):
                results.append(self._test_openai_tools(task, task_name, TaskComplexity.MEDIUM))
            elif framework == "langchain" and langchain and os.getenv("OPENAI_API_KEY"):
                results.append(self._test_langchain_tools(task, task_name, TaskComplexity.MEDIUM))

        return results

    def compare_multi_step_task(self) -> List[ComparisonResult]:
        """Compare frameworks on multi-step reasoning task"""
        task = """
        You are planning a small dinner party. Given:
        - 6 guests attending
        - Budget of $120
        - Need appetizer, main course, and dessert
        - One guest is vegetarian

        Create a menu plan with estimated costs that stays within budget.
        """
        task_name = "Multi-step Planning"
        results = []

        # Test frameworks on complex reasoning
        if openai and os.getenv("OPENAI_API_KEY"):
            results.append(self._test_openai_direct(task, task_name, TaskComplexity.COMPLEX))

        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            results.append(self._test_anthropic_direct(task, task_name, TaskComplexity.COMPLEX))

        return results

    def _test_openai_direct(self, prompt: str, task_name: str, complexity: TaskComplexity) -> ComparisonResult:
        """Test OpenAI direct API"""
        start_time = time.time()

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            execution_time = time.time() - start_time
            output = response.choices[0].message.content.strip()

            return ComparisonResult(
                framework="openai_direct",
                task_name=task_name,
                complexity=complexity,
                execution_time=execution_time,
                success=True,
                output=output,
                token_count=response.usage.total_tokens if response.usage else None,
                cost_estimate=self._estimate_openai_cost(response.usage.total_tokens if response.usage else 0)
            )

        except Exception as e:
            return ComparisonResult(
                framework="openai_direct",
                task_name=task_name,
                complexity=complexity,
                execution_time=time.time() - start_time,
                success=False,
                output="",
                error_message=str(e)
            )

    def _test_anthropic_direct(self, prompt: str, task_name: str, complexity: TaskComplexity) -> ComparisonResult:
        """Test Anthropic direct API"""
        start_time = time.time()

        try:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            execution_time = time.time() - start_time
            output = response.content[0].text.strip()

            return ComparisonResult(
                framework="anthropic_direct",
                task_name=task_name,
                complexity=complexity,
                execution_time=execution_time,
                success=True,
                output=output,
                token_count=response.usage.input_tokens + response.usage.output_tokens,
                cost_estimate=self._estimate_anthropic_cost(response.usage.input_tokens, response.usage.output_tokens)
            )

        except Exception as e:
            return ComparisonResult(
                framework="anthropic_direct",
                task_name=task_name,
                complexity=complexity,
                execution_time=time.time() - start_time,
                success=False,
                output="",
                error_message=str(e)
            )

    def _test_langchain(self, prompt: str, task_name: str, complexity: TaskComplexity) -> ComparisonResult:
        """Test LangChain framework"""
        start_time = time.time()

        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=200)
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)

            execution_time = time.time() - start_time
            output = response.content.strip()

            return ComparisonResult(
                framework="langchain",
                task_name=task_name,
                complexity=complexity,
                execution_time=execution_time,
                success=True,
                output=output
            )

        except Exception as e:
            return ComparisonResult(
                framework="langchain",
                task_name=task_name,
                complexity=complexity,
                execution_time=time.time() - start_time,
                success=False,
                output="",
                error_message=str(e)
            )

    def _test_dspy(self, prompt: str, task_name: str, complexity: TaskComplexity) -> ComparisonResult:
        """Test DSPy framework"""
        start_time = time.time()

        try:
            # Configure DSPy
            lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), max_tokens=200)
            dspy.settings.configure(lm=lm)

            # Create simple QA signature
            class QA(dspy.Signature):
                question = dspy.InputField()
                answer = dspy.OutputField()

            # Create predictor
            predictor = dspy.Predict(QA)
            result = predictor(question=prompt)

            execution_time = time.time() - start_time
            output = result.answer.strip()

            return ComparisonResult(
                framework="dspy",
                task_name=task_name,
                complexity=complexity,
                execution_time=execution_time,
                success=True,
                output=output
            )

        except Exception as e:
            return ComparisonResult(
                framework="dspy",
                task_name=task_name,
                complexity=complexity,
                execution_time=time.time() - start_time,
                success=False,
                output="",
                error_message=str(e)
            )

    def _test_openai_tools(self, prompt: str, task_name: str, complexity: TaskComplexity) -> ComparisonResult:
        """Test OpenAI with function calling"""
        start_time = time.time()

        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Define calculator tool
            tools = [{
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform basic arithmetic calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto"
            )

            # Handle tool calls
            output = ""
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "calculate":
                        args = json.loads(tool_call.function.arguments)
                        try:
                            result = eval(args["expression"])
                            output += f"Calculation: {args['expression']} = {result}\\n"
                        except:
                            output += f"Error calculating: {args['expression']}\\n"

            if response.choices[0].message.content:
                output += response.choices[0].message.content

            execution_time = time.time() - start_time

            return ComparisonResult(
                framework="openai_tools",
                task_name=task_name,
                complexity=complexity,
                execution_time=execution_time,
                success=True,
                output=output.strip()
            )

        except Exception as e:
            return ComparisonResult(
                framework="openai_tools",
                task_name=task_name,
                complexity=complexity,
                execution_time=time.time() - start_time,
                success=False,
                output="",
                error_message=str(e)
            )

    def _test_langchain_tools(self, prompt: str, task_name: str, complexity: TaskComplexity) -> ComparisonResult:
        """Test LangChain with tools"""
        start_time = time.time()

        try:
            # Define calculator tool
            @tool
            def calculate(expression: str) -> float:
                """Perform basic arithmetic calculations"""
                try:
                    return eval(expression)
                except:
                    return "Error in calculation"

            # Create agent
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            tools = [calculate]

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant with access to a calculator."),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent = create_openai_tools_agent(llm, tools, prompt_template)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            result = agent_executor.invoke({"input": prompt})

            execution_time = time.time() - start_time
            output = result.get("output", "No output generated")

            return ComparisonResult(
                framework="langchain_tools",
                task_name=task_name,
                complexity=complexity,
                execution_time=execution_time,
                success=True,
                output=output
            )

        except Exception as e:
            return ComparisonResult(
                framework="langchain_tools",
                task_name=task_name,
                complexity=complexity,
                execution_time=time.time() - start_time,
                success=False,
                output="",
                error_message=str(e)
            )

    def _estimate_openai_cost(self, tokens: int) -> float:
        """Estimate OpenAI cost (GPT-3.5-turbo pricing)"""
        # Rough estimate: $0.002 per 1K tokens
        return (tokens / 1000) * 0.002

    def _estimate_anthropic_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate Anthropic cost (Claude Haiku pricing)"""
        # Rough estimate: $0.00025 per 1K input tokens, $0.00125 per 1K output tokens
        input_cost = (input_tokens / 1000) * 0.00025
        output_cost = (output_tokens / 1000) * 0.00125
        return input_cost + output_cost

    def print_comparison_results(self, results: List[ComparisonResult]):
        """Print formatted comparison results"""
        if not results:
            print("No results to display")
            return

        task_name = results[0].task_name
        print(f"\\n{'='*60}")
        print(f"📊 TASK: {task_name.upper()}")
        print(f"{'='*60}")

        # Sort by execution time
        results_sorted = sorted([r for r in results if r.success], key=lambda x: x.execution_time)

        for i, result in enumerate(results_sorted):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."

            print(f"\\n{rank} {result.framework.upper()}")
            print(f"   ⏱️  Execution time: {result.execution_time:.2f}s")
            print(f"   ✅ Success: {result.success}")

            if result.token_count:
                print(f"   🎫 Tokens used: {result.token_count}")
            if result.cost_estimate:
                print(f"   💰 Est. cost: ${result.cost_estimate:.4f}")

            # Show output preview
            output_preview = result.output[:100] + "..." if len(result.output) > 100 else result.output
            print(f"   📝 Output: {output_preview}")

        # Show failed attempts
        failed = [r for r in results if not r.success]
        if failed:
            print(f"\\n❌ Failed attempts:")
            for result in failed:
                print(f"   • {result.framework}: {result.error_message}")

    def print_framework_capabilities(self):
        """Print comprehensive framework capability comparison"""
        print("\\n" + "="*80)
        print("🔍 COMPREHENSIVE FRAMEWORK COMPARISON")
        print("="*80)

        # Create comparison table
        headers = ["Framework", "Ease", "Flex", "Perf", "Docs", "Comm", "Prod", "Learn"]
        print(f"\\n{'Framework':<20} {'Ease':<5} {'Flex':<5} {'Perf':<5} {'Docs':<5} {'Comm':<5} {'Prod':<5} {'Learn':<5}")
        print("-" * 70)

        for name, caps in self.frameworks.items():
            print(f"{caps.name:<20} "
                  f"{caps.ease_of_use:<5} "
                  f"{caps.flexibility:<5} "
                  f"{caps.performance:<5} "
                  f"{caps.documentation:<5} "
                  f"{caps.community_support:<5} "
                  f"{caps.production_ready:<5} "
                  f"{caps.learning_curve:<5}")

        print("\\nScale: 1-10 (higher is better, except Learning Curve where lower is easier)")

        # Detailed framework analysis
        for name, caps in self.frameworks.items():
            print(f"\\n{'='*50}")
            print(f"📋 {caps.name.upper()}")
            print(f"{'='*50}")

            print(f"🎯 Best for: {', '.join(caps.use_cases[:3])}")
            print(f"✅ Pros: {', '.join(caps.pros[:3])}")
            print(f"❌ Cons: {', '.join(caps.cons[:2])}")

            # Overall score calculation
            overall_score = (
                caps.ease_of_use + caps.flexibility + caps.performance +
                caps.documentation + caps.community_support + caps.production_ready +
                (10 - caps.learning_curve)  # Invert learning curve
            ) / 7
            print(f"📊 Overall Score: {overall_score:.1f}/10")

    def run_comprehensive_comparison(self):
        """Run comprehensive framework comparison"""
        print("🚀 Starting Comprehensive Framework Comparison...")

        # Print framework capabilities first
        self.print_framework_capabilities()

        print("\\n" + "="*80)
        print("⚡ PERFORMANCE BENCHMARKS")
        print("="*80)
        print("Running standardized tasks across available frameworks...")

        # Simple Q&A comparison
        qa_results = self.compare_simple_qa_task()
        if qa_results:
            self.print_comparison_results(qa_results)

        # Tool usage comparison
        tool_results = self.compare_tool_usage_task()
        if tool_results:
            self.print_comparison_results(tool_results)

        # Multi-step reasoning comparison
        complex_results = self.compare_multi_step_task()
        if complex_results:
            self.print_comparison_results(complex_results)

        # Summary recommendations
        print("\\n" + "="*80)
        print("🎯 FRAMEWORK SELECTION GUIDE")
        print("="*80)

        recommendations = {
            "Beginners": "OpenAI Direct API or SmolAgents - easiest to learn",
            "Rapid Prototyping": "LangChain - extensive tools and integrations",
            "Complex Workflows": "LangGraph - state management and conditional logic",
            "Team Scenarios": "CrewAI - multi-agent collaboration",
            "Performance Critical": "DSPy - automatic optimization",
            "Privacy/Security": "Local Models - complete control",
            "Production Ready": "OpenAI/Anthropic Direct - most stable",
            "Research/Academic": "DSPy - principled approach"
        }

        for use_case, recommendation in recommendations.items():
            print(f"• {use_case:<20}: {recommendation}")


def main():
    """Main demonstration function"""
    print("⚖️ Framework Comparison - Agentic AI Framework Analysis")
    print("=" * 70)

    print("\\nThis module compares major agentic AI frameworks:")
    print("• Performance benchmarks on standardized tasks")
    print("• Capability assessments across multiple dimensions")
    print("• Use case recommendations and selection guidance")

    # Check available frameworks
    available_frameworks = []
    if openai and os.getenv("OPENAI_API_KEY"):
        available_frameworks.append("OpenAI")
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        available_frameworks.append("Anthropic")
    if langchain:
        available_frameworks.append("LangChain")
    if dspy:
        available_frameworks.append("DSPy")

    print(f"\\n🔧 Available frameworks for testing: {', '.join(available_frameworks)}")

    if not available_frameworks:
        print("\\n⚠️  No frameworks available for testing.")
        print("Install packages and configure API keys to run comparisons:")
        print("• pip install openai anthropic langchain dspy-ai")
        print("• Set OPENAI_API_KEY and ANTHROPIC_API_KEY in .env")
        return

    try:
        comparator = FrameworkComparator()
        comparator.run_comprehensive_comparison()

        print("\\n" + "="*80)
        print("🎉 Framework Comparison Complete!")

        print("\\n🔑 Key Takeaways:")
        print("• Choose frameworks based on specific use cases")
        print("• Consider team expertise and learning curve")
        print("• Balance ease of use vs. advanced capabilities")
        print("• Evaluate performance requirements vs. development speed")
        print("• Factor in long-term maintenance and community support")

        print("\\nNext Steps:")
        print("• Try integration_patterns.py for multi-framework architectures")
        print("• See production_examples.py for deployment strategies")
        print("• Check module 10 exercises for hands-on practice")

    except KeyboardInterrupt:
        print("\\n\\n👋 Comparison interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during comparison: {e}")
        print("Make sure required packages are installed and API keys are configured")


if __name__ == "__main__":
    main()