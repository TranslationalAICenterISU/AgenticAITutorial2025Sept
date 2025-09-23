"""
SmolAgents Basics
Introduction to HuggingFace's official SmolAgents framework
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Note: SmolAgents requires 'smolagents' package
# Install with: pip install "smolagents[toolkit]"

def basic_smolagents_example():
    """Basic SmolAgents usage example"""
    print("🐭 SmolAgents Basic Example")
    print("=" * 40)

    try:
        from smolagents import CodeAgent, InferenceClientModel

        print("✅ SmolAgents imported successfully!")

        # Create a model (using HuggingFace Inference API)
        model = InferenceClientModel()

        # Create a basic agent
        agent = CodeAgent(tools=[], model=model)

        print("\n🤖 Created CodeAgent with InferenceClientModel")

        # Test with a simple mathematical task
        task = "Calculate the sum of numbers from 1 to 10"
        print(f"\n📋 Task: {task}")

        # Note: In a real scenario, you would have API keys configured
        print("ℹ️  This would execute if HuggingFace API key is configured")
        print("The agent would generate and execute Python code to solve the task")

        return True

    except ImportError as e:
        print("❌ SmolAgents not installed")
        print("Install with: pip install \"smolagents[toolkit]\"")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return False


def smolagents_with_tools():
    """SmolAgents with built-in tools"""
    print("\n🔧 SmolAgents with Tools")
    print("=" * 40)

    try:
        from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

        # Create model
        model = InferenceClientModel()

        # Create agent with tools
        web_search = WebSearchTool()
        agent = CodeAgent(tools=[web_search], model=model)

        print("✅ Created CodeAgent with WebSearchTool")

        # Example task that would use web search
        task = "What is the current weather in Paris?"
        print(f"\n📋 Task: {task}")
        print("ℹ️  With proper API keys, this would:")
        print("  1. Generate code to search for Paris weather")
        print("  2. Execute the search using WebSearchTool")
        print("  3. Parse and return the weather information")

        return True

    except ImportError:
        print("❌ WebSearchTool not available")
        print("Make sure to install: pip install \"smolagents[toolkit]\"")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def smolagents_code_execution():
    """Demonstrate SmolAgents code-first approach"""
    print("\n💻 SmolAgents Code-First Approach")
    print("=" * 40)

    try:
        from smolagents import CodeAgent, InferenceClientModel

        # This demonstrates the philosophy, even without execution
        print("🎯 SmolAgents Philosophy:")
        print("- Agents think and act in Python code")
        print("- No complex orchestration layers")
        print("- Direct code generation and execution")
        print("- ~1000 lines of framework code total")

        print("\n📝 Example of how SmolAgent thinks:")
        example_task = "Calculate compound interest for $1000 at 5% for 3 years"
        print(f"Task: {example_task}")

        print("\nAgent would generate code like:")
        print("```python")
        print("principal = 1000")
        print("rate = 0.05")
        print("time = 3")
        print("compound_interest = principal * (1 + rate) ** time")
        print("print(f'Final amount: ${compound_interest:.2f}')")
        print("```")

        print("\nThen execute it directly, no intermediate representations!")

        return True

    except ImportError:
        print("❌ SmolAgents not available")
        return False


def smolagents_model_options():
    """Show different model options with SmolAgents"""
    print("\n🔄 SmolAgents Model Options")
    print("=" * 40)

    try:
        from smolagents import CodeAgent

        print("🎯 Model Options:")
        print("1. InferenceClientModel - HuggingFace Inference API")
        print("2. OpenAI models via LiteLLM integration")
        print("3. Anthropic models via LiteLLM")
        print("4. Local models (transformers, ollama)")

        print("\n💡 Examples:")

        # HuggingFace option
        print("# HuggingFace Inference API")
        print("from smolagents import InferenceClientModel")
        print("model = InferenceClientModel()")
        print("agent = CodeAgent(tools=[], model=model)")

        # OpenAI option (if available)
        print("\n# OpenAI via LiteLLM (requires openai package)")
        print("from smolagents import LiteLLMModel")
        print("model = LiteLLMModel(model_id='gpt-4')")
        print("agent = CodeAgent(tools=[], model=model)")

        # Local model option
        print("\n# Local model")
        print("from smolagents import HfApiModel")
        print("model = HfApiModel()")
        print("agent = CodeAgent(tools=[], model=model)")

        return True

    except ImportError:
        print("❌ SmolAgents not available")
        return False


def smolagents_security_features():
    """Demonstrate SmolAgents security features"""
    print("\n🔒 SmolAgents Security Features")
    print("=" * 40)

    print("🛡️  Security Options:")
    print("1. E2B Sandbox - Cloud-based secure execution")
    print("2. Modal Sandbox - Serverless secure environment")
    print("3. Docker - Containerized execution")
    print("4. Pyodide + Deno - WebAssembly sandbox")

    print("\n💡 Example with E2B:")
    print("```python")
    print("from smolagents import CodeAgent, E2BExecutor")
    print("from smolagents import InferenceClientModel")
    print("")
    print("# Create secure executor")
    print("executor = E2BExecutor()")
    print("")
    print("# Create agent with secure execution")
    print("model = InferenceClientModel()")
    print("agent = CodeAgent(")
    print("    tools=[],")
    print("    model=model,")
    print("    executor=executor  # Secure sandbox")
    print(")")
    print("```")

    print("\n✅ Code executes safely in isolated environment")
    print("✅ No access to host system or sensitive data")
    print("✅ Network access can be controlled")


def fallback_demonstration():
    """Demonstrate SmolAgents concepts without requiring installation"""
    print("\n🎭 SmolAgents Concepts (Fallback Demo)")
    print("=" * 50)

    print("🎯 Key SmolAgents Concepts:")
    print("\n1. Minimalism:")
    print("   - ~1000 lines of core framework code")
    print("   - No complex abstractions")
    print("   - Direct code generation and execution")

    print("\n2. Code-First Approach:")
    print("   - Agents write Python code to solve problems")
    print("   - No intermediate action representations")
    print("   - Direct execution of generated code")

    print("\n3. Tool Integration:")
    print("   - Tools are Python functions")
    print("   - Agent generates code that calls tools")
    print("   - Seamless integration with existing Python ecosystem")

    print("\n4. Model Agnostic:")
    print("   - Works with any LLM (local or cloud)")
    print("   - HuggingFace, OpenAI, Anthropic, local models")
    print("   - Easy to swap models based on needs")

    print("\n5. Security:")
    print("   - Multiple sandboxing options")
    print("   - Safe code execution environments")
    print("   - Isolated from host system")

    print("\n📝 Typical SmolAgent Workflow:")
    print("1. Receive task/question")
    print("2. Generate Python code to solve it")
    print("3. Execute code in sandbox")
    print("4. Return results")
    print("5. Iterate if needed")

    print("\n💡 Use Cases:")
    print("- Data analysis and manipulation")
    print("- Mathematical computations")
    print("- Web scraping and API calls")
    print("- File processing")
    print("- Research and information gathering")


def main():
    """Main demonstration function"""
    print("🐭 SmolAgents Framework Exploration")
    print("=" * 50)

    # Try to demonstrate real SmolAgents
    success = basic_smolagents_example()

    if success:
        smolagents_with_tools()
        smolagents_code_execution()
        smolagents_model_options()
        smolagents_security_features()
    else:
        # Fallback to conceptual demonstration
        fallback_demonstration()

    print("\n🎯 Next Steps:")
    print("1. Install SmolAgents: pip install \"smolagents[toolkit]\"")
    print("2. Set up HuggingFace API key or other model access")
    print("3. Try the examples with real API keys")
    print("4. Explore advanced features like custom tools")
    print("5. Implement secure execution environments")

    print("\n📚 Resources:")
    print("- GitHub: https://github.com/huggingface/smolagents")
    print("- Documentation: https://huggingface.co/docs/smolagents")
    print("- Blog: https://huggingface.co/blog/smolagents")


if __name__ == "__main__":
    main()