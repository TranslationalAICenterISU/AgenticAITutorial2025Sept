"""
SmolAgents Advanced Examples
Demonstrates advanced features and patterns using HuggingFace SmolAgents
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def custom_tool_example():
    """Create and use custom tools with SmolAgents"""
    print("🔧 SmolAgents Custom Tools")
    print("=" * 40)

    try:
        from smolagents import CodeAgent, InferenceClientModel, tool

        print("✅ Imported SmolAgents components")

        # Create a custom tool using the @tool decorator
        @tool
        def calculate_fibonacci(n: int) -> int:
            """Calculate the nth Fibonacci number.

            Args:
                n: The position in the Fibonacci sequence

            Returns:
                The nth Fibonacci number
            """
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

        print("✅ Created custom Fibonacci tool")

        # Create agent with custom tool
        model = InferenceClientModel()
        agent = CodeAgent(tools=[calculate_fibonacci], model=model)

        print("✅ Created CodeAgent with custom tool")
        print("ℹ️  Agent can now use the Fibonacci tool in generated code")

        return True

    except ImportError:
        print("❌ SmolAgents not available")
        print("Install with: pip install \"smolagents[toolkit]\"")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def multi_modal_example():
    """Demonstrate multi-modal capabilities"""
    print("\n🎨 SmolAgents Multi-Modal Capabilities")
    print("=" * 40)

    try:
        from smolagents import CodeAgent, InferenceClientModel

        print("🎯 SmolAgents Multi-Modal Support:")
        print("- Text processing (default)")
        print("- Image analysis and generation")
        print("- Audio processing")
        print("- Video analysis")

        print("\n💡 Example workflow:")
        print("1. User uploads an image")
        print("2. Agent generates code to analyze the image")
        print("3. Code uses computer vision libraries")
        print("4. Results are processed and returned")

        # Show conceptual example
        print("\n📝 Agent would generate code like:")
        print("```python")
        print("from PIL import Image")
        print("import numpy as np")
        print("")
        print("# Load and analyze image")
        print("img = Image.open('uploaded_image.jpg')")
        print("img_array = np.array(img)")
        print("height, width, channels = img_array.shape")
        print("")
        print("print(f'Image dimensions: {width}x{height}')")
        print("print(f'Color channels: {channels}')")
        print("```")

        return True

    except ImportError:
        print("❌ SmolAgents not available")
        return False


def agent_with_memory():
    """Demonstrate persistent memory patterns"""
    print("\n🧠 SmolAgents with Memory")
    print("=" * 40)

    try:
        from smolagents import CodeAgent, InferenceClientModel

        print("🎯 Memory Strategies in SmolAgents:")
        print("1. Conversation history (built-in)")
        print("2. File-based persistence")
        print("3. Database storage")
        print("4. Vector embeddings for semantic memory")

        print("\n📝 Memory implementation example:")
        print("```python")
        print("import json")
        print("from datetime import datetime")
        print("")
        print("# Agent would generate code like:")
        print("memory_file = 'agent_memory.json'")
        print("")
        print("# Load previous memories")
        print("try:")
        print("    with open(memory_file, 'r') as f:")
        print("        memories = json.load(f)")
        print("except FileNotFoundError:")
        print("    memories = []")
        print("")
        print("# Add new memory")
        print("new_memory = {")
        print("    'timestamp': datetime.now().isoformat(),")
        print("    'task': 'current_task',")
        print("    'result': 'task_result'")
        print("}")
        print("memories.append(new_memory)")
        print("")
        print("# Save memories")
        print("with open(memory_file, 'w') as f:")
        print("    json.dump(memories, f)")
        print("```")

        return True

    except ImportError:
        print("❌ SmolAgents not available")
        return False


def error_handling_patterns():
    """Demonstrate robust error handling"""
    print("\n🛡️ SmolAgents Error Handling")
    print("=" * 40)

    print("🎯 Error Handling Strategies:")
    print("1. Code validation before execution")
    print("2. Graceful degradation")
    print("3. Retry mechanisms")
    print("4. Fallback approaches")

    print("\n💡 Agent generates defensive code:")
    print("```python")
    print("import sys")
    print("import traceback")
    print("")
    print("def safe_execute(operation, fallback_value=None):")
    print("    \"\"\"Safely execute operation with fallback\"\"\"")
    print("    try:")
    print("        return operation()")
    print("    except Exception as e:")
    print("        print(f'Error: {e}')")
    print("        if fallback_value is not None:")
    print("            return fallback_value")
    print("        raise")
    print("")
    print("# Usage in agent-generated code")
    print("result = safe_execute(")
    print("    lambda: complex_calculation(),")
    print("    fallback_value='Could not calculate'")
    print(")")
    print("```")


def performance_optimization():
    """Demonstrate performance optimization techniques"""
    print("\n⚡ SmolAgents Performance Optimization")
    print("=" * 40)

    print("🎯 Optimization Techniques:")
    print("1. Efficient code generation")
    print("2. Caching mechanisms")
    print("3. Batch processing")
    print("4. Asynchronous operations")

    print("\n💡 Agent generates optimized code:")
    print("```python")
    print("import asyncio")
    print("from concurrent.futures import ThreadPoolExecutor")
    print("import functools")
    print("")
    print("# Caching decorator")
    print("@functools.lru_cache(maxsize=128)")
    print("def expensive_calculation(n):")
    print("    # Expensive operation")
    print("    return sum(i**2 for i in range(n))")
    print("")
    print("# Batch processing")
    print("def process_batch(items, batch_size=100):")
    print("    for i in range(0, len(items), batch_size):")
    print("        batch = items[i:i+batch_size]")
    print("        yield [process_item(item) for item in batch]")
    print("")
    print("# Async operations")
    print("async def parallel_tasks(tasks):")
    print("    return await asyncio.gather(*tasks)")
    print("```")


def integration_patterns():
    """Show integration with other frameworks"""
    print("\n🔗 SmolAgents Integration Patterns")
    print("=" * 40)

    print("🎯 Framework Integration:")
    print("1. With LangChain - Use SmolAgent as a tool")
    print("2. With CrewAI - SmolAgent as specialized crew member")
    print("3. With FastAPI - RESTful agent endpoints")
    print("4. With Streamlit - Interactive agent UI")

    print("\n💡 LangChain Integration Example:")
    print("```python")
    print("from langchain.tools import BaseTool")
    print("from smolagents import CodeAgent, InferenceClientModel")
    print("")
    print("class SmolAgentTool(BaseTool):")
    print("    name = 'code_agent'")
    print("    description = 'Executes code to solve problems'")
    print("    ")
    print("    def __init__(self):")
    print("        super().__init__()")
    print("        model = InferenceClientModel()")
    print("        self.agent = CodeAgent(tools=[], model=model)")
    print("    ")
    print("    def _run(self, task: str) -> str:")
    print("        return self.agent.run(task)")
    print("```")

    print("\n💡 FastAPI Integration:")
    print("```python")
    print("from fastapi import FastAPI")
    print("from smolagents import CodeAgent, InferenceClientModel")
    print("")
    print("app = FastAPI()")
    print("model = InferenceClientModel()")
    print("agent = CodeAgent(tools=[], model=model)")
    print("")
    print("@app.post('/solve')")
    print("async def solve_task(task: str):")
    print("    result = agent.run(task)")
    print("    return {'result': result}")
    print("```")


def production_deployment():
    """Production deployment considerations"""
    print("\n🚀 Production Deployment")
    print("=" * 40)

    print("🎯 Production Checklist:")
    print("✅ Secure sandboxing (E2B, Docker, etc.)")
    print("✅ Rate limiting and quotas")
    print("✅ Monitoring and logging")
    print("✅ Error tracking and alerting")
    print("✅ Model fallbacks and redundancy")
    print("✅ API key rotation")
    print("✅ Cost optimization")

    print("\n🛡️ Security Measures:")
    print("- Code execution in isolated environments")
    print("- Input sanitization and validation")
    print("- Output filtering and safety checks")
    print("- Network access restrictions")
    print("- Resource usage limits")

    print("\n📊 Monitoring Metrics:")
    print("- Task completion rate")
    print("- Code execution success rate")
    print("- Average response time")
    print("- Resource usage patterns")
    print("- Error rates and types")


def main():
    """Main demonstration function"""
    print("🐭 SmolAgents Advanced Features")
    print("=" * 50)

    # Try advanced examples
    success = custom_tool_example()

    if success:
        multi_modal_example()
        agent_with_memory()
    else:
        print("\n📚 Advanced SmolAgents Concepts (Fallback)")
        print("=" * 50)

    # Always show these conceptual examples
    error_handling_patterns()
    performance_optimization()
    integration_patterns()
    production_deployment()

    print("\n🎯 Key Takeaways:")
    print("1. SmolAgents excels at code generation and execution")
    print("2. Custom tools extend agent capabilities")
    print("3. Multi-modal support enables rich interactions")
    print("4. Memory patterns enable persistent conversations")
    print("5. Integration patterns connect with other frameworks")
    print("6. Production requires careful security and monitoring")

    print("\n📚 Resources:")
    print("- SmolAgents GitHub: https://github.com/huggingface/smolagents")
    print("- HuggingFace Hub: Search for smolagents tools")
    print("- E2B Sandboxes: https://e2b.dev/")
    print("- Community Examples: https://huggingface.co/spaces")


if __name__ == "__main__":
    main()