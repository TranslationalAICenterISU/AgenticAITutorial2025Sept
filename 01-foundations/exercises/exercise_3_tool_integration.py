"""
Exercise 3: Tool Integration Patterns
Learn how agents interact with external tools and services

OBJECTIVE: Understand tool selection, execution, and error handling in agentic systems
DIFFICULTY: Intermediate
TIME: 30-35 minutes
"""

import json
import time
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """Base class for all tools"""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.usage_count = 0
        self.success_rate = 1.0

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate
        }


class CalculatorTool(Tool):
    """Mathematical calculations tool"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
            }
        )

    def execute(self, expression: str = "") -> ToolResult:
        """Execute calculation"""
        start_time = time.time()
        self.usage_count += 1

        try:
            # Simple safety check
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Expression contains invalid characters")

            result = eval(expression)
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={"expression": expression}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count

            return ToolResult(
                success=False,
                data=None,
                error_message=str(e),
                execution_time=execution_time,
                metadata={"expression": expression}
            )


class WebSearchTool(Tool):
    """Web search simulation tool"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results"}
            }
        )

    def execute(self, query: str = "", max_results: int = 5) -> ToolResult:
        """Execute web search (simulated)"""
        start_time = time.time()
        self.usage_count += 1

        try:
            # Simulate search delay
            time.sleep(0.1)

            # Mock search results based on query
            if "python" in query.lower():
                results = [
                    {"title": "Python.org", "url": "https://python.org", "snippet": "Official Python website"},
                    {"title": "Python Tutorial", "url": "https://docs.python.org/3/tutorial/", "snippet": "Python tutorial"},
                    {"title": "Learn Python", "url": "https://learnpython.org", "snippet": "Interactive Python tutorial"}
                ]
            elif "weather" in query.lower():
                results = [
                    {"title": "Weather.com", "url": "https://weather.com", "snippet": "Weather forecasts"},
                    {"title": "Local Weather", "url": "https://localweather.com", "snippet": "Current conditions"}
                ]
            else:
                results = [
                    {"title": f"Search result for '{query}'", "url": "https://example.com", "snippet": "Mock result"}
                ]

            # Limit results
            results = results[:max_results]
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data=results,
                execution_time=execution_time,
                metadata={"query": query, "result_count": len(results)}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e),
                execution_time=execution_time
            )


class FileManagerTool(Tool):
    """File operations tool"""

    def __init__(self):
        super().__init__(
            name="file_manager",
            description="Read, write, and manage files",
            parameters={
                "action": {"type": "string", "description": "Action to perform: read, write, list"},
                "filename": {"type": "string", "description": "Name of the file"},
                "content": {"type": "string", "description": "Content to write (for write action)"}
            }
        )
        self.mock_files = {
            "notes.txt": "These are my personal notes about the project.",
            "todo.txt": "1. Complete exercises\\n2. Review concepts\\n3. Practice implementation",
            "config.json": '{"theme": "dark", "language": "en", "notifications": true}'
        }

    def execute(self, action: str = "", filename: str = "", content: str = "") -> ToolResult:
        """Execute file operation"""
        start_time = time.time()
        self.usage_count += 1

        try:
            if action == "read":
                if filename in self.mock_files:
                    data = self.mock_files[filename]
                else:
                    raise FileNotFoundError(f"File '{filename}' not found")

            elif action == "write":
                self.mock_files[filename] = content
                data = f"Successfully wrote {len(content)} characters to {filename}"

            elif action == "list":
                data = list(self.mock_files.keys())

            else:
                raise ValueError(f"Unknown action: {action}")

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data=data,
                execution_time=execution_time,
                metadata={"action": action, "filename": filename}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                error_message=str(e),
                execution_time=execution_time
            )


class UnreliableTool(Tool):
    """Tool that sometimes fails - for testing error handling"""

    def __init__(self, failure_rate: float = 0.3):
        super().__init__(
            name="unreliable_service",
            description="A service that sometimes fails",
            parameters={
                "request": {"type": "string", "description": "Request to process"}
            }
        )
        self.failure_rate = failure_rate

    def execute(self, request: str = "") -> ToolResult:
        """Execute with potential failure"""
        start_time = time.time()
        self.usage_count += 1

        # Randomly fail based on failure rate
        if random.random() < self.failure_rate:
            execution_time = time.time() - start_time
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count

            return ToolResult(
                success=False,
                data=None,
                error_message="Service temporarily unavailable",
                execution_time=execution_time
            )

        # Success
        execution_time = time.time() - start_time
        return ToolResult(
            success=True,
            data=f"Processed request: {request}",
            execution_time=execution_time
        )


class ToolExecutor:
    """Manages and executes tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict[str, Any]] = []

    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        print(f"🔧 Registered tool: {tool.name}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return [tool.get_info() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error_message=f"Tool '{tool_name}' not found"
            )

        tool = self.tools[tool_name]
        result = tool.execute(**kwargs)

        # Log execution
        self.execution_history.append({
            "tool": tool_name,
            "parameters": kwargs,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })

        return result

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        stats = {}
        for tool_name, tool in self.tools.items():
            stats[tool_name] = {
                "usage_count": tool.usage_count,
                "success_rate": tool.success_rate,
                "description": tool.description
            }
        return stats


class SmartAgent:
    """Agent with intelligent tool selection and error handling"""

    def __init__(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor
        self.task_history: List[Dict[str, Any]] = []

    def analyze_task(self, task: str) -> str:
        """Analyze task and suggest best tool"""
        task_lower = task.lower()

        if any(word in task_lower for word in ['calculate', 'math', 'compute', '+', '-', '*', '/']):
            return "calculator"
        elif any(word in task_lower for word in ['search', 'find', 'lookup', 'google']):
            return "web_search"
        elif any(word in task_lower for word in ['file', 'read', 'write', 'save']):
            return "file_manager"
        elif "unreliable" in task_lower or "test" in task_lower:
            return "unreliable_service"
        else:
            return "unknown"

    def execute_task(self, task: str, max_retries: int = 3) -> Dict[str, Any]:
        """Execute task with error handling and retries"""
        print(f"🎯 Executing task: {task}")

        # Analyze and select tool
        suggested_tool = self.analyze_task(task)
        print(f"🤖 Suggested tool: {suggested_tool}")

        if suggested_tool == "unknown":
            return {
                "success": False,
                "result": None,
                "error": "Could not determine appropriate tool for task"
            }

        # Prepare parameters based on tool
        params = self._prepare_parameters(task, suggested_tool)
        print(f"📋 Parameters: {params}")

        # Execute with retries
        for attempt in range(max_retries):
            print(f"\\n🔄 Attempt {attempt + 1}/{max_retries}")

            result = self.tool_executor.execute_tool(suggested_tool, **params)

            if result.success:
                print(f"✅ Success! Result: {result.data}")
                task_record = {
                    "task": task,
                    "tool_used": suggested_tool,
                    "success": True,
                    "attempts": attempt + 1,
                    "result": result.data
                }
                self.task_history.append(task_record)
                return task_record

            else:
                print(f"❌ Failed: {result.error_message}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        # All attempts failed
        task_record = {
            "task": task,
            "tool_used": suggested_tool,
            "success": False,
            "attempts": max_retries,
            "result": None,
            "error": result.error_message
        }
        self.task_history.append(task_record)
        return task_record

    def _prepare_parameters(self, task: str, tool_name: str) -> Dict[str, Any]:
        """Prepare parameters for specific tool based on task"""
        if tool_name == "calculator":
            # Extract mathematical expression from task
            import re
            math_pattern = r'([0-9+\\-*/.() ]+)'
            matches = re.findall(math_pattern, task)
            expression = matches[0].strip() if matches else task
            return {"expression": expression}

        elif tool_name == "web_search":
            # Use task as search query
            return {"query": task, "max_results": 3}

        elif tool_name == "file_manager":
            # Simple heuristics for file operations
            if "read" in task.lower():
                return {"action": "read", "filename": "notes.txt"}
            elif "write" in task.lower():
                return {"action": "write", "filename": "output.txt", "content": task}
            else:
                return {"action": "list"}

        elif tool_name == "unreliable_service":
            return {"request": task}

        return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.task_history:
            return {"message": "No tasks executed yet"}

        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for task in self.task_history if task["success"])
        success_rate = successful_tasks / total_tasks

        tool_usage = {}
        for task in self.task_history:
            tool = task["tool_used"]
            if tool not in tool_usage:
                tool_usage[tool] = {"count": 0, "successes": 0}
            tool_usage[tool]["count"] += 1
            if task["success"]:
                tool_usage[tool]["successes"] += 1

        return {
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "tool_usage": tool_usage
        }


def tool_integration_exercise():
    """Exercise: Basic tool integration"""

    print("🔧 Tool Integration Exercise")
    print("="*30)

    # Set up tools
    executor = ToolExecutor()
    executor.register_tool(CalculatorTool())
    executor.register_tool(WebSearchTool())
    executor.register_tool(FileManagerTool())

    print("\\n📊 Available Tools:")
    for tool_info in executor.list_tools():
        print(f"   • {tool_info['name']}: {tool_info['description']}")

    # Test individual tools
    print("\\n🧪 Testing Individual Tools:")
    print("-" * 35)

    # Calculator test
    print("\\n1. Calculator Tool:")
    calc_result = executor.execute_tool("calculator", expression="25 * 4 + 100")
    print(f"   Task: Calculate 25 * 4 + 100")
    print(f"   Result: {calc_result.data if calc_result.success else calc_result.error_message}")

    # Web search test
    print("\\n2. Web Search Tool:")
    search_result = executor.execute_tool("web_search", query="Python programming", max_results=2)
    print(f"   Task: Search for Python programming")
    if search_result.success:
        print(f"   Found {len(search_result.data)} results:")
        for result in search_result.data:
            print(f"     • {result['title']}")
    else:
        print(f"   Error: {search_result.error_message}")

    # File manager test
    print("\\n3. File Manager Tool:")
    file_result = executor.execute_tool("file_manager", action="list")
    print(f"   Task: List available files")
    print(f"   Files: {file_result.data if file_result.success else file_result.error_message}")

    return executor


def error_handling_exercise():
    """Exercise: Error handling and retries"""

    print("\\n⚠️ Error Handling Exercise")
    print("="*30)

    # Set up executor with unreliable tool
    executor = ToolExecutor()
    executor.register_tool(CalculatorTool())
    executor.register_tool(UnreliableTool(failure_rate=0.7))  # High failure rate

    # Create smart agent
    agent = SmartAgent(executor)

    print("\\n🤖 Testing Smart Agent with Unreliable Service:")

    # Test tasks
    test_tasks = [
        "calculate 15 + 25",
        "test the unreliable service with some request",
        "calculate square root of 144"  # This will fail - no sqrt function
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\\n--- Task {i} ---")
        result = agent.execute_task(task, max_retries=3)
        print(f"Final result: {'✅ Success' if result['success'] else '❌ Failed'}")

    # Show performance summary
    print("\\n📈 Agent Performance Summary:")
    summary = agent.get_performance_summary()
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Tools used: {list(summary['tool_usage'].keys())}")

    return agent


def advanced_tool_patterns():
    """Exercise: Advanced tool patterns"""

    print("\\n🚀 Advanced Tool Patterns Exercise")
    print("="*35)

    class ConditionalTool(Tool):
        """Tool that behaves differently based on conditions"""

        def __init__(self):
            super().__init__(
                name="conditional_processor",
                description="Process data differently based on type",
                parameters={
                    "data": {"type": "any", "description": "Data to process"},
                    "processing_type": {"type": "string", "description": "Type of processing"}
                }
            )

        def execute(self, data: Any = None, processing_type: str = "default") -> ToolResult:
            start_time = time.time()

            if processing_type == "number":
                result = f"Processed number: {float(data) * 2}"
            elif processing_type == "text":
                result = f"Processed text: {str(data).upper()}"
            elif processing_type == "list":
                result = f"Processed list: {sorted(data) if isinstance(data, list) else [data]}"
            else:
                result = f"Default processing: {data}"

            return ToolResult(
                success=True,
                data=result,
                execution_time=time.time() - start_time
            )

    class ChainableTool(Tool):
        """Tool that can be chained with others"""

        def __init__(self, name: str, operation: Callable):
            super().__init__(
                name=name,
                description=f"Chainable operation: {name}",
                parameters={"input": {"type": "any", "description": "Input data"}}
            )
            self.operation = operation

        def execute(self, input_data: Any = None) -> ToolResult:
            try:
                result = self.operation(input_data)
                return ToolResult(success=True, data=result)
            except Exception as e:
                return ToolResult(success=False, data=None, error_message=str(e))

    # Set up advanced tools
    executor = ToolExecutor()
    executor.register_tool(ConditionalTool())

    # Chainable tools
    executor.register_tool(ChainableTool("double", lambda x: x * 2))
    executor.register_tool(ChainableTool("add_ten", lambda x: x + 10))
    executor.register_tool(ChainableTool("to_string", lambda x: f"Value: {x}"))

    print("\\n🔗 Tool Chaining Example:")
    # Chain: 5 -> double -> add_ten -> to_string
    value = 5
    print(f"Starting value: {value}")

    # Step 1: Double
    result1 = executor.execute_tool("double", input_data=value)
    print(f"After doubling: {result1.data}")

    # Step 2: Add ten
    result2 = executor.execute_tool("add_ten", input_data=result1.data)
    print(f"After adding ten: {result2.data}")

    # Step 3: Convert to string
    result3 = executor.execute_tool("to_string", input_data=result2.data)
    print(f"Final result: {result3.data}")

    print("\\n🎭 Conditional Processing Example:")
    test_data = [
        (42, "number"),
        ("hello world", "text"),
        ([3, 1, 4, 1, 5], "list"),
        ("unknown data", "default")
    ]

    for data, proc_type in test_data:
        result = executor.execute_tool("conditional_processor", data=data, processing_type=proc_type)
        print(f"   Input: {data} ({proc_type}) -> Output: {result.data}")

    # Show tool statistics
    print("\\n📊 Tool Usage Statistics:")
    stats = executor.get_tool_stats()
    for tool_name, tool_stats in stats.items():
        print(f"   {tool_name}: {tool_stats['usage_count']} uses, {tool_stats['success_rate']:.1%} success rate")


def main():
    """Run all tool integration exercises"""

    print("🔧 TOOL INTEGRATION EXERCISES")
    print("="*35)

    print("\\nLearn how agents interact with external tools:")
    print("1. Basic Tool Integration")
    print("2. Error Handling and Retries")
    print("3. Advanced Tool Patterns")

    exercises = [
        ("Basic Integration", tool_integration_exercise),
        ("Error Handling", error_handling_exercise),
        ("Advanced Patterns", advanced_tool_patterns)
    ]

    for i, (name, exercise_func) in enumerate(exercises, 1):
        print(f"\\n{'='*50}")
        print(f"EXERCISE {i}: {name.upper()}")
        print(f"{'='*50}")

        try:
            exercise_func()
        except Exception as e:
            print(f"❌ Exercise error: {e}")

        input("\\nPress Enter to continue to next exercise...")

    print("\\n🎉 Congratulations! You've completed the Tool Integration exercises!")
    print("\\n💡 Key Takeaways:")
    print("1. Tools extend agent capabilities beyond language generation")
    print("2. Error handling and retries are crucial for reliability")
    print("3. Tool selection should match the task requirements")
    print("4. Chaining tools enables complex workflows")
    print("5. Monitoring tool performance helps optimize agent behavior")

    print("\\n➡️ Continue to Module 2 (LLM APIs) when ready!")


if __name__ == "__main__":
    main()