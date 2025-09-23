"""
Model Context Protocol (MCP) Basics
Introduction to building MCP servers and clients for agent tool integration
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import time
from pathlib import Path

# Since MCP is still evolving, this is an educational implementation
# showing the core concepts and patterns


@dataclass
class MCPResource:
    """Represents an MCP resource"""
    uri: str
    name: str
    description: str
    mimeType: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


@dataclass
class MCPMessage:
    """Base MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPServer:
    """Basic MCP Server implementation"""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.resources: Dict[str, MCPResource] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.request_handlers = {
            "initialize": self.handle_initialize,
            "resources/list": self.handle_list_resources,
            "resources/read": self.handle_read_resource,
            "tools/list": self.handle_list_tools,
            "tools/call": self.handle_call_tool,
            "prompts/list": self.handle_list_prompts,
            "prompts/get": self.handle_get_prompt,
        }
        self.logger = logging.getLogger(f"mcp.server.{name}")

    def add_resource(self, resource: MCPResource):
        """Add a resource to the server"""
        self.resources[resource.uri] = resource
        self.logger.info(f"Added resource: {resource.name} ({resource.uri})")

    def add_tool(self, tool: MCPTool, handler=None):
        """Add a tool to the server"""
        self.tools[tool.name] = tool
        if handler:
            self.request_handlers[f"tool:{tool.name}"] = handler
        self.logger.info(f"Added tool: {tool.name}")

    def add_prompt(self, prompt: MCPPrompt):
        """Add a prompt template to the server"""
        self.prompts[prompt.name] = prompt
        self.logger.info(f"Added prompt: {prompt.name}")

    async def handle_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request"""
        try:
            method = message.get("method")
            params = message.get("params", {})
            request_id = message.get("id")

            self.logger.debug(f"Handling request: {method}")

            if method in self.request_handlers:
                result = await self.request_handlers[method](params)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }

        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {"subscribe": False, "listChanged": False},
                "tools": {"listChanged": False},
                "prompts": {"listChanged": False},
                "logging": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list resources request"""
        return {
            "resources": [asdict(resource) for resource in self.resources.values()]
        }

    async def handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle read resource request"""
        uri = params.get("uri")
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        # This is where you'd actually read the resource content
        # For demo purposes, we'll return mock content
        content = await self._read_resource_content(uri)

        return {
            "contents": [{
                "uri": uri,
                "mimeType": self.resources[uri].mimeType or "text/plain",
                "text": content
            }]
        }

    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list tools request"""
        return {
            "tools": [asdict(tool) for tool in self.tools.values()]
        }

    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request"""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")

        # Execute the tool
        result = await self._execute_tool(name, arguments)

        return {
            "content": [{
                "type": "text",
                "text": str(result)
            }],
            "isError": False
        }

    async def handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list prompts request"""
        return {
            "prompts": [asdict(prompt) for prompt in self.prompts.values()]
        }

    async def handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get prompt request"""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name not in self.prompts:
            raise ValueError(f"Prompt not found: {name}")

        # Generate the prompt with arguments
        prompt_content = await self._generate_prompt(name, arguments)

        return {
            "description": self.prompts[name].description,
            "messages": [{
                "role": "user",
                "content": {
                    "type": "text",
                    "text": prompt_content
                }
            }]
        }

    async def _read_resource_content(self, uri: str) -> str:
        """Read resource content (implement based on resource type)"""
        resource = self.resources[uri]

        if uri.startswith("file://"):
            # File system resource
            file_path = uri[7:]  # Remove file:// prefix
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        elif uri.startswith("config://"):
            # Configuration resource
            return f"Configuration data for {resource.name}"

        else:
            # Mock content for other resources
            return f"Mock content for resource: {resource.name}"

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool (implement tool-specific logic)"""
        if name == "calculator":
            expression = arguments.get("expression", "")
            try:
                # Simple eval for demo - use safe math library in production
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"

        elif name == "file_reader":
            filepath = arguments.get("path", "")
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                return f"File content ({len(content)} chars):\\n{content[:500]}..."
            except Exception as e:
                return f"Error reading file: {e}"

        elif name == "system_info":
            import platform
            import psutil
            info = {
                "platform": platform.system(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
            }
            return json.dumps(info, indent=2)

        else:
            return f"Tool '{name}' executed with arguments: {arguments}"

    async def _generate_prompt(self, name: str, arguments: Dict[str, Any]) -> str:
        """Generate prompt content with arguments"""
        prompt = self.prompts[name]

        if name == "code_review":
            code = arguments.get("code", "")
            language = arguments.get("language", "python")
            return f"Please review this {language} code:\\n\\n```{language}\\n{code}\\n```\\n\\nProvide feedback on code quality, potential issues, and suggestions for improvement."

        elif name == "summarize":
            text = arguments.get("text", "")
            max_length = arguments.get("max_length", 100)
            return f"Please summarize the following text in no more than {max_length} words:\\n\\n{text}"

        else:
            return f"Prompt '{name}' with arguments: {arguments}"


class MCPClient:
    """Basic MCP Client implementation"""

    def __init__(self, name: str):
        self.name = name
        self.server_capabilities = None
        self.next_id = 1
        self.logger = logging.getLogger(f"mcp.client.{name}")

    def get_next_id(self) -> int:
        """Get next request ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id

    async def initialize(self, server: MCPServer) -> Dict[str, Any]:
        """Initialize connection with server"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": self.name,
                    "version": "1.0.0"
                }
            }
        }

        response = await server.handle_request(request)
        if "error" in response:
            raise Exception(f"Initialization failed: {response['error']}")

        self.server_capabilities = response["result"]["capabilities"]
        self.logger.info("Client initialized successfully")
        return response["result"]

    async def list_resources(self, server: MCPServer) -> List[MCPResource]:
        """List available resources"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "resources/list"
        }

        response = await server.handle_request(request)
        if "error" in response:
            raise Exception(f"List resources failed: {response['error']}")

        resources = []
        for res_data in response["result"]["resources"]:
            resources.append(MCPResource(**res_data))
        return resources

    async def read_resource(self, server: MCPServer, uri: str) -> str:
        """Read a resource"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "resources/read",
            "params": {"uri": uri}
        }

        response = await server.handle_request(request)
        if "error" in response:
            raise Exception(f"Read resource failed: {response['error']}")

        contents = response["result"]["contents"]
        if contents:
            return contents[0]["text"]
        return ""

    async def list_tools(self, server: MCPServer) -> List[MCPTool]:
        """List available tools"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "tools/list"
        }

        response = await server.handle_request(request)
        if "error" in response:
            raise Exception(f"List tools failed: {response['error']}")

        tools = []
        for tool_data in response["result"]["tools"]:
            tools.append(MCPTool(**tool_data))
        return tools

    async def call_tool(self, server: MCPServer, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        response = await server.handle_request(request)
        if "error" in response:
            raise Exception(f"Tool call failed: {response['error']}")

        content = response["result"]["content"]
        if content and len(content) > 0:
            return content[0]["text"]
        return ""

    async def list_prompts(self, server: MCPServer) -> List[MCPPrompt]:
        """List available prompts"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": "prompts/list"
        }

        response = await server.handle_request(request)
        if "error" in response:
            raise Exception(f"List prompts failed: {response['error']}")

        prompts = []
        for prompt_data in response["result"]["prompts"]:
            prompts.append(MCPPrompt(**prompt_data))
        return prompts


async def demonstrate_mcp():
    """Demonstrate MCP server and client interaction"""

    print("🔗 Model Context Protocol (MCP) Demonstration")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create MCP server
    server = MCPServer("demo-server", "1.0.0")

    # Add resources
    server.add_resource(MCPResource(
        uri="file://demo.txt",
        name="Demo File",
        description="A demonstration text file",
        mimeType="text/plain"
    ))

    server.add_resource(MCPResource(
        uri="config://app-settings",
        name="App Settings",
        description="Application configuration",
        mimeType="application/json"
    ))

    # Add tools
    server.add_tool(MCPTool(
        name="calculator",
        description="Perform mathematical calculations",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    ))

    server.add_tool(MCPTool(
        name="system_info",
        description="Get system information",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ))

    # Add prompts
    server.add_prompt(MCPPrompt(
        name="code_review",
        description="Review code for quality and issues",
        arguments=[
            {"name": "code", "description": "Code to review", "required": True},
            {"name": "language", "description": "Programming language", "required": False}
        ]
    ))

    # Create MCP client
    client = MCPClient("demo-client")

    try:
        print("\\n🚀 Initializing MCP connection...")
        init_result = await client.initialize(server)
        print(f"✅ Connection initialized")
        print(f"   Protocol version: {init_result.get('protocolVersion')}")
        print(f"   Server: {init_result['serverInfo']['name']} v{init_result['serverInfo']['version']}")

        # List resources
        print("\\n📚 Available Resources:")
        resources = await client.list_resources(server)
        for resource in resources:
            print(f"   - {resource.name} ({resource.uri})")
            print(f"     Description: {resource.description}")

        # List tools
        print("\\n🔧 Available Tools:")
        tools = await client.list_tools(server)
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")

        # List prompts
        print("\\n💭 Available Prompts:")
        prompts = await client.list_prompts(server)
        for prompt in prompts:
            print(f"   - {prompt.name}: {prompt.description}")

        # Test tool calls
        print("\\n🧮 Testing Tool Calls:")
        print("-" * 30)

        # Calculator tool
        calc_result = await client.call_tool(server, "calculator", {"expression": "25 * 4 + 100"})
        print(f"Calculator (25 * 4 + 100): {calc_result}")

        # System info tool
        try:
            system_result = await client.call_tool(server, "system_info", {})
            print(f"System Info: {system_result}")
        except Exception as e:
            print(f"System Info: Error - {e}")

        # Test resource reading
        print("\\n📖 Reading Resources:")
        print("-" * 25)

        for resource in resources[:2]:  # Test first 2 resources
            try:
                content = await client.read_resource(server, resource.uri)
                print(f"{resource.name}: {content[:100]}..." if len(content) > 100 else f"{resource.name}: {content}")
            except Exception as e:
                print(f"{resource.name}: Error - {e}")

        print("\\n✅ MCP demonstration completed!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")

    print("\\n📋 MCP Key Concepts Demonstrated:")
    print("- Server-client architecture with JSON-RPC protocol")
    print("- Resource abstraction for data access")
    print("- Tool abstraction for function execution")
    print("- Prompt templates for reusable interactions")
    print("- Standardized error handling and capability negotiation")


class FilesystemMCPServer(MCPServer):
    """Example MCP server for filesystem access"""

    def __init__(self, base_path: str = "."):
        super().__init__("filesystem-server")
        self.base_path = Path(base_path).resolve()
        self._setup_filesystem_resources()

    def _setup_filesystem_resources(self):
        """Setup filesystem-based resources"""
        # Add files in base path as resources
        for file_path in self.base_path.glob("*.py"):
            relative_path = file_path.relative_to(self.base_path)
            self.add_resource(MCPResource(
                uri=f"file://{file_path}",
                name=str(relative_path),
                description=f"Python file: {relative_path}",
                mimeType="text/x-python"
            ))

        # Add filesystem tools
        self.add_tool(MCPTool(
            name="list_files",
            description="List files in directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "File pattern (optional)"}
                },
                "required": ["path"]
            }
        ))

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute filesystem tools"""
        if name == "list_files":
            path = arguments.get("path", ".")
            pattern = arguments.get("pattern", "*")

            try:
                dir_path = Path(path)
                if not dir_path.is_dir():
                    return f"Error: {path} is not a directory"

                files = list(dir_path.glob(pattern))
                file_list = [str(f.name) for f in files]
                return f"Files in {path}: {file_list}"

            except Exception as e:
                return f"Error listing files: {e}"

        return await super()._execute_tool(name, arguments)


async def demonstrate_filesystem_mcp():
    """Demonstrate filesystem MCP server"""

    print("\\n📁 Filesystem MCP Server Demonstration")
    print("=" * 45)

    # Create filesystem server
    fs_server = FilesystemMCPServer(".")
    client = MCPClient("fs-client")

    try:
        # Initialize
        await client.initialize(fs_server)
        print("✅ Filesystem MCP server initialized")

        # List Python file resources
        resources = await client.list_resources(fs_server)
        print(f"\\n📚 Found {len(resources)} Python file resources:")
        for resource in resources[:5]:  # Show first 5
            print(f"   - {resource.name}")

        # Test file listing tool
        print("\\n📂 Testing file listing tool:")
        result = await client.call_tool(fs_server, "list_files", {
            "path": ".",
            "pattern": "*.py"
        })
        print(f"Python files: {result}")

    except Exception as e:
        print(f"❌ Filesystem MCP error: {e}")


if __name__ == "__main__":
    async def main():
        await demonstrate_mcp()
        await demonstrate_filesystem_mcp()

    # Check if we need to install additional dependencies
    try:
        import psutil
    except ImportError:
        print("⚠️  psutil not installed - system_info tool will not work")
        print("   Install with: pip install psutil")

    asyncio.run(main())