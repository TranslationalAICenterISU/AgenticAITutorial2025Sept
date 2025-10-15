"""
Model Context Protocol (MCP) Basics - Using Official SDK
Introduction to building MCP servers and clients for agent tool integration
"""

import asyncio
import json
import logging
import platform
from pathlib import Path
from typing import Any

# Official MCP Python SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_server() -> Server:
    """Create a demo MCP server with resources, tools, and prompts"""

    server = Server("demo-server")

    # Define resources
    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        """List available resources"""
        return [
            types.Resource(
                uri="file://demo.txt",
                name="Demo File",
                description="A demonstration text file",
                mimeType="text/plain"
            ),
            types.Resource(
                uri="config://app-settings",
                name="App Settings",
                description="Application configuration",
                mimeType="application/json"
            )
        ]

    @server.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """Read a specific resource"""
        if uri == "file://demo.txt":
            return "This is demo content from the MCP server."
        elif uri == "config://app-settings":
            return json.dumps({
                "app_name": "MCP Demo",
                "version": "1.0.0",
                "features": ["resources", "tools", "prompts"]
            }, indent=2)
        else:
            raise ValueError(f"Resource not found: {uri}")

    # Define tools
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="calculator",
                description="Perform mathematical calculations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            types.Tool(
                name="system_info",
                description="Get system information",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any]
    ) -> list[types.TextContent]:
        """Execute a tool"""

        if name == "calculator":
            expression = arguments.get("expression", "")
            try:
                # Simple eval for demo - use safe math library in production
                result = eval(expression, {"__builtins__": {}}, {})
                return [types.TextContent(
                    type="text",
                    text=f"Result: {result}"
                )]
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

        elif name == "system_info":
            try:
                import psutil
                info = {
                    "platform": platform.system(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
                }
                return [types.TextContent(
                    type="text",
                    text=json.dumps(info, indent=2)
                )]
            except ImportError:
                return [types.TextContent(
                    type="text",
                    text="Error: psutil not installed. Install with: pip install psutil"
                )]
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

        else:
            raise ValueError(f"Unknown tool: {name}")

    # Define prompts
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        """List available prompts"""
        return [
            types.Prompt(
                name="code_review",
                description="Review code for quality and issues",
                arguments=[
                    types.PromptArgument(
                        name="code",
                        description="Code to review",
                        required=True
                    ),
                    types.PromptArgument(
                        name="language",
                        description="Programming language",
                        required=False
                    )
                ]
            ),
            types.Prompt(
                name="summarize",
                description="Summarize text content",
                arguments=[
                    types.PromptArgument(
                        name="text",
                        description="Text to summarize",
                        required=True
                    ),
                    types.PromptArgument(
                        name="max_length",
                        description="Maximum length in words",
                        required=False
                    )
                ]
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(
        name: str,
        arguments: dict[str, str] | None
    ) -> types.GetPromptResult:
        """Get a specific prompt"""
        args = arguments or {}

        if name == "code_review":
            code = args.get("code", "")
            language = args.get("language", "python")
            prompt_text = (
                f"Please review this {language} code:\n\n"
                f"```{language}\n{code}\n```\n\n"
                "Provide feedback on code quality, potential issues, "
                "and suggestions for improvement."
            )
        elif name == "summarize":
            text = args.get("text", "")
            max_length = args.get("max_length", "100")
            prompt_text = (
                f"Please summarize the following text in no more than "
                f"{max_length} words:\n\n{text}"
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

        return types.GetPromptResult(
            description=f"Generated {name} prompt",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=prompt_text
                    )
                )
            ]
        )

    return server


def create_filesystem_server(base_path: str = ".") -> Server:
    """Create an MCP server for filesystem access"""

    server = Server("filesystem-server")
    base = Path(base_path).resolve()

    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        """List Python files as resources"""
        resources = []
        for file_path in base.glob("*.py"):
            relative_path = file_path.relative_to(base)
            resources.append(types.Resource(
                uri=f"file://{file_path}",
                name=str(relative_path),
                description=f"Python file: {relative_path}",
                mimeType="text/x-python"
            ))
        return resources

    @server.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """Read a file resource"""
        if uri.startswith("file://"):
            file_path = uri[7:]
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Error reading file: {e}")
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List filesystem tools"""
        return [
            types.Tool(
                name="list_files",
                description="List files in directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "File pattern (e.g., '*.py')"
                        }
                    },
                    "required": ["path"]
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any]
    ) -> list[types.TextContent]:
        """Execute filesystem tools"""

        if name == "list_files":
            path = arguments.get("path", ".")
            pattern = arguments.get("pattern", "*")

            try:
                dir_path = Path(path)
                if not dir_path.is_dir():
                    return [types.TextContent(
                        type="text",
                        text=f"Error: {path} is not a directory"
                    )]

                files = list(dir_path.glob(pattern))
                file_list = [str(f.name) for f in files]

                return [types.TextContent(
                    type="text",
                    text=f"Files in {path}: {file_list}"
                )]
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error listing files: {e}"
                )]
        else:
            raise ValueError(f"Unknown tool: {name}")

    return server


async def run_server_stdio():
    """Run MCP server using stdio transport (for production use)"""
    server = create_demo_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


async def demonstrate_mcp_with_client():
    """Demonstrate MCP interaction using a client"""

    print("🔗 Model Context Protocol (MCP) Demonstration")
    print("=" * 50)
    print("\n⚠️  Note: This is a simplified demo showing MCP concepts.")
    print("In production, client would connect to server via stdio/SSE transport.\n")

    # Create server directly for demo purposes
    server = create_demo_server()

    print("✅ MCP Server created with capabilities:")
    print("   - Resources: File and configuration access")
    print("   - Tools: Calculator and system info")
    print("   - Prompts: Code review and summarization templates")

    print("\n📋 MCP Key Concepts:")
    print("=" * 50)
    print("1. Server Architecture:")
    print("   - Decorators: @server.list_tools(), @server.call_tool(), etc.")
    print("   - Type-safe: Uses mcp.types for all protocol messages")
    print("   - Async: Built on asyncio for high performance")

    print("\n2. Resources:")
    print("   - Provide data access (files, configs, databases)")
    print("   - URI-based addressing (file://, config://, etc.)")
    print("   - MIME type support for content negotiation")

    print("\n3. Tools:")
    print("   - Expose functions to LLMs")
    print("   - JSON Schema for input validation")
    print("   - Structured output with TextContent/ImageContent")

    print("\n4. Prompts:")
    print("   - Reusable prompt templates")
    print("   - Parameterized with arguments")
    print("   - Returns formatted messages for LLMs")

    print("\n5. Transports:")
    print("   - stdio: Standard input/output (subprocess)")
    print("   - SSE: Server-Sent Events (HTTP)")
    print("   - WebSocket: Bidirectional communication")

    print("\n📝 Example Tool Definition:")
    print("-" * 50)
    print("""
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculator",
            description="Perform calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "calculator":
        result = eval(arguments["expression"])
        return [types.TextContent(type="text", text=f"Result: {result}")]
    """)

    print("\n📝 Example Client Usage:")
    print("-" * 50)
    print("""
# Connect to MCP server via stdio
async with stdio_client(
    StdioServerParameters(command="python", args=["server.py"])
) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # List and call tools
        tools = await session.list_tools()
        result = await session.call_tool("calculator", {"expression": "2+2"})

        # Read resources
        resources = await session.list_resources()
        content = await session.read_resource(uri="file://data.txt")

        # Get prompts
        prompts = await session.list_prompts()
        prompt = await session.get_prompt("code_review", {"code": "..."})
    """)

    print("\n🚀 Production Usage:")
    print("-" * 50)
    print("1. Run server: python mcp_basics.py --server")
    print("2. Configure in Claude Desktop or other MCP clients")
    print("3. Server exposes tools/resources/prompts to LLM")
    print("4. LLM can call tools and access resources automatically")

    print("\n🔧 Filesystem MCP Server Example:")
    print("-" * 50)
    fs_server = create_filesystem_server(".")
    print("✅ Filesystem server created")
    print("   - Exposes .py files as resources")
    print("   - Provides list_files tool")
    print("   - Can be used to give LLMs filesystem access")

    print("\n✅ MCP demonstration completed!")


async def demonstrate_filesystem_mcp():
    """Demonstrate filesystem MCP server"""

    print("\n📁 Filesystem MCP Server Demonstration")
    print("=" * 45)

    server = create_filesystem_server(".")

    print("✅ Filesystem MCP server initialized")
    print("\n💡 Filesystem Server Features:")
    print("   - Lists Python files as MCP resources")
    print("   - Provides file reading capability")
    print("   - Includes list_files tool for directory listing")
    print("\n📝 This server can be used by LLMs to:")
    print("   - Read project files")
    print("   - Understand codebase structure")
    print("   - Provide code-aware assistance")


if __name__ == "__main__":
    import sys

    # Check for psutil
    try:
        import psutil
    except ImportError:
        print("⚠️  psutil not installed - system_info tool will not work")
        print("   Install with: pip install psutil\n")

    # Allow running as server or demo
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        print("🚀 Starting MCP server on stdio...")
        asyncio.run(run_server_stdio())
    else:
        async def main():
            await demonstrate_mcp_with_client()
            await demonstrate_filesystem_mcp()

        asyncio.run(main())
