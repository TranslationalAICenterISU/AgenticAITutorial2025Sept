# Module 12: Model Context Protocol (MCP)

## Overview
Learn MCP (Model Context Protocol), Anthropic's open standard for connecting AI assistants to external systems and data sources through a unified interface.

## Learning Objectives
- Understand the Model Context Protocol specification
- Build MCP servers to expose tools and resources
- Connect agents to external systems via MCP
- Implement bidirectional communication between agents and tools
- Design scalable, interoperable agent architectures

## Topics Covered

### 1. MCP Fundamentals
- Protocol specification and architecture
- Client-server communication model
- Resource, tool, and prompt abstractions
- Transport layers (stdio, SSE, WebSocket)

### 2. MCP Server Development
- Server implementation patterns
- Resource management and access
- Tool registration and execution
- Error handling and logging

### 3. MCP Client Integration
- Client connection management
- Resource discovery and access
- Tool invocation patterns
- Session management

### 4. Advanced MCP Features
- Sampling and LLM integration
- Progressive enhancement
- Security and authentication
- Performance optimization

### 5. Production Deployment
- Server hosting and scaling
- Client configuration management
- Monitoring and observability
- Integration with existing systems

### 6. Ecosystem Integration
- Claude Desktop integration
- VS Code extension patterns
- Custom application integration
- Multi-server orchestration

## Hands-On Activities
1. **First MCP Server**: Build a basic server with tools
2. **Resource Server**: Create server exposing file system resources
3. **Database MCP**: Connect agents to databases via MCP
4. **Multi-Server Setup**: Orchestrate multiple MCP servers
5. **Production Deployment**: Deploy MCP server in cloud environment

## Files in This Module
- `mcp_basics.py` - Core MCP concepts and simple server
- `mcp_server.py` - Complete server implementation
- `mcp_client.py` - Client connection examples
- `database_mcp.py` - Database integration server
- `file_system_mcp.py` - File system resource server
- `production_deployment.py` - Deployment examples
- `exercises/` - Hands-on coding exercises

## MCP Architecture

### Core Concepts
```
┌─────────────┐     MCP Protocol     ┌─────────────┐
│   Client    │ ←──────────────────→ │   Server    │
│ (AI Agent)  │                     │ (Tools/Data)│
└─────────────┘                     └─────────────┘
```

### Protocol Layers
1. **Transport**: How messages are sent (stdio, WebSocket, SSE)
2. **Protocol**: Message format and routing
3. **Application**: Specific tools, resources, and prompts

### Message Types
- **Resources**: Read-only data sources
- **Tools**: Executable functions
- **Prompts**: Reusable prompt templates
- **Sampling**: LLM integration requests

## Key Features

### Resources
- File system access
- Database queries
- API endpoint data
- Configuration files
- Log files and metrics

### Tools
- System operations
- External API calls
- Data processing functions
- Workflow automation
- Custom business logic

### Prompts
- Template management
- Context injection
- Prompt versioning
- Dynamic prompt generation
- A/B testing support

## Implementation Patterns

### Simple Server
```python
from mcp import Server
from mcp.types import Tool, TextContent

server = Server("my-mcp-server")

@server.list_tools()
async def list_tools():
    return [Tool(name="calculator", description="Math operations")]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "calculator":
        return TextContent(type="text", text=str(eval(arguments["expression"])))
```

### Client Connection
```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client(["python", "server.py"]) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()

        tools = await session.list_tools()
        result = await session.call_tool("calculator", {"expression": "2+2"})
```

## Use Cases

### Development Tools
- Code analysis and refactoring
- Build system integration
- Testing and deployment
- Documentation generation
- Performance monitoring

### Data Integration
- Database connectivity
- API aggregation
- File system access
- Cloud storage integration
- Real-time data streams

### Business Applications
- CRM integration
- Document management
- Workflow automation
- Analytics and reporting
- Customer support tools

### System Administration
- Server monitoring
- Log analysis
- Configuration management
- Security auditing
- Backup and recovery

## Security Considerations

### Authentication
- API key management
- OAuth integration
- Certificate-based auth
- Role-based access control
- Session management

### Authorization
- Resource-level permissions
- Tool execution policies
- Rate limiting
- Audit logging
- Sandboxing

### Data Protection
- Encryption in transit
- Sensitive data handling
- PII redaction
- Compliance requirements
- Data retention policies

## Performance Optimization

### Server Side
- Caching strategies
- Connection pooling
- Async operation handling
- Resource cleanup
- Memory management

### Client Side
- Connection reuse
- Request batching
- Response caching
- Error recovery
- Timeout handling

### Network
- Compression
- Keep-alive connections
- Load balancing
- Failover mechanisms
- Latency optimization

## Integration Examples

### With LangChain
```python
from langchain.tools import Tool
from mcp_client import MCPClient

class MCPTool(Tool):
    def __init__(self, mcp_client, tool_name):
        self.mcp_client = mcp_client
        super().__init__(name=tool_name, func=self._run)

    def _run(self, arguments):
        return self.mcp_client.call_tool(self.name, arguments)
```

### With CrewAI
```python
from crewai import Tool as CrewTool

def create_mcp_tools(mcp_client):
    tools = mcp_client.list_tools()
    return [
        CrewTool(
            name=tool.name,
            description=tool.description,
            func=lambda args: mcp_client.call_tool(tool.name, args)
        )
        for tool in tools
    ]
```

### With Local Models
```python
# MCP server for local model access
class LocalModelMCPServer:
    def __init__(self, model_name):
        self.model = load_local_model(model_name)

    async def generate_text(self, prompt):
        return await self.model.generate(prompt)
```

## Deployment Strategies

### Standalone Servers
- Docker containers
- Systemd services
- Cloud functions
- Kubernetes pods
- Edge deployment

### Embedded Servers
- Application plugins
- Library integration
- In-process servers
- Shared libraries
- Database extensions

### Distributed Architecture
- Microservices pattern
- API gateway integration
- Service mesh deployment
- Event-driven architecture
- Serverless functions

## Monitoring and Observability

### Metrics
- Request/response rates
- Latency percentiles
- Error rates
- Resource utilization
- Connection counts

### Logging
- Structured logging
- Request tracing
- Error tracking
- Audit trails
- Performance profiling

### Health Checks
- Service availability
- Resource connectivity
- Performance benchmarks
- Dependency status
- Version compatibility

## Prerequisites
- Understanding of client-server architecture
- JSON-RPC or similar protocol experience
- Async programming concepts (helpful)
- Completed foundational modules

## Next Steps
After completing this module, you'll understand how to:
- Build production-ready MCP servers
- Integrate MCP with existing agent frameworks
- Design scalable agent-tool architectures
- Deploy and monitor MCP-based systems
- Contribute to the MCP ecosystem