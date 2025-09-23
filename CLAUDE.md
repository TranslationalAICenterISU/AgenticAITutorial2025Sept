# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a comprehensive Agentic AI tutorial covering foundations, multiple frameworks, and hands-on implementation exercises. The tutorial is designed to teach both theoretical concepts and practical skills for building autonomous AI systems.

## Repository Structure

```
├── 01-foundations/          # Core Agentic AI concepts and theory
├── 02-llm-apis/            # OpenAI and Anthropic API fundamentals
├── 03-langchain/           # LangChain framework and tools
├── 04-langgraph/           # LangGraph for complex workflows
├── 05-google-adk/          # Google AI Development Kit
├── 06-crewai/              # CrewAI multi-agent framework
├── 07-smolagents/          # SmolAgents lightweight framework
├── 08-dspy/                # DSPy programming model
├── 09-parlant/             # Parlant conversation framework
├── 10-integration/         # Multi-framework integration
├── 11-local-models/        # Local models and small LLMs
├── 12-mcp/                 # Model Context Protocol
├── examples/               # Complete example implementations
├── exercises/              # Hands-on coding exercises
└── resources/              # Additional learning resources
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- API keys for OpenAI and Anthropic

### Quick Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python verify_setup.py
```

### Verification
Run the setup verification script to ensure all dependencies and API keys are properly configured:
```bash
python verify_setup.py
```

## Key Framework Coverage

The tutorial covers these major agentic AI frameworks:
- **OpenAI/Anthropic APIs**: Direct model integration
- **LangChain**: Tool integration and agent chains
- **LangGraph**: Complex workflow orchestration
- **Google ADK**: Google AI platform integration with Gemini
- **CrewAI**: Multi-agent collaborative systems
- **SmolAgents**: Lightweight agent implementations
- **DSPy**: Declarative self-improving programs
- **Parlant**: Conversation-focused agents
- **Local Models**: Ollama, HuggingFace for private deployment
- **MCP**: Model Context Protocol for standardized tool integration

## Development Workflow

### Running Examples
Each module contains runnable examples:
```bash
cd 01-foundations/
python simple_agent.py
```

### Testing Framework Code
Verify individual framework installations:
```bash
cd 02-llm-apis/
python openai_basics.py
```

### Tutorial Progression
Follow the modules in order for structured learning:
1. Start with `01-foundations/` for core concepts
2. Progress through `02-llm-apis/` to `12-mcp/` for framework-specific learning
3. Complete with integration exercises and final project

## Architecture Patterns

The repository demonstrates several key architectural patterns:
- **ReAct Pattern**: Reasoning and acting cycles
- **Multi-Agent Systems**: Collaborative agent workflows
- **Tool Integration**: External API and service integration
- **Memory Management**: Persistent state across interactions
- **Workflow Orchestration**: Complex multi-step processes

## Common Tasks

### Adding New Framework Examples
1. Create module directory: `mkdir XX-framework-name/`
2. Add README.md with learning objectives
3. Implement basic examples and exercises
4. Update main README.md with new module

### Testing API Integrations
Use the verification script to test all API connections:
```bash
python verify_setup.py
```

### Running Tutorial Sessions
Follow the tutorial modules in sequence for structured learning.

## Best Practices

### Code Organization
- Keep examples simple and well-commented
- Include error handling in all API interactions
- Provide both basic and advanced examples
- Document framework-specific considerations

### API Key Management
- Never commit API keys to repository
- Use environment variables for all credentials
- Provide clear setup instructions for learners
- Test with invalid keys to ensure proper error handling

## Final Integration Project

The tutorial culminates in a comprehensive integration project found in `exercises/integration/FINAL_PROJECT.md` where learners build a multi-framework system demonstrating real-world agentic AI application.