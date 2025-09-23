# Hands-On Agentic AI Tutorial

A comprehensive tutorial covering Agentic AI concepts and practical implementations using leading frameworks.

## Overview

This tutorial provides hands-on experience with Agentic AI systems, covering both theoretical foundations and practical implementations using popular frameworks including OpenAI, Anthropic, LangChain, LangGraph, Google ADK, CrewAI, SmolAgents, DSPy, Parlant, MCP (Model Context Protocol), and local model deployments (Ollama, HuggingFace).

## Quick Start

### Prerequisites
- Python 3.8+ installed
- Basic understanding of Python programming
- Familiarity with APIs and HTTP requests
- API keys for OpenAI and Anthropic

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TranslationalAICenterISU/AgenticAITutorial2025Sept
   cd AgenticAITutorial2025Sept
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv

   # Activate (Linux/Mac):
   source venv/bin/activate

   # Activate (Windows):
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env file with your actual API keys
   ```

5. **Verify setup:**
   ```bash
   python verify_setup.py
   ```

   You should see all green checkmarks indicating successful setup.

## Learning Path

### Core Modules

```
├── 01-foundations/          # Agentic AI concepts and theory
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

### Recommended Learning Sequence

**Phase 1: Foundations**
- Module 1: Core Concepts - Understanding agentic behavior
- Module 2: LLM APIs - Direct model integration

**Phase 2: Popular Frameworks**
- Module 3: LangChain - Rapid agent development
- Module 4: LangGraph - Complex workflows
- Module 5: Google ADK - Multimodal capabilities

**Phase 3: Specialized Frameworks**
- Module 6: CrewAI - Multi-agent collaboration
- Module 7: SmolAgents - Lightweight approach
- Module 8: DSPy - Self-improving systems
- Module 9: Parlant - Conversation systems

**Phase 4: Production & Integration**
- Module 10: Integration Patterns
- Module 11: Local Models
- Module 12: Model Context Protocol

### Getting Started with Each Module

Each module contains:
- `README.md` - Learning objectives and concepts
- `*_basics.py` - Core implementation examples
- `exercises/` - Hands-on coding challenges
- Additional specialized examples

To start learning:
```bash
cd 01-foundations
python simple_agent.py
```

## Learning Outcomes

By the end of this tutorial, you will be able to:
- Understand core Agentic AI concepts and architectures
- Build agents using multiple frameworks
- Choose the right framework for specific use cases
- Implement multi-agent systems and workflows
- Integrate different frameworks effectively
- Deploy and monitor agentic systems

## Framework Coverage

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

## Common Setup Issues & Solutions

### API Key Issues
```bash
# Check if API keys are set
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Anthropic:', bool(os.getenv('ANTHROPIC_API_KEY')))"
```

### Package Installation Issues
```bash
# If packages fail to install, try:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

### Virtual Environment Issues
```bash
# If activation doesn't work:
python -m venv --clear venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Import Errors
- Ensure virtual environment is activated
- Run `pip list` to verify package installation
- Check Python version: `python --version` (needs 3.8+)

### Framework-Specific Issues
See `TROUBLESHOOTING.md` for detailed solutions.

## Support

- **Setup Issues**: Run `python verify_setup.py` for diagnostics
- **Code Issues**: Check working examples in each module
- **Concept Questions**: Review module README files
- **Bug Reports**: Create GitHub issue

## Contributing

Contributions welcome! Please:
- Follow existing code style and patterns
- Update documentation for any new features
- Include error handling in examples

## Architecture Patterns Demonstrated

- **ReAct Pattern**: Reasoning and acting cycles
- **Multi-Agent Systems**: Collaborative agent workflows
- **Tool Integration**: External API and service integration
- **Memory Management**: Persistent state across interactions
- **Workflow Orchestration**: Complex multi-step processes

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

---

**Ready to start?** Navigate to `01-foundations/` and run your first agentic AI example!