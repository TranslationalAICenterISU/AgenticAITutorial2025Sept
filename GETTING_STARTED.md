# Getting Started with Agentic AI Workshop

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AgenticAITutorial2025Sept.git
cd AgenticAITutorial2025Sept
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# If you encounter issues, install core dependencies first:
pip install openai anthropic python-dotenv langchain langgraph
```

### 4. Set Up API Keys

#### 4.1 Copy Environment Template
```bash
cp .env.example .env
```

#### 4.2 Edit .env File
Open `.env` in your text editor and add your API keys:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional API Keys
GOOGLE_API_KEY=your-google-api-key-here
SERPER_API_KEY=your-serper-key-here
```

#### 4.3 How to Get API Keys

**OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login
3. Navigate to API Keys section
4. Create new secret key
5. Copy the key (starts with `sk-`)

**Anthropic API Key:**
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up/login
3. Navigate to API Keys
4. Create new key
5. Copy the key (starts with `sk-ant-`)

**Google AI API Key (Optional):**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign up/login
3. Create API key
4. Copy the key

### 5. Verify Setup
```bash
python verify_setup.py
```

You should see:
```
🚀 Agentic AI Workshop Setup Verification
==================================================

📦 Checking Package Installation:
✅ openai is installed
✅ anthropic is installed
✅ langchain is installed
...

🔑 Checking API Configuration:
✅ OpenAI API key is configured
✅ Anthropic API key is configured
...

✅ Setup verification completed successfully!
```

### 6. Test Your First Agent
```bash
cd 01-foundations
python simple_agent.py
```

## Workshop Structure

### Learning Path
1. **Start Here**: `01-foundations/` - Core concepts
2. **Build Foundation**: `02-llm-apis/` - Direct API usage
3. **Add Tools**: `03-langchain/` - Framework integration
4. **Advanced Patterns**: Continue through remaining modules
5. **Final Project**: `exercises/integration/` - Capstone project

### Module Overview
```
01-foundations/     ← Start here (Theory + basic patterns)
02-llm-apis/        ← OpenAI & Anthropic integration
03-langchain/       ← Popular framework
04-langgraph/       ← Advanced workflows
05-google-adk/      ← Google AI platform
06-crewai/          ← Multi-agent systems
07-smolagents/      ← Lightweight approach
08-dspy/            ← Self-improving programs
09-parlant/         ← Conversation systems
10-integration/     ← Best practices
11-local-models/    ← Local deployment
12-mcp/             ← Protocol standards
```

## Common Issues and Solutions

### Issue: `ModuleNotFoundError`
**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: API Key Errors
**Solution:**
```bash
# Check .env file exists and has correct format
ls -la .env
cat .env

# Ensure no spaces around = in .env file
# Wrong: OPENAI_API_KEY = sk-...
# Right: OPENAI_API_KEY=sk-...
```

### Issue: SSL Certificate Errors
**Solution:**
```bash
pip install --upgrade certifi
# On macOS, also run:
/Applications/Python\\ 3.x/Install\\ Certificates.command
```

### Issue: Permission Errors
**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use sudo (Linux/Mac)
sudo pip install -r requirements.txt
```

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **Internet**: Stable connection for API calls
- **Python**: 3.8 or higher

### For Local Models (Module 11)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: Optional but recommended (8GB+ VRAM)
- **Storage**: 50GB+ for model downloads

## Development Environment Setup

### VS Code Setup
1. Install Python extension
2. Install Jupyter extension
3. Set Python interpreter to your virtual environment
4. Install recommended extensions:
   - Python
   - Jupyter
   - Python Docstring Generator
   - GitLens

### Jupyter Setup
```bash
# Install Jupyter in your virtual environment
pip install jupyter ipykernel

# Add virtual environment to Jupyter
python -m ipykernel install --user --name=agentic-ai

# Start Jupyter
jupyter notebook
```

## Getting Help

### Documentation
- Each module has detailed README with examples
- Check `TROUBLESHOOTING.md` for common issues
- Review `WORKSHOP_SCHEDULE.md` for structured learning

### Code Examples
Every module contains:
- `README.md` - Module overview and learning objectives
- `*_basics.py` - Core concepts with working examples
- `exercises/` - Hands-on coding challenges
- Additional example files for specific use cases

### Support
- Create GitHub issue for bugs
- Check existing issues for solutions
- Follow workshop schedule for optimal learning path

## Next Steps

1. **Complete Setup Verification**: Ensure all green checkmarks
2. **Start with Foundations**: `cd 01-foundations && python simple_agent.py`
3. **Follow Learning Path**: Progress through modules sequentially
4. **Practice with Exercises**: Complete exercises in each module
5. **Build Final Project**: Integration challenge in `exercises/integration/`

## Tips for Success

1. **Take Your Time**: Each module builds on previous concepts
2. **Run Examples**: Don't just read code - execute it
3. **Experiment**: Modify examples to understand behavior
4. **Ask Questions**: Use GitHub issues for help
5. **Join Community**: Connect with other learners

Ready to build intelligent agents? Start with `01-foundations/`! 🚀