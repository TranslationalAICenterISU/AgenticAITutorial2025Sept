# Troubleshooting Guide

## Common Issues and Solutions

### 1. Installation Problems

#### Issue: `pip install` fails with permissions error
```bash
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

#### Issue: Package conflicts or version errors
```bash
ERROR: pip's dependency resolver does not currently consider installed packages
```

**Solution:**
```bash
# Clean install in fresh virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. API Key Issues

#### Issue: `API key not found` errors
```
ValueError: OPENAI_API_KEY not found in environment variables
```

**Solution:**
1. **Check .env file exists:**
   ```bash
   ls -la .env
   ```

2. **Verify .env file format (no spaces around =):**
   ```bash
   # Wrong:
   OPENAI_API_KEY = sk-your-key

   # Correct:
   OPENAI_API_KEY=sk-your-key
   ```

3. **Check file encoding (should be UTF-8):**
   ```bash
   file .env
   ```

4. **Reload environment:**
   ```bash
   # Restart your terminal or
   source venv/bin/activate
   ```

#### Issue: `Invalid API key` errors
```
AuthenticationError: Invalid API key provided
```

**Solutions:**
1. **Verify key format:**
   - OpenAI keys start with `sk-`
   - Anthropic keys start with `sk-ant-`

2. **Check key status:**
   - Go to provider's console
   - Verify key is active
   - Check usage limits

3. **Test key manually:**
   ```bash
   # Test OpenAI key
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"

   # Test Anthropic key
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"claude-3-haiku-20240307","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
   ```

### 3. Import Errors

#### Issue: `ModuleNotFoundError`
```python
ModuleNotFoundError: No module named 'openai'
```

**Solutions:**
1. **Verify virtual environment is activated:**
   ```bash
   which python
   # Should show path to venv/bin/python
   ```

2. **Install missing package:**
   ```bash
   pip install openai
   ```

3. **Check Python path:**
   ```bash
   python -c "import sys; print(sys.path)"
   ```

#### Issue: Import works but functions fail
```python
AttributeError: module 'openai' has no attribute 'ChatCompletion'
```

**Solution:**
Update to newer API version:
```bash
pip install --upgrade openai>=1.0.0
```

### 4. Local Model Issues

#### Issue: Ollama not found
```
ConnectionError: Could not connect to Ollama server
```

**Solutions:**
1. **Install Ollama:**
   - Visit [ollama.ai](https://ollama.ai)
   - Download for your OS
   - Follow installation instructions

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull required models:**
   ```bash
   ollama pull llama3.2:3b
   ollama pull mistral:7b
   ```

4. **Check Ollama is running:**
   ```bash
   curl http://localhost:11434/api/version
   ```

#### Issue: GPU out of memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use smaller models:**
   ```bash
   ollama pull llama3.2:3b  # instead of larger models
   ```

2. **Reduce context length:**
   ```python
   # In your code
   max_tokens = 512  # instead of 2048
   ```

3. **Use CPU inference:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

### 5. Network and Connection Issues

#### Issue: SSL Certificate errors
```
SSLError: HTTPSConnectionPool(host='api.openai.com', port=443)
```

**Solutions:**
1. **Update certificates:**
   ```bash
   pip install --upgrade certifi
   ```

2. **On macOS:**
   ```bash
   /Applications/Python\ 3.x/Install\ Certificates.command
   ```

3. **Corporate networks:**
   ```bash
   # Set proxy if needed
   export https_proxy=http://proxy.company.com:8080
   export http_proxy=http://proxy.company.com:8080
   ```

#### Issue: Rate limiting errors
```
RateLimitError: You exceeded your current quota
```

**Solutions:**
1. **Check usage:**
   - Visit provider's console
   - Check billing and usage

2. **Implement retry with backoff:**
   ```python
   import time
   import random

   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
       raise
   ```

3. **Use different models:**
   - Switch to less expensive models
   - Use local models for development

### 6. Framework-Specific Issues

#### Issue: LangChain import errors
```python
ImportError: cannot import name 'ChatOpenAI' from 'langchain.llms'
```

**Solution:**
Use correct import paths:
```python
# Old way (deprecated)
from langchain.llms import ChatOpenAI

# New way
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
```

#### Issue: LangGraph execution hangs
```python
# Agent seems to hang during execution
```

**Solutions:**
1. **Add timeout:**
   ```python
   app = workflow.compile()
   result = app.invoke(input_data, {"timeout": 30})
   ```

2. **Enable debug mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Check for infinite loops:**
   - Review conditional logic
   - Add maximum iteration limits

### 7. Exercise-Specific Issues

#### Issue: Exercise files won't run
```bash
python exercise_1_agent_analysis.py
# No output or errors
```

**Solutions:**
1. **Check Python version:**
   ```bash
   python --version
   # Should be 3.8+
   ```

2. **Run with verbose output:**
   ```bash
   python -v exercise_1_agent_analysis.py
   ```

3. **Check file permissions:**
   ```bash
   chmod +x exercise_1_agent_analysis.py
   ```

#### Issue: Interactive exercises not working
```python
KeyboardInterrupt during input()
```

**Solutions:**
1. **Use different Python environment:**
   ```bash
   # Try IPython
   pip install ipython
   ipython exercise_1_agent_analysis.py

   # Or Jupyter
   jupyter notebook
   ```

2. **Run in IDE:**
   - VS Code with Python extension
   - PyCharm
   - Other IDE with interactive support

### 8. Performance Issues

#### Issue: Slow API responses
```
Response takes >30 seconds
```

**Solutions:**
1. **Reduce max_tokens:**
   ```python
   max_tokens = 512  # instead of 2048
   ```

2. **Use faster models:**
   - GPT-3.5-turbo instead of GPT-4
   - Claude Haiku instead of Claude Opus

3. **Enable streaming:**
   ```python
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=messages,
       stream=True
   )
   ```

#### Issue: High memory usage
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Process in batches:**
   ```python
   batch_size = 10
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size]
       process_batch(batch)
   ```

2. **Clear memory:**
   ```python
   import gc

   # After processing
   del large_variable
   gc.collect()
   ```

### 9. Development Environment Issues

#### Issue: VS Code not recognizing virtual environment
**Solution:**
1. **Set Python interpreter:**
   - Ctrl+Shift+P → "Python: Select Interpreter"
   - Choose venv/bin/python

2. **Reload VS Code:**
   - Close and reopen VS Code
   - Or reload window (Ctrl+Shift+P → "Developer: Reload Window")

#### Issue: Jupyter notebook kernel issues
**Solution:**
```bash
# Install kernel for virtual environment
pip install ipykernel
python -m ipykernel install --user --name agentic-ai --display-name "Agentic AI Tutorial"

# Start Jupyter
jupyter notebook

# Select "Agentic AI Tutorial" kernel
```

### 10. Getting Help

#### When to seek help:
1. Error persists after trying solutions above
2. Hardware-specific issues (GPU drivers, etc.)
3. Corporate network/firewall issues
4. Novel error messages not covered here

#### How to get help:
1. **Create GitHub issue with:**
   - Full error message
   - Steps to reproduce
   - System information (OS, Python version)
   - What you've already tried

2. **Include diagnostic information:**
   ```bash
   # System info
   python --version
   pip --version
   pip list | grep -E "(openai|anthropic|langchain)"

   # Environment info
   echo $OPENAI_API_KEY | head -c 10  # Don't share full key!
   ls -la .env
   ```

3. **Community resources:**
   - Tutorial GitHub discussions
   - Stack Overflow with relevant tags
   - Framework-specific Discord/forums

## Prevention Tips

1. **Use virtual environments always**
2. **Keep dependencies updated regularly**
3. **Monitor API usage and billing**
4. **Test with small examples first**
5. **Read error messages carefully**
6. **Keep backups of working configurations**

## Quick Diagnostic Commands

```bash
# Environment check
python verify_setup.py

# Package versions
pip list | grep -E "(openai|anthropic|langchain)"

# API key format check (don't run with real keys)
echo $OPENAI_API_KEY | grep -E "^sk-"
echo $ANTHROPIC_API_KEY | grep -E "^sk-ant-"

# Network connectivity
curl -I https://api.openai.com
curl -I https://api.anthropic.com

# Local model check
curl http://localhost:11434/api/version
```

Remember: Most issues are environment-related. When in doubt, try a fresh virtual environment!