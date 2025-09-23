"""
Ollama Integration for Local Model Agents
Demonstrates how to use Ollama for running local models in agentic systems
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class OllamaModel:
    """Represents an Ollama model configuration"""
    name: str
    size: str
    description: str
    context_length: int
    recommended_use: str


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self._get_available_models()

    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            else:
                return []
        except requests.exceptions.RequestException:
            return []

    def is_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )

            print(f"Pulling model {model_name}...")
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if "status" in data:
                        print(f"Status: {data['status']}")
                    if data.get("status") == "success":
                        return True
            return True
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False

    def generate(self, model: str, prompt: str, system: str = None, **kwargs) -> str:
        """Generate text using Ollama model"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error generating response: {e}"

    def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with Ollama model using OpenAI-compatible format"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload
            )

            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error in chat: {e}"


class OllamaAgent:
    """Agent using Ollama for local inference"""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.client = OllamaClient()
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []

        # Check if Ollama is running
        if not self.client.is_running():
            print("❌ Ollama server is not running. Please start it with 'ollama serve'")
            return

        # Check if model is available
        if model_name not in self.client.available_models:
            print(f"Model {model_name} not found. Attempting to pull...")
            if not self.client.pull_model(model_name):
                print(f"❌ Failed to pull model {model_name}")
                return

        print(f"✅ Ollama agent initialized with {model_name}")

    def chat(self, user_message: str, system_prompt: str = None) -> str:
        """Have a conversation with the local model"""

        # Prepare messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Generate response
        response = self.client.chat(self.model_name, messages)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def simple_generate(self, prompt: str, system: str = None) -> str:
        """Simple text generation without conversation history"""
        return self.client.generate(self.model_name, prompt, system)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class LocalModelComparison:
    """Compare performance across different local models"""

    def __init__(self):
        self.client = OllamaClient()
        self.recommended_models = [
            OllamaModel("llama3.2:3b", "2.0GB", "Fast, efficient for most tasks", 2048, "General purpose"),
            OllamaModel("mistral:7b", "4.1GB", "Strong reasoning and instruction following", 4096, "Complex reasoning"),
            OllamaModel("codellama:7b", "3.8GB", "Specialized for code tasks", 4096, "Code generation"),
            OllamaModel("phi3:3.8b", "2.3GB", "Microsoft's efficient small model", 4096, "Resource-constrained environments"),
            OllamaModel("qwen2:7b", "4.4GB", "Strong multilingual capabilities", 4096, "Multilingual tasks"),
        ]

    def benchmark_models(self, test_prompt: str, models: List[str] = None) -> Dict[str, Dict]:
        """Benchmark multiple models on the same prompt"""

        if models is None:
            models = [model.name for model in self.recommended_models
                     if model.name in self.client.available_models]

        results = {}

        for model in models:
            print(f"\\nTesting {model}...")

            start_time = time.time()
            response = self.client.generate(model, test_prompt)
            end_time = time.time()

            results[model] = {
                "response": response,
                "response_time": end_time - start_time,
                "response_length": len(response),
                "words_per_second": len(response.split()) / (end_time - start_time)
            }

        return results

    def print_benchmark_results(self, results: Dict[str, Dict]):
        """Print formatted benchmark results"""
        print("\\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)

        for model, metrics in results.items():
            print(f"\\n📊 {model.upper()}")
            print(f"   ⏱️  Response Time: {metrics['response_time']:.2f}s")
            print(f"   📝 Response Length: {metrics['response_length']} chars")
            print(f"   🚀 Speed: {metrics['words_per_second']:.1f} words/sec")
            print(f"   💬 Response Preview: {metrics['response'][:100]}...")


def setup_ollama():
    """Guide user through Ollama setup"""
    print("🚀 Ollama Setup Guide")
    print("="*50)

    # Check if ollama command exists
    try:
        result = subprocess.run(["ollama", "--version"],
                              capture_output=True, text=True)
        print(f"✅ Ollama installed: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from https://ollama.ai")
        print("   After installation, run 'ollama serve' in a terminal")
        return False

    # Check if server is running
    client = OllamaClient()
    if not client.is_running():
        print("❌ Ollama server not running")
        print("   Please run 'ollama serve' in a separate terminal")
        return False

    print("✅ Ollama server is running")

    # Show available models
    available = client.available_models
    if available:
        print(f"✅ Available models: {', '.join(available)}")
    else:
        print("ℹ️  No models installed yet")

    return True


def demonstrate_ollama():
    """Demonstrate Ollama integration"""

    print("=== Ollama Local Model Demonstration ===\\n")

    if not setup_ollama():
        return

    # Initialize agent with a small, fast model
    agent = OllamaAgent("llama3.2:3b")  # Adjust model as needed

    if not agent.client.is_running():
        return

    # Test conversations
    system_prompt = """You are a helpful AI assistant running locally.
    Keep your responses concise but informative."""

    test_questions = [
        "What are the main advantages of using local AI models?",
        "Explain the trade-offs between model size and performance.",
        "How can I optimize local model inference speed?"
    ]

    print("Starting conversation with local model...\\n")

    for question in test_questions:
        print(f"👤 User: {question}")
        response = agent.chat(question, system_prompt)
        print(f"🤖 Local Assistant: {response}\\n")
        print("-" * 60 + "\\n")

    # Demonstrate model comparison
    print("\\n🔍 Comparing Local Models")
    print("-" * 40)

    comparison = LocalModelComparison()
    test_prompt = "Explain quantum computing in simple terms."

    # Only test models that are available
    available_models = [model for model in ["llama3.2:3b", "mistral:7b", "phi3:3.8b"]
                       if model in comparison.client.available_models]

    if available_models:
        results = comparison.benchmark_models(test_prompt, available_models)
        comparison.print_benchmark_results(results)
    else:
        print("No models available for comparison. Please install some models first.")
        print("Suggested commands:")
        print("  ollama pull llama3.2:3b")
        print("  ollama pull mistral:7b")


if __name__ == "__main__":
    demonstrate_ollama()