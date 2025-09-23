"""
Workshop Setup Verification Script
Verifies that all required packages are installed and API keys are configured
"""

import os
import sys
from typing import Tuple
import importlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        return True, f"✅ {package_name} is installed"
    except ImportError as e:
        return False, f"❌ {package_name} is not installed: {e}"


def check_api_key(key_name: str, service_name: str) -> Tuple[bool, str]:
    """Check if an API key is configured"""
    key_value = os.getenv(key_name)
    if key_value and key_value != f"your_{key_name.lower()}_here":
        # Don't display the actual key, just confirm it's set
        masked_key = key_value[:8] + "..." if len(key_value) > 8 else "***"
        return True, f"✅ {service_name} API key is configured ({masked_key})"
    else:
        return False, f"❌ {service_name} API key is not configured"


def test_api_connection(service: str) -> Tuple[bool, str]:
    """Test API connection (basic connectivity test)"""
    try:
        if service == "openai":
            import openai
            openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Just test that we can create a client - don't make actual API calls
            return True, "✅ OpenAI client initialized successfully"

        elif service == "anthropic":
            import anthropic
            anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return True, "✅ Anthropic client initialized successfully"

        elif service == "google":
            import google.generativeai as genai
            if os.getenv("GOOGLE_API_KEY"):
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                return True, "✅ Google AI client configured successfully"
            else:
                return False, "❌ Google AI API key not configured"

    except Exception as e:
        return False, f"❌ {service} client initialization failed: {e}"


def main():
    """Main verification function"""
    print("🚀 Agentic AI Workshop Setup Verification")
    print("=" * 50)

    # Track overall success
    all_passed = True

    # Core packages to check
    packages = [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("python-dotenv", "dotenv"),
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-anthropic", "langchain_anthropic"),
        ("langgraph", "langgraph"),
        ("crewai", "crewai"),
        ("google-generativeai", "google.generativeai"),
        ("transformers", "transformers"),
        ("dspy-ai", "dspy"),
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        ("tiktoken", "tiktoken"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]

    print("\\n📦 Checking Package Installation:")
    print("-" * 30)

    failed_packages = []
    for package_name, import_name in packages:
        success, message = check_package(package_name, import_name)
        print(message)
        if not success:
            failed_packages.append(package_name)
            all_passed = False

    # API Key checks
    print("\\n🔑 Checking API Configuration:")
    print("-" * 30)

    api_keys = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("GOOGLE_API_KEY", "Google AI"),
    ]

    missing_keys = []
    for key_name, service_name in api_keys:
        success, message = check_api_key(key_name, service_name)
        print(message)
        if not success:
            missing_keys.append(service_name)
            all_passed = False

    # Test API connections (basic initialization)
    print("\\n🌐 Testing API Connections:")
    print("-" * 30)

    if os.getenv("OPENAI_API_KEY"):
        success, message = test_api_connection("openai")
        print(message)
        if not success:
            all_passed = False

    if os.getenv("ANTHROPIC_API_KEY"):
        success, message = test_api_connection("anthropic")
        print(message)
        if not success:
            all_passed = False

    if os.getenv("GOOGLE_API_KEY"):
        success, message = test_api_connection("google")
        print(message)
        if not success:
            all_passed = False

    # Python version check
    print("\\n🐍 Python Environment:")
    print("-" * 20)
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major == 3 and python_version.minor >= 8:
        print("✅ Python version is compatible")
    else:
        print("❌ Python 3.8+ is required")
        all_passed = False

    # Summary
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 Setup verification completed successfully!")
        print("You're ready to start the Agentic AI workshop!")
    else:
        print("❌ Setup verification failed!")
        print("\\nPlease address the following issues:")

        if failed_packages:
            print("\\n📦 Install missing packages:")
            print("   pip install " + " ".join(failed_packages))

        if missing_keys:
            print("\\n🔑 Configure missing API keys in .env file:")
            for service in missing_keys:
                if service == "OpenAI":
                    print("   OPENAI_API_KEY=your_key_here")
                elif service == "Anthropic":
                    print("   ANTHROPIC_API_KEY=your_key_here")
                elif service == "Google AI":
                    print("   GOOGLE_API_KEY=your_key_here")

        print("\\nThen run this script again to verify the setup.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)