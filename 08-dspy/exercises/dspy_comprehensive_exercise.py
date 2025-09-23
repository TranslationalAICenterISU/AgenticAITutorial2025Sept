"""
DSPy Comprehensive Exercise - Build Self-Optimizing Language Programs

This exercise challenges you to master DSPy by building self-optimizing programs
that automatically improve their prompts and few-shot examples. You'll work with
signatures, modules, optimizers, and evaluation metrics.

Learning Objectives:
- Master DSPy's declarative programming model
- Build optimizable language programs
- Implement custom evaluation metrics
- Use different optimization strategies
- Create complex multi-hop reasoning systems
- Understand when and how to apply DSPy

Tasks:
1. Build a Question-Answering System with Optimization
2. Create a Multi-Hop Reasoning Chain
3. Implement Custom Signatures and Modules
4. Design Evaluation Metrics and Datasets
5. Build a Self-Improving Research Assistant
6. Compare Optimization Strategies
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import random

# DSPy imports
try:
    import dspy
    from dspy.evaluate import Evaluate
    from dspy.teleprompt import BootstrapFewShot, COPRO, MIPROv2
    from dspy.primitives.assertions import assert_transform_module, backtrack_handler
except ImportError:
    dspy = None
    print("❌ DSPy not available. Install with: pip install dspy-ai")

from dotenv import load_dotenv
load_dotenv()


def setup_dspy():
    """Configure DSPy with available language model"""
    if not dspy:
        return False

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        return False

    lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=300)
    dspy.settings.configure(lm=lm)
    print("✅ DSPy configured with OpenAI GPT-3.5-turbo")
    return True


# Task 1: Question-Answering System with Optimization
class QASignature(dspy.Signature):
    """
    TODO: Design a signature for question-answering

    Requirements:
    - Input: question
    - Output: answer
    - Include helpful descriptions
    """
    pass  # Replace with your implementation


class BasicQA(dspy.Module):
    """
    TODO: Implement a basic QA module

    Requirements:
    - Use the QASignature
    - Can be a simple Predict or ChainOfThought
    - Should be optimizable
    """

    def __init__(self):
        super().__init__()
        # TODO: Initialize your QA module

    def forward(self, question):
        # TODO: Implement forward pass
        pass


class OptimizedQA(dspy.Module):
    """
    TODO: Implement an advanced QA module with reasoning

    Requirements:
    - Use ChainOfThought or similar reasoning
    - Include confidence estimation
    - Handle edge cases
    """

    def __init__(self):
        super().__init__()
        # TODO: Initialize optimized QA module

    def forward(self, question):
        # TODO: Implement forward pass with reasoning
        pass


# Task 2: Multi-Hop Reasoning Chain
class HopSignature(dspy.Signature):
    """
    TODO: Design signature for single reasoning hop

    Requirements:
    - Take a question and context
    - Produce a reasoning step and sub-question
    """
    pass  # Replace with your implementation


class MultiHopQA(dspy.Module):
    """
    TODO: Implement multi-hop reasoning system

    Requirements:
    - Break complex questions into steps
    - Execute reasoning chain
    - Combine results for final answer
    - Handle variable number of hops
    """

    def __init__(self, max_hops: int = 3):
        super().__init__()
        self.max_hops = max_hops
        # TODO: Initialize multi-hop components

    def forward(self, question):
        # TODO: Implement multi-hop reasoning
        # Hint: Use a loop to perform multiple reasoning steps
        pass


# Task 3: Custom Signatures and Modules
class SummarizationSignature(dspy.Signature):
    """
    TODO: Create signature for text summarization

    Requirements:
    - Input: long text
    - Output: concise summary
    - Consider length constraints
    """
    pass


class AnalysisSignature(dspy.Signature):
    """
    TODO: Create signature for text analysis

    Requirements:
    - Input: text to analyze
    - Output: key insights, themes, sentiment
    """
    pass


class ResearchAssistant(dspy.Module):
    """
    TODO: Build a comprehensive research assistant

    Requirements:
    - Combine multiple reasoning steps
    - Use summarization and analysis
    - Handle different types of queries
    - Provide structured outputs
    """

    def __init__(self):
        super().__init__()
        # TODO: Initialize research assistant components
        # Hint: You might need QA, summarization, and analysis modules

    def research_topic(self, topic: str, depth: str = "medium"):
        """
        TODO: Research a topic with specified depth

        Requirements:
        - Generate relevant questions about the topic
        - Answer those questions
        - Summarize findings
        - Return structured research report
        """
        pass

    def analyze_text(self, text: str):
        """
        TODO: Analyze given text

        Requirements:
        - Extract key points
        - Identify themes
        - Assess sentiment/tone
        - Provide actionable insights
        """
        pass


# Task 4: Evaluation Metrics and Datasets
def exact_match_metric(gold, pred, trace=None):
    """
    TODO: Implement exact match metric

    Requirements:
    - Compare gold.answer with pred.answer
    - Handle case sensitivity
    - Return True/False for match
    """
    pass


def contains_answer_metric(gold, pred, trace=None):
    """
    TODO: Implement contains answer metric

    Requirements:
    - Check if prediction contains the gold answer
    - Handle partial matches
    - Be case-insensitive
    """
    pass


def quality_metric(gold, pred, trace=None):
    """
    TODO: Implement quality-based metric

    Requirements:
    - Evaluate answer quality beyond exact match
    - Consider completeness, accuracy, clarity
    - Return score between 0 and 1
    """
    pass


class DatasetGenerator:
    """
    TODO: Generate training and evaluation datasets

    Requirements:
    - Create QA pairs for different domains
    - Generate multi-hop reasoning examples
    - Include diverse question types
    - Support different difficulty levels
    """

    def __init__(self):
        # TODO: Initialize dataset generator
        pass

    def generate_qa_dataset(self, domain: str = "general", size: int = 20) -> List[dspy.Example]:
        """
        TODO: Generate QA dataset

        Requirements:
        - Create question-answer pairs
        - Ensure diversity in question types
        - Include different difficulty levels
        - Return list of dspy.Example objects
        """
        pass

    def generate_multihop_dataset(self, size: int = 15) -> List[dspy.Example]:
        """
        TODO: Generate multi-hop reasoning dataset

        Requirements:
        - Create questions requiring multiple reasoning steps
        - Include intermediate reasoning steps
        - Cover different reasoning patterns
        """
        pass


# Task 5: Optimization Strategies Comparison
class OptimizationComparator:
    """
    TODO: Compare different DSPy optimization strategies

    Requirements:
    - Test BootstrapFewShot, COPRO, and other optimizers
    - Measure performance improvements
    - Compare optimization times
    - Analyze which optimizers work best for different tasks
    """

    def __init__(self):
        self.results = {}

    def compare_optimizers(self, module: dspy.Module, trainset: List[dspy.Example],
                          testset: List[dspy.Example], metric: callable):
        """
        TODO: Compare different optimization strategies

        Requirements:
        - Test multiple optimizers on the same task
        - Measure before/after performance
        - Track optimization time
        - Return comparison results
        """
        pass

    def bootstrap_optimization(self, module: dspy.Module, trainset: List[dspy.Example],
                              metric: callable, max_demos: int = 5):
        """
        TODO: Implement BootstrapFewShot optimization

        Requirements:
        - Use BootstrapFewShot optimizer
        - Configure appropriate parameters
        - Return optimized module and metrics
        """
        pass

    def copro_optimization(self, module: dspy.Module, trainset: List[dspy.Example],
                          metric: callable):
        """
        TODO: Implement COPRO optimization

        Requirements:
        - Use COPRO optimizer for instruction optimization
        - Handle optimization parameters
        - Return optimized module and metrics
        """
        pass


# Task 6: Advanced DSPy Patterns
class ChainOfThoughtQA(dspy.Module):
    """
    TODO: Implement Chain-of-Thought reasoning

    Requirements:
    - Use dspy.ChainOfThought
    - Show reasoning steps
    - Handle complex questions
    """

    def __init__(self):
        super().__init__()
        # TODO: Initialize CoT module

    def forward(self, question):
        # TODO: Implement CoT forward pass
        pass


class ProgramOfThoughtQA(dspy.Module):
    """
    TODO: Implement Program-of-Thought reasoning

    Requirements:
    - Generate code/programs to solve problems
    - Execute generated programs
    - Handle math and logic problems
    """

    def __init__(self):
        super().__init__()
        # TODO: Initialize PoT module

    def forward(self, question):
        # TODO: Implement PoT forward pass
        # Hint: Generate code and execute it safely
        pass


class ReActAgent(dspy.Module):
    """
    TODO: Implement ReAct (Reasoning + Acting) pattern

    Requirements:
    - Interleave reasoning and action steps
    - Use external tools when needed
    - Handle multi-step problems
    """

    def __init__(self, tools: List[callable] = None):
        super().__init__()
        self.tools = tools or []
        # TODO: Initialize ReAct components

    def forward(self, question):
        # TODO: Implement ReAct reasoning loop
        # Hint: Alternate between thinking and acting
        pass


# Exercise Testing Framework
class DSPyExerciseTests:
    """Test suite for DSPy exercise implementation"""

    def __init__(self):
        self.test_results = []

    async def run_all_tests(self):
        """Run comprehensive test suite"""
        if not setup_dspy():
            print("Cannot run tests - DSPy not configured")
            return

        print("🧪 DSPy Comprehensive Exercise Tests")
        print("=" * 50)

        await self.test_basic_qa()
        await self.test_multihop_reasoning()
        await self.test_custom_modules()
        await self.test_optimization()
        await self.test_advanced_patterns()

        self.show_results()

    async def test_basic_qa(self):
        """Test basic QA implementation"""
        print("\\n📝 Testing Basic QA Implementation...")

        # TODO: Implement tests for BasicQA and OptimizedQA
        # Test ideas:
        # - Create QA instances
        # - Test with sample questions
        # - Verify output format
        # - Check reasoning quality

        try:
            # Example test structure:
            qa = BasicQA()
            # result = qa(question="What is the capital of France?")
            # assert result.answer, "QA should produce an answer"
            self.test_results.append(("Basic QA", True, "Implemented correctly"))
        except Exception as e:
            self.test_results.append(("Basic QA", False, f"Error: {e}"))

    async def test_multihop_reasoning(self):
        """Test multi-hop reasoning"""
        print("\\n🔗 Testing Multi-Hop Reasoning...")

        # TODO: Implement tests for MultiHopQA
        # Test ideas:
        # - Test complex questions requiring multiple steps
        # - Verify reasoning chain
        # - Check intermediate steps
        # - Validate final answers

        try:
            # Example test structure:
            multihop = MultiHopQA()
            # result = multihop(question="Complex multi-step question")
            self.test_results.append(("Multi-Hop QA", True, "Reasoning chain works"))
        except Exception as e:
            self.test_results.append(("Multi-Hop QA", False, f"Error: {e}"))

    async def test_custom_modules(self):
        """Test custom modules and signatures"""
        print("\\n🛠️ Testing Custom Modules...")

        # TODO: Implement tests for custom signatures and modules
        # Test ideas:
        # - Test ResearchAssistant
        # - Verify summarization
        # - Check analysis capabilities
        # - Test different input types

        try:
            # Example test structure:
            assistant = ResearchAssistant()
            # result = assistant.research_topic("Machine Learning")
            self.test_results.append(("Custom Modules", True, "Modules work correctly"))
        except Exception as e:
            self.test_results.append(("Custom Modules", False, f"Error: {e}"))

    async def test_optimization(self):
        """Test optimization strategies"""
        print("\\n⚡ Testing Optimization Strategies...")

        # TODO: Implement optimization tests
        # Test ideas:
        # - Test different optimizers
        # - Measure performance improvements
        # - Compare optimization strategies
        # - Validate optimized modules

        try:
            # Example test structure:
            comparator = OptimizationComparator()
            # results = comparator.compare_optimizers(...)
            self.test_results.append(("Optimization", True, "Optimizers working"))
        except Exception as e:
            self.test_results.append(("Optimization", False, f"Error: {e}"))

    async def test_advanced_patterns(self):
        """Test advanced DSPy patterns"""
        print("\\n🧠 Testing Advanced Patterns...")

        # TODO: Implement tests for advanced patterns
        # Test ideas:
        # - Test ChainOfThoughtQA
        # - Test ProgramOfThoughtQA
        # - Test ReActAgent
        # - Verify reasoning quality

        try:
            # Example test structure:
            cot = ChainOfThoughtQA()
            # result = cot(question="Reasoning question")
            self.test_results.append(("Advanced Patterns", True, "Patterns implemented"))
        except Exception as e:
            self.test_results.append(("Advanced Patterns", False, f"Error: {e}"))

    def show_results(self):
        """Show comprehensive test results"""
        print("\\n" + "=" * 60)
        print("📊 DSPy Exercise Test Results")
        print("=" * 60)

        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)

        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status:<10} {test_name:<20} {message}")

        print(f"\\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("\\n🎉 Congratulations! All tests passed!")
            print("Your DSPy implementation is working correctly.")
        else:
            print("\\n📝 Keep working on the failed components.")
            print("Review the TODO sections and implement missing functionality.")


# Demo Functions
def demo_basic_usage():
    """Demonstrate basic DSPy usage"""
    print("\\n🎯 Basic DSPy Usage Demo")
    print("=" * 40)

    if not setup_dspy():
        return

    print("""
    # Basic DSPy Signature
    class QA(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    # Basic Module
    class BasicQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.Predict(QA)

        def forward(self, question):
            return self.generate_answer(question=question)

    # Usage
    qa = BasicQA()
    result = qa(question="What is DSPy?")
    print(result.answer)
    """)


def demo_optimization():
    """Demonstrate DSPy optimization"""
    print("\\n⚡ DSPy Optimization Demo")
    print("=" * 40)

    print("""
    # Create training data
    trainset = [
        dspy.Example(question="What is 2+2?", answer="4"),
        dspy.Example(question="Capital of France?", answer="Paris"),
        # ... more examples
    ]

    # Define metric
    def accuracy(gold, pred, trace=None):
        return gold.answer.lower() == pred.answer.lower()

    # Optimize
    optimizer = BootstrapFewShot(metric=accuracy, max_bootstrapped_demos=3)
    optimized_qa = optimizer.compile(BasicQA(), trainset=trainset)

    # The optimized version performs better!
    """)


def main():
    """Main exercise instructions"""
    print("🎯 DSPy Comprehensive Exercise")
    print("=" * 50)

    print("""
    Welcome to the DSPy Comprehensive Exercise!

    DSPy revolutionizes how we build language model applications by treating
    them as optimizable programs rather than static prompts.

    🎯 Learning Goals:
    • Master DSPy's declarative programming model
    • Build self-optimizing language programs
    • Understand signatures, modules, and optimizers
    • Implement custom evaluation metrics
    • Create complex reasoning systems

    📋 Tasks Overview:

    1. ✅ Question-Answering System
       - Implement QASignature and BasicQA
       - Create OptimizedQA with reasoning
       - Test with sample questions

    2. ✅ Multi-Hop Reasoning
       - Build MultiHopQA system
       - Handle complex reasoning chains
       - Break questions into steps

    3. ✅ Custom Signatures & Modules
       - Design specialized signatures
       - Build ResearchAssistant module
       - Implement summarization and analysis

    4. ✅ Evaluation & Datasets
       - Create evaluation metrics
       - Generate training/test datasets
       - Build quality assessment functions

    5. ✅ Optimization Comparison
       - Test different optimizers
       - Compare BootstrapFewShot vs COPRO
       - Measure performance improvements

    6. ✅ Advanced Patterns
       - Implement Chain-of-Thought
       - Build Program-of-Thought reasoning
       - Create ReAct agent pattern

    🚀 Getting Started:

    1. Make sure DSPy is installed: `pip install dspy-ai`
    2. Configure your OpenAI API key in .env
    3. Start with Task 1 - implement the TODO sections
    4. Test each component as you build it
    5. Run the comprehensive test suite
    6. Experiment with optimization strategies

    💡 DSPy Key Concepts:

    • Signatures: Define input/output specifications
    • Modules: Reusable components that can be optimized
    • Optimizers: Automatically improve prompts and examples
    • Metrics: Define what "better" means for your task

    📊 Success Criteria:
    • All TODO sections implemented
    • Tests pass successfully
    • Optimizers show measurable improvements
    • Code follows DSPy best practices

    ⚡ Pro Tips:
    • Start simple, then add complexity
    • Define good evaluation metrics early
    • Collect quality training examples
    • Test optimization on real tasks
    • Use Chain-of-Thought for reasoning tasks

    🎉 Advanced Challenges:
    • Build domain-specific optimizers
    • Create custom assertion functions
    • Implement multi-modal reasoning
    • Build agent-like behavior patterns
    """)

    print("\\n🔧 Quick Start Commands:")
    print("1. Run tests: python -c 'import asyncio; asyncio.run(DSPyExerciseTests().run_all_tests())'")
    print("2. See demos: python this_file.py --demo")
    print("3. Start coding: Fill in the TODO sections!")

    # Show available components
    if dspy:
        print("\\n✅ DSPy is available - you're ready to start!")
    else:
        print("\\n❌ DSPy not found. Install with: pip install dspy-ai")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_basic_usage()
        demo_optimization()
    else:
        main()