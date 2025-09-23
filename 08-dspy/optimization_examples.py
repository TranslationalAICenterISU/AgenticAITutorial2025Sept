"""
DSPy Optimization Examples - Automatic Prompt and Example Optimization

This module demonstrates DSPy's key feature: automatic optimization of prompts and few-shot examples.
DSPy can automatically improve your language model programs by:
- Selecting better few-shot examples
- Optimizing prompt templates
- Finding better reasoning strategies
- Improving overall performance on your metrics

This is what makes DSPy different from traditional prompt engineering.
"""

import os
import sys
import random
from typing import List, Dict, Any
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, COPRO, MIPROv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_dspy():
    """Set up DSPy with OpenAI language model"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        return False

    lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=200)
    dspy.settings.configure(lm=lm)
    print("✅ DSPy configured with OpenAI GPT-3.5-turbo")
    return True


# 1. Define a more complex task: Math Word Problems
class MathWordProblem(dspy.Signature):
    """Solve math word problems step by step."""
    problem = dspy.InputField(desc="A math word problem")
    solution = dspy.OutputField(desc="Step-by-step solution with final numerical answer")


class MathSolver(dspy.Module):
    """Math word problem solver with chain-of-thought reasoning"""

    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(MathWordProblem)

    def forward(self, problem):
        prediction = self.solve(problem=problem)
        return dspy.Prediction(solution=prediction.solution)


# 2. Create training and evaluation datasets
def get_math_examples():
    """Generate math word problem examples for training and testing"""
    examples = [
        dspy.Example(
            problem="Sarah has 12 apples. She gives 3 apples to her friend and buys 5 more. How many apples does she have now?",
            solution="Sarah starts with 12 apples. She gives away 3, so she has 12 - 3 = 9 apples. Then she buys 5 more, so she has 9 + 5 = 14 apples. Answer: 14"
        ),
        dspy.Example(
            problem="A rectangle has a length of 8 meters and width of 5 meters. What is its area?",
            solution="Area of rectangle = length × width. Area = 8 × 5 = 40 square meters. Answer: 40"
        ),
        dspy.Example(
            problem="Tom runs 3 miles every day for 7 days. How many miles does he run in total?",
            solution="Tom runs 3 miles per day for 7 days. Total miles = 3 × 7 = 21 miles. Answer: 21"
        ),
        dspy.Example(
            problem="A pizza costs $12 and is cut into 8 slices. What is the cost per slice?",
            solution="Total cost = $12, Number of slices = 8. Cost per slice = $12 ÷ 8 = $1.50. Answer: $1.50"
        ),
        dspy.Example(
            problem="Lisa has 20 stickers. She uses 1/4 of them to decorate her notebook. How many stickers did she use?",
            solution="Lisa has 20 stickers. She uses 1/4 of them. 1/4 of 20 = 20 ÷ 4 = 5 stickers. Answer: 5"
        ),
        dspy.Example(
            problem="A car travels 60 miles per hour for 3 hours. How far does it travel?",
            solution="Distance = speed × time. Distance = 60 mph × 3 hours = 180 miles. Answer: 180"
        ),
        dspy.Example(
            problem="There are 24 students in a class. If 1/3 are absent, how many students are present?",
            solution="Total students = 24. Absent students = 1/3 of 24 = 8. Present students = 24 - 8 = 16. Answer: 16"
        ),
        dspy.Example(
            problem="A box contains 15 red balls and 10 blue balls. What fraction of the balls are red?",
            solution="Red balls = 15, Blue balls = 10. Total balls = 15 + 10 = 25. Fraction red = 15/25 = 3/5. Answer: 3/5"
        )
    ]
    return examples


def get_qa_examples():
    """Get question-answering examples for comparison"""
    return [
        dspy.Example(question="What is the capital of France?", answer="Paris"),
        dspy.Example(question="Who wrote 'Romeo and Juliet'?", answer="William Shakespeare"),
        dspy.Example(question="What is the chemical symbol for gold?", answer="Au"),
        dspy.Example(question="How many continents are there?", answer="7"),
        dspy.Example(question="What is the largest mammal?", answer="Blue whale"),
        dspy.Example(question="In which year did World War II end?", answer="1945"),
        dspy.Example(question="What is the square root of 64?", answer="8"),
        dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci")
    ]


# 3. Define evaluation metrics
def math_accuracy_metric(gold, pred, trace=None):
    """Evaluate math problem solving accuracy"""
    try:
        # Extract numbers from the solution
        import re
        gold_numbers = re.findall(r'\d+\.?\d*', gold.solution)
        pred_numbers = re.findall(r'\d+\.?\d*', pred.solution)

        if not gold_numbers or not pred_numbers:
            return False

        # Check if the final answer (usually the last number) matches
        gold_answer = gold_numbers[-1]
        pred_answer = pred_numbers[-1]

        return abs(float(gold_answer) - float(pred_answer)) < 0.01

    except (ValueError, IndexError):
        # Fallback to string matching
        return gold.solution.lower().strip() in pred.solution.lower()


def simple_qa_metric(gold, pred, trace=None):
    """Simple QA accuracy metric"""
    return gold.answer.lower().strip() in pred.answer.lower()


# 4. Optimization demonstration functions
def demonstrate_bootstrap_optimization():
    """Demonstrate BootstrapFewShot optimization"""
    print("\n" + "="*60)
    print("🎯 BOOTSTRAP FEW-SHOT OPTIMIZATION")
    print("="*60)

    # Create datasets
    math_examples = get_math_examples()
    train_set = math_examples[:5]  # Use first 5 for training
    test_set = math_examples[5:]   # Use remaining for testing

    print(f"Training examples: {len(train_set)}")
    print(f"Test examples: {len(test_set)}")

    # Create unoptimized module
    math_solver = MathSolver()

    print("\n📊 Testing BEFORE optimization:")
    correct_before = 0
    for i, example in enumerate(test_set[:2]):  # Test on first 2 to save API calls
        try:
            result = math_solver(problem=example.problem)
            is_correct = math_accuracy_metric(example, result)
            correct_before += is_correct

            print(f"\nProblem {i+1}: {example.problem}")
            print(f"Expected: {example.solution}")
            print(f"Got: {result.solution}")
            print(f"Correct: {'✅' if is_correct else '❌'}")

        except Exception as e:
            print(f"Error solving problem {i+1}: {e}")

    accuracy_before = correct_before / min(len(test_set), 2)
    print(f"\nAccuracy before optimization: {accuracy_before:.2%}")

    # Optimize using BootstrapFewShot
    print("\n🔧 Optimizing with BootstrapFewShot...")
    try:
        optimizer = BootstrapFewShot(metric=math_accuracy_metric, max_bootstrapped_demos=3)
        optimized_solver = optimizer.compile(math_solver, trainset=train_set)

        print("✅ Optimization complete!")

        # Test optimized version
        print("\n📈 Testing AFTER optimization:")
        correct_after = 0
        for i, example in enumerate(test_set[:2]):
            try:
                result = optimized_solver(problem=example.problem)
                is_correct = math_accuracy_metric(example, result)
                correct_after += is_correct

                print(f"\nOptimized Problem {i+1}: {example.problem}")
                print(f"Expected: {example.solution}")
                print(f"Got: {result.solution}")
                print(f"Correct: {'✅' if is_correct else '❌'}")

            except Exception as e:
                print(f"Error with optimized solver on problem {i+1}: {e}")

        accuracy_after = correct_after / min(len(test_set), 2)
        print(f"\nAccuracy after optimization: {accuracy_after:.2%}")

        improvement = accuracy_after - accuracy_before
        print(f"Improvement: {improvement:+.2%}")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("Note: Optimization requires sufficient training examples and API credits")


def demonstrate_evaluation_driven_optimization():
    """Show how metrics drive optimization"""
    print("\n" + "="*60)
    print("📏 EVALUATION-DRIVEN OPTIMIZATION")
    print("="*60)

    # Create a simple QA task for faster demonstration
    qa_examples = get_qa_examples()
    train_set = qa_examples[:4]
    test_set = qa_examples[4:6]  # Small test set for demo

    # Define a simple QA module
    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.Predict("question -> answer")

        def forward(self, question):
            prediction = self.generate_answer(question=question)
            return prediction

    qa_module = SimpleQA()

    # Evaluate before optimization
    print("🔍 Evaluating before optimization...")
    evaluator = Evaluate(devset=test_set, metric=simple_qa_metric, num_threads=1, display_progress=True)

    try:
        score_before = evaluator(qa_module)
        print(f"Score before optimization: {score_before:.2%}")

        # Show individual predictions
        print("\n📋 Individual predictions (before):")
        for i, example in enumerate(test_set):
            try:
                result = qa_module(question=example.question)
                is_correct = simple_qa_metric(example, result)
                print(f"{i+1}. Q: {example.question}")
                print(f"   Expected: {example.answer}")
                print(f"   Got: {result.answer}")
                print(f"   Correct: {'✅' if is_correct else '❌'}")
            except Exception as e:
                print(f"   Error: {e}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def demonstrate_optimization_concepts():
    """Explain DSPy optimization concepts"""
    print("\n" + "="*60)
    print("🧠 DSPy OPTIMIZATION CONCEPTS")
    print("="*60)

    print("DSPy offers several optimization strategies:\n")

    print("1. 📚 BootstrapFewShot:")
    print("   • Automatically selects the best few-shot examples")
    print("   • Uses your training data to find effective demonstrations")
    print("   • Works by bootstrapping: uses model predictions as training data")

    print("\n2. 🎯 COPRO (Coordinate Prompt Optimization):")
    print("   • Optimizes the instruction/prompt text itself")
    print("   • Finds better ways to phrase instructions")
    print("   • Particularly good for complex reasoning tasks")

    print("\n3. 🔄 MIPROv2 (Multi-Prompt Instruction Optimization):")
    print("   • Advanced optimization combining multiple techniques")
    print("   • Optimizes both instructions and examples")
    print("   • Best performance but requires more computation")

    print("\n4. 📊 Evaluation Metrics:")
    print("   • Define what 'better' means for your task")
    print("   • Can be accuracy, F1, custom business metrics")
    print("   • Optimization is driven by improving these metrics")

    print("\n🔑 Key Benefits:")
    print("   • No manual prompt engineering needed")
    print("   • Systematic improvement based on data")
    print("   • Reproducible results")
    print("   • Adapts to your specific use case")


def demonstrate_manual_vs_automatic():
    """Compare manual prompt engineering vs automatic optimization"""
    print("\n" + "="*60)
    print("🤖 MANUAL vs AUTOMATIC OPTIMIZATION")
    print("="*60)

    # Manual approach - carefully crafted prompt
    class ManualMathSolver(dspy.Module):
        def __init__(self):
            super().__init__()
            # Manually crafted detailed instruction
            self.solve = dspy.Predict(
                "problem -> solution",
                instructions="You are a math tutor. Solve the math word problem step by step. Show your work clearly. Always end with 'Answer: [number]'."
            )

        def forward(self, problem):
            return self.solve(problem=problem)

    # Automatic approach - let DSPy optimize
    class AutoMathSolver(dspy.Module):
        def __init__(self):
            super().__init__()
            # Simple signature - DSPy will optimize the prompt
            self.solve = dspy.ChainOfThought("problem -> solution")

        def forward(self, problem):
            return self.solve(problem=problem)

    print("Manual Approach:")
    print("• Carefully craft detailed instructions")
    print("• Manually select few-shot examples")
    print("• Iterate based on trial and error")
    print("• Time-consuming and subjective")

    print("\nAutomatic Approach (DSPy):")
    print("• Define simple input/output specification")
    print("• Provide training examples with desired outputs")
    print("• Let optimizer find best prompts and examples")
    print("• Systematic and data-driven")

    # Quick demonstration
    test_problem = "A store sells 5 apples for $2. How much would 15 apples cost?"

    manual_solver = ManualMathSolver()
    auto_solver = AutoMathSolver()

    print(f"\nTest Problem: {test_problem}")

    try:
        manual_result = manual_solver(problem=test_problem)
        print(f"\nManual Approach Result:")
        print(f"Solution: {manual_result.solution}")
    except Exception as e:
        print(f"Manual approach error: {e}")

    try:
        auto_result = auto_solver(problem=test_problem)
        print(f"\nAutomatic Approach Result:")
        print(f"Solution: {auto_result.solution}")
    except Exception as e:
        print(f"Automatic approach error: {e}")


def main():
    """Main demonstration function"""
    print("🎯 DSPy Optimization Examples")
    print("Automatic Prompt and Example Optimization")
    print("=" * 60)

    # Setup DSPy
    if not setup_dspy():
        return

    print("\nDSPy's Key Innovation: Automatic Optimization")
    print("• Traditional: Manual prompt engineering")
    print("• DSPy: Define task + examples → Automatic optimization")
    print("• Result: Better performance with less manual work")

    try:
        # Run demonstrations
        demonstrate_optimization_concepts()
        demonstrate_manual_vs_automatic()
        demonstrate_evaluation_driven_optimization()

        # Note about full optimization
        print("\n" + "="*60)
        print("⚡ FULL OPTIMIZATION DEMO")
        print("="*60)
        print("For a complete optimization demo with BootstrapFewShot:")
        print("• Requires 20-50+ training examples")
        print("• Takes 2-5 minutes and multiple API calls")
        print("• Shows significant improvement on complex tasks")

        response = input("\nRun full optimization demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demonstrate_bootstrap_optimization()

        print("\n" + "="*60)
        print("🎉 DSPy Optimization Examples Complete!")
        print("\nKey Takeaways:")
        print("• DSPy automates what traditionally requires manual work")
        print("• Optimization is driven by metrics, not intuition")
        print("• Larger training sets → Better optimization results")
        print("• Chain-of-thought often improves with optimization")
        print("• Best for tasks where you have clear success criteria")

        print("\nNext Steps:")
        print("• Try DSPy on your own tasks")
        print("• Collect training examples for your use case")
        print("• Define good evaluation metrics")
        print("• Experiment with different optimizers")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Common issues:")
        print("• API key not configured")
        print("• Network connectivity problems")
        print("• Insufficient API credits")


if __name__ == "__main__":
    main()