"""
DSPy Basics - Introduction to Declarative Self-improving Language Programs

This module demonstrates the core concepts of DSPy:
- Signatures: Input/output specifications for language model tasks
- Modules: Reusable components that can be optimized
- Optimizers: Automatic prompt and example optimization
- Evaluation: Metrics-driven improvement

DSPy automatically optimizes prompts and few-shot examples based on your success metrics.
"""

import os
import sys
from typing import List, Dict, Any
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure DSPy with OpenAI
def setup_dspy():
    """Set up DSPy with OpenAI language model"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        return False

    # Configure the language model
    lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=250)
    dspy.settings.configure(lm=lm)
    print("✅ DSPy configured with OpenAI GPT-3.5-turbo")
    return True


# 1. Basic Signatures - Define input/output specifications
class BasicQA(dspy.Signature):
    """Answer questions with short, factual responses."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A concise, factual answer")


class Summarize(dspy.Signature):
    """Summarize long text into key points."""
    text = dspy.InputField()
    summary = dspy.OutputField(desc="A brief summary of the main points")


class Sentiment(dspy.Signature):
    """Classify the sentiment of text."""
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")


# 2. Basic Modules - Reusable components
class SimpleQA(dspy.Module):
    """A simple question-answering module"""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(BasicQA)

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)


class TextSummarizer(dspy.Module):
    """A text summarization module"""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.Predict(Summarize)

    def forward(self, text):
        prediction = self.summarize(text=text)
        return dspy.Prediction(summary=prediction.summary)


class SentimentClassifier(dspy.Module):
    """A sentiment classification module"""

    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(Sentiment)

    def forward(self, text):
        prediction = self.classify(text=text)
        return dspy.Prediction(sentiment=prediction.sentiment)


# 3. Chain-of-Thought Reasoning
class ChainOfThoughtQA(dspy.Module):
    """Question answering with chain-of-thought reasoning"""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(
            answer=prediction.answer,
            reasoning=prediction.rationale if hasattr(prediction, 'rationale') else "No reasoning provided"
        )


# 4. Multi-hop Reasoning
class MultiHopQA(dspy.Module):
    """Multi-hop question answering that can break down complex questions"""

    def __init__(self):
        super().__init__()
        self.generate_query = dspy.Predict("question -> search_query")
        self.generate_answer = dspy.ChainOfThought(BasicQA)

    def forward(self, question):
        # First, generate a search query (simulated)
        search_prediction = self.generate_query(question=question)

        # Then answer based on the question
        answer_prediction = self.generate_answer(question=question)

        return dspy.Prediction(
            answer=answer_prediction.answer,
            search_query=search_prediction.search_query if hasattr(search_prediction, 'search_query') else "No query generated"
        )


# 5. Evaluation Metrics
def exact_match_metric(gold, pred, trace=None):
    """Simple exact match metric for evaluation"""
    return gold.answer.lower().strip() == pred.answer.lower().strip()


def contains_metric(gold, pred, trace=None):
    """Check if prediction contains key information from gold answer"""
    return gold.answer.lower() in pred.answer.lower()


# 6. Example Training Data
def get_qa_examples():
    """Get sample QA examples for training/evaluation"""
    return [
        dspy.Example(question="What is the capital of France?", answer="Paris"),
        dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare"),
        dspy.Example(question="What is 2+2?", answer="4"),
        dspy.Example(question="What is the largest planet in our solar system?", answer="Jupiter"),
        dspy.Example(question="In what year did World War II end?", answer="1945"),
    ]


def get_sentiment_examples():
    """Get sample sentiment examples"""
    return [
        dspy.Example(text="I love this product! It's amazing!", sentiment="positive"),
        dspy.Example(text="This is terrible and doesn't work.", sentiment="negative"),
        dspy.Example(text="The weather is okay today.", sentiment="neutral"),
        dspy.Example(text="Best purchase I've ever made!", sentiment="positive"),
        dspy.Example(text="Completely disappointed with this service.", sentiment="negative"),
    ]


# 7. Demonstration Functions
def demonstrate_basic_qa():
    """Demonstrate basic question answering"""
    print("\n" + "="*50)
    print("🔍 BASIC QUESTION ANSWERING")
    print("="*50)

    qa_module = SimpleQA()

    questions = [
        "What is the capital of Japan?",
        "Who invented the telephone?",
        "What is photosynthesis?"
    ]

    for question in questions:
        try:
            result = qa_module(question=question)
            print(f"\nQ: {question}")
            print(f"A: {result.answer}")
        except Exception as e:
            print(f"Error answering '{question}': {e}")


def demonstrate_chain_of_thought():
    """Demonstrate chain-of-thought reasoning"""
    print("\n" + "="*50)
    print("🧠 CHAIN-OF-THOUGHT REASONING")
    print("="*50)

    cot_qa = ChainOfThoughtQA()

    complex_questions = [
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "Why do seasons change on Earth?",
        "What happens when you mix red and blue paint?"
    ]

    for question in complex_questions:
        try:
            result = cot_qa(question=question)
            print(f"\nQ: {question}")
            print(f"A: {result.answer}")
            if hasattr(result, 'reasoning'):
                print(f"Reasoning: {result.reasoning}")
        except Exception as e:
            print(f"Error with chain-of-thought for '{question}': {e}")


def demonstrate_sentiment_analysis():
    """Demonstrate sentiment classification"""
    print("\n" + "="*50)
    print("😊 SENTIMENT ANALYSIS")
    print("="*50)

    sentiment_classifier = SentimentClassifier()

    texts = [
        "I absolutely love this new restaurant!",
        "The service was terrible and the food was cold.",
        "It was an average experience, nothing special.",
        "This is the best day of my life!",
        "I'm feeling disappointed about the results."
    ]

    for text in texts:
        try:
            result = sentiment_classifier(text=text)
            print(f"\nText: '{text}'")
            print(f"Sentiment: {result.sentiment}")
        except Exception as e:
            print(f"Error analyzing sentiment for '{text}': {e}")


def demonstrate_text_summarization():
    """Demonstrate text summarization"""
    print("\n" + "="*50)
    print("📝 TEXT SUMMARIZATION")
    print("="*50)

    summarizer = TextSummarizer()

    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to
    the natural intelligence displayed by humans and animals. Leading AI textbooks define
    the field as the study of "intelligent agents": any device that perceives its environment
    and takes actions that maximize its chance of successfully achieving its goals.
    Colloquially, the term "artificial intelligence" is often used to describe machines
    that mimic "cognitive" functions that humans associate with the human mind, such as
    "learning" and "problem solving". As machines become increasingly capable, tasks
    considered to require "intelligence" are often removed from the definition of AI,
    a phenomenon known as the AI effect.
    """

    try:
        result = summarizer(text=long_text.strip())
        print(f"Original text length: {len(long_text)} characters")
        print(f"Summary: {result.summary}")
        print(f"Summary length: {len(result.summary)} characters")
    except Exception as e:
        print(f"Error summarizing text: {e}")


def demonstrate_dspy_optimization():
    """Demonstrate DSPy's optimization capabilities"""
    print("\n" + "="*50)
    print("🎯 DSPY OPTIMIZATION DEMO")
    print("="*50)

    # Create training and test sets
    qa_examples = get_qa_examples()
    train_set = qa_examples[:3]  # First 3 examples for training
    test_set = qa_examples[3:]   # Remaining for testing

    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    # Create unoptimized module
    qa_module = SimpleQA()

    # Test before optimization
    print("\n📊 Testing before optimization:")
    for example in test_set[:2]:  # Test on first 2 examples
        try:
            result = qa_module(question=example.question)
            print(f"Q: {example.question}")
            print(f"Expected: {example.answer}")
            print(f"Got: {result.answer}")
            print(f"Match: {'✅' if example.answer.lower() in result.answer.lower() else '❌'}")
        except Exception as e:
            print(f"Error during testing: {e}")

    # Note: Full optimization would require more examples and compute time
    print("\n💡 Note: Full DSPy optimization requires larger datasets and more compute time.")
    print("In production, you would use BootstrapFewShot or other optimizers with 50+ examples.")


def main():
    """Main demonstration function"""
    print("🚀 DSPy Basics - Declarative Self-improving Language Programs")
    print("=" * 60)

    # Setup DSPy
    if not setup_dspy():
        return

    print("\nDSPy combines:")
    print("• Signatures: Define what you want the model to do")
    print("• Modules: Reusable components that can be optimized")
    print("• Optimizers: Automatic prompt and example selection")
    print("• Evaluation: Metrics-driven improvement")

    try:
        # Run demonstrations
        demonstrate_basic_qa()
        demonstrate_chain_of_thought()
        demonstrate_sentiment_analysis()
        demonstrate_text_summarization()
        demonstrate_dspy_optimization()

        print("\n" + "="*60)
        print("🎉 DSPy Basics Demo Complete!")
        print("\nKey Takeaways:")
        print("• DSPy separates 'what' (signatures) from 'how' (implementation)")
        print("• Modules are automatically optimizable")
        print("• Chain-of-thought reasoning improves complex tasks")
        print("• Optimization requires good metrics and training data")
        print("\nNext: Run optimization_examples.py to see automatic prompt optimization")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure you have:")
        print("• Valid OPENAI_API_KEY in .env file")
        print("• DSPy package installed: pip install dspy-ai")


if __name__ == "__main__":
    main()