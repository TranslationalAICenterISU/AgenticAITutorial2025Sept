"""
Integration Patterns - Multi-Framework Architecture Designs

This module demonstrates common patterns for integrating multiple agentic AI frameworks
in production systems. Each pattern solves different architectural challenges and
represents best practices learned from real-world implementations.

Patterns covered:
1. API Gateway Pattern - Central routing and management
2. Pipeline Pattern - Sequential processing through frameworks
3. Microservices Pattern - Framework-specific services
4. Event-Driven Pattern - Async communication between agents
5. Hybrid Reasoning Pattern - Combine different AI capabilities
6. Fallback Chain Pattern - Graceful degradation
7. Consensus Pattern - Multi-framework agreement
8. Load Balancing Pattern - Distribute across frameworks
"""

import os
import sys
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from abc import ABC, abstractmethod

# Import frameworks with fallbacks
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.tools import tool
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    ChatOpenAI = None

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool as crewai_tool
except ImportError:
    Agent = None

try:
    from smolagents import CodeAgent, InferenceClientModel
except ImportError:
    CodeAgent = None

from dotenv import load_dotenv
load_dotenv()


class IntegrationPattern(Enum):
    """Integration pattern types"""
    API_GATEWAY = "api_gateway"
    PIPELINE = "pipeline"
    MICROSERVICES = "microservices"
    EVENT_DRIVEN = "event_driven"
    HYBRID_REASONING = "hybrid_reasoning"
    FALLBACK_CHAIN = "fallback_chain"
    CONSENSUS = "consensus"
    LOAD_BALANCING = "load_balancing"


@dataclass
class ProcessingResult:
    """Result from framework processing"""
    framework: str
    success: bool
    output: Any
    execution_time: float
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Abstract base classes for integration patterns
class FrameworkAdapter(ABC):
    """Abstract adapter for different frameworks"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def process(self, input_data: Any) -> ProcessingResult:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass


class IntegrationManager(ABC):
    """Abstract integration manager"""

    def __init__(self, adapters: List[FrameworkAdapter]):
        self.adapters = {adapter.name: adapter for adapter in adapters}

    @abstractmethod
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        pass


# Concrete Framework Adapters
class OpenAIAdapter(FrameworkAdapter):
    """OpenAI framework adapter"""

    def __init__(self):
        super().__init__("openai")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if openai else None

    async def process(self, input_data: Any) -> ProcessingResult:
        if not self.client:
            return ProcessingResult(
                framework=self.name,
                success=False,
                output=None,
                execution_time=0,
                error_message="OpenAI client not available"
            )

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": str(input_data)}],
                max_tokens=200
            )

            return ProcessingResult(
                framework=self.name,
                success=True,
                output=response.choices[0].message.content.strip(),
                execution_time=time.time() - start_time,
                confidence=0.9,  # Simulated confidence
                metadata={"model": "gpt-3.5-turbo", "tokens": response.usage.total_tokens}
            )

        except Exception as e:
            return ProcessingResult(
                framework=self.name,
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def health_check(self) -> bool:
        return self.client is not None and bool(os.getenv("OPENAI_API_KEY"))


class AnthropicAdapter(FrameworkAdapter):
    """Anthropic framework adapter"""

    def __init__(self):
        super().__init__("anthropic")
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if anthropic else None

    async def process(self, input_data: Any) -> ProcessingResult:
        if not self.client:
            return ProcessingResult(
                framework=self.name,
                success=False,
                output=None,
                execution_time=0,
                error_message="Anthropic client not available"
            )

        start_time = time.time()
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": str(input_data)}]
            )

            return ProcessingResult(
                framework=self.name,
                success=True,
                output=response.content[0].text.strip(),
                execution_time=time.time() - start_time,
                confidence=0.85,  # Simulated confidence
                metadata={"model": "claude-3-haiku", "tokens": response.usage.input_tokens + response.usage.output_tokens}
            )

        except Exception as e:
            return ProcessingResult(
                framework=self.name,
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def health_check(self) -> bool:
        return self.client is not None and bool(os.getenv("ANTHROPIC_API_KEY"))


class LangChainAdapter(FrameworkAdapter):
    """LangChain framework adapter"""

    def __init__(self):
        super().__init__("langchain")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo") if ChatOpenAI and os.getenv("OPENAI_API_KEY") else None

    async def process(self, input_data: Any) -> ProcessingResult:
        if not self.llm:
            return ProcessingResult(
                framework=self.name,
                success=False,
                output=None,
                execution_time=0,
                error_message="LangChain client not available"
            )

        start_time = time.time()
        try:
            message = HumanMessage(content=str(input_data))
            response = self.llm.invoke([message])

            return ProcessingResult(
                framework=self.name,
                success=True,
                output=response.content.strip(),
                execution_time=time.time() - start_time,
                confidence=0.8,  # Simulated confidence
                metadata={"framework": "langchain", "model": "gpt-3.5-turbo"}
            )

        except Exception as e:
            return ProcessingResult(
                framework=self.name,
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def health_check(self) -> bool:
        return self.llm is not None


# Integration Pattern Implementations

class APIGatewayManager(IntegrationManager):
    """API Gateway Pattern - Central routing and load balancing"""

    def __init__(self, adapters: List[FrameworkAdapter]):
        super().__init__(adapters)
        self.routing_rules = {}
        self.load_balancer_index = 0

    def add_routing_rule(self, condition: Callable[[Any], bool], framework: str):
        """Add routing rule based on input condition"""
        self.routing_rules[framework] = condition

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Route request to appropriate framework"""
        routing_strategy = kwargs.get('strategy', 'round_robin')

        if routing_strategy == 'rule_based':
            # Apply routing rules
            for framework, condition in self.routing_rules.items():
                if condition(input_data) and framework in self.adapters:
                    result = await self.adapters[framework].process(input_data)
                    return {"strategy": "rule_based", "framework": framework, "result": result}

        elif routing_strategy == 'round_robin':
            # Simple round-robin load balancing
            healthy_adapters = [name for name, adapter in self.adapters.items() if adapter.health_check()]
            if not healthy_adapters:
                return {"error": "No healthy adapters available"}

            selected = healthy_adapters[self.load_balancer_index % len(healthy_adapters)]
            self.load_balancer_index += 1

            result = await self.adapters[selected].process(input_data)
            return {"strategy": "round_robin", "framework": selected, "result": result}

        elif routing_strategy == 'fastest_first':
            # Route to fastest available adapter
            health_check_results = {name: adapter.health_check() for name, adapter in self.adapters.items()}
            healthy_adapters = [name for name, status in health_check_results.items() if status]

            if not healthy_adapters:
                return {"error": "No healthy adapters available"}

            # For demo, just pick first healthy adapter
            # In production, you'd track performance metrics
            selected = healthy_adapters[0]
            result = await self.adapters[selected].process(input_data)
            return {"strategy": "fastest_first", "framework": selected, "result": result}

        return {"error": "Unknown routing strategy"}


class PipelineManager(IntegrationManager):
    """Pipeline Pattern - Sequential processing through frameworks"""

    def __init__(self, adapters: List[FrameworkAdapter]):
        super().__init__(adapters)
        self.pipeline_stages = []

    def add_stage(self, framework: str, transform_fn: Optional[Callable] = None):
        """Add a stage to the pipeline"""
        self.pipeline_stages.append({
            'framework': framework,
            'transform': transform_fn or (lambda x: x)
        })

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute pipeline stages sequentially"""
        results = []
        current_input = input_data

        for i, stage in enumerate(self.pipeline_stages):
            framework_name = stage['framework']
            transform_fn = stage['transform']

            if framework_name not in self.adapters:
                return {"error": f"Framework {framework_name} not available", "completed_stages": results}

            # Process current input through framework
            result = await self.adapters[framework_name].process(current_input)
            results.append({
                "stage": i + 1,
                "framework": framework_name,
                "result": result
            })

            if not result.success:
                return {"error": f"Stage {i+1} failed", "completed_stages": results}

            # Transform output for next stage
            current_input = transform_fn(result.output)

        return {"success": True, "final_output": current_input, "pipeline_results": results}


class FallbackChainManager(IntegrationManager):
    """Fallback Chain Pattern - Graceful degradation"""

    def __init__(self, adapters: List[FrameworkAdapter]):
        super().__init__(adapters)
        self.fallback_order = []

    def set_fallback_order(self, framework_names: List[str]):
        """Set the order of frameworks to try"""
        self.fallback_order = framework_names

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Try frameworks in fallback order until one succeeds"""
        attempts = []
        timeout = kwargs.get('timeout', 30)  # seconds

        for framework_name in self.fallback_order:
            if framework_name not in self.adapters:
                continue

            adapter = self.adapters[framework_name]
            if not adapter.health_check():
                attempts.append({
                    "framework": framework_name,
                    "status": "unhealthy",
                    "result": None
                })
                continue

            try:
                # Try processing with timeout
                result = await asyncio.wait_for(
                    adapter.process(input_data),
                    timeout=timeout
                )

                attempts.append({
                    "framework": framework_name,
                    "status": "attempted",
                    "result": result
                })

                if result.success:
                    return {
                        "success": True,
                        "framework": framework_name,
                        "result": result,
                        "attempts": attempts
                    }

            except asyncio.TimeoutError:
                attempts.append({
                    "framework": framework_name,
                    "status": "timeout",
                    "result": None
                })
            except Exception as e:
                attempts.append({
                    "framework": framework_name,
                    "status": "error",
                    "error": str(e),
                    "result": None
                })

        return {
            "success": False,
            "error": "All frameworks in fallback chain failed",
            "attempts": attempts
        }


class ConsensusManager(IntegrationManager):
    """Consensus Pattern - Multi-framework agreement"""

    def __init__(self, adapters: List[FrameworkAdapter]):
        super().__init__(adapters)
        self.consensus_threshold = 0.5  # Majority

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Get consensus from multiple frameworks"""
        consensus_type = kwargs.get('consensus_type', 'majority')
        timeout = kwargs.get('timeout', 30)

        # Run all adapters concurrently
        tasks = []
        healthy_adapters = []

        for name, adapter in self.adapters.items():
            if adapter.health_check():
                healthy_adapters.append(name)
                tasks.append(asyncio.create_task(adapter.process(input_data)))

        if not tasks:
            return {"error": "No healthy adapters available"}

        # Wait for all to complete (or timeout)
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            return {"error": "Consensus timeout reached"}

        # Process results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, ProcessingResult) and result.success:
                successful_results.append({
                    "framework": healthy_adapters[i],
                    "result": result
                })

        if len(successful_results) < 2:
            return {"error": "Need at least 2 successful results for consensus"}

        if consensus_type == 'majority':
            return self._majority_consensus(successful_results)
        elif consensus_type == 'weighted':
            return self._weighted_consensus(successful_results)
        elif consensus_type == 'unanimous':
            return self._unanimous_consensus(successful_results)

        return {"error": "Unknown consensus type"}

    def _majority_consensus(self, results: List[Dict]) -> Dict[str, Any]:
        """Simple majority consensus (most common response)"""
        outputs = [r["result"].output for r in results]
        output_counts = {}

        for output in outputs:
            output_counts[output] = output_counts.get(output, 0) + 1

        most_common = max(output_counts, key=output_counts.get)
        consensus_count = output_counts[most_common]

        return {
            "consensus_type": "majority",
            "consensus_output": most_common,
            "agreement_count": consensus_count,
            "total_responses": len(results),
            "confidence": consensus_count / len(results),
            "all_results": results
        }

    def _weighted_consensus(self, results: List[Dict]) -> Dict[str, Any]:
        """Weighted consensus based on confidence scores"""
        total_weight = 0
        weighted_outputs = {}

        for r in results:
            confidence = r["result"].confidence or 0.5
            output = r["result"].output
            total_weight += confidence

            if output in weighted_outputs:
                weighted_outputs[output] += confidence
            else:
                weighted_outputs[output] = confidence

        best_output = max(weighted_outputs, key=weighted_outputs.get)
        best_weight = weighted_outputs[best_output]

        return {
            "consensus_type": "weighted",
            "consensus_output": best_output,
            "consensus_weight": best_weight,
            "total_weight": total_weight,
            "confidence": best_weight / total_weight,
            "all_results": results
        }

    def _unanimous_consensus(self, results: List[Dict]) -> Dict[str, Any]:
        """Require all frameworks to agree"""
        outputs = [r["result"].output for r in results]
        unique_outputs = set(outputs)

        if len(unique_outputs) == 1:
            return {
                "consensus_type": "unanimous",
                "consensus_output": outputs[0],
                "unanimous": True,
                "confidence": 1.0,
                "all_results": results
            }
        else:
            return {
                "consensus_type": "unanimous",
                "consensus_output": None,
                "unanimous": False,
                "confidence": 0.0,
                "disagreement": list(unique_outputs),
                "all_results": results
            }


class HybridReasoningManager(IntegrationManager):
    """Hybrid Reasoning Pattern - Combine different AI capabilities"""

    def __init__(self, adapters: List[FrameworkAdapter]):
        super().__init__(adapters)
        self.reasoning_stages = {
            'analysis': [],
            'synthesis': [],
            'verification': []
        }

    def configure_reasoning_stages(self, analysis_frameworks: List[str],
                                 synthesis_frameworks: List[str],
                                 verification_frameworks: List[str]):
        """Configure which frameworks handle which reasoning stages"""
        self.reasoning_stages['analysis'] = analysis_frameworks
        self.reasoning_stages['synthesis'] = synthesis_frameworks
        self.reasoning_stages['verification'] = verification_frameworks

    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute hybrid reasoning workflow"""
        results = {}

        # Stage 1: Analysis
        analysis_prompt = f"Analyze the following: {input_data}"
        analysis_results = []

        for framework in self.reasoning_stages['analysis']:
            if framework in self.adapters and self.adapters[framework].health_check():
                result = await self.adapters[framework].process(analysis_prompt)
                analysis_results.append({"framework": framework, "result": result})

        results['analysis'] = analysis_results

        # Stage 2: Synthesis
        if analysis_results:
            # Combine analysis results for synthesis
            analysis_summary = " ".join([
                r["result"].output for r in analysis_results if r["result"].success
            ])

            synthesis_prompt = f"Synthesize insights from: {analysis_summary}"
            synthesis_results = []

            for framework in self.reasoning_stages['synthesis']:
                if framework in self.adapters and self.adapters[framework].health_check():
                    result = await self.adapters[framework].process(synthesis_prompt)
                    synthesis_results.append({"framework": framework, "result": result})

            results['synthesis'] = synthesis_results

            # Stage 3: Verification
            if synthesis_results:
                synthesis_output = synthesis_results[0]["result"].output if synthesis_results[0]["result"].success else ""
                verification_prompt = f"Verify and critique: {synthesis_output}"
                verification_results = []

                for framework in self.reasoning_stages['verification']:
                    if framework in self.adapters and self.adapters[framework].health_check():
                        result = await self.adapters[framework].process(verification_prompt)
                        verification_results.append({"framework": framework, "result": result})

                results['verification'] = verification_results

        return {
            "reasoning_type": "hybrid",
            "stages_completed": list(results.keys()),
            "results": results,
            "final_output": self._extract_final_output(results)
        }

    def _extract_final_output(self, results: Dict) -> str:
        """Extract final output from hybrid reasoning results"""
        if 'verification' in results and results['verification']:
            # Use verification stage output if available
            for result in results['verification']:
                if result["result"].success:
                    return result["result"].output

        if 'synthesis' in results and results['synthesis']:
            # Fall back to synthesis stage
            for result in results['synthesis']:
                if result["result"].success:
                    return result["result"].output

        # Last resort: use first analysis result
        if 'analysis' in results and results['analysis']:
            for result in results['analysis']:
                if result["result"].success:
                    return result["result"].output

        return "No successful results from hybrid reasoning"


# Demonstration functions
async def demonstrate_api_gateway_pattern():
    """Demonstrate API Gateway pattern"""
    print("\\n" + "="*60)
    print("🚪 API GATEWAY PATTERN")
    print("="*60)

    adapters = []
    if openai and os.getenv("OPENAI_API_KEY"):
        adapters.append(OpenAIAdapter())
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        adapters.append(AnthropicAdapter())
    if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        adapters.append(LangChainAdapter())

    if not adapters:
        print("No adapters available for demonstration")
        return

    gateway = APIGatewayManager(adapters)

    # Add routing rules
    gateway.add_routing_rule(
        lambda x: "creative" in str(x).lower(),
        "openai"
    )
    gateway.add_routing_rule(
        lambda x: "analysis" in str(x).lower(),
        "anthropic"
    )

    test_inputs = [
        "Write a creative story about robots",
        "Analyze the pros and cons of renewable energy",
        "What is the capital of France?"
    ]

    for input_data in test_inputs:
        print(f"\\n📨 Input: {input_data}")

        # Test different routing strategies
        for strategy in ['rule_based', 'round_robin']:
            result = await gateway.execute(input_data, strategy=strategy)
            print(f"   {strategy}: {result.get('framework', 'N/A')} -> {result.get('result', {}).get('success', False)}")


async def demonstrate_pipeline_pattern():
    """Demonstrate Pipeline pattern"""
    print("\\n" + "="*60)
    print("🔄 PIPELINE PATTERN")
    print("="*60)

    adapters = []
    if openai and os.getenv("OPENAI_API_KEY"):
        adapters.append(OpenAIAdapter())
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        adapters.append(AnthropicAdapter())

    if len(adapters) < 2:
        print("Need at least 2 adapters for pipeline demonstration")
        return

    pipeline = PipelineManager(adapters)

    # Create a simple pipeline: analyze -> summarize
    pipeline.add_stage("openai", lambda x: f"Summarize this analysis: {x}")
    pipeline.add_stage("anthropic", lambda x: f"Provide key takeaways from: {x}")

    input_data = "The impact of artificial intelligence on modern society"
    print(f"\\n📨 Input: {input_data}")

    result = await pipeline.execute(input_data)
    if result.get('success'):
        print("\\n✅ Pipeline completed successfully")
        print(f"   Final output: {result['final_output'][:100]}...")
        print(f"   Stages completed: {len(result['pipeline_results'])}")
    else:
        print(f"\\n❌ Pipeline failed: {result.get('error')}")


async def demonstrate_fallback_chain_pattern():
    """Demonstrate Fallback Chain pattern"""
    print("\\n" + "="*60)
    print("🔄 FALLBACK CHAIN PATTERN")
    print("="*60)

    adapters = []
    if openai and os.getenv("OPENAI_API_KEY"):
        adapters.append(OpenAIAdapter())
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        adapters.append(AnthropicAdapter())
    if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        adapters.append(LangChainAdapter())

    if not adapters:
        print("No adapters available for demonstration")
        return

    fallback_manager = FallbackChainManager(adapters)
    fallback_manager.set_fallback_order([adapter.name for adapter in adapters])

    input_data = "Explain quantum computing in simple terms"
    print(f"\\n📨 Input: {input_data}")

    result = await fallback_manager.execute(input_data)
    print(f"\\n{'✅' if result.get('success') else '❌'} Fallback chain result:")
    print(f"   Used framework: {result.get('framework', 'None')}")
    print(f"   Attempts made: {len(result.get('attempts', []))}")

    for i, attempt in enumerate(result.get('attempts', [])):
        status = attempt.get('status', 'unknown')
        framework = attempt.get('framework', 'unknown')
        print(f"   {i+1}. {framework}: {status}")


async def demonstrate_consensus_pattern():
    """Demonstrate Consensus pattern"""
    print("\\n" + "="*60)
    print("🤝 CONSENSUS PATTERN")
    print("="*60)

    adapters = []
    if openai and os.getenv("OPENAI_API_KEY"):
        adapters.append(OpenAIAdapter())
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        adapters.append(AnthropicAdapter())

    if len(adapters) < 2:
        print("Need at least 2 adapters for consensus demonstration")
        return

    consensus_manager = ConsensusManager(adapters)

    input_data = "What is 2+2?"
    print(f"\\n📨 Input: {input_data}")

    result = await consensus_manager.execute(input_data, consensus_type='majority')
    print("\\n🗳️ Majority consensus result:")
    print(f"   Consensus output: {result.get('consensus_output')}")
    print(f"   Agreement: {result.get('agreement_count', 0)}/{result.get('total_responses', 0)}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")


async def demonstrate_hybrid_reasoning_pattern():
    """Demonstrate Hybrid Reasoning pattern"""
    print("\\n" + "="*60)
    print("🧠 HYBRID REASONING PATTERN")
    print("="*60)

    adapters = []
    if openai and os.getenv("OPENAI_API_KEY"):
        adapters.append(OpenAIAdapter())
    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        adapters.append(AnthropicAdapter())

    if len(adapters) < 2:
        print("Need at least 2 adapters for hybrid reasoning demonstration")
        return

    hybrid_manager = HybridReasoningManager(adapters)
    hybrid_manager.configure_reasoning_stages(
        analysis_frameworks=["openai"],
        synthesis_frameworks=["anthropic"],
        verification_frameworks=["openai"]
    )

    input_data = "The ethical implications of AI in healthcare"
    print(f"\\n📨 Input: {input_data}")

    result = await hybrid_manager.execute(input_data)
    print("\\n🧠 Hybrid reasoning result:")
    print(f"   Stages completed: {result.get('stages_completed')}")
    print(f"   Final output: {result.get('final_output', '')[:150]}...")


async def main():
    """Main demonstration function"""
    print("🔗 Integration Patterns - Multi-Framework Architecture Designs")
    print("=" * 70)

    print("\\nPatterns demonstrated:")
    print("• API Gateway - Central routing and load balancing")
    print("• Pipeline - Sequential processing through frameworks")
    print("• Fallback Chain - Graceful degradation")
    print("• Consensus - Multi-framework agreement")
    print("• Hybrid Reasoning - Combine different AI capabilities")

    # Check available adapters
    available_count = 0
    if openai and os.getenv("OPENAI_API_KEY"):
        available_count += 1
        print("✅ OpenAI available")
    else:
        print("❌ OpenAI not available")

    if anthropic and os.getenv("ANTHROPIC_API_KEY"):
        available_count += 1
        print("✅ Anthropic available")
    else:
        print("❌ Anthropic not available")

    if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        print("✅ LangChain available")
    else:
        print("❌ LangChain not available")

    if available_count < 1:
        print("\\n⚠️ No frameworks available. Please configure API keys.")
        return

    try:
        # Run demonstrations
        await demonstrate_api_gateway_pattern()
        await demonstrate_pipeline_pattern()
        await demonstrate_fallback_chain_pattern()
        await demonstrate_consensus_pattern()
        await demonstrate_hybrid_reasoning_pattern()

        print("\\n" + "="*70)
        print("🎉 Integration Patterns Demonstration Complete!")

        print("\\n🔑 Key Takeaways:")
        print("• Use API Gateway for central routing and load balancing")
        print("• Use Pipeline for sequential processing workflows")
        print("• Use Fallback Chain for high availability and fault tolerance")
        print("• Use Consensus for critical decisions requiring agreement")
        print("• Use Hybrid Reasoning for complex multi-stage problems")

        print("\\nProduction Considerations:")
        print("• Implement proper error handling and retries")
        print("• Add comprehensive monitoring and metrics")
        print("• Consider performance and latency requirements")
        print("• Plan for scaling and load balancing")
        print("• Implement proper security and authentication")

    except KeyboardInterrupt:
        print("\\n\\n👋 Demonstration interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during demonstration: {e}")


if __name__ == "__main__":
    asyncio.run(main())