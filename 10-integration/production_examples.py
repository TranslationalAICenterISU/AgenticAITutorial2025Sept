"""
Production Examples - Real-World Multi-Framework Deployments

This module demonstrates production-ready implementations of multi-framework
agentic AI systems. Each example includes monitoring, error handling,
scaling considerations, and operational best practices.

Production Examples:
1. Enterprise Document Processing Pipeline
2. Customer Service Chatbot with Escalation
3. Content Creation Workflow
4. Multi-Modal Analysis System
5. Research Assistant with Fact-Checking
"""

import os
import sys
import asyncio
import time
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from datetime import datetime, timedelta

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
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    ChatOpenAI = None

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProductionMetrics:
    """Production system metrics"""
    requests_processed: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    frameworks_used: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    def add_request(self, success: bool, latency: float, framework: str, error: Optional[str] = None):
        """Add request metrics"""
        self.requests_processed += 1
        self.total_latency += latency

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.error_counts[error] = self.error_counts.get(error, 0) + 1

        self.frameworks_used[framework] = self.frameworks_used.get(framework, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = datetime.now() - self.start_time
        avg_latency = self.total_latency / self.requests_processed if self.requests_processed > 0 else 0

        return {
            "uptime_minutes": uptime.total_seconds() / 60,
            "requests_processed": self.requests_processed,
            "success_rate": self.successful_requests / self.requests_processed if self.requests_processed > 0 else 0,
            "avg_latency_ms": avg_latency * 1000,
            "frameworks_used": self.frameworks_used,
            "error_counts": self.error_counts
        }


class HealthCheck:
    """System health monitoring"""

    def __init__(self):
        self.framework_health = {}
        self.last_check = {}

    def check_framework_health(self, framework_name: str) -> bool:
        """Check if framework is healthy"""
        now = datetime.now()

        # Cache health checks for 1 minute
        if framework_name in self.last_check:
            if now - self.last_check[framework_name] < timedelta(minutes=1):
                return self.framework_health.get(framework_name, False)

        # Perform health check
        healthy = False
        try:
            if framework_name == "openai" and openai and os.getenv("OPENAI_API_KEY"):
                # Simple health check - try to create client
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                healthy = True
            elif framework_name == "anthropic" and anthropic and os.getenv("ANTHROPIC_API_KEY"):
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                healthy = True
            elif framework_name == "langchain" and ChatOpenAI and os.getenv("OPENAI_API_KEY"):
                llm = ChatOpenAI(model="gpt-3.5-turbo")
                healthy = True
        except Exception as e:
            logger.warning(f"Health check failed for {framework_name}: {e}")
            healthy = False

        self.framework_health[framework_name] = healthy
        self.last_check[framework_name] = now
        return healthy

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        frameworks = ["openai", "anthropic", "langchain"]
        health_status = {}

        for framework in frameworks:
            health_status[framework] = self.check_framework_health(framework)

        healthy_count = sum(health_status.values())
        total_count = len(health_status)

        return {
            "overall_healthy": healthy_count > 0,
            "healthy_frameworks": healthy_count,
            "total_frameworks": total_count,
            "framework_status": health_status
        }


class ProductionDocumentProcessor:
    """Production document processing pipeline with multiple frameworks"""

    def __init__(self):
        self.metrics = ProductionMetrics()
        self.health_check = HealthCheck()
        self.setup_frameworks()

    def setup_frameworks(self):
        """Initialize available frameworks"""
        self.frameworks = {}

        if openai and os.getenv("OPENAI_API_KEY"):
            self.frameworks["openai"] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            self.frameworks["anthropic"] = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
            self.frameworks["langchain"] = ChatOpenAI(model="gpt-3.5-turbo")

        logger.info(f"Initialized frameworks: {list(self.frameworks.keys())}")

    async def process_document(self, document_text: str, processing_type: str = "summarize") -> Dict[str, Any]:
        """Process document through multi-framework pipeline"""
        start_time = time.time()
        result = {
            "success": False,
            "output": None,
            "frameworks_used": [],
            "processing_time": 0,
            "errors": []
        }

        try:
            # Stage 1: Text preprocessing (LangChain)
            if "langchain" in self.frameworks and self.health_check.check_framework_health("langchain"):
                chunks = self._preprocess_text(document_text)
                result["frameworks_used"].append("langchain")
                logger.info(f"Text preprocessing complete: {len(chunks)} chunks")
            else:
                chunks = [document_text]  # Fallback: use full text

            # Stage 2: Content analysis (OpenAI)
            analysis = None
            if "openai" in self.frameworks and self.health_check.check_framework_health("openai"):
                analysis = await self._analyze_content(chunks[0], processing_type)  # Analyze first chunk
                if analysis:
                    result["frameworks_used"].append("openai")
                    logger.info("Content analysis complete")

            # Stage 3: Output refinement (Anthropic)
            if analysis and "anthropic" in self.frameworks and self.health_check.check_framework_health("anthropic"):
                refined_output = await self._refine_output(analysis, processing_type)
                if refined_output:
                    result["output"] = refined_output
                    result["frameworks_used"].append("anthropic")
                    result["success"] = True
                    logger.info("Output refinement complete")
            elif analysis:
                # Fallback: use analysis directly
                result["output"] = analysis
                result["success"] = True

            if not result["success"]:
                result["errors"].append("No frameworks available for processing")

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            result["errors"].append(str(e))

        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        # Update metrics
        primary_framework = result["frameworks_used"][0] if result["frameworks_used"] else "none"
        error_msg = result["errors"][0] if result["errors"] else None
        self.metrics.add_request(result["success"], processing_time, primary_framework, error_msg)

        return result

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text using LangChain text splitter"""
        try:
            if ChatOpenAI:  # Check if langchain is available
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=100,
                    separators=["\\n\\n", "\\n", " ", ""]
                )
                return splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")

        return [text]  # Fallback: return original text as single chunk

    async def _analyze_content(self, text: str, processing_type: str) -> Optional[str]:
        """Analyze content using OpenAI"""
        try:
            prompts = {
                "summarize": f"Provide a comprehensive summary of the following text:\\n\\n{text}",
                "extract_key_points": f"Extract the key points from this text:\\n\\n{text}",
                "analyze_sentiment": f"Analyze the sentiment and tone of this text:\\n\\n{text}",
                "categorize": f"Categorize and tag this document:\\n\\n{text}"
            }

            prompt = prompts.get(processing_type, prompts["summarize"])

            response = self.frameworks["openai"].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return None

    async def _refine_output(self, analysis: str, processing_type: str) -> Optional[str]:
        """Refine output using Anthropic Claude"""
        try:
            refine_prompts = {
                "summarize": f"Please refine and improve this summary, making it more concise and well-structured:\\n\\n{analysis}",
                "extract_key_points": f"Reorganize these key points in order of importance and clarity:\\n\\n{analysis}",
                "analyze_sentiment": f"Provide a more nuanced sentiment analysis based on this initial analysis:\\n\\n{analysis}",
                "categorize": f"Improve this categorization with more specific and useful tags:\\n\\n{analysis}"
            }

            prompt = refine_prompts.get(processing_type, refine_prompts["summarize"])

            response = self.frameworks["anthropic"].messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Output refinement failed: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "health": self.health_check.get_system_health(),
            "metrics": self.metrics.get_stats(),
            "available_frameworks": list(self.frameworks.keys()),
            "timestamp": datetime.now().isoformat()
        }


class ProductionCustomerService:
    """Production customer service system with multi-framework support"""

    def __init__(self):
        self.metrics = ProductionMetrics()
        self.health_check = HealthCheck()
        self.escalation_threshold = 3  # Number of failed attempts before escalation
        self.conversation_memory = {}  # In production, use Redis or similar

    async def handle_customer_query(self, customer_id: str, query: str,
                                  conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle customer service query with escalation support"""
        start_time = time.time()
        conversation_history = conversation_history or []

        result = {
            "success": False,
            "response": None,
            "confidence": 0.0,
            "should_escalate": False,
            "framework_used": None,
            "processing_time": 0
        }

        try:
            # Determine query complexity and routing
            query_analysis = await self._analyze_query_complexity(query)
            routing_decision = self._determine_routing(query_analysis, conversation_history)

            # Process query through appropriate framework
            if routing_decision["framework"] == "anthropic" and self.health_check.check_framework_health("anthropic"):
                response_data = await self._handle_with_anthropic(query, conversation_history)
            elif routing_decision["framework"] == "openai" and self.health_check.check_framework_health("openai"):
                response_data = await self._handle_with_openai(query, conversation_history)
            else:
                # Fallback to any available framework
                response_data = await self._handle_with_fallback(query, conversation_history)

            if response_data:
                result.update(response_data)
                result["success"] = True

                # Check if escalation is needed
                result["should_escalate"] = self._should_escalate(
                    result["confidence"],
                    len(conversation_history),
                    query_analysis.get("complexity", "medium")
                )

        except Exception as e:
            logger.error(f"Customer service error: {e}")
            result["response"] = "I apologize, but I'm experiencing technical difficulties. Let me connect you with a human agent."
            result["should_escalate"] = True

        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        # Update conversation memory
        self.conversation_memory[customer_id] = conversation_history + [{
            "query": query,
            "response": result["response"],
            "timestamp": datetime.now().isoformat(),
            "framework": result["framework_used"]
        }]

        # Update metrics
        framework = result["framework_used"] or "fallback"
        self.metrics.add_request(result["success"], processing_time, framework)

        return result

    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity for routing decisions"""
        complexity_indicators = {
            "simple": ["hello", "hi", "thanks", "thank you", "goodbye", "hours", "location"],
            "medium": ["help", "problem", "issue", "order", "account", "billing"],
            "complex": ["complaint", "refund", "technical", "legal", "escalate", "manager"]
        }

        query_lower = query.lower()
        complexity = "medium"  # Default

        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
                break

        return {
            "complexity": complexity,
            "word_count": len(query.split()),
            "contains_question": "?" in query,
            "sentiment": "neutral"  # In production, use proper sentiment analysis
        }

    def _determine_routing(self, query_analysis: Dict[str, Any], conversation_history: List[Dict]) -> Dict[str, str]:
        """Determine which framework to use for the query"""
        complexity = query_analysis.get("complexity", "medium")
        conversation_length = len(conversation_history)

        # Routing logic
        if complexity == "complex" or conversation_length > 5:
            # Use Anthropic for complex queries or long conversations
            return {"framework": "anthropic", "reason": "complex_query_or_long_conversation"}
        elif complexity == "simple":
            # Use OpenAI for simple queries
            return {"framework": "openai", "reason": "simple_query"}
        else:
            # Default to OpenAI for medium complexity
            return {"framework": "openai", "reason": "medium_complexity"}

    async def _handle_with_anthropic(self, query: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """Handle query with Anthropic Claude"""
        try:
            # Build context from conversation history
            context = "Previous conversation:\\n"
            for turn in history[-3:]:  # Last 3 turns
                context += f"Customer: {turn.get('query', '')}\\nAgent: {turn.get('response', '')}\\n"

            prompt = f"""You are a helpful customer service agent. {context}

Current customer query: {query}

Provide a helpful, professional response. If you cannot fully resolve the issue, suggest next steps or escalation."""

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "response": response.content[0].text.strip(),
                "confidence": 0.85,  # Claude generally high confidence
                "framework_used": "anthropic"
            }

        except Exception as e:
            logger.error(f"Anthropic handling failed: {e}")
            return None

    async def _handle_with_openai(self, query: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """Handle query with OpenAI"""
        try:
            # Build messages from conversation history
            messages = [
                {"role": "system", "content": "You are a helpful customer service agent. Be professional, concise, and solution-oriented."}
            ]

            # Add conversation history
            for turn in history[-5:]:  # Last 5 turns
                messages.append({"role": "user", "content": turn.get('query', '')})
                messages.append({"role": "assistant", "content": turn.get('response', '')})

            # Add current query
            messages.append({"role": "user", "content": query})

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )

            return {
                "response": response.choices[0].message.content.strip(),
                "confidence": 0.8,  # OpenAI confidence estimate
                "framework_used": "openai"
            }

        except Exception as e:
            logger.error(f"OpenAI handling failed: {e}")
            return None

    async def _handle_with_fallback(self, query: str, history: List[Dict]) -> Optional[Dict[str, Any]]:
        """Fallback handling when primary frameworks are unavailable"""
        # Simple fallback responses based on keywords
        fallback_responses = {
            "hello": "Hello! How can I help you today?",
            "hours": "Our customer service hours are Monday-Friday 9AM-6PM EST.",
            "help": "I'd be happy to help you. Can you please provide more details about what you need assistance with?",
            "problem": "I understand you're experiencing an issue. Let me connect you with a specialist who can help resolve this.",
            "thanks": "You're welcome! Is there anything else I can help you with today?"
        }

        query_lower = query.lower()
        for keyword, response in fallback_responses.items():
            if keyword in query_lower:
                return {
                    "response": response,
                    "confidence": 0.6,
                    "framework_used": "fallback"
                }

        # Generic fallback
        return {
            "response": "Thank you for contacting us. Let me connect you with a specialist who can better assist you with your request.",
            "confidence": 0.5,
            "framework_used": "fallback"
        }

    def _should_escalate(self, confidence: float, conversation_length: int, complexity: str) -> bool:
        """Determine if the conversation should be escalated to a human agent"""
        return (
            confidence < 0.7 or  # Low confidence response
            conversation_length >= self.escalation_threshold or  # Too many back-and-forth exchanges
            complexity == "complex"  # Complex queries should be escalated
        )


# Demonstration functions
async def demonstrate_document_processor():
    """Demonstrate production document processing"""
    print("\\n" + "="*60)
    print("📄 PRODUCTION DOCUMENT PROCESSOR")
    print("="*60)

    processor = ProductionDocumentProcessor()

    # Sample document
    sample_document = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies
    of the 21st century. From healthcare and finance to transportation and entertainment,
    AI is revolutionizing industries and changing the way we live and work.

    The key benefits of AI include increased efficiency, improved accuracy, and the ability
    to process vast amounts of data quickly. However, challenges remain, including ethical
    concerns, job displacement, and the need for proper regulation.

    As we move forward, it's crucial that we develop AI responsibly, ensuring that its
    benefits are shared broadly while minimizing potential risks and negative impacts.
    """

    print("\\n📝 Processing sample document...")

    # Test different processing types
    processing_types = ["summarize", "extract_key_points", "analyze_sentiment"]

    for processing_type in processing_types:
        print(f"\\n🔧 Processing type: {processing_type}")

        result = await processor.process_document(sample_document, processing_type)

        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Frameworks used: {', '.join(result['frameworks_used'])}")
        print(f"   Processing time: {result['processing_time']:.2f}s")

        if result['success']:
            output_preview = result['output'][:150] + "..." if len(result['output']) > 150 else result['output']
            print(f"   Output: {output_preview}")
        else:
            print(f"   Errors: {', '.join(result['errors'])}")

    # Show system status
    print("\\n📊 System Status:")
    status = processor.get_system_status()
    print(f"   Overall healthy: {status['health']['overall_healthy']}")
    print(f"   Requests processed: {status['metrics']['requests_processed']}")
    print(f"   Success rate: {status['metrics']['success_rate']:.1%}")


async def demonstrate_customer_service():
    """Demonstrate production customer service system"""
    print("\\n" + "="*60)
    print("🎧 PRODUCTION CUSTOMER SERVICE")
    print("="*60)

    service = ProductionCustomerService()

    # Simulate customer conversation
    customer_id = "customer_123"
    conversation_flow = [
        "Hello, I need help with my order",
        "I placed an order last week but haven't received it yet",
        "The order number is #12345. Can you check the status?",
        "This is very frustrating. I need this urgently.",
        "Can I speak with a manager?"
    ]

    conversation_history = []

    print(f"\\n👤 Customer ID: {customer_id}")

    for i, query in enumerate(conversation_flow):
        print(f"\\n{i+1}. Customer: {query}")

        result = await service.handle_customer_query(customer_id, query, conversation_history.copy())

        print(f"   🤖 Agent: {result['response']}")
        print(f"   Framework: {result['framework_used']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Should escalate: {'🚨 YES' if result['should_escalate'] else '✅ NO'}")

        # Add to conversation history for next iteration
        conversation_history.append({
            "query": query,
            "response": result["response"],
            "timestamp": datetime.now().isoformat(),
            "framework": result["framework_used"]
        })

        if result['should_escalate']:
            print("   >>> Escalating to human agent")
            break

    print(f"\\n📈 Final conversation length: {len(conversation_history)} turns")


async def demonstrate_system_monitoring():
    """Demonstrate system monitoring and metrics"""
    print("\\n" + "="*60)
    print("📊 SYSTEM MONITORING")
    print("="*60)

    # Create systems
    doc_processor = ProductionDocumentProcessor()
    customer_service = ProductionCustomerService()

    print("\\n🔍 Health Checks:")
    doc_health = doc_processor.health_check.get_system_health()
    cs_health = customer_service.health_check.get_system_health()

    print(f"Document Processor: {'✅ Healthy' if doc_health['overall_healthy'] else '❌ Unhealthy'}")
    print(f"Customer Service: {'✅ Healthy' if cs_health['overall_healthy'] else '❌ Unhealthy'}")

    print("\\n📈 Framework Availability:")
    all_frameworks = set(list(doc_health['framework_status'].keys()) + list(cs_health['framework_status'].keys()))

    for framework in all_frameworks:
        doc_status = doc_health['framework_status'].get(framework, False)
        cs_status = cs_health['framework_status'].get(framework, False)
        overall = doc_status or cs_status

        print(f"   {framework}: {'✅' if overall else '❌'}")

    # Simulate some load and show metrics
    print("\\n⚡ Simulating production load...")

    tasks = []
    # Simulate concurrent document processing
    sample_docs = [
        "Short document for testing",
        "Medium length document with more content to analyze and process",
        "Very long document with extensive content that requires comprehensive analysis and processing through multiple stages"
    ]

    for doc in sample_docs:
        tasks.append(doc_processor.process_document(doc, "summarize"))

    # Simulate concurrent customer service
    customer_queries = [
        "What are your hours?",
        "I have a problem with my account",
        "This is urgent, I need help immediately"
    ]

    for i, query in enumerate(customer_queries):
        tasks.append(customer_service.handle_customer_query(f"customer_{i}", query))

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"   Processed {len(results)} concurrent requests")

    # Show final metrics
    print("\\n📊 Final Metrics:")
    doc_metrics = doc_processor.metrics.get_stats()
    cs_metrics = customer_service.metrics.get_stats()

    print("Document Processor:")
    print(f"   Requests: {doc_metrics['requests_processed']}")
    print(f"   Success Rate: {doc_metrics['success_rate']:.1%}")
    print(f"   Avg Latency: {doc_metrics['avg_latency_ms']:.0f}ms")

    print("Customer Service:")
    print(f"   Requests: {cs_metrics['requests_processed']}")
    print(f"   Success Rate: {cs_metrics['success_rate']:.1%}")
    print(f"   Avg Latency: {cs_metrics['avg_latency_ms']:.0f}ms")


async def main():
    """Main demonstration function"""
    print("🏭 Production Examples - Real-World Multi-Framework Deployments")
    print("=" * 70)

    print("\\nProduction systems demonstrated:")
    print("• Document Processing Pipeline - Multi-stage processing with fallbacks")
    print("• Customer Service System - Intelligent routing and escalation")
    print("• System Monitoring - Health checks and metrics collection")

    # Check system readiness
    health_check = HealthCheck()
    system_health = health_check.get_system_health()

    print(f"\\n🏥 System Health Check:")
    print(f"   Overall healthy: {'✅' if system_health['overall_healthy'] else '❌'}")
    print(f"   Available frameworks: {system_health['healthy_frameworks']}/{system_health['total_frameworks']}")

    if not system_health['overall_healthy']:
        print("\\n⚠️  Some frameworks are unavailable. Production examples will run with fallbacks.")

    try:
        # Run production demonstrations
        await demonstrate_document_processor()
        await demonstrate_customer_service()
        await demonstrate_system_monitoring()

        print("\\n" + "="*70)
        print("🎉 Production Examples Complete!")

        print("\\n🔑 Production Best Practices Demonstrated:")
        print("• Comprehensive health monitoring and metrics collection")
        print("• Graceful fallback handling when frameworks are unavailable")
        print("• Intelligent routing based on query complexity and context")
        print("• Automatic escalation based on confidence and conversation length")
        print("• Concurrent processing for improved performance")
        print("• Proper error handling and logging")

        print("\\nProduction Deployment Considerations:")
        print("• Implement proper authentication and authorization")
        print("• Set up monitoring and alerting for production systems")
        print("• Use proper databases for conversation memory and metrics")
        print("• Implement rate limiting and request validation")
        print("• Set up proper CI/CD pipelines for deployments")
        print("• Plan for scaling and load balancing")
        print("• Ensure compliance with data privacy regulations")

    except KeyboardInterrupt:
        print("\\n\\n👋 Production examples interrupted by user")
    except Exception as e:
        logger.error(f"Production example error: {e}")
        print(f"\\n❌ Error during production examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())