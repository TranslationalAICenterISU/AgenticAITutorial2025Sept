#!/usr/bin/env python3
"""
Comprehensive Testing Script for Agentic AI Tutorial
====================================================

This script tests all example files across all tutorial modules to ensure they:
1. Import correctly without errors
2. Have basic functionality working
3. Handle missing dependencies gracefully
4. Provide meaningful error messages

Usage:
    python test_all_examples.py                    # Test all modules
    python test_all_examples.py --module 03        # Test specific module
    python test_all_examples.py --quick            # Quick test (imports only)
    python test_all_examples.py --detailed         # Detailed test with execution
"""

import os
import sys
import importlib
import traceback
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures


@dataclass
class TestResult:
    """Test result for a single file"""
    file_path: str
    test_type: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    output: Optional[str] = None


@dataclass
class ModuleTestReport:
    """Test report for a module"""
    module_name: str
    total_files: int
    tested_files: int
    passed_files: int
    failed_files: int
    execution_time: float
    results: List[TestResult] = field(default_factory=list)


class TutorialTester:
    """Comprehensive tester for tutorial examples"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.modules = self._discover_modules()
        self.test_results = {}

    def _discover_modules(self) -> Dict[str, Path]:
        """Discover all tutorial modules"""
        modules = {}
        for path in self.base_dir.iterdir():
            if path.is_dir() and path.name.startswith(('0', '1')):
                # Match patterns like 01-foundations, 02-llm-apis, etc.
                if '-' in path.name:
                    modules[path.name] = path
        return modules

    def test_file_import(self, file_path: Path) -> TestResult:
        """Test if a Python file can be imported"""
        start_time = time.time()

        try:
            # Get module path relative to base directory
            relative_path = file_path.relative_to(self.base_dir)
            module_parts = list(relative_path.parts[:-1])  # Remove filename
            module_parts.append(relative_path.stem)  # Add filename without extension
            module_name = '.'.join(module_parts)

            # Temporarily add base directory to Python path
            old_path = sys.path[:]
            sys.path.insert(0, str(self.base_dir))

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Check for common required attributes/functions
                warnings = []
                if hasattr(module, 'main') and callable(getattr(module, 'main')):
                    warnings.append("Has main() function")
                if hasattr(module, '__doc__') and module.__doc__:
                    warnings.append("Has module documentation")

                execution_time = time.time() - start_time
                return TestResult(
                    file_path=str(file_path),
                    test_type="import",
                    success=True,
                    execution_time=execution_time,
                    warnings=warnings
                )

            finally:
                sys.path = old_path
                # Clean up imported module
                if module_name in sys.modules:
                    del sys.modules[module_name]

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                file_path=str(file_path),
                test_type="import",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_file_syntax(self, file_path: Path) -> TestResult:
        """Test if a Python file has valid syntax"""
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Compile to check syntax
            compile(content, str(file_path), 'exec')

            execution_time = time.time() - start_time
            return TestResult(
                file_path=str(file_path),
                test_type="syntax",
                success=True,
                execution_time=execution_time
            )

        except SyntaxError as e:
            execution_time = time.time() - start_time
            return TestResult(
                file_path=str(file_path),
                test_type="syntax",
                success=False,
                execution_time=execution_time,
                error_message=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                file_path=str(file_path),
                test_type="syntax",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_file_execution(self, file_path: Path, timeout: int = 30) -> TestResult:
        """Test if a Python file can execute without errors"""
        start_time = time.time()

        try:
            # Use subprocess to run the file in isolation
            result = subprocess.run(
                [sys.executable, str(file_path), '--help'],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )

            execution_time = time.time() - start_time

            # Check if execution was successful or if it's expected to fail with --help
            if result.returncode == 0 or '--help' in result.stderr or 'help' in result.stdout:
                return TestResult(
                    file_path=str(file_path),
                    test_type="execution",
                    success=True,
                    execution_time=execution_time,
                    output=result.stdout[:500] if result.stdout else None
                )
            else:
                return TestResult(
                    file_path=str(file_path),
                    test_type="execution",
                    success=False,
                    execution_time=execution_time,
                    error_message=result.stderr[:500] if result.stderr else "Unknown execution error",
                    output=result.stdout[:500] if result.stdout else None
                )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestResult(
                file_path=str(file_path),
                test_type="execution",
                success=False,
                execution_time=execution_time,
                error_message=f"Execution timeout after {timeout} seconds"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                file_path=str(file_path),
                test_type="execution",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def test_module(self, module_name: str, test_type: str = "import") -> ModuleTestReport:
        """Test all Python files in a module"""
        print(f"\\n🧪 Testing module: {module_name}")
        print("=" * 50)

        if module_name not in self.modules:
            print(f"❌ Module {module_name} not found")
            return ModuleTestReport(
                module_name=module_name,
                total_files=0,
                tested_files=0,
                passed_files=0,
                failed_files=0,
                execution_time=0
            )

        module_path = self.modules[module_name]
        python_files = list(module_path.rglob("*.py"))

        if not python_files:
            print(f"⚠️  No Python files found in {module_name}")
            return ModuleTestReport(
                module_name=module_name,
                total_files=0,
                tested_files=0,
                passed_files=0,
                failed_files=0,
                execution_time=0
            )

        print(f"Found {len(python_files)} Python files")

        start_time = time.time()
        results = []
        passed = 0
        failed = 0

        for file_path in python_files:
            print(f"  Testing: {file_path.relative_to(self.base_dir)}")

            if test_type == "syntax":
                result = self.test_file_syntax(file_path)
            elif test_type == "execution":
                result = self.test_file_execution(file_path)
            else:  # default to import test
                result = self.test_file_import(file_path)

            results.append(result)

            if result.success:
                passed += 1
                status = "✅ PASS"
                if result.warnings:
                    status += f" ({', '.join(result.warnings)})"
                print(f"    {status}")
            else:
                failed += 1
                print(f"    ❌ FAIL: {result.error_message}")
                if result.output:
                    print(f"    Output: {result.output[:100]}...")

        execution_time = time.time() - start_time

        report = ModuleTestReport(
            module_name=module_name,
            total_files=len(python_files),
            tested_files=len(results),
            passed_files=passed,
            failed_files=failed,
            execution_time=execution_time,
            results=results
        )

        print(f"\\n📊 Module Results: {passed}/{len(python_files)} passed ({passed/len(python_files)*100:.1f}%)")
        return report

    def test_all_modules(self, test_type: str = "import") -> Dict[str, ModuleTestReport]:
        """Test all tutorial modules"""
        print("🚀 Agentic AI Tutorial - Comprehensive Testing")
        print("=" * 60)

        print(f"Testing {len(self.modules)} modules with {test_type} tests:")
        for module_name in sorted(self.modules.keys()):
            print(f"  • {module_name}")

        reports = {}
        total_start_time = time.time()

        for module_name in sorted(self.modules.keys()):
            report = self.test_module(module_name, test_type)
            reports[module_name] = report
            self.test_results[module_name] = report

        total_execution_time = time.time() - total_start_time

        # Generate summary
        self._print_summary(reports, total_execution_time)
        return reports

    def _print_summary(self, reports: Dict[str, ModuleTestReport], total_time: float):
        """Print comprehensive test summary"""
        print("\\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)

        total_files = sum(r.total_files for r in reports.values())
        total_passed = sum(r.passed_files for r in reports.values())
        total_failed = sum(r.failed_files for r in reports.values())

        print(f"\\n📊 Overall Results:")
        print(f"   Total modules tested: {len(reports)}")
        print(f"   Total files tested: {total_files}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Success rate: {total_passed/total_files*100:.1f}%" if total_files > 0 else "   No files to test")
        print(f"   Total execution time: {total_time:.2f} seconds")

        print(f"\\n📈 Module Breakdown:")
        print(f"{'Module':<20} {'Files':<8} {'Passed':<8} {'Failed':<8} {'Rate':<8}")
        print("-" * 60)

        for module_name in sorted(reports.keys()):
            report = reports[module_name]
            if report.total_files > 0:
                success_rate = f"{report.passed_files/report.total_files*100:.1f}%"
            else:
                success_rate = "N/A"

            print(f"{module_name:<20} {report.total_files:<8} {report.passed_files:<8} {report.failed_files:<8} {success_rate:<8}")

        # Show failed tests
        failed_results = []
        for report in reports.values():
            failed_results.extend([r for r in report.results if not r.success])

        if failed_results:
            print(f"\\n❌ Failed Tests ({len(failed_results)}):")
            print("-" * 40)
            for result in failed_results:
                file_name = Path(result.file_path).name
                print(f"   {file_name}: {result.error_message}")

        print(f"\\n🎯 Recommendations:")
        if total_failed == 0:
            print("   🎉 All tests passed! Tutorial examples are ready.")
        elif total_failed < total_files * 0.1:
            print("   ✅ Most tests passed. Fix remaining issues for optimal experience.")
        elif total_failed < total_files * 0.3:
            print("   ⚠️  Some issues found. Review failed tests before tutorial.")
        else:
            print("   🚨 Many issues found. Comprehensive review needed before tutorial.")

        print(f"\\n📝 Next Steps:")
        if failed_results:
            print("   1. Review failed test output above")
            print("   2. Fix import errors and missing dependencies")
            print("   3. Ensure all API keys are properly configured")
            print("   4. Re-run tests to verify fixes")
        else:
            print("   1. Tutorial examples are ready to use!")
            print("   2. Ensure participants have required API keys")
            print("   3. Share setup instructions from README.md")

    def generate_report(self, output_file: str = "test_report.json"):
        """Generate detailed JSON report"""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "modules": {}
        }

        for module_name, report in self.test_results.items():
            report_data["modules"][module_name] = {
                "total_files": report.total_files,
                "tested_files": report.tested_files,
                "passed_files": report.passed_files,
                "failed_files": report.failed_files,
                "execution_time": report.execution_time,
                "results": [
                    {
                        "file_path": r.file_path,
                        "test_type": r.test_type,
                        "success": r.success,
                        "execution_time": r.execution_time,
                        "error_message": r.error_message,
                        "warnings": r.warnings,
                        "output": r.output
                    }
                    for r in report.results
                ]
            }

        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\\n📄 Detailed report saved to: {output_file}")

    def check_dependencies(self):
        """Check if required packages are installed"""
        print("\\n🔍 Checking Dependencies")
        print("=" * 30)

        required_packages = [
            ("openai", "OpenAI API integration"),
            ("anthropic", "Anthropic API integration"),
            ("langchain", "LangChain framework"),
            ("langgraph", "LangGraph workflows"),
            ("crewai", "CrewAI multi-agent"),
            ("dspy", "DSPy optimization", "dspy-ai"),
            ("transformers", "HuggingFace transformers"),
            ("python-dotenv", "Environment variables", "python-dotenv")
        ]

        missing_packages = []
        available_packages = []

        for package_info in required_packages:
            package_name = package_info[0]
            description = package_info[1]
            pip_name = package_info[2] if len(package_info) > 2 else package_name

            try:
                importlib.import_module(package_name)
                available_packages.append((package_name, description))
                print(f"   ✅ {package_name:<15} - {description}")
            except ImportError:
                missing_packages.append((pip_name, description))
                print(f"   ❌ {package_name:<15} - {description}")

        print(f"\\n📊 Dependency Summary:")
        print(f"   Available: {len(available_packages)}/{len(required_packages)}")
        print(f"   Missing: {len(missing_packages)}")

        if missing_packages:
            print(f"\\n💡 Install missing packages:")
            packages_to_install = " ".join([p[0] for p in missing_packages])
            print(f"   pip install {packages_to_install}")

        return len(missing_packages) == 0

    def check_api_keys(self):
        """Check if API keys are configured"""
        print("\\n🔑 Checking API Keys")
        print("=" * 25)

        from dotenv import load_dotenv
        load_dotenv()

        required_keys = [
            ("OPENAI_API_KEY", "OpenAI API access"),
            ("ANTHROPIC_API_KEY", "Anthropic Claude access"),
            ("GOOGLE_API_KEY", "Google AI/Gemini access (optional)"),
        ]

        configured_keys = []
        missing_keys = []

        for key_name, description in required_keys:
            key_value = os.getenv(key_name)
            if key_value and key_value != f"your_{key_name.lower()}_here":
                configured_keys.append((key_name, description))
                # Mask the key for security
                masked_key = key_value[:8] + "..." if len(key_value) > 8 else "***"
                print(f"   ✅ {key_name:<20} - {description} ({masked_key})")
            else:
                missing_keys.append((key_name, description))
                print(f"   ❌ {key_name:<20} - {description}")

        print(f"\\n📊 API Key Summary:")
        print(f"   Configured: {len(configured_keys)}")
        print(f"   Missing: {len(missing_keys)}")

        if missing_keys:
            print(f"\\n💡 Configure missing API keys in .env file:")
            for key_name, _ in missing_keys:
                print(f"   {key_name}=your_key_here")

        return len(missing_keys) <= 1  # Allow one missing key (Google is optional)


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Test Agentic AI Tutorial Examples")
    parser.add_argument("--module", help="Test specific module (e.g., 01-foundations)")
    parser.add_argument("--quick", action="store_true", help="Quick test (imports only)")
    parser.add_argument("--detailed", action="store_true", help="Detailed test with execution")
    parser.add_argument("--syntax", action="store_true", help="Syntax checking only")
    parser.add_argument("--report", help="Generate JSON report file")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency checks")

    args = parser.parse_args()

    tester = TutorialTester()

    # Determine test type
    if args.detailed:
        test_type = "execution"
    elif args.syntax:
        test_type = "syntax"
    else:
        test_type = "import"

    # Run dependency and API key checks unless skipped
    if not args.no_deps:
        deps_ok = tester.check_dependencies()
        keys_ok = tester.check_api_keys()

        if not deps_ok:
            print("\\n⚠️  Some dependencies are missing. Tests may fail.")
        if not keys_ok:
            print("\\n⚠️  Some API keys are missing. API-dependent tests may fail.")

    # Run tests
    if args.module:
        tester.test_module(args.module, test_type)
    else:
        tester.test_all_modules(test_type)

    # Generate report if requested
    if args.report:
        tester.generate_report(args.report)

    print("\\n🏁 Testing complete!")


if __name__ == "__main__":
    main()