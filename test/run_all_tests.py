#!/usr/bin/env python3
"""
tests/run_all_tests.py
Main Test Runner for AI Text Classification System
Executes all test suites and provides comprehensive reporting.
"""

import os
import sys
import time
import traceback
import subprocess
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test modules
try:
    from test_config import run_config_tests
    from test_data_processor import run_data_processor_tests
    from test_ai_classifier import run_ai_classifier_tests
    from test_clustering import run_clustering_tests
    from test_evaluation import run_evaluation_tests
    from test_pipeline import run_pipeline_tests
    from test_utilities import run_utilities_tests
    from test_data_generator import main as generate_test_data
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the tests directory and all test files are present.")
    sys.exit(1)

class TestSuiteRunner:
    """Main test suite runner with comprehensive reporting."""
    
    def __init__(self, verbose=False, skip_slow=False, generate_data=True):
        """
        Initialize test runner.
        
        Args:
            verbose: Enable verbose output
            skip_slow: Skip slow tests (like pipeline integration)
            generate_data: Generate test data before running tests
        """
        self.verbose = verbose
        self.skip_slow = skip_slow
        self.generate_data = generate_data
        
        self.results = {}
        self.total_start_time = time.time()
        self.failed_tests = []
        
    def print_banner(self):
        """Print test suite banner."""
        print("=" * 80)
        print("🧪 AI TEXT CLASSIFICATION SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
        print(f"Skip slow tests: {'YES' if self.skip_slow else 'NO'}")
        print(f"Generate test data: {'YES' if self.generate_data else 'NO'}")
        print("=" * 80)
        print()
    
    def run_test_suite(self, suite_name, test_function, is_slow=False):
        """
        Run a single test suite with error handling and timing.
        
        Args:
            suite_name: Name of the test suite
            test_function: Function to execute
            is_slow: Whether this is a slow test
        """
        if is_slow and self.skip_slow:
            print(f"⏭️  SKIPPING {suite_name} (slow test)")
            self.results[suite_name] = {
                'status': 'skipped',
                'duration': 0,
                'error': None
            }
            return
        
        print(f"🚀 STARTING {suite_name}")
        suite_start_time = time.time()
        
        try:
            if self.verbose:
                test_function()
            else:
                # Capture output for non-verbose mode
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    test_function()
                
                # Only show output if there were issues
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()
                
                if stderr_content:
                    print(f"⚠️  Warnings/Errors in {suite_name}:")
                    print(stderr_content)
            
            suite_duration = time.time() - suite_start_time
            self.results[suite_name] = {
                'status': 'passed',
                'duration': suite_duration,
                'error': None
            }
            
            print(f"✅ COMPLETED {suite_name} in {suite_duration:.2f}s")
            
        except Exception as e:
            suite_duration = time.time() - suite_start_time
            error_msg = str(e)
            
            self.results[suite_name] = {
                'status': 'failed',
                'duration': suite_duration,
                'error': error_msg
            }
            
            self.failed_tests.append(suite_name)
            
            print(f"❌ FAILED {suite_name} in {suite_duration:.2f}s")
            print(f"   Error: {error_msg}")
            
            if self.verbose:
                print("   Full traceback:")
                traceback.print_exc()
        
        print()
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        print("🔍 CHECKING DEPENDENCIES")
        print("-" * 40)
        
        dependencies = {
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing',
            'sklearn': 'Machine learning',
            'yaml': 'Configuration files',
            'pytest': 'Testing framework'
        }
        
        missing_deps = []
        
        for dep, description in dependencies.items():
            try:
                __import__(dep)
                print(f"✅ {dep:12} - {description}")
            except ImportError:
                print(f"❌ {dep:12} - {description} (MISSING)")
                missing_deps.append(dep)
        
        # Check optional dependencies
        optional_deps = {
            'hdbscan': 'HDBSCAN clustering',
            'umap': 'UMAP dimensionality reduction',
            'openai': 'OpenAI API integration',
            'tiktoken': 'Token counting',
            'matplotlib': 'Basic plotting',
            'seaborn': 'Statistical plotting'
        }
        
        print("\nOptional dependencies:")
        for dep, description in optional_deps.items():
            try:
                __import__(dep)
                print(f"✅ {dep:12} - {description}")
            except ImportError:
                print(f"⚠️  {dep:12} - {description} (missing - some tests may be skipped)")
        
        print()
        
        if missing_deps:
            print(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
            print("Install with: pip install " + " ".join(missing_deps))
            return False
        
        return True
    
    def check_environment(self):
        """Check environment setup."""
        print("🌍 CHECKING ENVIRONMENT")
        print("-" * 40)
        
        # Check Python version
        print(f"Python version: {sys.version}")
        
        # Check if we're in the right directory
        current_dir = os.getcwd()
        expected_files = ['config.py', 'main.py', 'modules']
        
        missing_files = []
        for file in expected_files:
            if not os.path.exists(os.path.join('..', file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Warning: Not in expected directory structure")
            print(f"   Missing: {', '.join(missing_files)}")
            print(f"   Current directory: {current_dir}")
        else:
            print("✅ Directory structure looks correct")
        
        # Check for test data directory
        if not os.path.exists('data'):
            print("📁 Creating test data directory...")
            os.makedirs('data', exist_ok=True)
        
        print()
    
    def generate_test_data_if_needed(self):
        """Generate test data if requested."""
        if not self.generate_data:
            return
            
        print("📊 GENERATING TEST DATA")
        print("-" * 40)
        
        try:
            # Check if test data already exists
            test_data_file = "data/test_job_dataset.dta"
            if os.path.exists(test_data_file):
                print(f"✅ Test data already exists: {test_data_file}")
            else:
                print("Generating fresh test data...")
                generate_test_data()
                
        except Exception as e:
            print(f"⚠️  Test data generation failed: {e}")
            print("   Tests will continue with existing data or skip data-dependent tests")
        
        print()
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 RUNNING ALL TEST SUITES")
        print("=" * 80)
        print()
        
        # Define test suites
        test_suites = [
            ("Configuration Tests", run_config_tests, False),
            ("Data Processing Tests", run_data_processor_tests, False),
            ("AI Classification Tests", run_ai_classifier_tests, False),
            ("Traditional Clustering Tests", run_clustering_tests, False),
            ("Evaluation & Visualization Tests", run_evaluation_tests, False),
            ("Utilities Tests", run_utilities_tests, False),
            ("Pipeline Integration Tests", run_pipeline_tests, True),  # Slow test
        ]
        
        # Run each test suite
        for suite_name, test_function, is_slow in test_suites:
            self.run_test_suite(suite_name, test_function, is_slow)
    
    def generate_summary_report(self):
        """Generate and display summary report."""
        total_duration = time.time() - self.total_start_time
        
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 80)
        
        # Count results
        passed = len([r for r in self.results.values() if r['status'] == 'passed'])
        failed = len([r for r in self.results.values() if r['status'] == 'failed'])
        skipped = len([r for r in self.results.values() if r['status'] == 'skipped'])
        total = len(self.results)
        
        print(f"📈 Overall Results:")
        print(f"   ✅ Passed:  {passed}/{total}")
        print(f"   ❌ Failed:  {failed}/{total}")
        print(f"   ⏭️  Skipped: {skipped}/{total}")
        print(f"   ⏱️  Total time: {total_duration:.2f}s")
        print()
        
        # Detailed results
        print("📋 Detailed Results:")
        print("-" * 40)
        for suite_name, result in self.results.items():
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'skipped': '⏭️'
            }.get(result['status'], '❓')
            
            print(f"{status_icon} {suite_name:30} {result['duration']:6.2f}s")
            if result['error']:
                print(f"    Error: {result['error']}")
        
        print()
        
        # Failed tests details
        if self.failed_tests:
            print("🚨 FAILED TESTS:")
            print("-" * 40)
            for failed_test in self.failed_tests:
                error = self.results[failed_test]['error']
                print(f"❌ {failed_test}")
                print(f"   {error}")
            print()
        
        # Performance summary
        print("⚡ Performance Summary:")
        print("-" * 40)
        fastest = min(self.results.values(), key=lambda x: x['duration'])
        slowest = max(self.results.values(), key=lambda x: x['duration'])
        
        fastest_name = [name for name, result in self.results.items() if result == fastest][0]
        slowest_name = [name for name, result in self.results.items() if result == slowest][0]
        
        print(f"🚀 Fastest: {fastest_name} ({fastest['duration']:.2f}s)")
        print(f"🐌 Slowest: {slowest_name} ({slowest['duration']:.2f}s)")
        print()
        
        # Final status
        if failed == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("The AI Text Classification System is working correctly.")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED")
            print("Please review the failed tests and fix any issues.")
        
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return failed == 0
    
    def save_report_to_file(self):
        """Save test report to file."""
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write(f"AI Text Classification System - Test Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for suite_name, result in self.results.items():
                    f.write(f"{suite_name}: {result['status'].upper()} ({result['duration']:.2f}s)\n")
                    if result['error']:
                        f.write(f"  Error: {result['error']}\n")
                f.write("\n")
                
                if self.failed_tests:
                    f.write("Failed Tests:\n")
                    for test in self.failed_tests:
                        f.write(f"- {test}\n")
                
            print(f"📝 Test report saved to: {report_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save report to file: {e}")

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='AI Text Classification System Test Runner')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow tests (like pipeline integration)')
    parser.add_argument('--no-data-gen', action='store_true',
                       help='Skip test data generation')
    parser.add_argument('--save-report', action='store_true',
                       help='Save test report to file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (skip slow tests and data generation)')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.skip_slow = True
        args.no_data_gen = True
    
    # Initialize test runner
    runner = TestSuiteRunner(
        verbose=args.verbose,
        skip_slow=args.skip_slow,
        generate_data=not args.no_data_gen
    )
    
    try:
        # Print banner
        runner.print_banner()
        
        # Pre-flight checks
        runner.check_environment()
        
        if not runner.check_dependencies():
            print("❌ Dependency check failed. Please install missing dependencies.")
            return 1
        
        # Generate test data if needed
        runner.generate_test_data_if_needed()
        
        # Run all tests
        runner.run_all_tests()
        
        # Generate summary
        success = runner.generate_summary_report()
        
        # Save report if requested
        if args.save_report:
            runner.save_report_to_file()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n💥 Unexpected error in test runner: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())