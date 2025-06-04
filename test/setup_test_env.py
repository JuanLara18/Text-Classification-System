# tests/setup_test_env.py
# Test environment setup script

#!/usr/bin/env python3
"""
Setup script for test environment.
Creates necessary directories and files for testing.
"""

import os
import sys
import shutil

def create_test_directories():
    """Create necessary test directories."""
    directories = [
        'tests/data',
        'tests/output', 
        'tests/results',
        'tests/configs',
        'tests/cache',
        'tests/logs',
        'tests/checkpoints',
        'tests/coverage_html'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_test_configs():
    """Create test configuration files if they don't exist."""
    config_files = [
        'tests/configs/test_config_ai.yaml',
        'tests/configs/test_config_clustering.yaml', 
        'tests/configs/test_config_hybrid.yaml'
    ]
    
    existing_configs = [f for f in config_files if os.path.exists(f)]
    
    if existing_configs:
        print(f"✅ Test configs already exist: {len(existing_configs)}/{len(config_files)}")
    else:
        print("⚠️  Test configs not found. Run test_data_generator.py to create them.")

def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = [
        'pytest', 'pandas', 'numpy', 'scikit-learn', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_gitignore_if_needed():
    """Create .gitignore for test directories."""
    gitignore_content = """# Test outputs
tests/output/
tests/results/
tests/cache/
tests/logs/
tests/checkpoints/
tests/coverage_html/
tests/data/*.dta
tests/data/*.csv

# Test reports
test_report_*.txt
*.coverage

# Temporary files
tests/temp/
tests/tmp/
"""
    
    gitignore_path = 'tests/.gitignore'
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(f"✅ Created {gitignore_path}")
    else:
        print(f"✅ {gitignore_path} already exists")

def main():
    """Main setup function."""
    print("🔧 Setting up test environment...")
    print("=" * 50)
    
    # Create directories
    create_test_directories()
    print()
    
    # Check dependencies
    print("📦 Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    # Check configs
    print("⚙️  Checking test configurations...")
    create_test_configs()
    print()
    
    # Create gitignore
    print("📝 Creating .gitignore...")
    create_gitignore_if_needed()
    print()
    
    # Final status
    if deps_ok:
        print("✅ Test environment setup complete!")
        print("\n🚀 Ready to run tests:")
        print("   python tests/run_all_tests.py")
        print("   python tests/run_all_tests.py --quick")
        print("   python tests/run_all_tests.py --verbose")
    else:
        print("⚠️  Test environment setup completed with missing dependencies.")
        print("   Install missing packages before running tests.")

if __name__ == "__main__":
    main()