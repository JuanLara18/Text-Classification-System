#!/usr/bin/env python3
"""
Simplified Setup Script for AI Text Classification System
Creates essential directories and environment file.
"""

import os

def print_banner():
    """Print setup banner."""
    print("=" * 50)
    print("AI Text Classification System")
    print("=" * 50)
    print("Setting up essential files and directories...")
    print()

def create_directories():
    """Create essential directories only."""
    print("* Creating directories...")
    
    directories = [
        "input",           # For input data files
        "output",          # For output results
        "logs",            # For log files
        "cache",           # For caching data
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  + {directory}/")
    print()

def create_env_file():
    """Create environment file."""
    print("* Creating .env file...")
    
    env_content = '''# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Uncomment and set if needed
# LOG_LEVEL=INFO
'''
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("  + .env file created (ADD YOUR API KEY HERE)")
    else:
        print("   = .env file already exists")
    print()

def print_next_steps():
    """Print what the user needs to do next."""
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Add your OpenAI API key to .env file:")
    print("   OPENAI_API_KEY=sk-your-actual-key-here")
    print()
    print("2. Create and activate a new environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # ./venv/Scripts/activate in Windows")
    print()
    print("3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("4. Install additional package:")
    print("   python nltk_download.py")
    print()
    print("5. Configure your classification settings:")
    print("   python app.py")
    print()
    print("6. Download the generated config.yaml from the web interface")
    print()
    print("7. Run the classification:")
    print("   python main.py --config config.yaml")
    print()
    print(" ** See README.md for detailed instructions")

def main():
    """Main setup function."""
    print_banner()
    create_directories()
    create_env_file()
    print_next_steps()

if __name__ == "__main__":
    main()