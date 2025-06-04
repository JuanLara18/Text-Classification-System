#!/usr/bin/env python3
"""
Simple Setup Script for AI Text Classification System
Creates essential directories, config template, and environment file.
"""

import os

def print_banner():
    """Print setup banner."""
    print("=" * 50)
    print("🤖 AI Text Classification System")
    print("=" * 50)
    print("Setting up essential files and directories...")
    print()

def create_directories():
    """Create essential directories only."""
    print("📁 Creating directories...")
    
    directories = [
        "input",           # For input data files
        "output",          # For output results
        "logs",            # For log files
        "cache",           # For caching data
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    print()

def create_config_template():
    """Create a simple, ready-to-use config template."""
    print("⚙️ Creating config.yaml template...")
    
    config_content = '''# AI Text Classification System Configuration
# Quick Setup: Just update the file paths and API key below

#######################
# ESSENTIAL SETTINGS (MUST CHANGE THESE)
#######################
input_file: "input/your_data_file.dta"           # ⚠️ UPDATE: Path to your Stata file
output_file: "output/classified_data.dta"        # ⚠️ UPDATE: Where to save results
text_columns:                                     # ⚠️ UPDATE: Your text column names
  - "position_name"                               # Example - change to your column name
  - "job_description"                             # Example - add/remove as needed

#######################
# AI CLASSIFICATION SETUP
#######################
clustering_perspectives:
  # AI-powered job position classifier (example)
  position_classifier:
    type: "openai_classification"
    columns: 
      - "position_name"                           # Must match text_columns above
    target_categories:                            # ⚠️ UPDATE: Your categories
      - "Management & Executive"
      - "Engineering & Technical"
      - "Sales & Marketing" 
      - "Human Resources"
      - "Finance & Accounting"
      - "Operations"
      - "Customer Service"
      - "Administration"
      - "Other"
    output_column: "job_category"                 # Name of new classification column
    
    # OpenAI settings (optimized for cost and speed)
    llm_config:
      provider: "openai"
      model: "gpt-4o-mini"                        # Most cost-effective model
      temperature: 0.0                            # Consistent results
      max_tokens: 10                              # Short responses
      api_key_env: "OPENAI_API_KEY"              # Environment variable name
    
    # Classification settings
    classification_config:
      batch_size: 50                              # Process 50 at a time
      unknown_category: "Other"
      prompt_template: |
        Classify this job position into one category:
        Categories: {categories}
        Position: {text}
        Respond with only the category name.

#######################
# SYSTEM SETTINGS (Usually don't need to change)
#######################
results_dir: "output"
preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  min_word_length: 2

performance:
  batch_size: 100
  parallel_jobs: 4
  cache_embeddings: true
  cache_directory: "cache"

ai_classification:
  cost_management:
    max_cost_per_run: 10.0                       # Stop if cost exceeds $10
  caching:
    enabled: true
    cache_directory: "cache"
  rate_limiting:
    requests_per_minute: 100

spark:
  executor_memory: "4g"
  driver_memory: "4g"
  executor_cores: 2

logging:
  level: "INFO"
  log_file: "logs/classification.log"
  console_output: true

options:
  seed: 42
'''
    
    with open("config.yaml", "w") as f:
        f.write(config_content)
    
    print("   ✅ config.yaml created")
    print()

def create_env_template():
    """Create environment file template."""
    print("🔑 Creating .env template...")
    
    env_content = '''# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Uncomment and set if needed
# LOG_LEVEL=INFO
'''
    
    with open(".env.template", "w") as f:
        f.write(env_content)
    
    # Create actual .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("   ✅ .env file created (ADD YOUR API KEY HERE)")
    else:
        print("   ✅ .env.template created (.env already exists)")
    print()

def create_gitignore():
    """Create essential .gitignore."""
    print("🚫 Creating .gitignore...")
    
    gitignore_content = '''# Environment and secrets
.env
*.key
credentials.*

# Data files
input/
output/
*.dta
*.csv
*.xlsx

# Cache and logs
cache/
logs/
*.log

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# System
.DS_Store
Thumbs.db
'''
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("   ✅ .gitignore created")
    print()

def print_next_steps():
    """Print what the user needs to do next."""
    print("🎯 NEXT STEPS:")
    print("=" * 50)
    print("1. Add your OpenAI API key to .env file:")
    print("   OPENAI_API_KEY=sk-your-actual-key-here")
    print()
    print("2. Edit config.yaml and update:")
    print("   - input_file: path to your .dta file")
    print("   - text_columns: your column names")
    print("   - target_categories: your classification categories")
    print()
    print("3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("4. Run the classification:")
    print("   python main.py --config config.yaml")
    print()
    print("📖 See README.md for detailed instructions")

def main():
    """Main setup function."""
    print_banner()
    create_directories()
    create_config_template()
    create_env_template()
    create_gitignore()
    print_next_steps()

if __name__ == "__main__":
    main()
