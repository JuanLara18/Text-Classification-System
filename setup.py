#!/usr/bin/env python3
"""
AI Classification System - Complete Setup Script
Run this script to automatically set up your AI classification system.

Usage: python setup.py
"""

import os

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🚀 AI Classification System Setup")
    print("=" * 60)
    print("Setting up your AI-powered text classification system...")
    print()

def create_directory_structure():
    """Create the complete directory structure."""
    print("📁 Creating directory structure...")
    
    directories = [
        "modules",
        "tests", 
        "input",
        "output",
        "output/classification_results",
        "cache",
        "ai_cache",
        "checkpoints",
        "logs",
        "config_templates",
        "scripts",
        "backup"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    print()

def create_config_file():
    """Create the main configuration file optimized for user's specs."""
    config_content = """# AI Classification System Configuration
# Optimized for: 128GB RAM, 12 CPU, GPT-3.5 Turbo
# Target: HR position classification

#######################
# File paths and basics
#######################
input_file: "input/HR_monthly_panel_translated.dta"
output_file: "output/HR_monthly_panel_classified.dta"
results_dir: "output/classification_results"

# Text columns to classify
text_columns:
  - position_name_english

#######################
# Text Preprocessing
#######################
preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: false
  custom_stopwords: []
  min_word_length: 2
  max_length: 5000

#######################
# Feature Extraction (for clustering perspectives)
#######################
feature_extraction:
  method: "embedding"
  embedding:
    model: "sentence-transformers"
    sentence_transformers:
      model_name: "all-MiniLM-L6-v2"
    dimensionality_reduction:
      method: "umap"
      n_components: 30
      random_state: 42

#######################
# Classification Perspectives
#######################
clustering_perspectives:
  # AI-powered HR position classification
  hr_position_classifier:
    type: "openai_classification"
    columns: 
      - position_name_english
    target_categories:
      - "Executive & Strategy"
      - "Legal, Compliance & Risk"
      - "Human Resources (HR)"
      - "Administration"
      - "Information Technology (IT)"
      - "Operations/Manufacturing"
      - "Supply Chain"
      - "Marketing & Communications"
      - "Sales"
      - "Research and Development (R&D)"
      - "Accounting and Finance"
      - "Customer Service & Support"
      - "Unknown"
    output_column: "position_category_ai"
    
    # OpenAI Configuration - Optimized for cost and accuracy
    llm_config:
      provider: "openai"
      model: "gpt-3.5-turbo-0125"  # Most cost-effective
      temperature: 0.0             # Deterministic results
      max_tokens: 30               # Short responses
      timeout: 30
      max_retries: 3
      api_key_env: "OPENAI_API_KEY"
    
    # Classification settings
    classification_config:
      unknown_category: "Unknown"
      batch_size: 25               # Optimized for your system
      include_unknown_in_categories: true
      prompt_template: |
        Classify this job position into exactly one category:
        
        Categories:
        - Executive & Strategy
        - Legal, Compliance & Risk  
        - Human Resources (HR)
        - Administration
        - Information Technology (IT)
        - Operations/Manufacturing
        - Supply Chain
        - Marketing & Communications
        - Sales
        - Research and Development (R&D)
        - Accounting and Finance
        - Customer Service & Support
        - Unknown
        
        Position: {text}
        
        Respond with ONLY the category name from the list above.
        Category:
    
    # Validation settings
    validation:
      strict_category_matching: true
      fallback_strategy: "unknown"

#######################
# AI Classification Global Settings
#######################
ai_classification:
  # Cost management
  cost_management:
    max_cost_per_run: 50.0        # USD limit
    track_token_usage: true
    cost_alerts: true
  
  # Intelligent caching
  caching:
    enabled: true
    cache_duration_days: 60       # Long cache for stable results
    cache_directory: "ai_cache"
    
  # Rate limiting (optimized for GPT-3.5)
  rate_limiting:
    requests_per_minute: 120      # Higher for GPT-3.5
    batch_delay_seconds: 0.5
    
  # Monitoring
  monitoring:
    log_api_calls: true
    track_classification_accuracy: true

#######################
# Performance Settings (Optimized for 128GB RAM, 12 CPU)
#######################
performance:
  batch_size: 512                 # Large batches for your RAM
  parallel_jobs: 12               # Use all CPU cores
  cache_embeddings: true
  cache_directory: "cache"
  sample_rate: 1.0               # Process full dataset

#######################
# Spark Configuration (Optimized for your hardware)
#######################
spark:
  executor_memory: "32g"          # Use ~25% of your RAM
  driver_memory: "16g"            # Generous driver memory
  executor_cores: 4               # 3 executors on 12 cores
  default_parallelism: 24         # 2x cores for parallel tasks
  
#######################
# Checkpointing
#######################
checkpoint:
  enabled: true
  interval: 1
  directory: "checkpoints"
  max_checkpoints: 5

#######################
# Logging
#######################
logging:
  level: "INFO"
  log_file: "logs/classification_process.log"
  console_output: true

#######################
# Miscellaneous options
#######################
options:
  seed: 42
  save_intermediate: true
  clean_intermediate_on_success: false
"""
    
    with open("config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ Created config.yaml (optimized for your specs)")

def create_env_template():
    """Create environment template file."""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Set custom cache location
# AI_CACHE_DIR=/path/to/cache

# Optional: Set log level
# LOG_LEVEL=INFO
"""
    
    with open(".env.template", "w") as f:
        f.write(env_content)
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env template (add your OpenAI API key here)")


def main():
    """Main setup function."""
    print_banner()
    
    # Core setup steps
    create_directory_structure()
    create_config_file()
    create_env_template()

if __name__ == "__main__":
    main()