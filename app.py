#!/usr/bin/env python3
"""
Streamlit Configuration App for AI Text Classification System
Easy-to-use web interface for generating configuration files.
"""

import streamlit as st
import yaml
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Text Classification Setup",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 AI Text Classification System")
    st.markdown("**Easy Configuration Interface**")
    
    # Sidebar navigation
    st.sidebar.title("Configuration Sections")
    sections = [
        "📁 Data Files",
        "🏷️ Classification Setup", 
        "⚙️ Advanced Settings"
    ]
    
    selected_section = st.sidebar.radio("Navigate to:", sections)
    
    # Initialize session state for configuration
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()
    
    # Main content based on selection
    if selected_section == "📁 Data Files":
        data_files_section()
    elif selected_section == "🏷️ Classification Setup":
        classification_setup_section()
    elif selected_section == "⚙️ Advanced Settings":
        advanced_settings_section()
    
    # Generate and download configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Configuration")
    
    if st.sidebar.button("Generate Config File", type="primary"):
        config_yaml = generate_config_yaml()
        st.sidebar.download_button(
            label="Download config.yaml",
            data=config_yaml,
            file_name="config.yaml",
            mime="text/yaml",
            help="Download the generated configuration file"
        )
        st.sidebar.success("✅ Configuration generated!")
    
    # Preview current configuration
    with st.sidebar.expander("🔍 Preview Configuration", expanded=False):
        st.code(generate_config_yaml(), language='yaml')

def get_default_config():
    """Return default configuration structure."""
    return {
        'input_file': 'input/your_data_file.dta',
        'output_file': 'output/classified_data.dta',
        'results_dir': 'output',
        'text_columns': ['position_name', 'job_description'],
        'clustering_perspectives': {},
        'preprocessing': {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': False,
            'min_word_length': 2,
            'max_length': 10000
        },
        'feature_extraction': {
            'method': 'embedding',
            'embedding': {
                'model': 'sentence-transformers',
                'sentence_transformers': {
                    'model_name': 'all-MiniLM-L6-v2'
                }
            }
        },
        'ai_classification': {
            'cost_management': {
                'max_cost_per_run': 10.0
            },
            'rate_limiting': {
                'requests_per_minute': 100
            },
            'caching': {
                'enabled': True,
                'cache_directory': 'cache'
            }
        },
        'performance': {
            'batch_size': 50,
            'parallel_jobs': 4,
            'cache_embeddings': True
        },
        'spark': {
            'executor_memory': '4g',
            'driver_memory': '4g',
            'executor_cores': 2
        },
        'logging': {
            'level': 'INFO',
            'console_output': True,
            'log_file': 'logs/classification.log'
        }
    }

def data_files_section():
    """Configure data files and basic settings."""
    st.header("📁 Data Files Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input & Output")
        
        st.session_state.config['input_file'] = st.text_input(
            "Input File Path",
            value=st.session_state.config['input_file'],
            help="Path to your data file (.dta, .csv, .xlsx)"
        )
        
        st.session_state.config['output_file'] = st.text_input(
            "Output File Path", 
            value=st.session_state.config['output_file'],
            help="Where to save the classified data"
        )
        
        st.session_state.config['results_dir'] = st.text_input(
            "Results Directory",
            value=st.session_state.config['results_dir'],
            help="Directory for reports and visualizations"
        )
    
    with col2:
        st.subheader("Text Columns")
        st.markdown("Specify which columns contain the text you want to classify.")
        
        # Text columns input
        text_cols_str = st.text_area(
            "Text Columns (one per line)",
            value='\n'.join(st.session_state.config['text_columns']),
            height=100,
            help="List the column names that contain text data"
        )
        
        if text_cols_str:
            st.session_state.config['text_columns'] = [
                col.strip() for col in text_cols_str.split('\n') if col.strip()
            ]
        
        # Preview
        if st.session_state.config['text_columns']:
            st.info(f"📝 Text columns: {', '.join(st.session_state.config['text_columns'])}")

def classification_setup_section():
    """Configure classification perspectives."""
    st.header("🏷️ Classification Setup")
    
    st.markdown("Create classification perspectives. Each perspective processes your text data in a specific way.")
    
    # Add new perspective
    with st.expander("➕ Add New Perspective", expanded=True):
        add_new_perspective()
    
    # Manage existing perspectives
    if st.session_state.config['clustering_perspectives']:
        st.subheader("📋 Current Perspectives")
        manage_existing_perspectives()
    else:
        st.info("🎯 No perspectives configured yet. Add one above to get started!")

def add_new_perspective():
    """Add a new classification perspective."""
    st.subheader("Create New Perspective")
    
    col1, col2 = st.columns(2)
    
    with col1:
        perspective_name = st.text_input(
            "Perspective Name",
            placeholder="e.g., job_classifier",
            help="Unique name for this classification task"
        )
        
        perspective_type = st.selectbox(
            "Classification Type",
            ["AI Classification (OpenAI)", "Traditional Clustering"],
            help="Choose between AI-powered classification or traditional clustering"
        )
        
        columns_for_perspective = st.multiselect(
            "Columns to Use",
            options=st.session_state.config['text_columns'],
            help="Select which text columns to use for this perspective"
        )
    
    with col2:
        output_column = st.text_input(
            "Output Column Name",
            placeholder="e.g., job_category",
            help="Name of the new column that will contain the classifications"
        )
        
        if perspective_type == "AI Classification (OpenAI)":
            st.markdown("**AI Classification Settings**")
            
            # Categories input
            categories_str = st.text_area(
                "Target Categories (one per line)",
                placeholder="Engineering\nSales\nMarketing\nOther",
                height=100,
                help="List the categories you want to classify into"
            )
            
            # Model selection
            model = st.selectbox(
                "OpenAI Model",
                ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4"],
                help="gpt-4o-mini is recommended for cost-effectiveness"
            )
            
        else:  # Traditional Clustering
            st.markdown("**Clustering Settings**")
            
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["hdbscan", "kmeans", "agglomerative"],
                help="HDBSCAN is recommended for discovering natural groups"
            )
            
            if algorithm == "hdbscan":
                min_cluster_size = st.number_input(
                    "Minimum Cluster Size",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    help="Minimum number of samples in a cluster"
                )
            elif algorithm == "kmeans":
                n_clusters = st.number_input(
                    "Number of Clusters",
                    min_value=2,
                    max_value=50,
                    value=8,
                    help="How many clusters to create"
                )
    
    # Add perspective button
    if st.button("Add Perspective", type="primary"):
        if perspective_name and columns_for_perspective and output_column:
            if perspective_type == "AI Classification (OpenAI)":
                if categories_str:
                    categories = [cat.strip() for cat in categories_str.split('\n') if cat.strip()]
                    add_ai_perspective(perspective_name, columns_for_perspective, output_column, categories, model)
                    st.success(f"✅ AI perspective '{perspective_name}' added!")
                else:
                    st.error("Please specify target categories.")
            else:
                add_clustering_perspective(perspective_name, columns_for_perspective, output_column, algorithm, locals())
                st.success(f"✅ Clustering perspective '{perspective_name}' added!")
        else:
            st.error("Please fill in all required fields.")

def add_ai_perspective(name, columns, output_col, categories, model):
    """Add AI classification perspective to config."""
    st.session_state.config['clustering_perspectives'][name] = {
        'type': 'openai_classification',
        'columns': columns,
        'target_categories': categories,
        'output_column': output_col,
        'llm_config': {
            'provider': 'openai',
            'model': model,
            'temperature': 0.0,
            'max_tokens': 10,
            'api_key_env': 'OPENAI_API_KEY'
        },
        'classification_config': {
            'batch_size': 50,
            'unknown_category': 'Other',
            'prompt_template': '''Classify this text into one category:
Categories: {categories}
Text: {text}
Respond with only the category name.'''
        }
    }

def add_clustering_perspective(name, columns, output_col, algorithm, local_vars):
    """Add clustering perspective to config."""
    perspective = {
        'type': 'clustering',
        'columns': columns,
        'algorithm': algorithm,
        'output_column': output_col,
        'params': {}
    }
    
    if algorithm == 'hdbscan':
        perspective['params']['min_cluster_size'] = local_vars.get('min_cluster_size', 50)
        perspective['params']['min_samples'] = 25
    elif algorithm == 'kmeans':
        perspective['params']['n_clusters'] = local_vars.get('n_clusters', 8)
    
    st.session_state.config['clustering_perspectives'][name] = perspective

def manage_existing_perspectives():
    """Manage existing perspectives."""
    for name, config in list(st.session_state.config['clustering_perspectives'].items()):
        with st.expander(f"📊 {name} ({config.get('type', 'clustering')})", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Type:** {config.get('type', 'clustering').replace('_', ' ').title()}")
                st.write(f"**Columns:** {', '.join(config.get('columns', []))}")
                st.write(f"**Output:** {config.get('output_column', 'N/A')}")
            
            with col2:
                if config.get('type') == 'openai_classification':
                    categories = config.get('target_categories', [])
                    st.write(f"**Categories:** {len(categories)} categories")
                    if categories:
                        st.write(f"• {', '.join(categories[:3])}{'...' if len(categories) > 3 else ''}")
                else:
                    algorithm = config.get('algorithm', 'Unknown')
                    st.write(f"**Algorithm:** {algorithm}")
                    params = config.get('params', {})
                    if params:
                        param_str = ', '.join([f"{k}: {v}" for k, v in params.items()])
                        st.write(f"**Parameters:** {param_str}")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                    del st.session_state.config['clustering_perspectives'][name]
                    st.rerun()

def advanced_settings_section():
    """Configure advanced settings."""
    st.header("⚙️ Advanced Settings")
    
    tabs = st.tabs([
        "🔤 Text Processing", 
        "🧠 AI Settings", 
        "⚡ Performance", 
        "💾 System"
    ])
    
    with tabs[0]:
        text_processing_settings()
    
    with tabs[1]:
        ai_settings()
    
    with tabs[2]:
        performance_settings()
    
    with tabs[3]:
        system_settings()

def text_processing_settings():
    """Configure text preprocessing options."""
    st.subheader("🔤 Text Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.config['preprocessing']['lowercase'] = st.checkbox(
            "Convert to lowercase",
            value=st.session_state.config['preprocessing']['lowercase'],
            help="Convert all text to lowercase for consistency"
        )
        
        st.session_state.config['preprocessing']['remove_punctuation'] = st.checkbox(
            "Remove punctuation",
            value=st.session_state.config['preprocessing']['remove_punctuation']
        )
        
        st.session_state.config['preprocessing']['remove_stopwords'] = st.checkbox(
            "Remove stopwords",
            value=st.session_state.config['preprocessing']['remove_stopwords'],
            help="Remove common words like 'the', 'and', etc."
        )
        
        st.session_state.config['preprocessing']['lemmatize'] = st.checkbox(
            "Apply lemmatization",
            value=st.session_state.config['preprocessing']['lemmatize'],
            help="Convert words to their base form (e.g., 'running' → 'run')"
        )
    
    with col2:
        st.session_state.config['preprocessing']['min_word_length'] = st.number_input(
            "Minimum word length",
            min_value=1,
            max_value=10,
            value=st.session_state.config['preprocessing']['min_word_length'],
            help="Ignore words shorter than this"
        )
        
        st.session_state.config['preprocessing']['max_length'] = st.number_input(
            "Maximum text length",
            min_value=1000,
            max_value=50000,
            value=st.session_state.config['preprocessing']['max_length'],
            help="Truncate texts longer than this"
        )
        
        # Feature extraction method
        st.session_state.config['feature_extraction']['method'] = st.selectbox(
            "Feature extraction method",
            ["embedding", "tfidf", "hybrid"],
            index=["embedding", "tfidf", "hybrid"].index(
                st.session_state.config['feature_extraction']['method']
            ),
            help="How to convert text to numerical features"
        )

def ai_settings():
    """Configure AI-specific settings."""
    st.subheader("🧠 AI Classification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Cost Management**")
        
        st.session_state.config['ai_classification']['cost_management']['max_cost_per_run'] = st.number_input(
            "Maximum cost per run ($)",
            min_value=1.0,
            max_value=100.0,
            value=st.session_state.config['ai_classification']['cost_management']['max_cost_per_run'],
            step=1.0,
            help="Stop processing if cost exceeds this amount"
        )
        
        st.markdown("**🎯 Caching**")
        
        st.session_state.config['ai_classification']['caching']['enabled'] = st.checkbox(
            "Enable caching",
            value=st.session_state.config['ai_classification']['caching']['enabled'],
            help="Cache API responses to avoid repeated calls"
        )
        
        if st.session_state.config['ai_classification']['caching']['enabled']:
            st.session_state.config['ai_classification']['caching']['cache_directory'] = st.text_input(
                "Cache directory",
                value=st.session_state.config['ai_classification']['caching']['cache_directory']
            )
    
    with col2:
        st.markdown("**🚦 Rate Limiting**")
        
        st.session_state.config['ai_classification']['rate_limiting']['requests_per_minute'] = st.number_input(
            "Requests per minute",
            min_value=10,
            max_value=1000,
            value=st.session_state.config['ai_classification']['rate_limiting']['requests_per_minute'],
            help="Maximum API requests per minute"
        )
        
        st.markdown("**📊 Quality Settings**")
        
        enable_unique_processing = st.checkbox(
            "Enable unique value processing",
            value=True,
            help="Process only unique text values to reduce costs (recommended)"
        )
        
        if enable_unique_processing:
            st.info("💡 Unique processing can reduce costs by 50-90% for datasets with repeated text values.")

def performance_settings():
    """Configure performance settings."""
    st.subheader("⚡ Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔄 Processing**")
        
        st.session_state.config['performance']['batch_size'] = st.number_input(
            "Batch size",
            min_value=10,
            max_value=200,
            value=st.session_state.config['performance']['batch_size'],
            help="Number of items to process at once"
        )
        
        st.session_state.config['performance']['parallel_jobs'] = st.number_input(
            "Parallel jobs",
            min_value=1,
            max_value=16,
            value=st.session_state.config['performance']['parallel_jobs'],
            help="Number of parallel processing threads"
        )
        
        st.session_state.config['performance']['cache_embeddings'] = st.checkbox(
            "Cache embeddings",
            value=st.session_state.config['performance']['cache_embeddings'],
            help="Cache text embeddings for faster processing"
        )
    
    with col2:
        st.markdown("**💾 Spark Settings**")
        
        st.session_state.config['spark']['driver_memory'] = st.selectbox(
            "Driver memory",
            ["2g", "4g", "8g", "16g"],
            index=["2g", "4g", "8g", "16g"].index(st.session_state.config['spark']['driver_memory']),
            help="Memory allocation for Spark driver"
        )
        
        st.session_state.config['spark']['executor_memory'] = st.selectbox(
            "Executor memory", 
            ["2g", "4g", "8g", "16g"],
            index=["2g", "4g", "8g", "16g"].index(st.session_state.config['spark']['executor_memory']),
            help="Memory allocation for Spark executors"
        )
        
        st.session_state.config['spark']['executor_cores'] = st.number_input(
            "Executor cores",
            min_value=1,
            max_value=8,
            value=st.session_state.config['spark']['executor_cores'],
            help="Number of CPU cores per executor"
        )

def system_settings():
    """Configure system settings."""
    st.subheader("💾 System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Logging**")
        
        st.session_state.config['logging']['level'] = st.selectbox(
            "Log level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(st.session_state.config['logging']['level']),
            help="Level of detail in logs"
        )
        
        st.session_state.config['logging']['console_output'] = st.checkbox(
            "Console output",
            value=st.session_state.config['logging']['console_output'],
            help="Show logs in console"
        )
        
        st.session_state.config['logging']['log_file'] = st.text_input(
            "Log file path",
            value=st.session_state.config['logging']['log_file'],
            help="Path to save log file"
        )
    
    with col2:
        st.markdown("**🔄 Checkpoints**")
        
        checkpoint_enabled = st.checkbox(
            "Enable checkpoints",
            value=True,
            help="Save progress to resume if interrupted"
        )
        
        if checkpoint_enabled:
            if 'checkpoint' not in st.session_state.config:
                st.session_state.config['checkpoint'] = {
                    'enabled': True,
                    'directory': 'checkpoints',
                    'max_checkpoints': 5
                }
            
            st.session_state.config['checkpoint']['directory'] = st.text_input(
                "Checkpoint directory",
                value=st.session_state.config['checkpoint']['directory']
            )
            
            st.session_state.config['checkpoint']['max_checkpoints'] = st.number_input(
                "Max checkpoints to keep",
                min_value=1,
                max_value=20,
                value=st.session_state.config['checkpoint']['max_checkpoints']
            )

def generate_config_yaml():
    """Generate YAML configuration string."""
    # Clean up the configuration before generating YAML
    config = clean_config(st.session_state.config)
    
    # Add header comment
    yaml_content = f"""# AI Text Classification System Configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# Quick Start:
# 1. Update input_file and text_columns for your data
# 2. Add your OpenAI API key to .env file  
# 3. Run: python main.py --config config.yaml

"""
    
    yaml_content += yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return yaml_content

def clean_config(config):
    """Clean and validate configuration before export."""
    cleaned = config.copy()
    
    # Ensure required fields
    if not cleaned.get('clustering_perspectives'):
        st.warning("⚠️ No classification perspectives configured! Add at least one perspective.")
    
    # Add default options if missing
    if 'options' not in cleaned:
        cleaned['options'] = {
            'seed': 42,
            'save_intermediate': True,
            'clean_intermediate_on_success': False
        }
    
    return cleaned

if __name__ == "__main__":
    main()
