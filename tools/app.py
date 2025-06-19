#!/usr/bin/env python3
"""
Enhanced Professional Streamlit Application for Text Classification System
A comprehensive interface for both traditional clustering and AI-powered classification.
"""

import streamlit as st
import pandas as pd
import yaml
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our classification system
try:
    import warnings
    # Suppress torch warnings that don't affect functionality
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", message=".*torch.classes.*")
    
    from main import ClassificationPipeline
    from config import ConfigManager
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import classification system: {e}")
    SYSTEM_AVAILABLE = False
except Exception as e:
    # Handle other potential issues
    st.warning(f"Classification system loaded with warnings: {e}")
    try:
        from main import ClassificationPipeline
        from config import ConfigManager
        SYSTEM_AVAILABLE = True
    except:
        SYSTEM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Text Classification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling (dark mode compatible)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: var(--text-color, #1f77b4);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: var(--text-color, #2c3e50);
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .method-card {
        background-color: var(--background-color, rgba(248, 249, 250, 0.8));
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: var(--text-color, #2c3e50);
    }
    
    .success-box {
        background-color: var(--success-bg, rgba(212, 237, 218, 0.8));
        border: 1px solid #c3e6cb;
        color: var(--success-text, #155724);
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: var(--warning-bg, rgba(255, 243, 205, 0.8));
        border: 1px solid #ffeaa7;
        color: var(--warning-text, #856404);
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: var(--info-bg, rgba(227, 242, 253, 0.8));
        border: 1px solid #bbdefb;
        color: var(--info-text, #0d47a1);
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stTab {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #4fc3f7 !important;
        }
        
        .section-header {
            color: #e0e0e0 !important;
        }
        
        .method-card {
            background-color: rgba(66, 66, 66, 0.8) !important;
            color: #e0e0e0 !important;
        }
        
        .success-box {
            background-color: rgba(76, 175, 80, 0.2) !important;
            color: #a5d6a7 !important;
        }
        
        .warning-box {
            background-color: rgba(255, 152, 0, 0.2) !important;
            color: #ffcc02 !important;
        }
        
        .info-box {
            background-color: rgba(33, 150, 243, 0.2) !important;
            color: #90caf9 !important;
        }
    }
    
    /* Force dark mode styles for Streamlit dark theme */
    [data-theme="dark"] .main-header {
        color: #4fc3f7 !important;
    }
    
    [data-theme="dark"] .section-header {
        color: #e0e0e0 !important;
    }
    
    [data-theme="dark"] .method-card {
        background-color: rgba(66, 66, 66, 0.8) !important;
        color: #e0e0e0 !important;
    }
    
    [data-theme="dark"] .success-box {
        background-color: rgba(76, 175, 80, 0.2) !important;
        color: #a5d6a7 !important;
    }
    
    [data-theme="dark"] .warning-box {
        background-color: rgba(255, 152, 0, 0.2) !important;
        color: #ffcc02 !important;
    }
    
    [data-theme="dark"] .info-box {
        background-color: rgba(33, 150, 243, 0.2) !important;
        color: #90caf9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = {
            'input_file': '',
            'output_file': '',
            'results_dir': '',
            'text_columns': [],
            'preprocessing': {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_stopwords': True,
                'lemmatize': False,
                'custom_stopwords': [],
                'min_word_length': 2,
                'max_length': 10000
            },
            'feature_extraction': {
                'method': 'hybrid',
                'tfidf': {
                    'max_features': 5000,
                    'ngram_range': [1, 2],
                    'min_df': 5
                },
                'embedding': {
                    'model': 'sentence-transformers',
                    'sentence_transformers': {
                        'model_name': 'all-MiniLM-L6-v2'
                    },
                    'dimensionality_reduction': {
                        'method': 'umap',
                        'n_components': 50,
                        'random_state': 42
                    }
                }
            },
            'clustering_perspectives': {},
            'ai_classification': {
                'cost_management': {
                    'max_cost_per_run': 50.0
                },
                'caching': {
                    'enabled': True,
                    'cache_directory': 'ai_cache',
                    'cache_duration_days': 365
                },
                'rate_limiting': {
                    'requests_per_minute': 100,
                    'concurrent_requests': 4
                }
            },
            'evaluation': {
                'metrics': ['silhouette_score', 'davies_bouldin_score'],
                'visualizations': ['embeddings_plot', 'distribution_plot'],
                'output_format': ['html', 'json']
            },
            'spark': {
                'executor_memory': '4g',
                'driver_memory': '4g',
                'executor_cores': 2,
                'default_parallelism': 4
            },
            'checkpoint': {
                'enabled': True,
                'interval': 1,
                'directory': 'checkpoints',
                'max_checkpoints': 5
            },
            'logging': {
                'level': 'INFO',
                'console_output': True,
                'log_file': 'classification_process.log'
            },
            'options': {
                'seed': 42,
                'save_intermediate': True,
                'clean_intermediate_on_success': False
            }
        }
    
    if 'execution_state' not in st.session_state:
        st.session_state.execution_state = {
            'running': False,
            'progress': 0,
            'current_step': '',
            'results': None,
            'error': None,
            'start_time': None,
            'end_time': None
        }

# Helper functions
def help_tooltip(text):
    """Create a help tooltip icon with text"""
    return f" ℹ️" if text else ""

def create_help_expander(title, content):
    """Create an expandable help section"""
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.markdown(content)

def validate_file_path(path, extension=None):
    """Validate file path format (not existence since this is web-based)"""
    if not path:
        return False, "File path is required"
    
    if extension and not path.endswith(extension):
        return False, f"File must have {extension} extension"
    
    return True, ""

def load_sample_data_info(file_path):
    """Load sample information from uploaded file"""
    try:
        if file_path.endswith('.dta'):
            # Use iterator approach for better memory handling
            with pd.read_stata(file_path, iterator=True, convert_categoricals=False) as reader:
                sample_df = reader.read(5)  # Read first 5 rows
        elif file_path.endswith('.csv'):
            sample_df = pd.read_csv(file_path, nrows=5)
        else:
            # Try CSV as fallback
            sample_df = pd.read_csv(file_path, nrows=5)
        
        return sample_df, sample_df.columns.tolist()
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages
        if "FileNotFoundError" in str(type(e)):
            error_msg = "File not found. Please check the file path."
        elif "PermissionError" in str(type(e)):
            error_msg = "Permission denied. Please check file permissions."
        elif "pandas" in error_msg.lower():
            error_msg = f"Unable to read file format. Ensure it's a valid CSV or Stata file. Details: {error_msg}"
        
        return None, error_msg

# Main application pages
def show_introduction_page():
    """Introduction and theory page"""
    st.markdown('<div class="main-header">🔍 Text Classification System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Professional Text Classification System
    
    This advanced system provides comprehensive text analysis capabilities using both traditional machine learning 
    and cutting-edge AI approaches. Choose the method that best fits your data and research objectives.
    """)
    
    # Overview section
    st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 What This System Does</h4>
        <ul>
        <li><strong>Automatic Text Classification:</strong> Categorize large volumes of text data</li>
        <li><strong>Multiple Approaches:</strong> Traditional clustering and AI-powered classification</li>
        <li><strong>Flexible Configuration:</strong> Customize every aspect of the analysis</li>
        <li><strong>Professional Results:</strong> Comprehensive reports and visualizations</li>
        <li><strong>Performance Optimized:</strong> Handle large datasets efficiently</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>📈 Key Benefits</h4>
        <ul>
        <li><strong>Scale:</strong> Process thousands to millions of text records</li>
        <li><strong>Accuracy:</strong> State-of-the-art algorithms and models</li>
        <li><strong>Flexibility:</strong> Multiple classification perspectives</li>
        <li><strong>Cost-Effective:</strong> Intelligent caching and optimization</li>
        <li><strong>Reproducible:</strong> Consistent, documented results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Methods explanation
    st.markdown('<div class="section-header">🔬 Classification Methods Explained</div>', unsafe_allow_html=True)
    
    # Traditional Clustering Methods
    st.markdown("### 🤖 Traditional Machine Learning Clustering")
    
    methods = [
        {
            "name": "K-Means Clustering",
            "icon": "🎯",
            "description": "Partitions data into K clusters by minimizing within-cluster variance",
            "when_to_use": "When you know approximately how many categories you expect",
            "pros": "Fast, interpretable, works well with spherical clusters",
            "cons": "Requires specifying number of clusters, assumes spherical clusters",
            "best_for": "Well-defined categories, balanced cluster sizes"
        },
        {
            "name": "HDBSCAN Clustering", 
            "icon": "🌟",
            "description": "Hierarchical density-based clustering that finds clusters of varying densities",
            "when_to_use": "When you don't know the number of clusters and expect varying densities",
            "pros": "Finds optimal number of clusters, handles noise, variable cluster densities",
            "cons": "Can be sensitive to parameters, may create many small clusters",
            "best_for": "Exploratory analysis, natural groupings, handling outliers"
        },
        {
            "name": "Agglomerative Clustering",
            "icon": "🌳", 
            "description": "Bottom-up hierarchical clustering that merges similar data points",
            "when_to_use": "When you want to understand hierarchical relationships",
            "pros": "Creates hierarchical structure, flexible distance metrics",
            "cons": "Can be computationally expensive, sensitive to outliers",
            "best_for": "Understanding data hierarchy, non-spherical clusters"
        }
    ]
    
    for method in methods:
        st.markdown(f"""
        <div class="method-card">
        <h4>{method['icon']} {method['name']}</h4>
        <p><strong>How it works:</strong> {method['description']}</p>
        <p><strong>When to use:</strong> {method['when_to_use']}</p>
        <p><strong>✅ Pros:</strong> {method['pros']}</p>
        <p><strong>⚠️ Cons:</strong> {method['cons']}</p>
        <p><strong>🎯 Best for:</strong> {method['best_for']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Classification Methods
    st.markdown("### 🧠 AI-Powered Classification")
    
    st.markdown("""
    <div class="method-card">
    <h4>🤖 OpenAI GPT Classification</h4>
    <p><strong>How it works:</strong> Uses advanced language models (GPT-3.5, GPT-4) to understand and classify text based on natural language instructions</p>
    <p><strong>When to use:</strong> When you have specific categories in mind or need human-like understanding</p>
    <p><strong>✅ Pros:</strong> 
    • Extremely accurate and context-aware<br>
    • Works with custom categories<br>
    • Handles complex, nuanced text<br>
    • No training data required
    </p>
    <p><strong>⚠️ Cons:</strong> 
    • Costs money per classification<br>
    • Requires internet connection<br>
    • May be slower for very large datasets
    </p>
    <p><strong>🎯 Best for:</strong> High-quality classification, complex categories, business-critical applications</p>
    <p><strong>💰 Cost Optimization:</strong> The system includes smart caching and unique value processing to minimize API costs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Extraction Methods
    st.markdown('<div class="section-header">🔧 Feature Extraction Methods</div>', unsafe_allow_html=True)
    
    feature_methods = [
        {
            "name": "TF-IDF (Term Frequency-Inverse Document Frequency)",
            "description": "Creates numerical features based on word importance across documents",
            "best_for": "Traditional clustering, interpretable results, keyword-based analysis"
        },
        {
            "name": "Sentence Embeddings", 
            "description": "Uses deep learning models to create semantic vector representations",
            "best_for": "Capturing semantic meaning, similar concepts with different words"
        },
        {
            "name": "Hybrid Approach",
            "description": "Combines both TF-IDF and embeddings for comprehensive representation", 
            "best_for": "Maximum accuracy, leveraging both statistical and semantic features"
        }
    ]
    
    for method in feature_methods:
        st.markdown(f"""
        <div class="info-box">
        <h4>📊 {method['name']}</h4>
        <p>{method['description']}</p>
        <p><strong>Best for:</strong> {method['best_for']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Decision Guide
    st.markdown('<div class="section-header">🧭 How to Choose the Right Method</div>', unsafe_allow_html=True)
    
    decision_guide = """
    ### 🤔 Decision Framework
    
    **Choose AI Classification when:**
    - You have specific categories in mind
    - You need high accuracy and human-like understanding
    - Budget allows for API costs
    - Working with complex, nuanced text
    - Quality is more important than speed
    
    **Choose Traditional Clustering when:**
    - You want to discover natural groupings in data
    - You're doing exploratory analysis
    - You need faster processing of very large datasets
    - You want full control over the algorithm
    - You prefer interpretable, statistical methods
    
    **Use Multiple Perspectives when:**
    - You want to compare different approaches
    - You need both exploratory and targeted analysis
    - You want to validate results across methods
    """
    
    st.markdown(decision_guide)
    
    # Getting Started
    st.markdown('<div class="section-header">🚀 Getting Started</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>📋 Quick Start Checklist</h4>
    <ol>
    <li><strong>Prepare your data:</strong> CSV or Stata (.dta) file with text columns</li>
    <li><strong>Go to Configuration:</strong> Set up your analysis parameters</li>
    <li><strong>Choose methods:</strong> Select clustering and/or AI classification</li>
    <li><strong>Configure perspectives:</strong> Define how you want to analyze your text</li>
    <li><strong>Run analysis:</strong> Execute and monitor progress</li>
    <li><strong>Review results:</strong> Examine classifications and reports</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def show_configuration_page():
    """Main configuration page"""
    st.markdown('<div class="main-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # File Configuration
    st.markdown('<div class="section-header">📁 File Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Data")
        
        # File path input
        input_file = st.text_input(
            "Input data file path",
            value=st.session_state.config.get('input_file', ''),
            placeholder="e.g., C:/data/my_text_data.csv or /home/user/data.dta",
            help="Full path to your CSV or Stata (.dta) file containing text data"
        )
        
        if input_file:
            st.session_state.config['input_file'] = input_file
        
        # Manual text columns input since we can't inspect the file
        st.subheader("Text Columns")
        text_columns_input = st.text_input(
            "Text column names (comma-separated)",
            value=', '.join(st.session_state.config.get('text_columns', [])),
            placeholder="e.g., description, comments, feedback, title",
            help="Enter the names of columns that contain text to classify, separated by commas"
        )
        
        if text_columns_input:
            text_columns = [col.strip() for col in text_columns_input.split(',') if col.strip()]
            st.session_state.config['text_columns'] = text_columns
            
            if text_columns:
                st.success(f"✅ {len(text_columns)} text columns configured: {', '.join(text_columns)}")
        else:
            st.session_state.config['text_columns'] = []
    
    with col2:
        st.subheader("Output Configuration")
        
        output_file = st.text_input(
            "Output file path",
            value=st.session_state.config.get('output_file', ''),
            placeholder="e.g., C:/results/classified_data.dta or /home/user/output.csv",
            help="Where to save the classified data (will be created if doesn't exist)"
        )
        st.session_state.config['output_file'] = output_file
        
        results_dir = st.text_input(
            "Results directory",
            value=st.session_state.config.get('results_dir', ''),
            placeholder="e.g., C:/results/analysis_reports or /home/user/reports",
            help="Directory for reports, visualizations, and analysis results"
        )
        st.session_state.config['results_dir'] = results_dir
    
    # Text Preprocessing
    st.markdown('<div class="section-header">🔤 Text Preprocessing</div>', unsafe_allow_html=True)
    
    create_help_expander(
        "Text Preprocessing Help",
        """
        **Text preprocessing** cleans and standardizes your text data before analysis:
        
        - **Lowercase:** Converts all text to lowercase (recommended for most cases)
        - **Remove Punctuation:** Removes punctuation marks (! . , ? etc.)
        - **Remove Stopwords:** Removes common words like 'the', 'and', 'is'
        - **Lemmatize:** Reduces words to their root form (running → run)
        - **Min Word Length:** Ignores words shorter than this length
        - **Max Length:** Truncates very long texts to this character limit
        """
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.config['preprocessing']['lowercase'] = st.checkbox(
            "Convert to lowercase",
            value=st.session_state.config['preprocessing']['lowercase'],
            help="Convert all text to lowercase for consistency"
        )
        
        st.session_state.config['preprocessing']['remove_punctuation'] = st.checkbox(
            "Remove punctuation",
            value=st.session_state.config['preprocessing']['remove_punctuation'],
            help="Remove punctuation marks from text"
        )
    
    with col2:
        st.session_state.config['preprocessing']['remove_stopwords'] = st.checkbox(
            "Remove stopwords",
            value=st.session_state.config['preprocessing']['remove_stopwords'],
            help="Remove common words like 'the', 'and', 'is'"
        )
        
        st.session_state.config['preprocessing']['lemmatize'] = st.checkbox(
            "Lemmatize words",
            value=st.session_state.config['preprocessing']['lemmatize'],
            help="Reduce words to their root form (slower but more accurate)"
        )
    
    with col3:
        st.session_state.config['preprocessing']['min_word_length'] = st.number_input(
            "Minimum word length",
            min_value=1,
            max_value=10,
            value=st.session_state.config['preprocessing']['min_word_length'],
            help="Ignore words shorter than this"
        )
        
        st.session_state.config['preprocessing']['max_length'] = st.number_input(
            "Maximum text length",
            min_value=100,
            max_value=50000,
            value=st.session_state.config['preprocessing']['max_length'],
            help="Truncate texts longer than this"
        )
    
    # Custom stopwords
    custom_stopwords = st.text_area(
        "Custom stopwords (one per line)",
        value='\n'.join(st.session_state.config['preprocessing'].get('custom_stopwords', [])),
        placeholder="company\nbrand\nspecific\nterms\nto\nignore",
        help="Add your own words to ignore (one word per line)"
    )
    if custom_stopwords:
        st.session_state.config['preprocessing']['custom_stopwords'] = custom_stopwords.strip().split('\n')
    
    # Feature Extraction
    st.markdown('<div class="section-header">🔧 Feature Extraction</div>', unsafe_allow_html=True)
    
    create_help_expander(
        "Feature Extraction Help",
        """
        **Feature extraction** converts text into numerical features that algorithms can process:
        
        - **TF-IDF:** Statistical approach based on word frequency and importance
        - **Embeddings:** Semantic approach using deep learning models
        - **Hybrid:** Combines both approaches for best results
        
        **TF-IDF Settings:**
        - **Max Features:** Maximum number of word features to consider
        - **N-gram Range:** Word sequences to include (1,1 = single words, 1,2 = single words + pairs)
        - **Min DF:** Minimum documents a word must appear in
        
        **Embedding Settings:**
        - **Model:** The pre-trained model to use for creating embeddings
        - **Dimensionality Reduction:** Reduce embedding size for faster processing
        """
    )
    
    feature_method = st.selectbox(
        "Feature extraction method",
        options=['tfidf', 'embedding', 'hybrid'],
        index=['tfidf', 'embedding', 'hybrid'].index(st.session_state.config['feature_extraction']['method']),
        help="Method to convert text to numerical features"
    )
    st.session_state.config['feature_extraction']['method'] = feature_method
    
    if feature_method in ['tfidf', 'hybrid']:
        st.subheader("TF-IDF Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.config['feature_extraction']['tfidf']['max_features'] = st.number_input(
                "Max features",
                min_value=100,
                max_value=50000,
                value=st.session_state.config['feature_extraction']['tfidf']['max_features'],
                help="Maximum number of word features"
            )
        
        with col2:
            ngram_start = st.number_input("N-gram start", min_value=1, max_value=3, value=1)
            ngram_end = st.number_input("N-gram end", min_value=1, max_value=3, value=2)
            st.session_state.config['feature_extraction']['tfidf']['ngram_range'] = [ngram_start, ngram_end]
        
        with col3:
            st.session_state.config['feature_extraction']['tfidf']['min_df'] = st.number_input(
                "Min document frequency",
                min_value=1,
                max_value=100,
                value=st.session_state.config['feature_extraction']['tfidf']['min_df'],
                help="Minimum documents a word must appear in"
            )
    
    if feature_method in ['embedding', 'hybrid']:
        st.subheader("Embedding Configuration")
        
        embedding_model = st.selectbox(
            "Embedding model",
            options=['sentence-transformers', 'openai'],
            index=0 if st.session_state.config['feature_extraction']['embedding']['model'] == 'sentence-transformers' else 1,
            help="Model to use for creating semantic embeddings"
        )
        st.session_state.config['feature_extraction']['embedding']['model'] = embedding_model
        
        if embedding_model == 'sentence-transformers':
            model_name = st.selectbox(
                "Sentence transformer model",
                options=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'multi-qa-MiniLM-L6-cos-v1'],
                help="Pre-trained sentence transformer model"
            )
            st.session_state.config['feature_extraction']['embedding']['sentence_transformers']['model_name'] = model_name
        
        # Dimensionality reduction
        use_dim_reduction = st.checkbox(
            "Use dimensionality reduction",
            value=bool(st.session_state.config['feature_extraction']['embedding']['dimensionality_reduction']),
            help="Reduce embedding dimensions for faster processing"
        )
        
        if use_dim_reduction:
            col1, col2 = st.columns(2)
            with col1:
                dim_method = st.selectbox(
                    "Reduction method",
                    options=['umap', 'pca', 'tsne'],
                    help="Method for reducing dimensions"
                )
                st.session_state.config['feature_extraction']['embedding']['dimensionality_reduction']['method'] = dim_method
            
            with col2:
                n_components = st.number_input(
                    "Number of components",
                    min_value=2,
                    max_value=200,
                    value=st.session_state.config['feature_extraction']['embedding']['dimensionality_reduction']['n_components'],
                    help="Final number of dimensions"
                )
                st.session_state.config['feature_extraction']['embedding']['dimensionality_reduction']['n_components'] = n_components

def show_perspectives_page():
    """Clustering and AI classification perspectives configuration"""
    st.markdown('<div class="main-header">🎯 Classification Perspectives</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Perspectives** define different ways to classify your text data. You can create multiple perspectives 
    to analyze your data from different angles or compare methods.
    """)
    
    # Initialize perspectives if empty
    if 'clustering_perspectives' not in st.session_state.config:
        st.session_state.config['clustering_perspectives'] = {}
    
    # Perspective management
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📋 Current Perspectives")
        
        if st.session_state.config['clustering_perspectives']:
            for name, config in st.session_state.config['clustering_perspectives'].items():
                perspective_type = config.get('type', 'clustering')
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    
                    with col_a:
                        if perspective_type == 'clustering':
                            algorithm = config.get('algorithm', 'unknown')
                            st.write(f"🤖 **{name}** - {algorithm.upper()} clustering")
                        else:
                            categories = config.get('target_categories', [])
                            st.write(f"🧠 **{name}** - AI classification ({len(categories)} categories)")
                    
                    with col_b:
                        if st.button(f"Edit", key=f"edit_{name}"):
                            st.session_state.editing_perspective = name
                    
                    with col_c:
                        if st.button(f"Delete", key=f"delete_{name}"):
                            del st.session_state.config['clustering_perspectives'][name]
                            st.rerun()
        else:
            st.info("No perspectives configured yet. Create your first perspective below.")
    
    with col2:
        st.subheader("➕ Add New")
        
        if st.button("🤖 New Clustering", use_container_width=True):
            st.session_state.creating_perspective = 'clustering'
        
        if st.button("🧠 New AI Classification", use_container_width=True):
            st.session_state.creating_perspective = 'ai_classification'
    
    # Create/Edit perspective forms
    if hasattr(st.session_state, 'creating_perspective'):
        show_perspective_form(st.session_state.creating_perspective)
    
    elif hasattr(st.session_state, 'editing_perspective'):
        perspective_name = st.session_state.editing_perspective
        perspective_config = st.session_state.config['clustering_perspectives'][perspective_name]
        perspective_type = perspective_config.get('type', 'clustering')
        show_perspective_form(perspective_type, perspective_name, perspective_config)

def show_perspective_form(perspective_type, perspective_name=None, existing_config=None):
    """Show form for creating/editing perspectives"""
    
    if perspective_type == 'clustering':
        st.markdown('<div class="section-header">🤖 Clustering Perspective Configuration</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">🧠 AI Classification Perspective Configuration</div>', unsafe_allow_html=True)
    
    with st.form(f"perspective_form_{perspective_type}"):
        # Basic configuration
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Perspective name",
                value=perspective_name or "",
                placeholder="e.g., content_categories, error_types, sentiment_analysis",
                help="Unique name for this perspective"
            )
            
            if st.session_state.config.get('text_columns'):
                columns = st.multiselect(
                    "Text columns to use",
                    options=st.session_state.config['text_columns'],
                    default=existing_config.get('columns', st.session_state.config['text_columns']) if existing_config else st.session_state.config['text_columns'],
                    help="Which text columns to include in this analysis"
                )
            else:
                columns = []
                st.warning("⚠️ Please configure text columns in the Configuration section first")
        
        with col2:
            output_column = st.text_input(
                "Output column name",
                value=existing_config.get('output_column', f"{name}_result") if existing_config else f"{name}_result",
                placeholder="e.g., category_result, classification_output, cluster_id",
                help="Name of the column that will contain the classification results"
            )
        
        if perspective_type == 'clustering':
            # Clustering-specific configuration
            st.subheader("🔧 Clustering Algorithm")
            
            algorithm = st.selectbox(
                "Algorithm",
                options=['kmeans', 'hdbscan', 'agglomerative'],
                index=['kmeans', 'hdbscan', 'agglomerative'].index(existing_config.get('algorithm', 'hdbscan')) if existing_config else 1,
                help="Clustering algorithm to use"
            )
            
            # Algorithm-specific parameters
            params = existing_config.get('params', {}) if existing_config else {}
            
            if algorithm == 'kmeans':
                st.subheader("K-Means Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    n_clusters = st.number_input(
                        "Number of clusters",
                        min_value=2,
                        max_value=50,
                        value=params.get('n_clusters', 8),
                        help="Number of clusters to create"
                    )
                    params['n_clusters'] = n_clusters
                
                with col2:
                    # Option to auto-determine optimal k
                    auto_k = st.checkbox(
                        "Auto-determine optimal K",
                        help="Automatically find the best number of clusters using silhouette score"
                    )
                    
                    if auto_k:
                        k_min = st.number_input("K range min", min_value=2, value=2)
                        k_max = st.number_input("K range max", min_value=3, value=15)
                        params['evaluate_k_range'] = [k_min, k_max]
            
            elif algorithm == 'hdbscan':
                st.subheader("HDBSCAN Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_cluster_size = st.number_input(
                        "Minimum cluster size",
                        min_value=5,
                        max_value=1000,
                        value=params.get('min_cluster_size', 50),
                        help="Minimum number of points required to form a cluster"
                    )
                    params['min_cluster_size'] = min_cluster_size
                
                with col2:
                    min_samples = st.number_input(
                        "Minimum samples",
                        min_value=1,
                        max_value=100,
                        value=params.get('min_samples', 10),
                        help="Minimum number of samples in a neighborhood"
                    )
                    params['min_samples'] = min_samples
                
                max_clusters = st.number_input(
                    "Maximum clusters",
                    min_value=5,
                    max_value=200,
                    value=params.get('max_clusters', 50),
                    help="Maximum number of clusters to prevent over-fragmentation"
                )
                params['max_clusters'] = max_clusters
            
            elif algorithm == 'agglomerative':
                st.subheader("Agglomerative Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    n_clusters = st.number_input(
                        "Number of clusters",
                        min_value=2,
                        max_value=50,
                        value=params.get('n_clusters', 8),
                        help="Number of clusters to create"
                    )
                    params['n_clusters'] = n_clusters
                
                with col2:
                    linkage = st.selectbox(
                        "Linkage method",
                        options=['ward', 'complete', 'average', 'single'],
                        index=['ward', 'complete', 'average', 'single'].index(params.get('linkage', 'ward')),
                        help="Method for calculating distance between clusters"
                    )
                    params['linkage'] = linkage
            
            # Create perspective config
            perspective_config = {
                'type': 'clustering',
                'columns': columns,
                'algorithm': algorithm,
                'params': params,
                'output_column': output_column
            }
        
        else:  # AI Classification
            st.subheader("🧠 AI Classification Configuration")
            
            # Target categories
            st.subheader("📋 Target Categories")
            categories_text = st.text_area(
                "Categories (one per line)",
                value='\n'.join(existing_config.get('target_categories', [])) if existing_config else "",
                placeholder="Positive Feedback\nNegative Feedback\nNeutral Comment\nQuestion\nComplaint\nSuggestion",
                help="Enter each category on a separate line",
                height=150
            )
            
            target_categories = [cat.strip() for cat in categories_text.split('\n') if cat.strip()]
            
            if len(target_categories) < 2:
                st.warning("⚠️ Please provide at least 2 categories")
            
            # LLM Configuration
            st.subheader("🤖 Language Model Settings")
            
            llm_config = existing_config.get('llm_config', {}) if existing_config else {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                model = st.selectbox(
                    "Model",
                    options=['gpt-4o-mini', 'gpt-3.5-turbo-0125', 'gpt-4o', 'gpt-4-turbo'],
                    index=0,  # Default to gpt-4o-mini (fastest/cheapest)
                    help="OpenAI model to use for classification"
                )
                llm_config['model'] = model
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=llm_config.get('temperature', 0.0),
                    step=0.1,
                    help="Higher values make output more random (0.0 = deterministic)"
                )
                llm_config['temperature'] = temperature
            
            with col2:
                max_tokens = st.number_input(
                    "Max tokens",
                    min_value=5,
                    max_value=100,
                    value=llm_config.get('max_tokens', 20),
                    help="Maximum tokens in the response"
                )
                llm_config['max_tokens'] = max_tokens
                
                api_key_env = st.text_input(
                    "API key environment variable",
                    value=llm_config.get('api_key_env', 'OPENAI_API_KEY'),
                    placeholder="OPENAI_API_KEY",
                    help="Environment variable containing your OpenAI API key"
                )
                llm_config['api_key_env'] = api_key_env
            
            # Classification Configuration
            st.subheader("⚙️ Classification Settings")
            
            classification_config = existing_config.get('classification_config', {}) if existing_config else {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.number_input(
                    "Batch size",
                    min_value=1,
                    max_value=200,
                    value=classification_config.get('batch_size', 50),
                    help="Number of texts to process in each batch"
                )
                classification_config['batch_size'] = batch_size
            
            with col2:
                unknown_category = st.text_input(
                    "Unknown category name",
                    value=classification_config.get('unknown_category', 'Other/Unknown'),
                    placeholder="Other/Unknown",
                    help="Category name for texts that don't fit other categories"
                )
                classification_config['unknown_category'] = unknown_category
            
            # Custom prompt template
            st.subheader("📝 Custom Prompt Template")
            
            default_prompt = """Classify this text into ONE category from the list.

Categories:
{categories}

Text: "{text}"

Answer with the category name ONLY."""
            
            prompt_template = st.text_area(
                "Prompt template",
                value=classification_config.get('prompt_template', default_prompt),
                help="Custom prompt template. Use {categories} and {text} as placeholders.",
                height=150
            )
            classification_config['prompt_template'] = prompt_template
            
            # Create perspective config
            perspective_config = {
                'type': 'openai_classification',
                'columns': columns,
                'target_categories': target_categories,
                'output_column': output_column,
                'llm_config': llm_config,
                'classification_config': classification_config
            }
        
        # Form submission
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.form_submit_button("💾 Save Perspective", use_container_width=True):
                if name and columns and output_column:
                    st.session_state.config['clustering_perspectives'][name] = perspective_config
                    
                    # Clear form state
                    if hasattr(st.session_state, 'creating_perspective'):
                        delattr(st.session_state, 'creating_perspective')
                    if hasattr(st.session_state, 'editing_perspective'):
                        delattr(st.session_state, 'editing_perspective')
                    
                    st.success(f"✅ Perspective '{name}' saved successfully!")
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields")
        
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                # Clear form state
                if hasattr(st.session_state, 'creating_perspective'):
                    delattr(st.session_state, 'creating_perspective')
                if hasattr(st.session_state, 'editing_perspective'):
                    delattr(st.session_state, 'editing_perspective')
                st.rerun()

def show_advanced_settings_page():
    """Advanced settings and performance configuration"""
    st.markdown('<div class="main-header">⚙️ Advanced Settings</div>', unsafe_allow_html=True)
    
    # AI Classification Settings
    if any(p.get('type') == 'openai_classification' for p in st.session_state.config.get('clustering_perspectives', {}).values()):
        st.markdown('<div class="section-header">🧠 AI Classification Settings</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Cost Management")
            
            max_cost = st.number_input(
                "Maximum cost per run ($)",
                min_value=1.0,
                max_value=1000.0,
                value=st.session_state.config['ai_classification']['cost_management']['max_cost_per_run'],
                step=5.0,
                help="Maximum amount to spend on AI classification in a single run"
            )
            st.session_state.config['ai_classification']['cost_management']['max_cost_per_run'] = max_cost
            
            st.subheader("🚀 Rate Limiting")
            
            requests_per_minute = st.number_input(
                "Requests per minute",
                min_value=1,
                max_value=1000,
                value=st.session_state.config['ai_classification']['rate_limiting']['requests_per_minute'],
                help="Maximum API requests per minute"
            )
            st.session_state.config['ai_classification']['rate_limiting']['requests_per_minute'] = requests_per_minute
            
            concurrent_requests = st.number_input(
                "Concurrent requests",
                min_value=1,
                max_value=20,
                value=st.session_state.config['ai_classification']['rate_limiting']['concurrent_requests'],
                help="Number of simultaneous API requests"
            )
            st.session_state.config['ai_classification']['rate_limiting']['concurrent_requests'] = concurrent_requests
        
        with col2:
            st.subheader("💾 Caching")
            
            cache_enabled = st.checkbox(
                "Enable caching",
                value=st.session_state.config['ai_classification']['caching']['enabled'],
                help="Cache API responses to avoid repeated costs"
            )
            st.session_state.config['ai_classification']['caching']['enabled'] = cache_enabled
            
            if cache_enabled:
                cache_directory = st.text_input(
                    "Cache directory",
                    value=st.session_state.config['ai_classification']['caching']['cache_directory'],
                    placeholder="ai_cache",
                    help="Directory to store cached responses"
                )
                st.session_state.config['ai_classification']['caching']['cache_directory'] = cache_directory
                
                cache_duration = st.number_input(
                    "Cache duration (days)",
                    min_value=1,
                    max_value=3650,
                    value=st.session_state.config['ai_classification']['caching']['cache_duration_days'],
                    help="How long to keep cached responses"
                )
                st.session_state.config['ai_classification']['caching']['cache_duration_days'] = cache_duration
    
    # Spark Configuration
    st.markdown('<div class="section-header">⚡ Performance Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Spark Configuration")
        
        executor_memory = st.selectbox(
            "Executor memory",
            options=['2g', '4g', '8g', '16g', '32g'],
            index=['2g', '4g', '8g', '16g', '32g'].index(st.session_state.config['spark']['executor_memory']),
            help="Memory allocated to each Spark executor"
        )
        st.session_state.config['spark']['executor_memory'] = executor_memory
        
        driver_memory = st.selectbox(
            "Driver memory", 
            options=['2g', '4g', '8g', '16g', '32g'],
            index=['2g', '4g', '8g', '16g', '32g'].index(st.session_state.config['spark']['driver_memory']),
            help="Memory allocated to Spark driver"
        )
        st.session_state.config['spark']['driver_memory'] = driver_memory
        
        executor_cores = st.number_input(
            "Executor cores",
            min_value=1,
            max_value=16,
            value=st.session_state.config['spark']['executor_cores'],
            help="Number of CPU cores per executor"
        )
        st.session_state.config['spark']['executor_cores'] = executor_cores
    
    with col2:
        st.subheader("💾 Checkpointing")
        
        checkpoint_enabled = st.checkbox(
            "Enable checkpointing",
            value=st.session_state.config['checkpoint']['enabled'],
            help="Save intermediate results to resume if interrupted"
        )
        st.session_state.config['checkpoint']['enabled'] = checkpoint_enabled
        
        if checkpoint_enabled:
            checkpoint_dir = st.text_input(
                "Checkpoint directory",
                value=st.session_state.config['checkpoint']['directory'],
                placeholder="checkpoints",
                help="Directory to store checkpoint files"
            )
            st.session_state.config['checkpoint']['directory'] = checkpoint_dir
            
            max_checkpoints = st.number_input(
                "Max checkpoints to keep",
                min_value=1,
                max_value=20,
                value=st.session_state.config['checkpoint']['max_checkpoints'],
                help="Maximum number of checkpoint files to retain"
            )
            st.session_state.config['checkpoint']['max_checkpoints'] = max_checkpoints
    
    # Evaluation Settings
    st.markdown('<div class="section-header">📊 Evaluation & Output</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Metrics")
        
        available_metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
        selected_metrics = st.multiselect(
            "Evaluation metrics",
            options=available_metrics,
            default=st.session_state.config['evaluation']['metrics'],
            help="Metrics to calculate for clustering evaluation"
        )
        st.session_state.config['evaluation']['metrics'] = selected_metrics
        
        available_viz = ['embeddings_plot', 'silhouette_plot', 'distribution_plot']
        selected_viz = st.multiselect(
            "Visualizations",
            options=available_viz,
            default=st.session_state.config['evaluation']['visualizations'],
            help="Types of visualizations to generate"
        )
        st.session_state.config['evaluation']['visualizations'] = selected_viz
    
    with col2:
        st.subheader("📄 Output Formats")
        
        available_formats = ['html', 'json', 'csv']
        selected_formats = st.multiselect(
            "Report formats",
            options=available_formats,
            default=st.session_state.config['evaluation']['output_format'],
            help="Output formats for evaluation reports"
        )
        st.session_state.config['evaluation']['output_format'] = selected_formats
    
    # Logging Settings
    st.markdown('<div class="section-header">📝 Logging</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox(
            "Log level",
            options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(st.session_state.config['logging']['level']),
            help="Minimum level of log messages to display"
        )
        st.session_state.config['logging']['level'] = log_level
        
        console_output = st.checkbox(
            "Console output",
            value=st.session_state.config['logging']['console_output'],
            help="Display log messages in console"
        )
        st.session_state.config['logging']['console_output'] = console_output
    
    with col2:
        log_file = st.text_input(
            "Log file path",
            value=st.session_state.config['logging']['log_file'],
            placeholder="classification_process.log",
            help="File to save log messages (leave empty to disable)"
        )
        st.session_state.config['logging']['log_file'] = log_file
    
    # Other Options
    st.markdown('<div class="section-header">🎛️ Other Options</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=999999,
            value=st.session_state.config['options']['seed'],
            help="Seed for reproducible results"
        )
        st.session_state.config['options']['seed'] = seed
    
    with col2:
        save_intermediate = st.checkbox(
            "Save intermediate results",
            value=st.session_state.config['options']['save_intermediate'],
            help="Save intermediate processing results"
        )
        st.session_state.config['options']['save_intermediate'] = save_intermediate
    
    with col3:
        clean_intermediate = st.checkbox(
            "Clean intermediate on success",
            value=st.session_state.config['options']['clean_intermediate_on_success'],
            help="Delete intermediate files after successful completion"
        )
        st.session_state.config['options']['clean_intermediate_on_success'] = clean_intermediate

def show_execution_page():
    """Execution and monitoring page"""
    st.markdown('<div class="main-header">🚀 Execute Classification</div>', unsafe_allow_html=True)
    
    # Configuration validation
    st.markdown('<div class="section-header">✅ Configuration Validation</div>', unsafe_allow_html=True)
    
    validation_results = validate_configuration()
    
    if validation_results['valid']:
        st.markdown("""
        <div class="success-box">
        <h4>✅ Configuration Valid</h4>
        <p>All required settings are configured correctly. Ready to execute!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration Summary
        show_configuration_summary()
        
        # Execution Controls
        st.markdown('<div class="section-header">🎮 Execution Controls</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Export configuration
            if st.button("📥 Export Configuration", use_container_width=True):
                config_yaml = yaml.dump(st.session_state.config, default_flow_style=False)
                st.download_button(
                    label="💾 Download Config YAML",
                    data=config_yaml,
                    file_name=f"classification_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                    mime="text/yaml"
                )
        
        with col2:
            # Test run
            test_run = st.checkbox(
                "🧪 Test run",
                help="Process only a small sample for testing"
            )
        
        with col3:
            # Force recalculate
            force_recalc = st.checkbox(
                "🔄 Force recalculate",
                help="Ignore existing checkpoints and recalculate everything"
            )
        
        # Main execution button
        if not st.session_state.execution_state['running']:
            if st.button("🚀 START CLASSIFICATION", type="primary", use_container_width=True):
                if SYSTEM_AVAILABLE:
                    start_classification(test_run, force_recalc)
                else:
                    st.error("❌ Classification system not available. Please check installation.")
        else:
            st.button("⏸️ CLASSIFICATION RUNNING...", disabled=True, use_container_width=True)
            
            # Show progress
            show_execution_progress()
    
    else:
        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Configuration Issues</h4>
        <p>Please fix the following issues before executing:</p>
        <ul>
        {"".join(f"<li>{issue}</li>" for issue in validation_results['issues'])}
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Show results if available
    if st.session_state.execution_state['results']:
        show_execution_results()

def validate_configuration():
    """Validate the current configuration"""
    issues = []
    config = st.session_state.config
    
    # Basic configuration validation (not file existence since this is web-based)
    if not config.get('input_file'):
        issues.append("Input file path is required")
    
    if not config.get('output_file'):
        issues.append("Output file path is required")
    
    if not config.get('text_columns'):
        issues.append("At least one text column must be specified")
    
    # Perspectives validation
    perspectives = config.get('clustering_perspectives', {})
    if not perspectives:
        issues.append("At least one classification perspective must be configured")
    
    # AI classification specific validation
    ai_perspectives = [p for p in perspectives.values() if p.get('type') == 'openai_classification']
    if ai_perspectives:
        # Note: In web environment, we can't check environment variables
        # This is just a reminder for the user
        st.info("💡 Remember to set your OPENAI_API_KEY environment variable when running the classification")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }

def show_configuration_summary():
    """Show a summary of the current configuration"""
    config = st.session_state.config
    
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Files & Data")
            st.write(f"**Input:** {config.get('input_file', 'Not set')}")
            st.write(f"**Output:** {config.get('output_file', 'Not set')}")
            st.write(f"**Text columns:** {', '.join(config.get('text_columns', []))}")
            
            st.subheader("🔧 Processing")
            st.write(f"**Feature method:** {config['feature_extraction']['method']}")
            preprocessing = config['preprocessing']
            enabled_preprocessing = [k for k, v in preprocessing.items() if v and k not in ['custom_stopwords', 'min_word_length', 'max_length']]
            st.write(f"**Preprocessing:** {', '.join(enabled_preprocessing) if enabled_preprocessing else 'None'}")
        
        with col2:
            st.subheader("🎯 Perspectives")
            perspectives = config.get('clustering_perspectives', {})
            
            clustering_count = sum(1 for p in perspectives.values() if p.get('type', 'clustering') == 'clustering')
            ai_count = sum(1 for p in perspectives.values() if p.get('type') == 'openai_classification')
            
            st.write(f"**Traditional clustering:** {clustering_count}")
            st.write(f"**AI classification:** {ai_count}")
            st.write(f"**Total perspectives:** {len(perspectives)}")
            
            if ai_count > 0:
                total_categories = sum(len(p.get('target_categories', [])) for p in perspectives.values() if p.get('type') == 'openai_classification')
                st.write(f"**Total AI categories:** {total_categories}")

def start_classification(test_run=False, force_recalc=False):
    """Start the classification process"""
    st.session_state.execution_state.update({
        'running': True,
        'progress': 0,
        'current_step': 'Initializing...',
        'results': None,
        'error': None,
        'start_time': datetime.now()
    })
    
    # Create temporary config file
    config_data = st.session_state.config.copy()
    
    # Add test run configuration
    if test_run:
        config_data['testing'] = {
            'enabled': True,
            'sample_size': 100,
            'sample_method': 'random'
        }
    
    # Save config to temporary file
    temp_config_path = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # Start classification in separate thread
    def run_classification():
        try:
            # Create and run pipeline
            pipeline = ClassificationPipeline(temp_config_path)
            success = pipeline.run()
            
            if success:
                st.session_state.execution_state.update({
                    'running': False,
                    'progress': 100,
                    'current_step': 'Completed successfully!',
                    'results': {
                        'success': True,
                        'output_file': config_data['output_file'],
                        'results_dir': config_data.get('results_dir', ''),
                        'execution_time': (datetime.now() - st.session_state.execution_state['start_time']).total_seconds()
                    },
                    'end_time': datetime.now()
                })
            else:
                st.session_state.execution_state.update({
                    'running': False,
                    'error': 'Classification failed. Check logs for details.',
                    'end_time': datetime.now()
                })
        
        except Exception as e:
            st.session_state.execution_state.update({
                'running': False,
                'error': str(e),
                'end_time': datetime.now()
            })
        
        finally:
            # Clean up temporary config file
            try:
                os.remove(temp_config_path)
            except:
                pass
    
    # Start in thread (Note: in production, use proper async handling)
    thread = threading.Thread(target=run_classification)
    thread.daemon = True
    thread.start()

def show_execution_progress():
    """Show execution progress and status"""
    execution_state = st.session_state.execution_state
    
    # Progress bar (simulated - in real implementation, would connect to actual progress)
    progress_bar = st.progress(execution_state['progress'] / 100)
    st.write(f"**Current step:** {execution_state['current_step']}")
    
    if execution_state['start_time']:
        elapsed = datetime.now() - execution_state['start_time']
        st.write(f"**Elapsed time:** {str(elapsed).split('.')[0]}")
    
    # Real-time log display (placeholder)
    with st.expander("📜 Execution Logs", expanded=True):
        st.text_area(
            "Live logs",
            value="Classification process started...\nLoading data...\nProcessing perspectives...",
            height=200,
            disabled=True
        )
    
    # Auto-refresh
    time.sleep(2)
    st.rerun()

def show_execution_results():
    """Show execution results and downloads"""
    results = st.session_state.execution_state['results']
    
    st.markdown('<div class="section-header">🎉 Results</div>', unsafe_allow_html=True)
    
    if results.get('success'):
        st.markdown("""
        <div class="success-box">
        <h4>🎉 Classification Completed Successfully!</h4>
        <p>Your text data has been classified and results are ready for download.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Results metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3>⏱️ Execution Time</h3>
            <h2>{:.1f}s</h2>
            </div>
            """.format(results.get('execution_time', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3>📊 Perspectives</h3>
            <h2>{}</h2>
            </div>
            """.format(len(st.session_state.config.get('clustering_perspectives', {}))), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h3>✅ Status</h3>
            <h2>Success</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Download options
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(results['output_file']):
                with open(results['output_file'], 'rb') as f:
                    st.download_button(
                        label="📊 Download Classified Data",
                        data=f.read(),
                        file_name=os.path.basename(results['output_file']),
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        
        with col2:
            results_dir = results.get('results_dir', '')
            if results_dir and os.path.exists(results_dir):
                st.write(f"📁 **Results directory:** {results_dir}")
                st.write("Find detailed reports and visualizations in the results directory.")
    
    else:
        error_msg = st.session_state.execution_state.get('error', 'Unknown error')
        st.markdown(f"""
        <div class="warning-box">
        <h4>❌ Classification Failed</h4>
        <p><strong>Error:</strong> {error_msg}</p>
        <p>Please check your configuration and try again.</p>
        </div>
        """, unsafe_allow_html=True)

def show_results_page():
    """Results analysis and visualization page"""
    st.markdown('<div class="main-header">📊 Results Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.execution_state.get('results'):
        st.info("🔍 No results available yet. Please run a classification first.")
        return
    
    results = st.session_state.execution_state['results']
    
    if not results.get('success'):
        st.error("❌ Last execution failed. Please check the execution page for details.")
        return
    
    # Load and display results
    output_file = results['output_file']
    
    if os.path.exists(output_file):
        try:
            # Load results data
            if output_file.endswith('.dta'):
                df = pd.read_stata(output_file, convert_categoricals=False)
            else:
                df = pd.read_csv(output_file)
            
            st.subheader("📈 Classification Results Summary")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Records", f"{len(df):,}")
            
            # Find classification columns
            text_columns = st.session_state.config.get('text_columns', [])
            classification_columns = [col for col in df.columns if col not in text_columns and '_preprocessed' not in col]
            
            with col2:
                st.metric("🎯 Classification Columns", len(classification_columns))
            
            with col3:
                if classification_columns:
                    # Show unique categories in first classification column
                    first_class_col = classification_columns[0]
                    unique_categories = df[first_class_col].nunique()
                    st.metric("📋 Unique Categories", unique_categories)
            
            with col4:
                completion_rate = df[classification_columns].notna().any(axis=1).mean() * 100
                st.metric("✅ Completion Rate", f"{completion_rate:.1f}%")
            
            # Detailed analysis for each perspective
            st.subheader("🔍 Perspective Analysis")
            
            perspectives = st.session_state.config.get('clustering_perspectives', {})
            
            for perspective_name, perspective_config in perspectives.items():
                output_column = perspective_config.get('output_column', f"{perspective_name}_result")
                
                if output_column in df.columns:
                    with st.expander(f"📊 {perspective_name} Analysis", expanded=True):
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Distribution chart
                            value_counts = df[output_column].value_counts().head(20)
                            
                            fig = px.bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                title=f"Distribution - {perspective_name}",
                                labels={'x': 'Category', 'y': 'Count'}
                            )
                            fig.update_layout(
                                xaxis_tickangle=-45,
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Statistics
                            st.markdown("**📈 Statistics**")
                            
                            total_classified = df[output_column].notna().sum()
                            st.write(f"**Classified:** {total_classified:,} ({total_classified/len(df)*100:.1f}%)")
                            
                            unique_categories = df[output_column].nunique()
                            st.write(f"**Categories:** {unique_categories}")
                            
                            if perspective_config.get('type') == 'openai_classification':
                                target_categories = perspective_config.get('target_categories', [])
                                st.write(f"**Target categories:** {len(target_categories)}")
                                
                                # Show category coverage
                                used_categories = set(df[output_column].dropna().unique())
                                target_set = set(target_categories)
                                coverage = len(used_categories & target_set) / len(target_set) * 100
                                st.write(f"**Coverage:** {coverage:.1f}%")
            
            # Data preview
            st.subheader("👀 Data Preview")
            
            # Show columns to display
            display_columns = st.multiselect(
                "Select columns to display",
                options=df.columns.tolist(),
                default=text_columns + classification_columns[:3]
            )
            
            if display_columns:
                sample_size = st.slider("Sample size", 10, min(1000, len(df)), 50)
                sample_df = df[display_columns].sample(n=sample_size) if len(df) > sample_size else df[display_columns]
                st.dataframe(sample_df, use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download filtered data
                if st.button("📊 Download Sample Data", use_container_width=True):
                    if display_columns:
                        csv_data = sample_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv_data,
                            file_name=f"classification_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                # Download full results
                if st.button("📁 Download Full Results", use_container_width=True):
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label="💾 Download Full Dataset",
                            data=f.read(),
                            file_name=os.path.basename(output_file),
                            mime="application/octet-stream"
                        )
            
            with col3:
                # Generate summary report
                if st.button("📋 Generate Summary Report", use_container_width=True):
                    summary_report = generate_summary_report(df, perspectives)
                    st.download_button(
                        label="💾 Download Report",
                        data=summary_report,
                        file_name=f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"❌ Error loading results: {str(e)}")
    else:
        st.error(f"❌ Results file not found: {output_file}")

def generate_summary_report(df, perspectives):
    """Generate a text summary report"""
    report_lines = [
        "CLASSIFICATION RESULTS SUMMARY REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATASET OVERVIEW:",
        f"- Total records: {len(df):,}",
        f"- Total columns: {len(df.columns)}",
        "",
        "CLASSIFICATION PERSPECTIVES:",
    ]
    
    for perspective_name, perspective_config in perspectives.items():
        output_column = perspective_config.get('output_column', f"{perspective_name}_result")
        
        if output_column in df.columns:
            total_classified = df[output_column].notna().sum()
            unique_categories = df[output_column].nunique()
            completion_rate = total_classified / len(df) * 100
            
            report_lines.extend([
                f"\n{perspective_name}:",
                f"- Type: {perspective_config.get('type', 'clustering')}",
                f"- Classified records: {total_classified:,} ({completion_rate:.1f}%)",
                f"- Unique categories: {unique_categories}",
            ])
            
            if perspective_config.get('type') == 'openai_classification':
                target_categories = perspective_config.get('target_categories', [])
                report_lines.append(f"- Target categories: {len(target_categories)}")
            
            # Top categories
            top_categories = df[output_column].value_counts().head(5)
            report_lines.append("- Top categories:")
            for category, count in top_categories.items():
                pct = count / total_classified * 100
                report_lines.append(f"  • {category}: {count:,} ({pct:.1f}%)")
    
    return "\n".join(report_lines)

# Main application
def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.markdown("# 🔍 Navigation")
    
    pages = {
        "🏠 Introduction": show_introduction_page,
        "⚙️ Configuration": show_configuration_page,
        "🎯 Perspectives": show_perspectives_page,
        "🔧 Advanced Settings": show_advanced_settings_page,
        "🚀 Execute": show_execution_page,
        "📊 Results": show_results_page
    }
    
    selected_page = st.sidebar.radio("Go to:", list(pages.keys()))
    
    # Quick status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📋 Quick Status")
    
    # Configuration status
    validation = validate_configuration()
    if validation['valid']:
        st.sidebar.markdown("✅ **Configuration:** Valid")
    else:
        st.sidebar.markdown(f"⚠️ **Configuration:** {len(validation['issues'])} issues")
    
    # Perspectives status
    perspectives_count = len(st.session_state.config.get('clustering_perspectives', {}))
    st.sidebar.markdown(f"🎯 **Perspectives:** {perspectives_count}")
    
    # Execution status
    if st.session_state.execution_state['running']:
        st.sidebar.markdown("🔄 **Status:** Running")
    elif st.session_state.execution_state['results']:
        if st.session_state.execution_state['results'].get('success'):
            st.sidebar.markdown("✅ **Status:** Completed")
        else:
            st.sidebar.markdown("❌ **Status:** Failed")
    else:
        st.sidebar.markdown("⏸️ **Status:** Ready")
    
    # System information
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ System Info")
    st.sidebar.markdown(f"**System Available:** {'✅ Yes' if SYSTEM_AVAILABLE else '❌ No'}")
    
    if not SYSTEM_AVAILABLE:
        st.sidebar.error("⚠️ Classification system not available")
    
    # Show selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()