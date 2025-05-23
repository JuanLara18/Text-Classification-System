# Text Classification System

A robust, modular solution for automatically classifying text columns in data files using state-of-the-art natural language processing and machine learning techniques.

## Overview

This project implements a comprehensive text classification system designed to analyze datasets containing text columns and apply various clustering algorithms to identify meaningful categories. It takes translated text data, processes it through multiple "perspectives" (combinations of text columns, features, and algorithms), and generates rich, multi-dimensional classifications.

The system enhances your dataset with new classification columns while providing detailed visualizations and reports to help you understand the discovered categories.

## Key Features

- **Multi-Perspective Classification**: Apply different algorithms with varied parameters for multiple ways of looking at your data
- **Advanced Text Processing**: Clean and normalize text data for optimal results
- **State-of-the-Art Feature Extraction**:
  - Sentence embeddings using pre-trained models
  - TF-IDF vectorization with n-gram support
  - OpenAI embeddings integration (optional)
- **Multiple Clustering Algorithms**:
  - K-Means for spherical clusters
  - Hierarchical/Agglomerative clustering for nested categorization
  - HDBSCAN for density-based clustering that handles outliers
- **Comprehensive Evaluation**:
  - Quality metrics for cluster evaluation
  - Interactive visualizations
  - Detailed HTML, JSON, and CSV reports
- **Intelligent Cluster Labeling**:
  - TF-IDF-based key term extraction
  - OpenAI-powered semantic labels (requires API key)
- **Production-Ready Features**:
  - Checkpointing for fault tolerance
  - Performance monitoring
  - Distributed processing with PySpark

## Installation

### Prerequisites

- Python 3.8 or higher
- Java 8 or higher (for PySpark)
- Sufficient RAM (see Resource Guide below)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JuanLara18/Text-Classification-System.git
   cd text-classification-system
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (for OpenAI features):
   ```bash
   # On Linux/macOS
   export OPENAI_API_KEY=your_api_key_here
   # Or load from file
   export OPENAI_API_KEY=$(cat /path/to/OPENAI_API_KEY.env)
   
   # On Windows
   set OPENAI_API_KEY=your_api_key_here
   ```

5. **Install NLTK resources**:
   ```bash
   python nltk_download.py
   ```

## Resource Guide

Choose appropriate settings based on your available resources:

| Resource Level | RAM           | CPU Cores | Recommended Config Changes                                       |
|----------------|---------------|-----------|------------------------------------------------------------------|
| Low            | 4-8 GB        | 2-4       | Reduce batch_size to 16, executor/driver memory to "2g"          |
| Medium         | 8-16 GB       | 4-8       | Use default batch_size (128), memory "4g"                        |
| High           | 16-32 GB      | 8+        | Increase batch_size to 256, memory "8g" or "16g"                 |
| Production     | 32+ GB        | 16+       | Increase batch_size to 512, memory "16g" or higher               |

For large datasets (>100,000 rows):
- Consider using a sample first to test your configuration
- Enable caching to avoid recomputing embeddings
- Increase memory allocation if possible

## Quick Start

1. **Prepare your configuration file** (copy from the example):
   ```bash
   cp config.yaml my_config.yaml
   ```

2. **Edit the configuration file** to specify your input/output files and desired settings:
   ```yaml
   # Minimal required changes
   input_file: "path/to/your/input.dta"
   output_file: "path/to/desired/output.dta"
   results_dir: "path/to/results"
   
   # Define columns to classify
   text_columns:
     - column_name_1
     - column_name_2
   ```

3. **Run the classification system**:
   ```bash
   python main.py --config my_config.yaml
   ```

4. **For a clean run** (ignore previous checkpoints):
   ```bash
   python main.py --config my_config.yaml --force-recalculate
   ```

5. **Examine the results** in the specified output directory:
   - Check the output Stata file with new classification columns
   - Review the HTML reports for visualizations and insights
   - Explore the JSON and CSV files for more detailed metrics

## Configuration Guide

The system is highly configurable through the YAML configuration file. Here are the key sections:

### Basic Settings

```yaml
# File paths
input_file: "path/to/input.dta"       # Input Stata file
output_file: "path/to/output.dta"     # Output Stata file with classifications
results_dir: "path/to/results"        # Directory for reports and visualizations

# Text columns to classify
text_columns:
  - column_name_1
  - column_name_2
  - column_name_3
```

### Text Preprocessing

Control how text is cleaned before analysis:

```yaml
preprocessing:
  lowercase: true                # Convert all text to lowercase
  remove_punctuation: true       # Remove punctuation marks
  remove_stopwords: true         # Remove common words like "the", "and", etc.
  lemmatize: false               # Word lemmatization (slower but can improve results)
  custom_stopwords: []           # Add domain-specific stopwords if needed
  min_word_length: 2             # Remove very short words
  max_length: 10000              # Truncate long texts to preserve memory
```

### Feature Extraction

Configure how text is converted to numerical features:

```yaml
feature_extraction:
  method: "embedding"            # Options: "tfidf", "embedding", "hybrid"
  embedding:
    model: "sentence-transformers"  # Options: "openai", "sentence-transformers"
    sentence_transformers:
      model_name: "all-MiniLM-L6-v2"  # Smaller, faster model
      # or "paraphrase-multilingual-mpnet-base-v2" for better quality
    dimensionality_reduction:
      method: "pca"              # Options: "umap", "pca", "tsne"
      n_components: 50           # Number of dimensions to reduce to
```

### Clustering Perspectives

Define different ways to categorize your data:

```yaml
clustering_perspectives:
  # Example: Content categories based on specific columns
  content_categories:
    columns:
      - column_name_1
      - column_name_2
    weight: [0.7, 0.3]           # Relative importance of each column
    algorithm: "kmeans"          # Clustering algorithm to use
    params:
      n_clusters: 10             # Number of clusters to create
    output_column: "content_category_id"  # Column name in output file
    evaluate_k_range: [5, 15]    # Optional: Find optimal cluster count
```

Available algorithms:
- `kmeans`: Fast, creates spherical clusters of similar size
- `hdbscan`: Finds natural clusters of varying density, handles outliers
- `agglomerative`: Creates a hierarchy of clusters, good for nested categories

### Cluster Labeling

Configure how the system names discovered clusters:

```yaml
cluster_labeling:
  method: "tfidf"                # Options: "openai", "tfidf"
  openai:                        # Only used if method is "openai"
    model: "gpt-3.5-turbo"       # Model to use
    temperature: 0.3             # Creativity setting (lower is more focused)
    api_key_env: "OPENAI_API_KEY" # Environment variable with API key
  tfidf:
    top_terms: 5                 # Number of terms to include in labels
```

### Performance Settings

Optimize for your available resources:

```yaml
performance:
  batch_size: 128                # Process this many samples at once
  parallel_jobs: -1              # Use all available cores
  cache_embeddings: true         # Save embeddings to avoid recomputation
  cache_directory: "cache"       # Where to store cached data

spark:
  executor_memory: "4g"          # Memory per executor
  driver_memory: "4g"            # Memory for driver
  executor_cores: 2              # Cores per executor
```

## Advanced Usage

### Command Line Options

The system supports several command line arguments to override configuration settings:

```bash
# Basic usage
python main.py --config config.yaml

# Override input/output files
python main.py --config config.yaml --input new_input.dta --output new_output.dta

# Force recalculation (ignore checkpoints)
python main.py --config config.yaml --force-recalculate

# Set log level
python main.py --config config.yaml --log-level debug

# Set random seed
python main.py --config config.yaml --seed 123

# Run only specific perspectives
python main.py --config config.yaml --perspectives content_categories,detailed_topics
```

### Working with Large Datasets

For datasets with more than 100,000 rows:

1. **Enable caching**:
   ```yaml
   performance:
     cache_embeddings: true
     cache_directory: "cache"
   ```

2. **Increase memory allocation**:
   ```yaml
   spark:
     executor_memory: "16g"
     driver_memory: "16g"
   ```

3. **Use batch processing**:
   ```yaml
   performance:
     batch_size: 256
   ```

4. **Consider dimensionality reduction**:
   ```yaml
   embedding:
     dimensionality_reduction:
       method: "pca"
       n_components: 50
   ```

### Using OpenAI for Enhanced Features

To use OpenAI for improved embeddings and cluster labeling:

1. Set your API key:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   # Or load from file
   export OPENAI_API_KEY=$(cat /path/to/OPENAI_API_KEY.env)
   
   # Make sure Java options don't interfere
   unset JAVA_OPTIONS
   ```

2. Update your configuration:
   ```yaml
   feature_extraction:
     method: "embedding"
     embedding:
       model: "openai"
       openai:
         model_name: "text-embedding-ada-002"
   
   cluster_labeling:
     method: "openai"
     openai:
       model: "gpt-3.5-turbo"
       temperature: 0.3
   ```

## Understanding the Results

The system produces several outputs:

1. **Enhanced Dataset**: The original dataset with added columns for each clustering perspective:
   - `{perspective_name}_id`: Numeric cluster ID
   - `{perspective_name}_label`: Human-readable cluster label

2. **HTML Reports**: Interactive reports with visualizations:
   - Cluster embeddings plot: 2D visualization of cluster relationships
   - Silhouette plot: Measures how well-separated clusters are
   - Distribution plot: Shows cluster sizes and proportions
   - Detailed cluster analysis: Key terms and characteristics

3. **Metrics**: Evaluation metrics in JSON and CSV formats:
   - Silhouette score: Measures cohesion and separation (higher is better)
   - Davies-Bouldin score: Measures average similarity between clusters (lower is better)
   - Calinski-Harabasz score: Measures the ratio of between-cluster to within-cluster dispersion (higher is better)

4. **Cross-Perspective Analysis**: For setups with multiple perspectives:
   - Correlation heatmaps showing relationships between different clusterings
   - Combined report analyzing multiple perspectives together

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce `batch_size` in configuration
   - Lower `executor_memory` and `driver_memory` values
   - Enable `dimensionality_reduction` for embeddings

2. **Slow Performance**:
   - Enable `cache_embeddings`
   - Use a simpler embedding model
   - Reduce the number of text columns or perspectives

3. **Poor Clustering Quality**:
   - Try different algorithms (K-Means → HDBSCAN)
   - Adjust the number of clusters
   - Add custom stopwords to remove domain-specific noise

4. **File Not Found Errors**:
   - Check that all paths are correct (absolute vs. relative)
   - Ensure input files exist
   - Verify permissions on directories

### Logging

Enable detailed logging to diagnose issues:

```yaml
logging:
  level: "DEBUG"
  log_file: "classification_process.log"
  console_output: true
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
