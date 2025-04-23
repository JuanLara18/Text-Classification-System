# Text Classification System

A robust, scalable, and modular system for automatically classifying text columns in data files using various clustering algorithms and advanced natural language processing techniques.

## Overview

This project implements a sophisticated text classification system that processes datasets containing text columns and applies machine learning algorithms to create meaningful, multi-perspective classifications. It builds upon an existing translation infrastructure and follows a similar architectural pattern.

The system analyzes text data using state-of-the-art embedding techniques and multiple clustering algorithms to generate insights about the inherent structure and categories within your text data. By applying different "perspectives" (combinations of text columns, features, and algorithms), it creates a rich, multi-dimensional understanding of your content.

## Key Features

- **Multi-Perspective Classification**: Apply different algorithms with varied parameters to get multiple classification views of the same dataset
- **Advanced Text Processing**: Comprehensive preprocessing pipeline for cleaning and normalizing text data
- **Sophisticated Feature Extraction**:
  - OpenAI embeddings (text-embedding-ada-002)
  - Sentence Transformers for local embedding generation
  - TF-IDF vectorization with n-gram support
  - Hybrid approaches combining multiple methods
- **Multiple Clustering Algorithms**:
  - K-Means for spherical clusters
  - Hierarchical/Agglomerative clustering for dendrogram-based solutions
  - HDBSCAN for density-based clustering that handles noise points
- **Distributed Processing**: PySpark integration for handling large datasets efficiently
- **Comprehensive Evaluation**:
  - Silhouette score, Davies-Bouldin index, Calinski-Harabasz index
  - Interactive visualizations of cluster distributions and boundaries
  - Detailed HTML, JSON, and CSV reports
- **Automatic Cluster Labeling**:
  - TF-IDF-based extraction of key terms
  - OpenAI-powered semantic labels (requires API key)
- **Robust System Design**:
  - Checkpointing for fault tolerance and resume capability
  - Detailed logging and performance monitoring
  - Modular architecture for easy extension
  - Comprehensive configuration system

## Installation

### Prerequisites

- Python 3.8 or higher
- Java 8 or higher (for PySpark)
- Sufficient RAM for processing large datasets (8GB minimum recommended)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/text-classification-system.git
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

4. **Set up environment variables** (Optional, for OpenAI usage):
   ```bash
   # On Linux/macOS
   export OPENAI_API_KEY=your_api_key_here
   
   # On Windows
   set OPENAI_API_KEY=your_api_key_here
   ```

5. **Verify installation**:
   ```bash
   python -c "import pyspark, sklearn, hdbscan, umap, sentence_transformers; print('Installation successful!')"
   ```

## Quick Start Guide

Here's how to quickly get started with the classification system:

1. **Prepare your data**:
   - The system expects a Stata (.dta) file format
   - Ensure text columns you want to classify are present in the data
   - Make sure the file is accessible at your specified path

2. **Create a basic configuration file** (save as `my_config.yaml`):
   ```yaml
   # Basic configuration
   input_file: "path/to/your/input.dta"
   output_file: "path/to/your/output.dta"
   results_dir: "path/to/results"
   
   # Text columns to classify
   text_columns:
     - title_column
     - description_column
   
   # A simple clustering perspective
   clustering_perspectives:
     content_categories:
       columns:
         - title_column
         - description_column
       algorithm: "kmeans"
       params:
         n_clusters: 10
       output_column: "content_category_id"
   ```

3. **Run the classification**:
   ```bash
   python main.py --config my_config.yaml
   ```

4. **Examine the results**:
   - Check the output file for new classification columns
   - Review the visualizations and reports in the results directory

## Detailed Usage Guide

### Data Preparation

The system works with Stata (.dta) files. Ensure your text columns are properly encoded and contain meaningful content. Missing values are handled automatically.

If your data is in another format, convert it to Stata format first using pandas:

```python
import pandas as pd

# Read your data
df = pd.read_csv("your_data.csv")

# Save as Stata
df.to_stata("your_data.dta")
```

### Configuration File

The configuration file is the heart of the system. It allows you to customize every aspect of the classification process.

#### Essential Configuration Sections:

1. **File Paths**:
   ```yaml
   input_file: "path/to/your/input.dta"
   output_file: "path/to/your/output.dta"
   results_dir: "path/to/results"
   ```

2. **Text Columns**:
   ```yaml
   text_columns:
     - title_column
     - description_column
     - content_column
   ```

3. **Preprocessing Options**:
   ```yaml
   preprocessing:
     lowercase: true
     remove_punctuation: true
     remove_stopwords: true
     lemmatize: false
     custom_stopwords: ["example", "custom", "words"]
     min_word_length: 2
     max_length: 10000  # Truncate long texts to this length
   ```

4. **Feature Extraction**:
   ```yaml
   feature_extraction:
     method: "hybrid"  # Options: "tfidf", "embedding", "hybrid"
     tfidf:
       max_features: 5000
       ngram_range: [1, 2]
       min_df: 5
     embedding:
       model: "sentence-transformers"  # Options: "openai", "sentence-transformers"
       sentence_transformers:
         model_name: "all-MiniLM-L6-v2"
       openai:
         model_name: "text-embedding-ada-002"
         api_key_env: "OPENAI_API_KEY"
       dimensionality_reduction:
         method: "umap"  # Options: "umap", "pca", "tsne"
         n_components: 50
         random_state: 42
   ```

5. **Clustering Perspectives**:
   ```yaml
   clustering_perspectives:
     # Perspective 1: Content categories (using short description and title)
     content_categories:
       columns:
         - title_column
         - description_column
       weight: [0.3, 0.7]  # Relative weights for columns
       algorithm: "kmeans"
       params:
         n_clusters: 10
         random_state: 42
       output_column: "content_category_id"
       evaluate_k_range: [5, 15]  # Optional: Try different k values and evaluate
     
     # Perspective 2: Detailed topics (using full content)
     detailed_topics:
       columns:
         - content_column
       algorithm: "hdbscan"
       params:
         min_cluster_size: 15
         min_samples: 5
         metric: "cosine"
       output_column: "detailed_topic_id"
   ```

6. **Cluster Labeling**:
   ```yaml
   cluster_labeling:
     method: "tfidf"  # Options: "openai", "tfidf", "manual"
     openai:
       model: "gpt-3.5-turbo"
       temperature: 0.3
       max_tokens: 30
       prompt_template: "Based on these examples from a cluster, provide a short and descriptive label for the cluster (max 5 words): {examples}"
       examples_per_cluster: 10
     tfidf:
       top_terms: 5
   ```

7. **Evaluation Settings**:
   ```yaml
   evaluation:
     metrics:
       - "silhouette_score"
       - "davies_bouldin_score"
       - "calinski_harabasz_score"
     visualizations:
       - "embeddings_plot"
       - "silhouette_plot"
       - "distribution_plot"
     output_format:
       - "html"
       - "json"
       - "csv"
   ```

8. **Performance and Optimization**:
   ```yaml
   performance:
     batch_size: 100
     parallel_jobs: -1  # Use all available cores
     cache_embeddings: true
     cache_directory: "cache"
   
   spark:
     executor_memory: "4g"
     driver_memory: "4g"
     executor_cores: 2
     default_parallelism: 4
   
   checkpoint:
     enabled: true
     interval: 1  # Checkpoint after each major processing step
     directory: "checkpoints"
     max_checkpoints: 5
   ```

### Command Line Arguments

The system supports several command line arguments to override configuration settings:

```
python main.py --config config.yaml [OPTIONS]

Optional arguments:
  --input INPUT_FILE       Path to the input file
  --output OUTPUT_FILE     Path to the output file
  --results-dir RESULTS_DIR
                          Directory for results and visualizations
  --log-level {debug,info,warning,error}
                          Logging level
  --no-checkpoints         Disable checkpointing
  --seed SEED              Random seed for reproducibility
  --force-recalculate      Force recalculation even if checkpoints exist
  --skip-evaluation        Skip evaluation and reporting steps
  --export-config EXPORT_CONFIG
                          Export the complete configuration to a file
  --perspectives SELECTED_PERSPECTIVES
                          Comma-separated list of perspectives to process
```

### Processing Pipeline Explained

The classification system follows these steps:

1. **Configuration Loading**:
   - Parse YAML configuration and command-line overrides
   - Validate settings and create necessary directories

2. **Data Loading**:
   - Read the Stata file into a PySpark DataFrame
   - Validate the presence of specified text columns

3. **Text Preprocessing**:
   - Apply text cleaning (lowercase, punctuation removal)
   - Remove stopwords and short words if configured
   - Apply lemmatization if enabled

4. **Feature Extraction**:
   - For TF-IDF: Create vectorized representations with n-grams
   - For embeddings: Generate using Sentence Transformers or OpenAI API
   - For hybrid: Create both and combine as needed
   - Apply dimensionality reduction if configured

5. **Clustering for Each Perspective**:
   - Select configured algorithm (K-Means, HDBSCAN, Agglomerative)
   - Fit the model with appropriate parameters
   - Assign cluster IDs to each data point

6. **Cluster Labeling**:
   - For TF-IDF labeling: Extract top terms from each cluster
   - For OpenAI labeling: Generate descriptive labels based on examples
   - Add labels to output as additional columns

7. **Evaluation**:
   - Calculate quality metrics (Silhouette score, etc.)
   - Generate visualizations (embeddings plots, distribution plots)
   - Create comprehensive reports in configured formats

8. **Results Saving**:
   - Add cluster assignments and labels to the original dataset
   - Save as a new Stata file at the specified location
   - Save reports and visualizations to the results directory

### Using Multiple Perspectives

One of the system's most powerful features is the ability to create multiple classification perspectives. Each perspective can:

- Use different text columns or combinations
- Apply different clustering algorithms and parameters
- Focus on different aspects of the data

This allows you to get a more complete understanding of your data structure.

**Example**: For a dataset of learning materials, you might want:
1. A **content_category** perspective using title and short description with K-Means
2. A **detailed_topic** perspective using full content text with HDBSCAN
3. A **difficulty_level** perspective using specific columns related to complexity

Each perspective will create its own column in the output file.

### Optimizing for Large Datasets

For large datasets (>100,000 rows or very long text fields), consider these adjustments:

1. **Increase Spark Memory**:
   ```yaml
   spark:
     executor_memory: "8g"
     driver_memory: "8g"
   ```

2. **Enable Caching**:
   ```yaml
   performance:
     cache_embeddings: true
     cache_directory: "cache"
   ```

3. **Use Batch Processing**:
   ```yaml
   performance:
     batch_size: 500
   ```

4. **Consider Dimensionality Reduction**:
   ```yaml
   embedding:
     dimensionality_reduction:
       method: "umap"
       n_components: 50
   ```

### Interpreting Results

The system generates several types of outputs to help you interpret the clustering results:

1. **Cluster Assignments**: New columns in the output dataset with cluster IDs and labels
2. **Evaluation Metrics**:
   - **Silhouette Score**: Measures how similar an object is to its cluster compared to others (higher is better, range -1 to 1)
   - **Davies-Bouldin Index**: Measures average similarity between clusters (lower is better)
   - **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion (higher is better)
3. **Visualizations**:
   - **Embeddings Plot**: 2D projection of the feature space colored by cluster
   - **Silhouette Plot**: Visual representation of silhouette scores by cluster
   - **Distribution Plot**: Cluster size distribution

### Example Use Cases

1. **Content Categorization**:
   - Automatically organize articles, documents, or learning materials into topics
   - Discover natural groupings in your content library

2. **Customer Feedback Analysis**:
   - Cluster customer reviews or feedback into themes
   - Identify common issues or praise points

3. **Research Data Organization**:
   - Organize research papers or abstracts by topic
   - Discover research trends and connections

4. **Product Description Classification**:
   - Categorize product descriptions into logical groups
   - Identify gaps in product offerings

## Extending the System

### Adding a New Clustering Algorithm

1. Create a new class in `modules/classifier.py` that inherits from `BaseClusterer`:
   ```python
   class MyNewClusterer(BaseClusterer):
       def __init__(self, config, logger, perspective_config):
           super().__init__(config, logger)
           # Initialize your specific parameters
           
       def fit(self, features):
           # Implement your clustering logic
           return self
           
       def predict(self, features):
           # Implement prediction logic
           return predictions
   ```

2. Register your algorithm in the `ClassifierManager.create_clusterer` method.

### Implementing a New Feature Extraction Method

Extend the `FeatureExtractor` class in `modules/data_processor.py` with your new method:

```python
def extract_my_features(self, texts):
    # Implement your feature extraction logic
    return features
```

Then update the `extract_features` method to use your new extraction method.

### Creating New Evaluation Metrics

Add your metric to the `ClusteringEvaluator` class in `modules/evaluation.py`:

```python
def calculate_my_metric(self, features, cluster_assignments):
    # Implement your metric calculation
    return score
```

Update the `evaluate_clustering` method to include your metric.

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors**:
   - Reduce batch size in configuration
   - Increase Spark memory allocation
   - Use dimensionality reduction for embeddings
   - Process a sample of your data first

2. **Slow Performance**:
   - Enable caching for embeddings
   - Reduce the number of features in TF-IDF
   - Use a smaller embedding model
   - Parallelize processing by increasing executor cores

3. **Poor Clustering Quality**:
   - Try different algorithms (K-Means → HDBSCAN)
   - Adjust the number of clusters or algorithm parameters
   - Improve preprocessing (add custom stopwords)
   - Use different embedding models

4. **File Not Found Errors**:
   - Check paths in configuration (absolute vs. relative)
   - Ensure all directories exist
   - Verify file permissions

### Logging and Debugging

The system provides detailed logs that can help diagnose issues:

1. Set the log level to DEBUG for more information:
   ```yaml
   logging:
     level: "DEBUG"
     log_file: "classification_process.log"
   ```

2. Check the log file for error messages and warnings

3. Use the `--force-recalculate` flag to bypass checkpoints if you suspect they're corrupted

## Performance Considerations

- **Memory Usage**: Embedding generation can be memory-intensive. Monitor RAM usage and adjust batch sizes accordingly.
- **Disk Space**: Caching embeddings can use significant disk space for large datasets.
- **API Costs**: If using OpenAI embeddings, be aware of API usage costs.

## Project Structure

```
Python/Translation-Classification/Classification/
├── app.py                # Alternative entry point or API interface
├── config.py             # Configuration management
├── main.py               # Main entry point for the classification pipeline
├── modules/              # Core functionality modules
│   ├── __init__.py
│   ├── classifier.py     # Classification algorithms and logic
│   ├── data_processor.py # Data preprocessing and feature extraction
│   ├── evaluation.py     # Metrics and evaluation functionality
│   └── utilities.py      # Common utility functions
├── README.md             # Project documentation
└── requirements.txt      # Dependencies

Directory structure created during execution:
├── cache/                # Cached embeddings and features
├── checkpoints/          # Pipeline state checkpoints
├── results/              # Evaluation results and visualizations
```

## API Integration

The system provides a simple API interface through `app.py` for integration with other systems:

1. Start the API server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. Make requests to run classification:
   ```
   POST /classify
   Content-Type: application/json

   {
     "config_file": "path/to/config.yaml"
   }
   ```

## Advanced Configuration Examples

### Multi-Language Classification

```yaml
preprocessing:
  lowercase: true
  remove_stopwords: true
  language: "spanish"  # Can be 'english', 'spanish', 'french', etc.

feature_extraction:
  embedding:
    model: "sentence-transformers"
    sentence_transformers:
      model_name: "paraphrase-multilingual-mpnet-base-v2"  # Multilingual model
```

### Optimal K-Means Determination

```yaml
clustering_perspectives:
  content_categories:
    algorithm: "kmeans"
    evaluate_k_range: [5, 20]  # Will test k values from 5 to 20
    params:
      random_state: 42
```

### High-Quality Embeddings with OpenAI

```yaml
feature_extraction:
  method: "embedding"
  embedding:
    model: "openai"
    openai:
      model_name: "text-embedding-ada-002"
      api_key_env: "OPENAI_API_KEY"
      batch_size: 20
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License