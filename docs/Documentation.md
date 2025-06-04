# Text Classification System: Technical Documentation

## Architecture Overview

The Text Classification System is a modular application designed to automatically classify text columns in data files using various clustering algorithms. The system follows a pipeline architecture with clear separation of concerns, making it maintainable and extensible.

### High-Level Flow

1. **Configuration Loading**: Parse command-line arguments and load configuration from YAML
2. **Data Processing**: Load and preprocess text data
3. **Feature Extraction**: Convert text to numerical features
4. **Clustering**: Apply multiple clustering perspectives
5. **Evaluation**: Calculate quality metrics and visualize results
6. **Reporting**: Generate comprehensive reports
7. **Result Export**: Save enhanced dataset with classification columns

## Project Structure

```
Python/Translation-Classification/Classification/
├── app.py                # Alternative API interface
├── config.py             # Configuration management
├── main.py               # Main entry point
├── modules/              # Core functionality modules
│   ├── __init__.py
│   ├── classifier.py     # Classification algorithms
│   ├── data_processor.py # Data preprocessing and features
│   ├── evaluation.py     # Metrics and visualization
│   └── utilities.py      # Common utility functions
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

## Core Components

### 1. Configuration System (`config.py`)

The `ConfigManager` class handles loading, validating, and providing access to configuration settings.

#### Key Methods
- `load_config()`: Loads configuration from YAML file
- `validate_config()`: Ensures all required parameters are present
- `get_clustering_perspectives()`: Retrieves clustering configurations
- `get_feature_extraction_config()`: Gets feature extraction settings

### 2. Utilities (`modules/utilities.py`)

Contains utility classes used throughout the application.

#### Classes
- `Logger`: Handles logging with configurable levels and outputs
- `SparkSessionManager`: Manages PySpark sessions
- `FileOperationUtilities`: Handles file system operations
- `PerformanceMonitor`: Tracks execution time and resource usage
- `CheckpointManager`: Provides fault tolerance through checkpointing

### 3. Data Processing (`modules/data_processor.py`)

Handles data loading, preprocessing, and feature extraction.

#### Classes
- `DataProcessor`: Main class for loading and processing data
  - `load_data()`: Loads Stata files into PySpark DataFrames
  - `preprocess_text_columns()`: Applies text preprocessing
  - `extract_features()`: Extracts numerical features from text

- `TextPreprocessor`: Handles text cleaning and normalization
  - `preprocess_text()`: Applies preprocessing to a text string
- `FeatureExtractor`: Converts text to numerical features
  - `extract_tfidf_features()`: Creates TF-IDF representations
  - `extract_embeddings()`: Creates embeddings using models
  - `extract_sentence_transformer_embeddings()`: Uses local embedding models
  - `extract_openai_embeddings()`: Uses OpenAI API
  - `reduce_dimensionality()`: Applies dimension reduction (UMAP, PCA, t-SNE)

### 4. Classification (`modules/classifier.py`)

Implements various clustering algorithms and manages the classification process.

#### Classes
- `ClassifierManager`: Orchestrates the classification process
  - `classify_perspective()`: Applies a clustering perspective
  - `create_clusterer()`: Creates specific clustering algorithm instances
  - `add_cluster_labels()`: Adds human-readable labels to clusters

- `BaseClusterer`: Abstract base class for clustering algorithms
  - `fit()`: Trains the clustering model
  - `predict()`: Makes cluster assignments
  - `get_labels()`: Returns cluster assignments

- Algorithm-specific classes:
  - `KMeansClusterer`: K-means clustering
  - `HDBSCANClusterer`: Density-based clustering
  - `AgglomerativeClusterer`: Hierarchical clustering

- `ClusterLabeler`: Creates human-readable labels for clusters
  - `generate_labels()`: Creates labels based on cluster content
  - `generate_tfidf_labels()`: Creates labels from top TF-IDF terms
  - `generate_openai_labels()`: Uses OpenAI to create semantic labels

### 5. Evaluation (`modules/evaluation.py`)

Evaluates clustering quality and generates visualizations and reports.

#### Classes
- `ClusteringEvaluator`: Calculates quality metrics
  - `evaluate_clustering()`: Computes various metrics
  - `calculate_silhouette_score()`: Measures cluster separation
  - `calculate_davies_bouldin_score()`: Measures cluster similarity
  - `calculate_calinski_harabasz_score()`: Measures variance ratios

- `ClusteringVisualizer`: Creates visualizations
  - `create_embeddings_plot()`: Creates 2D projection of clusters
  - `create_silhouette_plot()`: Visualizes silhouette coefficients
  - `create_distribution_plot()`: Shows cluster size distribution
  - `create_cluster_correlation_heatmap()`: Compares perspectives

- `EvaluationReporter`: Generates reports in multiple formats
  - `generate_report()`: Creates comprehensive reports
  - `generate_html_report()`: Creates interactive HTML reports
  - `save_metrics_to_json()`: Exports metrics to JSON
  - `save_metrics_to_csv()`: Exports metrics to CSV

- `ClusterAnalyzer`: Extracts insights from clusters
  - `analyze_cluster_content()`: Extracts key characteristics
  - `extract_key_terms()`: Identifies representative terms
  - `generate_cluster_summary()`: Creates human-readable summaries

### 6. Main Application (`main.py`)

Contains the main application entry point and pipeline orchestration.

#### Classes
- `ClassificationPipeline`: Orchestrates the end-to-end process
  - `setup()`: Initializes components
  - `run()`: Executes the complete pipeline
  - `load_and_preprocess_data()`: Handles data loading and preprocessing
  - `apply_clustering_perspectives()`: Applies multiple clustering approaches
  - `evaluate_and_report()`: Handles evaluation and reporting
  - `perform_cross_perspective_analysis()`: Analyzes multiple perspectives
  - `save_results()`: Saves results to output files

## Pipeline Flow Details

### 1. Initialization
When the application starts, it:
- Parses command-line arguments
- Loads and validates configuration
- Sets up logging and performance monitoring
- Initializes the Spark session
- Creates necessary directories

### 2. Data Loading and Preprocessing
The pipeline then:
- Loads the Stata file into a PySpark DataFrame
- Validates that required columns exist
- Applies text preprocessing to each column
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Word filtering

### 3. Feature Extraction
For each text column:
- Extracts features using the configured method
  - TF-IDF vectorization
  - Sentence transformer embeddings
  - OpenAI embeddings (if configured)
- Applies dimensionality reduction if configured
- Caches features to avoid redundant computation

### 4. Clustering
For each perspective:
- Combines features from selected columns
- Creates and fits a clusterer for the specified algorithm
- Gets cluster assignments
- Adds cluster assignments to the DataFrame
- Generates human-readable labels for each cluster

### 5. Evaluation
For each perspective:
- Calculates quality metrics
  - Silhouette score
  - Davies-Bouldin index
  - Calinski-Harabasz index
- Generates visualizations
  - Embeddings plot
  - Silhouette plot
  - Distribution plot
- Creates comprehensive reports
  - HTML reports with visualizations
  - JSON metrics
  - CSV metrics

### 6. Cross-Perspective Analysis
If multiple perspectives are used:
- Analyzes relationships between perspectives
- Creates correlation heatmaps
- Generates a combined report

### 7. Result Export
Finally, the pipeline:
- Saves the enhanced DataFrame to Stata format
- Creates a timestamp file to mark completion
- Cleans up resources

## Configuration Options

### Input/Output
- `input_file`: Path to the input Stata file
- `output_file`: Path to the output Stata file
- `results_dir`: Directory for visualizations and reports

### Text Processing
- `text_columns`: List of columns to classify
- `preprocessing`: Text cleaning options
  - `lowercase`, `remove_punctuation`, `remove_stopwords`, etc.

### Feature Extraction
- `feature_extraction`: Feature creation settings
  - `method`: "tfidf", "embedding", or "hybrid"
  - `tfidf`: TF-IDF specific settings
  - `embedding`: Embedding model settings
  - `dimensionality_reduction`: Dimension reduction options

### Clustering
- `clustering_perspectives`: Different clustering approaches
  - Columns, weights, algorithms, parameters
  - Output column names

### Evaluation
- `evaluation`: Evaluation metrics and outputs
  - `metrics`: Which metrics to calculate
  - `visualizations`: Which visualizations to create
  - `output_format`: Report formats to generate

### Performance
- `performance`: Performance optimization settings
  - `batch_size`, `parallel_jobs`, `cache_embeddings`
- `spark`: PySpark configuration
  - `executor_memory`, `driver_memory`, etc.

## Extensibility

The system is designed to be easily extended:

### Adding New Clustering Algorithms
1. Create a new class that inherits from `BaseClusterer`
2. Implement `fit()` and `predict()` methods
3. Register in `ClassifierManager.create_clusterer()`

### Adding New Feature Extraction Methods
1. Add methods to the `FeatureExtractor` class
2. Update `extract_features()` to use the new method

### Adding New Evaluation Metrics
1. Add calculation methods to `ClusteringEvaluator`
2. Update `evaluate_clustering()` to include the new metrics

### Adding New Visualizations
1. Add methods to `ClusteringVisualizer`
2. Update reporting to include the new visualizations

## Data Flow Diagram

```
                   ┌─────────────────┐
                   │ Configuration   │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Data Processing │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │Feature Extraction│
                   └────────┬────────┘
                            │
           ┌───────────────┼───────────────┐
           │               │               │
┌──────────▼─────────┐┌────▼────────────┐┌─▼──────────────┐
│Perspective 1       ││Perspective 2    ││Perspective 3   │
│Classification      ││Classification   ││Classification  │
└──────────┬─────────┘└────┬────────────┘└───┬────────────┘
           │               │                 │
           └───────────────┼─────────────────┘
                           │
                  ┌────────▼────────┐
                  │   Evaluation    │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │    Reporting    │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Result Export  │
                  └─────────────────┘
```

## Error Handling and Fault Tolerance

The system includes several mechanisms for handling errors and ensuring robustness:

### Checkpointing
- The `CheckpointManager` provides checkpointing at key stages
- Progress can be resumed after failure
- Optional cleanup of old checkpoints

### Exception Handling
- Comprehensive try/except blocks throughout the code
- Graceful degradation when components fail
- Detailed error logging

### Validation
- Configuration validation before processing
- Input file validation
- Column existence checks

## Performance Considerations

The system includes various optimizations for performance:

### Memory Management
- Batch processing for large datasets
- Dimensionality reduction for high-dimensional features
- Caching of intermediate results

### Distributed Processing
- PySpark integration for horizontal scaling
- Configurable memory allocation
- Parallel processing where possible

### Monitoring
- The `PerformanceMonitor` tracks execution time
- Memory usage tracking
- Detailed performance reporting

## Limitations and Considerations

### Technical Limitations
- The system assumes input data is in Stata (.dta) format
- OpenAI features require an API key
- Resource requirements scale with dataset size

### Algorithm Considerations
- Different clustering algorithms are suited to different data distributions
- K-means assumes spherical clusters of similar size
- HDBSCAN handles varying densities but is more complex
- Agglomerative clustering reveals hierarchical structure

### Configuration Tuning
- Clustering quality depends on appropriate parameter selection
- Feature extraction method impacts clustering results
- Preprocessing choices affect final classifications

## Debugging and Troubleshooting

### Logging
- Use `--log-level debug` for detailed logging
- Check `classification_process.log` for errors
- Monitor Java process for memory issues

### Common Issues
- Memory errors: Reduce batch size, enable dimension reduction
- Slow performance: Enable caching, use simpler models
- Poor clustering: Try different algorithms, adjust parameters

## Best Practices

### Configuration
- Start with a small sample to test configurations
- Use `--force-recalculate` for clean runs
- Experiment with different perspectives for best results

### Resource Allocation
- Allocate memory based on dataset size
- Enable caching for large datasets
- Use dimensionality reduction for high-dimensional data

### Cluster Interpretation
- Use visualizations to understand cluster relationships
- Examine top terms to understand cluster themes
- Compare multiple perspectives for richer insights
