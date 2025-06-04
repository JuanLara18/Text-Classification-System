# AI Text Classification System

🤖 **Automatically classify text data using AI-powered categorization**

This tool takes your data files (like Excel or Stata files) with text columns and automatically adds new classification columns using OpenAI's GPT models. Perfect for categorizing job positions, products, customer feedback, or any text data.

## Quick Start (5 minutes)

### 1. Setup
```bash
# Clone and setup
git clone <repository-url>
cd text-classification-system
python setup.py

# Install dependencies
pip install -r requirements.txt
```

### 2. Get OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to your `.env` file:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Configure Your Data
Edit `config.yaml`:
```yaml
# Update these 3 things:
input_file: "input/your_data.dta"           # Path to your data file
text_columns: ["job_title", "description"]  # Your text column names
target_categories:                          # Your classification categories
  - "Engineering"
  - "Sales" 
  - "Marketing"
  - "HR"
  # Add your categories here
```

### 4. Run Classification
```bash
python main.py --config config.yaml
```

**That's it!** Your classified data will be in the `output/` folder.

---

## Essential Commands

```bash
# Basic usage
python main.py --config config.yaml

# Force recalculation (ignore cache)
python main.py --config config.yaml --force-recalculate

# Test with debug output
python main.py --config config.yaml --log-level debug

# Process specific input/output files
python main.py --config config.yaml --input data/myfile.dta --output results/classified.dta
```

## What You Get

The system adds new columns to your data:
- `{classification_name}` - The assigned category
- Detailed HTML reports with visualizations
- Cost tracking and performance metrics

Example: If you classify job positions, you'll get a new column like `job_category` with values like "Engineering", "Sales", etc.

---

## Configuration Guide

### Basic Configuration Structure
```yaml
# Required: Your data and columns
input_file: "path/to/your/data.dta"
output_file: "path/to/output.dta" 
text_columns: ["column1", "column2"]

# Required: Classification setup
clustering_perspectives:
  your_classifier_name:
    type: "openai_classification"
    columns: ["column1"]              # Which columns to classify
    target_categories: ["Cat1", "Cat2"]  # Your categories
    output_column: "new_category_column"  # Name of new column
```

### Supported File Formats
- **Stata files** (`.dta`) - Primary format
- **CSV files** (`.csv`) - Basic support
- **Excel files** (`.xlsx`) - Convert to Stata first

### Cost Management
The system includes built-in cost controls:
```yaml
ai_classification:
  cost_management:
    max_cost_per_run: 10.0  # Stop if cost exceeds $10
```

**Typical costs:**
- Small dataset (1,000 rows): ~$0.50
- Medium dataset (10,000 rows): ~$2.00  
- Large dataset (100,000 rows): ~$15.00

### Performance Settings
```yaml
performance:
  batch_size: 50        # Process 50 items at once
  parallel_jobs: 4      # Use 4 CPU cores
  cache_embeddings: true # Cache results to avoid reprocessing
```

## Advanced Usage

### Multiple Classifications
You can create multiple classification perspectives:
```yaml
clustering_perspectives:
  job_categories:
    columns: ["job_title"]
    target_categories: ["Engineering", "Sales", "Marketing"]
    output_column: "job_category"
  
  skill_levels:
    columns: ["job_description"] 
    target_categories: ["Entry Level", "Mid Level", "Senior Level"]
    output_column: "skill_level"
```

### Traditional Clustering (No AI)
For exploratory analysis without predefined categories:
```yaml
clustering_perspectives:
  content_clusters:
    type: "clustering"              # Traditional clustering
    algorithm: "hdbscan"            # or "kmeans"
    columns: ["text_column"]
    params:
      min_cluster_size: 50
    output_column: "discovered_cluster"
```

### Custom Prompts
Customize how the AI classifies your data:
```yaml
classification_config:
  prompt_template: |
    You are an expert in job classification.
    Classify this job position into exactly one category.
    
    Categories: {categories}
    Job Position: {text}
    
    Consider the job title, responsibilities, and required skills.
    Respond with only the category name.
```

## Troubleshooting

### Common Issues

**"API key not found"**
```bash
# Check your .env file
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

**"File not found"**
```bash
# Check your file path in config.yaml
ls input/  # List files in input directory
```

**"Memory error"**
```yaml
# Reduce batch size in config.yaml
performance:
  batch_size: 25  # Reduce from 50
```

**"Too expensive"**
```yaml
# Set lower cost limit
ai_classification:
  cost_management:
    max_cost_per_run: 5.0  # Reduce from 10.0
```

### Getting Help

1. **Check logs**: Look in `logs/classification.log`
2. **Enable debug mode**: Use `--log-level debug`
3. **Test small sample**: Process just 100 rows first

## Technical Details

### System Architecture
- **Text Processing**: NLTK + custom preprocessing
- **AI Classification**: OpenAI GPT models with intelligent caching
- **Traditional Clustering**: scikit-learn, HDBSCAN, UMAP
- **Data Processing**: PySpark for large datasets
- **Unique Value Optimization**: Dramatically reduces API calls by processing only unique text values

### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores  
- **Large datasets**: 16GB+ RAM, 8+ CPU cores

### Feature Extraction Methods
```yaml
feature_extraction:
  method: "embedding"  # or "tfidf", "hybrid"
  embedding:
    model: "sentence-transformers"
    sentence_transformers:
      model_name: "all-MiniLM-L6-v2"  # Fast, good quality
```

### Supported Algorithms
- **AI Classification**: GPT-4o-mini, GPT-3.5-turbo, GPT-4
- **Traditional Clustering**: K-Means, HDBSCAN, Agglomerative
- **Evaluation Metrics**: Silhouette score, Davies-Bouldin index

## Examples

### Example 1: Job Position Classification
```yaml
input_file: "data/hr_data.dta"
text_columns: ["position_title", "job_description"]

clustering_perspectives:
  job_classifier:
    type: "openai_classification"
    columns: ["position_title"]
    target_categories:
      - "Software Engineering"
      - "Data Science" 
      - "Product Management"
      - "Sales"
      - "Marketing"
      - "Human Resources"
      - "Finance"
      - "Operations"
      - "Other"
    output_column: "job_category"
```

### Example 2: Customer Feedback Analysis  
```yaml
input_file: "data/feedback.dta"
text_columns: ["customer_comment"]

clustering_perspectives:
  sentiment_classifier:
    type: "openai_classification"
    columns: ["customer_comment"]
    target_categories:
      - "Positive"
      - "Negative" 
      - "Neutral"
      - "Feature Request"
      - "Bug Report"
    output_column: "feedback_type"
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Need help?** Create an issue or check the logs in `logs/classification.log`
