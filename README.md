# AI Text Classification System

![Repo Size](https://img.shields.io/github/repo-size/JuanLara18/Text-Classification-System)
![License](https://img.shields.io/github/license/JuanLara18/Text-Classification-System)
![Last Commit](https://img.shields.io/github/last-commit/JuanLara18/Text-Classification-System)

Easily label large text files using AI and classic clustering methods. The
system relies on OpenAI models along with scikit-learn, HDBSCAN and PySpark to
process your data, add new category columns and create detailed reports.

## Quick Start

### 0. Clone this repository
```bash
git clone https://github.com/JuanLara18/Text-Classification-System.git
cd Text-Classification-System
```

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### 2. Run the setup script
```bash
python setup.py
```
This command creates the basic folders, a sample `config.yaml` and an `.env`
file.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI key
Edit `.env` and set:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 5. Configure the classification
Open `config.yaml` to tell the system how it should label your text.  A
**perspective** is a set of rules for one classification task.  You can define
several perspectives in the file and run them all at once.

**AI classification with ChatGPT**
- Set `type: "openai_classification"`.
- List the `columns` you want to classify and the `target_categories` you expect
  (e.g. job titles, product types, etc.).
- Optionally tweak the `classification_config` section to change the prompt or
  batch size.

**Traditional clustering**
- Set `type: "clustering"` and choose an `algorithm` such as `hdbscan` or
  `kmeans`.
- Provide algorithm parameters under `params` if needed.
- Use this when you do not have predefined categories and want to discover
  groups automatically.

**Hybrid approach**
- Mix both AI and clustering perspectives in the same file.
- Useful for comparing the results and refining your categories.

Remember to update `input_file`, `output_file`, `text_columns` and any category
lists or algorithm settings relevant to your use case.

### 6. Run the classifier
```bash
python main.py --config config.yaml
```

Your classified data will be saved in the `output/` folder.

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
