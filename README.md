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

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key

**Step 3.1: Get your API key**
- Visit: https://platform.openai.com/api-keys
- Create a new API key
- Copy the key (starts with `sk-`)

**Step 3.2: Create a .env file**
```bash
# Create .env file in the project root
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
```

**Step 3.3: Load the environment variables (Choose the method for your system)**

**Option A: Install python-dotenv (Recommended - Works on all platforms)**
```bash
pip install python-dotenv
```
*The system will automatically load your .env file when running the classifier.*

**Option B: Load manually (Mac/Linux)**
```bash
# Method 1: Using export
export $(grep -v '^#' .env | xargs)

# Method 2: Using source (edit .env to add 'export' first)
# Edit your .env file to: export OPENAI_API_KEY=sk-your-actual-key-here
source .env
```

**Option C: Load manually (Windows)**
```cmd
# Command Prompt
set OPENAI_API_KEY=sk-your-actual-key-here

# PowerShell
$env:OPENAI_API_KEY="sk-your-actual-key-here"
```

**⚠️ Important:** Replace `sk-your-actual-key-here` with your real OpenAI API key.

### 4. Configure your classification (Choose one option)

**Option A: Web Interface (Recommended for beginners)**
- Visit: https://text-classification-config.streamlit.app
- Upload your data and configure your classification settings through the web interface
- Download the generated `config.yaml` file to your project folder

**Option B: Local Web Interface**
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501` to configure your settings locally.

**Option C: Manual Configuration**
- Copy and edit the sample `config.yaml` file in the repository
- Configure your input/output files, text columns, and classification perspectives

### 5. Prepare your classification settings

The system supports two main approaches:

**AI Classification with ChatGPT**
- Set `type: "openai_classification"`
- Define your `target_categories` (e.g., job titles, product types, sentiment)
- Specify which `columns` to classify
- Perfect for when you know what categories you want

**Traditional Clustering**
- Set `type: "clustering"`
- Choose an `algorithm` such as `hdbscan` or `kmeans`
- Use this to discover hidden patterns in your data
- Great for exploratory analysis when you don't know the categories beforehand

### 6. Run the classifier
```bash
python main.py --config config.yaml
```

Your results will be automatically saved in the `output/` folder. The system creates all necessary folders automatically based on your configuration.

### 🔍 Verification

To verify everything is working correctly:

1. **Check your API key is loaded:**
   ```bash
   # Mac/Linux
   echo $OPENAI_API_KEY
   
   # Windows Command Prompt
   echo %OPENAI_API_KEY%
   
   # Windows PowerShell
   echo $env:OPENAI_API_KEY
   ```

2. **Test with a small dataset first** to ensure everything works before processing large files.

### 💡 Tips

- **For beginners:** Use the web interface (Option A) to generate your configuration
- **For advanced users:** Edit the `config.yaml` file directly for more control
- **Multiple perspectives:** You can define both AI and clustering approaches in the same configuration file to compare results
- **Large datasets:** The system supports processing large files efficiently using Spark
- **Resumable processing:** If interrupted, the system can resume from checkpoints

### 🆘 Troubleshooting

**API Key Issues:**
- Ensure your API key starts with `sk-`
- Check that the environment variable is set correctly
- Verify you have sufficient OpenAI API credits

**Import Errors:**
- Make sure you activated your virtual environment
- Try reinstalling dependencies: `pip install -r requirements.txt`

**Configuration Issues:**
- Use the web interface to generate a valid configuration
- Check that your input file path is correct
- Ensure your text columns exist in your dataset

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
