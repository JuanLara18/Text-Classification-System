# classifai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![GitHub stars](https://img.shields.io/github/stars/JuanLara18/classifai?style=social)](https://github.com/JuanLara18/classifai/stargazers)

**Classify text at any scale.** From a quick zero-shot call to a local LLM to production-scale fine-tuned transformers — one tool, one config, one command.

classifai handles the full classification pipeline: load any dataset → classify or cluster the text → get a clean report with metrics and visualizations. Swap backends without changing your workflow.

---

## Why classifai?

Most classification tools lock you into one approach. When your dataset grows, or your API bill spikes, or you need to run offline, you're starting over. classifai is built around interchangeable backends:

| Backend | Best for | Requires |
|---|---|---|
| `sklearn` | Baseline, millions of rows, no GPU | Nothing extra |
| `zero-shot` | No labeled data, quick exploration | HuggingFace |
| `llm` | Best accuracy, flexible categories | OpenAI / Anthropic / Ollama |
| `setfit` | 8–32 labeled examples per class | `sentence-transformers` |
| `transformers` | Production accuracy, GPU-optimized | PyTorch + GPU |

Same config file. Same output format. Different backend.

---

## Quick start

```bash
git clone https://github.com/JuanLara18/classifai.git
cd classifai
pip install -r requirements.txt
```

Set your API key if using the LLM backend:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

Run on the included example:

```bash
python main.py --config config.example.yaml
```

Results land in `output/` — a classified CSV and an HTML report.

---

## Configuration

All behavior is controlled by a single YAML file. A complete annotated example is in [`config.example.yaml`](config.example.yaml).

### Minimal config (AI classification)

```yaml
input_file:  "data/my_data.csv"
output_file: "data/my_data_classified.csv"
text_columns: [text]

clustering_perspectives:
  category:
    type: "openai_classification"
    columns: [text]
    target_categories: ["Billing", "Technical Support", "Other"]
    output_column: "routed_to"
    llm_config:
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
```

### Minimal config (clustering — no labels needed)

```yaml
input_file:  "data/my_data.csv"
output_file: "data/my_data_clustered.csv"
text_columns: [text]

clustering_perspectives:
  topics:
    type: "clustering"
    algorithm: "hdbscan"
    columns: [text]
    output_column: "topic_cluster"
```

### Mix both in one run

Define multiple perspectives in the same file — AI classification and clustering run in parallel and their results are compared in the final report.

---

## What you get

For each run classifai produces:

- **Classified dataset** — original file with new label columns added
- **HTML report** — distribution plots, embedding visualizations, cluster analysis, cost summary
- **JSON results** — machine-readable metrics (silhouette score, Davies-Bouldin, per-category counts)

### Cost management

The LLM backend never exceeds your budget:

```yaml
ai_classification:
  cost_management:
    max_cost_per_run: 5.00   # hard stop at $5
```

Typical costs with `gpt-4o-mini`:

| Dataset size | Estimated cost |
|---|---|
| 1,000 rows | ~$0.10 |
| 10,000 rows | ~$0.80 |
| 100,000 rows | ~$6.00 |

---

## Supported file formats

- CSV (`.csv`)
- Stata (`.dta`)
- Excel (`.xlsx`)

---

## Run options

```bash
# Standard run
python main.py --config config.yaml

# Ignore cache and recompute everything
python main.py --config config.yaml --force-recalculate

# Verbose logging
python main.py --config config.yaml --log-level debug

# Override input/output from CLI
python main.py --config config.yaml --input data/new.csv --output results/out.csv
```

---

## Advanced usage

### Multiple classification perspectives

```yaml
clustering_perspectives:

  # Classify by department
  department:
    type: "openai_classification"
    columns: [subject, body]
    target_categories: ["Billing", "Technical Support", "Account", "Other"]
    output_column: "department"

  # Classify by urgency
  urgency:
    type: "openai_classification"
    columns: [subject, body]
    target_categories: ["Critical", "High", "Normal", "Low"]
    output_column: "urgency"

  # Discover unknown patterns
  patterns:
    type: "clustering"
    algorithm: "hdbscan"
    columns: [body]
    output_column: "issue_pattern"
```

### Custom prompts

```yaml
classification_config:
  prompt_template: |
    You are a support routing expert.
    Classify the ticket below into exactly one department.

    Departments: {categories}
    Ticket: {text}

    Department name only:
```

### Feature extraction for clustering

```yaml
feature_extraction:
  method: "hybrid"        # tfidf + sentence embeddings
  tfidf:
    max_features: 2000
    ngram_range: [1, 2]
  embedding:
    model: "sentence-transformers"
    sentence_transformers:
      model_name: "all-MiniLM-L6-v2"
```

---

## Architecture

```
classifai/
├── main.py                 # Pipeline entry point
├── config.py               # Configuration management
├── config.example.yaml     # Annotated example
├── modules/
│   ├── ai_classifier.py    # LLM-based classification (OpenAI)
│   ├── classifier.py       # Clustering algorithms
│   ├── data_processor.py   # Data loading and preprocessing
│   ├── evaluation.py       # Metrics and HTML reports
│   └── utilities.py        # Logging, caching, checkpoints
└── tools/
    └── app.py              # Streamlit config generator UI
```

---

## Real-world use case

classifai was developed as part of a research project at **Harvard Business School**, where it was used to classify manufacturing maintenance records from an industrial dataset. The pipeline processed thousands of work order descriptions in English and German, testing multiple classification taxonomies — from 2 broad categories to 20 granular failure modes — using GPT-based classification and traditional clustering in parallel.

The goal was to surface patterns in manufacturing defects buried in unstructured text fields. The underlying data is confidential, but the methodology, pipeline, and configuration system are exactly what was used in that analysis.

---

## Troubleshooting

**API key not found**
```bash
cat .env   # should show OPENAI_API_KEY=sk-...
```

**Memory error during clustering**
```yaml
performance:
  batch_size: 25    # reduce from default 50
```

**Cost limit reached before completion**
```yaml
ai_classification:
  cost_management:
    max_cost_per_run: 20.0
```

**Logs and debug mode**
```bash
python main.py --config config.yaml --log-level debug
# logs are also written to logs/classification.log
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started, add a new backend, or report a bug.

---

## License

MIT — see [LICENSE](LICENSE).
