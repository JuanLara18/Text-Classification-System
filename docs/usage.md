# Usage examples

## Installation

```bash
git clone https://github.com/JuanLara18/classifai
cd classifai
pip install -e ".[llm,clustering,viz]"
```

Set your API key (skip for Ollama):

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 1. Classify with a known list of categories (LLM)

The fastest path — give classifai your categories and a text column:

```python
import pandas as pd
from classifai.backends import LLMBackend

df = pd.read_csv("data/support_tickets_sample.csv")

clf = LLMBackend(
    categories=["Billing", "Technical Support", "Account & Access", "Other"],
    model="gpt-4o-mini",
    provider="openai",
)

df["department"] = clf.predict(df["body"])
print(df[["subject", "department"]].head())
```

### Switch provider without changing anything else

```python
# Anthropic
clf = LLMBackend(categories=[...], model="claude-haiku-4-5-20251001", provider="anthropic")

# Ollama (free, local — no API key)
clf = LLMBackend(categories=[...], model="llama3", provider="ollama")
```

### Combine multiple columns as context

```python
clf = LLMBackend(categories=["Billing", "Technical Support", "Other"], model="gpt-4o-mini")

# Concatenate subject + body before classifying
text = df["subject"] + " — " + df["body"]
df["department"] = clf.predict(text)
```

---

## 2. Discover topics with clustering (no labels needed)

When you don't know the categories in advance:

```python
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv("data/support_tickets_sample.csv")

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["body"].tolist(), show_progress_bar=True)

# Cluster
km = KMeans(n_clusters=6, random_state=42)
df["cluster"] = km.fit_predict(embeddings)

print(df.groupby("cluster")["subject"].count())
```

For automatic cluster count, use HDBSCAN (finds noise automatically):

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
df["cluster"] = clusterer.fit_predict(embeddings)
# -1 = noise points
```

---

## 3. Run via config file (full pipeline)

Create a `config.yaml` (see `config.example.yaml`) and run:

```bash
python main.py --config config.yaml
```

Minimal config:

```yaml
input_file:   "data/support_tickets_sample.csv"
output_file:  "data/classified.csv"
text_columns: [subject, body]
results_dir:  "output"

clustering_perspectives:
  department:
    type: "openai_classification"
    columns: [subject, body]
    output_column: "department"
    target_categories:
      - Billing & Payments
      - Technical Support
      - Account & Access
      - Other
    llm_config:
      provider: "openai"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
```

The pipeline:
- shows a GPU/CPU banner at startup
- classifies only unique text values (90%+ fewer API calls on real datasets)
- caches results to `.cache/classifai/` — reruns are instant
- saves a labeled CSV and an HTML report in `output/`

---

## 4. Use the sample dataset

```python
import pandas as pd

df = pd.read_csv("data/support_tickets_sample.csv")
print(df.columns.tolist())
# ['id', 'created_at', 'subject', 'body', 'true_category']

print(df["true_category"].value_counts())
# Billing & Payments         10
# Technical Support          10
# Account & Access           10
# ...
```

Evaluate against ground truth after classification:

```python
from sklearn.metrics import classification_report

df["predicted"] = clf.predict(df["body"])
print(classification_report(df["true_category"], df["predicted"]))
```

---

## 5. Check GPU detection

```python
from classifai import device

info = device.get()
print(info.device)      # "cuda:0", "mps", or "cpu"
print(info.has_gpu)     # True / False
print(info.vram_gb)     # e.g. 8.0 (None on CPU)

device.print_banner()   # rich panel shown at pipeline startup
```

---

## 6. Cost control

```python
clf = LLMBackend(
    categories=["Billing", "Technical Support", "Other"],
    model="gpt-4o-mini",
    provider="openai",
    cache=True,              # disk cache — identical texts classified once
    cache_dir=".cache/clf",
    batch_size=50,           # texts per parallel batch
    max_workers=4,           # parallel API calls
)
```

Typical cost with `gpt-4o-mini`:

| Rows   | Unique texts | Cost    |
|--------|-------------|---------|
| 10 000 | ~2 000      | ~$0.05  |
| 100 000| ~15 000     | ~$0.40  |
| 1 000 000| ~80 000   | ~$2.00  |

---

## 7. Notebook

Open [`notebooks/quickstart.ipynb`](../notebooks/quickstart.ipynb) for an end-to-end walkthrough:
load → classify → evaluate accuracy → cluster with HDBSCAN/UMAP → visualize with Plotly.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanLara18/classifai/blob/main/notebooks/quickstart.ipynb)
