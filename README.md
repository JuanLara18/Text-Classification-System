# classifai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](pyproject.toml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanLara18/classifai/blob/main/notebooks/quickstart.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/JuanLara18/classifai?style=social)](https://github.com/JuanLara18/classifai/stargazers)

**Classify any text dataset with one config file.** Use OpenAI, Anthropic, or a free local model — and fall back to unsupervised clustering when you have no labels at all.

---

## What it does

Point classifai at a CSV, define your categories (or skip them for clustering), and get back a labeled dataset plus an HTML report.

```bash
python main.py --config config.example.yaml
```

It supports two modes that can run **in the same pass**:

| Mode | When to use |
|---|---|
| **AI classification** — OpenAI, Anthropic, or Ollama (free, local) | You know your categories |
| **Clustering** — HDBSCAN, K-Means, UMAP | You want to discover patterns |

---

## Install

```bash
git clone https://github.com/JuanLara18/classifai.git
cd classifai
pip install -r requirements.txt
```

For LLM classification, add your API key:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

To run **completely free and offline**, use [Ollama](https://ollama.com) instead — no key needed, just set `provider: ollama` in the config.

---

## Quick example

```yaml
# config.yaml
input_file:  "data/support_tickets.csv"
output_file: "data/classified.csv"
text_columns: [subject, body]

clustering_perspectives:

  # Label tickets by department (LLM)
  department:
    type: "openai_classification"
    columns: [subject, body]
    target_categories: [Billing, Technical Support, Account, Other]
    output_column: "routed_to"
    llm_config:
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"

  # Discover unknown patterns (no labels needed)
  topics:
    type: "clustering"
    algorithm: "hdbscan"
    columns: [body]
    output_column: "topic_cluster"
```

```bash
python main.py --config config.yaml
```

**Output:** `data/classified.csv` with two new columns (`routed_to`, `topic_cluster`) and an HTML report in `output/`.

---

## Key features

- **Guaranteed valid labels** — uses [instructor](https://python.useinstructor.com/) + Pydantic, so the LLM always returns one of your categories. No regex, no parsing errors.
- **Unique-value optimization** — classifies each distinct text only once, then maps results back. Reduces API calls by up to 90% on real datasets.
- **Multi-provider** — OpenAI, Anthropic, or Ollama (local, free). Same config, different `provider:` line.
- **Dual mode** — run AI classification and clustering in the same job and compare results.
- **Cost control** — set `max_cost_per_run` to hard-stop before overspending.
- **Resumable** — checkpoints let you continue interrupted runs without re-classifying.

---

## Notebooks

| | |
|---|---|
| [quickstart.ipynb](notebooks/quickstart.ipynb) — classify, cluster, visualize in one notebook | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanLara18/classifai/blob/main/notebooks/quickstart.ipynb) |

---

## Cost reference

`gpt-4o-mini` with unique-value optimization:

| Rows | Unique texts | Estimated cost |
|---|---|---|
| 10,000 | ~2,000 | ~$0.05 |
| 100,000 | ~15,000 | ~$0.40 |
| 1,000,000 | ~80,000 | ~$2.00 |

Set `provider: ollama` to pay nothing.

---

## File formats

CSV, Stata (`.dta`), Excel (`.xlsx`).

---

## Real-world use

classifai was built during a research project at **Harvard Business School** to classify BMW manufacturing maintenance records — thousands of work order descriptions in English and German, across taxonomies from 2 to 20 categories. The underlying data is confidential, but the methodology is exactly what's in this repo.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and new backends are especially welcome.

## License

MIT — see [LICENSE](LICENSE).
