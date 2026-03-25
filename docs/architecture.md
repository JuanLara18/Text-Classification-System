# Architecture

## Overview

classifai is a pipeline with two interchangeable classification modes that can run in the same job.

```mermaid
flowchart LR
    cfg([config.yaml]) --> P
    data([CSV · Stata · Excel]) --> P

    subgraph P[Pipeline]
        direction TB
        Dev[device.py\nGPU auto-detect] --> L
        L[Loader] --> Pre[Preprocessor]
        Pre --> Pers[Perspectives]
        Prog[progress.py\nrich bars + ETA] -.-> Pers
    end

    Pers -->|openai_classification| LLM[LLM Backend\nopenai · anthropic · ollama]
    Pers -->|clustering| CL[Clustering\nHDBSCAN · K-Means · Agglomerative]

    LLM --> Eval[Evaluator]
    CL  --> Eval

    Eval --> out([Labeled dataset\nHTML report · JSON metrics])
```

## Pipeline steps

```mermaid
flowchart TD
    A[Load config.yaml] --> B[Detect GPU/CPU\ndevice.print_banner]
    B --> C[Load dataset]
    C --> D[Preprocess text columns]
    D --> E{For each perspective}

    E -->|type: openai_classification| F[LLMBackend.from_config\ninstructor + Pydantic Enum]
    F --> G[Unique-value optimizer\nextract distinct texts]
    G --> H[Classify via LLM\nwith progress bar + ETA]
    H --> I[Map labels back to all rows]

    E -->|type: clustering| J[Extract features\nTF-IDF or sentence embeddings]
    J --> K[Run algorithm\nHDBSCAN · K-Means]
    K --> L[Label clusters with LLM]

    I --> M[Evaluate and generate report]
    L --> M
    M --> N[Save classified dataset]
```

## LLM backend: unique-value optimization

The key cost-saving mechanism — only distinct text values are sent to the model.

```mermaid
flowchart LR
    A[All rows\ne.g. 50 000] --> B[Unique-value\nextractor]
    B -->|Only unique texts\ne.g. 3 000| C[LLM\nvia instructor]
    C --> D[Disk cache]
    D --> E[Map results\nback to all rows]
    E --> F[Labeled dataset\n50 000 rows]
```

**Impact:** 90%+ reduction in API calls on real datasets where rows repeat.

## GPU auto-detection (`classifai/device.py`)

At startup the pipeline detects and logs the best available device — no manual configuration needed.

```
Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
```

```mermaid
flowchart LR
    D{torch available?}
    D -->|CUDA| GPU[DeviceInfo\ndevice=cuda:0\nvram_gb, cuda_version\ncuml_available]
    D -->|MPS| MPS[DeviceInfo\ndevice=mps]
    D -->|No GPU| CPU[DeviceInfo\ndevice=cpu\nram_gb via psutil]
```

If [cuML (RAPIDS)](https://docs.rapids.ai/install) is installed, UMAP and HDBSCAN also run on the GPU.

```python
from classifai import device

info = device.get()          # DeviceInfo singleton
print(info.device)           # "cuda:0", "mps", or "cpu"
print(info.has_gpu)          # True / False
device.print_banner()        # rich Panel shown at startup
```

## Progress bars (`classifai/progress.py`)

`PipelineProgress` wraps `rich.progress` and is used throughout the pipeline for embedding, classification, and clustering steps.

```python
from classifai.progress import PipelineProgress

prog = PipelineProgress()

with prog.task("Classifying", total=n_unique) as advance:
    for batch in batches:
        classify(batch)
        advance(len(batch))

prog.update_cost(0.0012)     # accumulate LLM cost per call
prog.print_cost_summary()    # prints total calls + $ estimate
```

Falls back to plain `print` when `rich` is not installed or `quiet=True`.

## Package structure

```mermaid
flowchart TD
    main.py --> classifai/pipeline.py

    subgraph classifai
        classifai/pipeline.py --> classifai/loader.py
        classifai/pipeline.py --> classifai/evaluator.py
        classifai/pipeline.py --> classifai/saver.py
        classifai/pipeline.py --> classifai/device.py
        classifai/pipeline.py --> classifai/progress.py
        classifai/pipeline.py --> classifai/backends/llm.py
    end

    subgraph backends[classifai/backends]
        classifai/backends/llm.py --> classifai/backends/base.py
    end

    subgraph legacy[modules - legacy]
        classifai/pipeline.py --> modules/classifier.py
        classifai/pipeline.py --> modules/data_processor.py
        classifai/pipeline.py --> modules/evaluation.py
    end

    config.py --> classifai/pipeline.py
```

The `modules/` directory is legacy code used for the `clustering` perspective. The `openai_classification` perspective now routes through `classifai/backends/llm.py` directly.

## Adding a new backend

Every backend implements `BaseBackend` from `classifai/backends/base.py`:

```mermaid
classDiagram
    class BaseBackend {
        +predict(texts: Series) Series
        +predict_batch(texts, batch_size) Series
    }
    class LLMBackend {
        +categories: list
        +provider: str
        +model: str
        +predict(texts) Series
        +from_config(perspective_config) LLMBackend
    }
    BaseBackend <|-- LLMBackend
    BaseBackend <|-- SklearnBackend
    BaseBackend <|-- SetFitBackend
```

To add a backend: create `classifai/backends/my_backend.py`, inherit `BaseBackend`, implement `predict()`, register in `classifai/backends/__init__.py`. See [CONTRIBUTING.md](../CONTRIBUTING.md).
