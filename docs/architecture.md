# Architecture

## Overview

classifai is a pipeline with two interchangeable classification modes that can run in the same job.

```mermaid
flowchart LR
    cfg([config.yaml]) --> P
    data([CSV · Stata · Excel]) --> P

    subgraph P[Pipeline]
        direction TB
        L[Loader] --> Pre[Preprocessor]
        Pre --> Pers[Perspectives]
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
    A[Load config.yaml] --> B[Load dataset]
    B --> C[Preprocess text columns]
    C --> D{For each perspective}

    D -->|type: openai_classification| E[Extract unique values]
    E --> F[Classify via LLM\ninstructor + Pydantic Enum]
    F --> G[Map labels back to all rows]

    D -->|type: clustering| H[Extract features\nTF-IDF or sentence embeddings]
    H --> I[Run algorithm\nHDBSCAN · K-Means]
    I --> J[Label clusters with LLM]

    G --> K[Evaluate and generate report]
    J --> K
    K --> L[Save classified dataset]
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

## Package structure

```mermaid
flowchart TD
    main.py --> classifai/pipeline.py
    classifai/pipeline.py --> classifai/loader.py
    classifai/pipeline.py --> classifai/evaluator.py
    classifai/pipeline.py --> classifai/saver.py
    classifai/pipeline.py --> classifai/backends/llm.py

    classifai/backends/llm.py --> modules/ai_classifier.py
    classifai/pipeline.py --> modules/classifier.py
    classifai/pipeline.py --> modules/data_processor.py
    classifai/pipeline.py --> modules/evaluation.py

    config.py --> classifai/pipeline.py
```

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
