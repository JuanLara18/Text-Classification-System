from classifai.backends import LLMBackend
from classifai import device
from classifai import progress

__all__ = ["LLMBackend", "device", "progress"]


def __getattr__(name: str):
    # Lazy-load ClassificationPipeline so its heavy legacy deps
    # (pyspark, sentence-transformers, etc.) don't break `import classifai`.
    if name == "ClassificationPipeline":
        from classifai.pipeline import ClassificationPipeline
        return ClassificationPipeline
    raise AttributeError(f"module 'classifai' has no attribute {name!r}")
