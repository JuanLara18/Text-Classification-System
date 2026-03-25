from classifai.backends import LLMBackend
from classifai import device
from classifai import progress
from classifai.reporter import generate_report, HTMLReporter

__all__ = ["LLMBackend", "device", "progress", "generate_report", "HTMLReporter"]


def __getattr__(name: str):
    # Lazy-load ClassificationPipeline so its heavy legacy deps
    # (pyspark, sentence-transformers, etc.) don't break `import classifai`.
    if name == "ClassificationPipeline":
        from classifai.pipeline import ClassificationPipeline
        return ClassificationPipeline
    raise AttributeError(f"module 'classifai' has no attribute {name!r}")
