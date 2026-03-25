"""Base class for all classifai backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseBackend(ABC):
    """
    Common interface every backend must implement.

    A backend receives a list of text strings and returns a list of
    category labels (one per input). The pipeline always calls
    `predict(texts)` regardless of which backend is active.
    """

    @abstractmethod
    def predict(self, texts: pd.Series) -> pd.Series:
        """
        Classify each text and return a Series of category labels.

        Args:
            texts: pandas Series of strings to classify.

        Returns:
            pandas Series of category label strings, same length as input.
        """

    def predict_batch(self, texts: pd.Series, batch_size: int = 100) -> pd.Series:
        """
        Default batched prediction. Backends can override for custom batching.
        Falls back to calling predict() on the full Series.
        """
        return self.predict(texts)
