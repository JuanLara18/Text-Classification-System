import re
from collections import defaultdict

class UniqueValueProcessor:
    """Fixed processor for unique value classification and mapping."""
    
    def __init__(self, logger):
        self.logger = logger
        self.value_map = None
    
    def prepare_unique_classification(self, texts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
        """Fixed: Better normalization and unique value extraction."""
        self.logger.info(f"Processing {len(texts)} texts for unique value extraction")
        
        value_to_indices = defaultdict(list)
        
        for i, text in enumerate(texts):
            # Fixed: Comprehensive normalization
            normalized_text = self._normalize_text(text)
            value_to_indices[normalized_text].append(i)
        
        # Extract unique values, preserving original formatting
        unique_texts = []
        self.value_map = {}
        
        for normalized_text, indices in value_to_indices.items():
            if normalized_text:  # Skip empty
                # Use first occurrence as canonical form
                original_text = texts[indices[0]]
                unique_texts.append(original_text)
                self.value_map[original_text] = indices
        
        # Handle empty values
        if "" in value_to_indices:
            self.value_map[""] = value_to_indices[""]
        
        reduction_ratio = len(unique_texts) / len(texts) if texts else 0
        self.logger.info(f"Reduced {len(texts)} texts to {len(unique_texts)} unique values "
                        f"({reduction_ratio:.2%} reduction)")
        
        return unique_texts, self.value_map
    
    def _normalize_text(self, text) -> str:
        """Fixed: Consistent text normalization."""
        if text is None or pd.isna(text):
            return ""
        
        # Convert to string and normalize
        text_str = str(text)
        
        # Fixed: Comprehensive normalization
        # Remove extra whitespace but preserve single spaces
        normalized = re.sub(r'\s+', ' ', text_str.strip())
        
        # Convert to lowercase for comparison (but preserve original case in storage)
        normalized = normalized.lower()
        
        return normalized if normalized else ""
    
    def map_results_to_original(self, unique_results: List[str], original_length: int) -> List[str]:
        """Map classification results back to original dataset."""
        if not self.value_map:
            raise ValueError("Must call prepare_unique_classification first")
        
        results = ["Unknown"] * original_length
        
        # Map unique results back
        for i, (unique_text, classification) in enumerate(zip(self.value_map.keys(), unique_results)):
            if unique_text in self.value_map:
                indices = self.value_map[unique_text]
                for idx in indices:
                    if idx < original_length:  # Safety check
                        results[idx] = classification
        
        return results
