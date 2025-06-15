# modules/ai_classifier.py
"""
AI-powered classification module using OpenAI and other LLM providers.
Integrates seamlessly with the existing Text Classification System.
"""

import os
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pickle
import pandas as pd
from collections import defaultdict, Counter
import openai
import tiktoken
import threading

import concurrent.futures
import threading

class TokenCounter:
    """Utility class for counting and managing OpenAI tokens with updated pricing."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost with updated pricing (December 2024)."""
        pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # Current pricing
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-0125": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.03, "output": 0.06},
        }
        
        model_pricing = pricing.get(self.model_name, pricing["gpt-4o-mini"])
        
        input_cost = (prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (completion_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost





class UniqueValueProcessor:
    """Processor that handles unique value classification and mapping."""
    
    def __init__(self, logger):
        self.logger = logger
        self.unique_values = None
        self.value_map = None
        self.reverse_map = None
    
    def prepare_unique_classification(self, texts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        Extract unique values and create mapping for efficient classification.
        FIXED: Less aggressive normalization to preserve meaningful distinctions.
        """
        self.logger.info(f"Processing {len(texts)} texts for unique value extraction")
        
        # Create mapping of unique values to their indices
        value_to_indices = defaultdict(list)
        for i, text in enumerate(texts):
            # FIXED: More conservative normalization - only handle None/NaN, keep original case and content
            if text is None or pd.isna(text):
                normalized_text = ""
            else:
                # Only strip whitespace, keep original case and punctuation
                normalized_text = str(text).strip()
                # Only treat as empty if it's actually empty after stripping
                if not normalized_text:
                    normalized_text = ""
            
            value_to_indices[normalized_text].append(i)
        
        # Get unique values (preserve original case and content)
        unique_texts = []
        self.value_map = {}
        
        for normalized_text, indices in value_to_indices.items():
            if normalized_text:  # Skip empty strings
                # Use the first occurrence as the canonical form (preserves original formatting)
                original_text = texts[indices[0]]
                unique_texts.append(original_text)
                self.value_map[original_text] = indices
        
        # Handle empty/null values
        if "" in value_to_indices:
            empty_indices = value_to_indices[""]
            self.value_map[""] = empty_indices
        
        reduction_ratio = len(unique_texts) / len(texts) if texts else 0
        self.logger.info(f"Reduced {len(texts)} texts to {len(unique_texts)} unique values "
                        f"({reduction_ratio:.2%} reduction)")
        
        return unique_texts, self.value_map
    
    def map_results_to_original(self, unique_results: List[str], original_length: int) -> List[str]:
        """Map classification results from unique values back to original dataset."""
        if not self.value_map:
            raise ValueError("Must call prepare_unique_classification first")
        
        # Initialize result array
        results = ["Unknown"] * original_length
        
        # Map unique results back to original positions
        for i, (unique_text, classification) in enumerate(zip(self.value_map.keys(), unique_results)):
            if unique_text in self.value_map:
                indices = self.value_map[unique_text]
                for idx in indices:
                    results[idx] = classification
        
        self.logger.info(f"Mapped {len(unique_results)} unique results to {original_length} original positions")
        return results


class OptimizedOpenAIClassifier:
    """Optimized OpenAI classifier with unique value processing and high-performance settings."""
    
    def __init__(self, config, logger, perspective_config):
        self.config = config
        self.logger = logger
        self.perspective_config = perspective_config
        
        # Initialize unique value processor
        self.unique_processor = UniqueValueProcessor(logger)
        
        # Extract configuration
        self.llm_config = perspective_config.get('llm_config', {})
        self.classification_config = perspective_config.get('classification_config', {})
        self.validation_config = perspective_config.get('validation', {})
        
        # LLM settings - OPTIMIZED for speed
        self.model = self.llm_config.get('model', 'gpt-4o-mini')
        self.temperature = self.llm_config.get('temperature', 0.0)
        self.max_tokens = self.llm_config.get('max_tokens', 20)  # Reduced for faster processing
        self.timeout = self.llm_config.get('timeout', 10)       # Reduced timeout
        self.max_retries = self.llm_config.get('max_retries', 2)
        self.backoff_factor = self.llm_config.get('backoff_factor', 1.5)
        
        # OPTIMIZED Classification settings
        self.target_categories = perspective_config.get('target_categories', [])
        self.unknown_category = self.classification_config.get('unknown_category', 'Other/Unknown')
        
        # ENHANCED batch processing for high-RAM systems
        base_batch_size = self.classification_config.get('batch_size', 50)
        self.batch_size = min(base_batch_size * 4, 200)  # Scale up for unique processing
        
        # Enhanced prompt template for faster processing
        self.prompt_template = self.classification_config.get('prompt_template', self._optimized_prompt_template())
        
        # Add unknown category if not present
        if (self.classification_config.get('include_unknown_in_categories', True) and 
            self.unknown_category not in self.target_categories):
            self.target_categories.append(self.unknown_category)
        
        # Validation settings
        self.strict_matching = self.validation_config.get('strict_category_matching', False)
        self.fallback_strategy = self.validation_config.get('fallback_strategy', 'unknown')
        
        # Initialize components
        self.token_counter = TokenCounter(self.model)
        
        # Enhanced cache configuration
        cache_config = config.get_config_value('ai_classification.caching', {})
        if cache_config.get('enabled', True):
            cache_dir = cache_config.get('cache_directory', 'ai_cache')
            cache_duration = cache_config.get('cache_duration_days', 365)  # Longer cache
            self.cache = ClassificationCache(cache_dir, cache_duration)
            
            # Pre-load cache into memory if configured
            if cache_config.get('preload_cache', True):
                self._preload_cache()
        else:
            self.cache = None
        
        # Cost and performance tracking
        self.total_cost = 0.0
        self.total_tokens = {'prompt': 0, 'completion': 0}
        self.api_calls = 0
        self.cache_hits = 0
        
        # ENHANCED rate limiting for parallel processing
        rate_config = config.get_config_value('ai_classification.rate_limiting', {})
        self.max_requests_per_minute = rate_config.get('requests_per_minute', 1000)
        self.batch_delay = rate_config.get('batch_delay_seconds', 0.02)  # Reduced delay
        self.concurrent_requests = rate_config.get('concurrent_requests', 15)  # Higher concurrency
        
        # Request timing
        self.request_times = []
        self.request_lock = threading.Lock()
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        self.logger.info(f"Optimized OpenAI Classifier initialized with model {self.model}")
        self.logger.info(f"Batch size: {self.batch_size}, Concurrent requests: {self.concurrent_requests}")
        self.logger.info(f"Target categories: {len(self.target_categories)} categories")
    
    def _optimized_prompt_template(self) -> str:
        """Optimized prompt template for faster processing."""
        return """Classify this job position into ONE category from the list.

Categories:
{categories}

Job Position: "{text}"

Answer with the category name ONLY."""
    
    def _preload_cache(self):
        """Pre-load cache into memory for faster access."""
        try:
            if self.cache and hasattr(self.cache, 'cache'):
                cache_size = len(self.cache.cache)
                self.logger.info(f"Pre-loaded {cache_size} cached classifications")
        except Exception as e:
            self.logger.warning(f"Could not preload cache: {e}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client with API key."""
        api_key_env = self.llm_config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.environ.get(api_key_env)
        
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        openai.api_key = api_key
    
    def _build_prompt(self, text: str) -> str:
        """Build optimized classification prompt."""
        # Create shorter category list for faster processing
        categories_str = ", ".join(self.target_categories)
        
        return self.prompt_template.format(
            categories=categories_str,
            text=text,
            unknown_category=self.unknown_category
        )
    
    def _validate_response(self, response: str) -> str:
        """Fixed: More precise response validation."""
        response = response.strip()
        response_lower = response.lower()
        
        # Fixed: Exact match first (most reliable)
        for category in self.target_categories:
            if response_lower == category.lower():
                return category
        
        # Fixed: Stricter partial matching
        for category in self.target_categories:
            cat_lower = category.lower()
            # Only match if response is clearly contained in category or vice versa
            if (len(response_lower) > 3 and cat_lower in response_lower) or \
               (len(cat_lower) > 3 and response_lower in cat_lower):
                return category
        
        # Fixed: Word-based matching for multi-word categories
        response_words = set(response_lower.split())
        for category in self.target_categories:
            cat_words = set(category.lower().split())
            # Match if significant word overlap
            if len(cat_words & response_words) >= max(1, len(cat_words) // 2):
                return category
        
        return self.unknown_category
     
    def _smart_rate_limit(self):
        """Fixed: Efficient rate limiting using deque."""
        current_time = time.time()
        
        with self.request_lock:
            # Fixed: Use deque for efficient cleanup
            if not hasattr(self, '_request_deque'):
                from collections import deque
                self._request_deque = deque()
            
            # Remove old requests (older than 1 minute)
            cutoff_time = current_time - 60
            while self._request_deque and self._request_deque[0] <= cutoff_time:
                self._request_deque.popleft()
            
            # Check rate limit
            if len(self._request_deque) >= self.max_requests_per_minute:
                sleep_time = 60 - (current_time - self._request_deque[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Add current request
            self._request_deque.append(current_time)
    
    def _classify_single_optimized(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Fixed: Better error handling and retry logic."""
        if not text or pd.isna(text) or str(text).strip() == "":
            return self.unknown_category, {'cached': False, 'cost': 0, 'tokens': 0}
        
        text = str(text).strip()
        
        # Check cache
        if self.cache:
            cached_result = self.cache.get(text, self.target_categories, self.model, self.prompt_template)
            if cached_result:
                self.cache_hits += 1
                return cached_result, {'cached': True, 'cost': 0, 'tokens': 0}
        
        prompt = self._build_prompt(text)
        prompt_tokens = self.token_counter.count_tokens(prompt)
        
        self._smart_rate_limit()
        
        # Fixed: Better error handling with specific retry logic
        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Classify quickly and accurately. Respond with only the category name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                
                classification = response.choices[0].message.content.strip()
                validated_classification = self._validate_response(classification)
                
                # Track usage
                completion_tokens = response.usage.completion_tokens
                cost = self.token_counter.estimate_cost(prompt_tokens, completion_tokens)
                
                self.total_tokens['prompt'] += prompt_tokens
                self.total_tokens['completion'] += completion_tokens
                self.total_cost += cost
                self.api_calls += 1
                
                # Cache result
                if self.cache:
                    self.cache.set(text, self.target_categories, self.model, self.prompt_template, validated_classification)
                
                return validated_classification, {
                    'cached': False, 'cost': cost, 'tokens': prompt_tokens + completion_tokens,
                    'raw_response': classification, 'validated_response': validated_classification
                }
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Fixed: Don't retry on permanent errors
                if any(perm_error in error_str for perm_error in 
                       ['invalid_api_key', 'insufficient_quota', 'model_not_found', 'invalid_request']):
                    self.logger.error(f"Permanent API error: {e}")
                    return self.unknown_category, {'cached': False, 'cost': 0, 'tokens': 0, 'error': str(e)}
                
                # Retry on temporary errors
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    self.logger.warning(f"Temporary API error (attempt {attempt + 1}): {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API call failed after {self.max_retries} attempts: {e}")
                    return self.unknown_category, {'cached': False, 'cost': 0, 'tokens': 0, 'error': str(e)}

    def classify_texts_with_unique_processing(self, texts: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        MAIN OPTIMIZATION: Classify texts using unique value processing and parallel execution.
        This is the key performance improvement that will dramatically speed up processing.
        """
        self.logger.info(f"Starting optimized classification of {len(texts)} texts")
        start_time = time.time()
        
        # Step 1: Extract unique values (MAJOR SPEEDUP)
        unique_texts, value_mapping = self.unique_processor.prepare_unique_classification(texts)
        
        if len(unique_texts) == 0:
            self.logger.warning("No valid texts to classify")
            return [self.unknown_category] * len(texts), {'total_cost': 0, 'processing_time': 0}
        
        # Step 2: Classify only unique values using parallel processing
        unique_classifications, classification_metadata = self._classify_unique_parallel(unique_texts)
        
        # Step 3: Map results back to original dataset
        final_classifications = self.unique_processor.map_results_to_original(unique_classifications, len(texts))
        
        # Compile final metadata
        processing_time = time.time() - start_time
        final_metadata = {
            **classification_metadata,
            'original_count': len(texts),
            'unique_count': len(unique_texts),
            'reduction_ratio': len(unique_texts) / len(texts) if texts else 0,
            'processing_time': processing_time,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / len(unique_texts) if unique_texts else 0,
            'classification_distribution': dict(Counter(final_classifications))
        }
        
        self.logger.info(f"Optimized classification completed in {processing_time:.2f}s")
        self.logger.info(f"Processed {len(unique_texts)} unique values from {len(texts)} total texts")
        self.logger.info(f"Cost: ${final_metadata['total_cost']:.4f}, Cache hit rate: {final_metadata['cache_hit_rate']:.1%}")
        
        # Save cache
        if self.cache:
            self.cache.save()
        
        return final_classifications, final_metadata
    
    def _classify_unique_parallel(self, unique_texts: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Classify unique texts using optimized parallel processing."""
        # Get parallel config - OPTIMIZED for high-CPU systems
        parallel_config = self.config.get_config_value('ai_classification.parallel_processing', {})
        max_workers = min(parallel_config.get('max_workers', 10), len(unique_texts))
        
        classifications = [None] * len(unique_texts)
        metadata = {
            'total_cost': 0,
            'total_tokens': 0,
            'cached_responses': 0,
            'api_calls': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Thread-safe counters
        lock = threading.Lock()
        
        def classify_batch(batch_data):
            batch_idx, batch_texts = batch_data
            batch_results = []
            batch_meta = {'cost': 0, 'tokens': 0, 'cached': 0, 'calls': 0, 'errors': 0}
            
            for text in batch_texts:
                classification, item_meta = self._classify_single_optimized(text)
                batch_results.append(classification)
                
                # Update batch metadata
                batch_meta['cost'] += item_meta.get('cost', 0)
                batch_meta['tokens'] += item_meta.get('tokens', 0)
                if item_meta.get('cached', False):
                    batch_meta['cached'] += 1
                if not item_meta.get('error'):
                    batch_meta['calls'] += 1
                else:
                    batch_meta['errors'] += 1
            
            return batch_idx, batch_results, batch_meta
        
        # Create optimized batches
        batches = []
        for i in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[i:i + self.batch_size]
            batches.append((i, batch))
        
        # Process with optimized thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(classify_batch, batch): batch for batch in batches}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx, batch_results, batch_meta = future.result()
                
                # Store results
                for i, result in enumerate(batch_results):
                    classifications[batch_idx + i] = result
                
                # Update metadata
                with lock:
                    metadata['total_cost'] += batch_meta['cost']
                    metadata['total_tokens'] += batch_meta['tokens']
                    metadata['cached_responses'] += batch_meta['cached']
                    metadata['api_calls'] += batch_meta['calls']
                    metadata['errors'] += batch_meta['errors']
                    
                    completed += 1
                    if completed % 5 == 0:  # More frequent updates
                        progress = completed / len(batches) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({completed}/{len(batches)} batches)")
        
        metadata['end_time'] = datetime.now().isoformat()
        return classifications, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive classifier statistics."""
        return {
            'model': self.model,
            'total_cost': self.total_cost,
            'total_tokens': self.total_tokens,
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'target_categories': self.target_categories,
            'batch_size': self.batch_size,
            'concurrent_requests': self.concurrent_requests,
            'cache_enabled': self.cache is not None
        }


import os
import json
import hashlib
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional


import os
import json
import hashlib
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ClassificationCache:
    """Fixed caching system for AI classification results."""
    
    def __init__(self, cache_dir: str, duration_days: int = 365):
        self.cache_dir = cache_dir
        self.duration_days = duration_days
        self.cache_file = os.path.join(cache_dir, "classification_cache.json")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Fixed: Use OrderedDict for LRU eviction
        self.memory_cache = OrderedDict()
        self.max_memory_cache = 10000
        
        # Fixed: Coarser locking strategy
        self._cache_lock = threading.RLock()
        
        # Load disk cache
        self.cache = self._load_cache()
        
        # Fixed: Track changes for reliable persistence
        self._unsaved_changes = 0
        self._last_save = time.time()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ClassificationCache initialized: {len(self.cache)} entries loaded")
    
    def _generate_key(self, text: str, categories: List[str], model: str, prompt: str) -> str:
        """Fixed: Generate consistent cache key including all parameters."""
        if not text or not categories:
            return ""
        
        # Fixed: Include all parameters and use full text hash
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        categories_hash = hashlib.md5('|'.join(sorted(categories)).encode('utf-8')).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]  # Short prompt hash
        
        # Create comprehensive key
        key_content = f"{text_hash}_{categories_hash}_{model}_{prompt_hash}"
        return hashlib.md5(key_content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, categories: List[str], model: str, prompt: str) -> Optional[str]:
        """Fixed: Get cached result with proper validation."""
        # Fixed: Validate all parameters
        if not text or not text.strip() or not categories or not model:
            return None
        
        key = self._generate_key(text, categories, model, prompt)
        if not key:
            return None
        
        # Fixed: Check memory cache with LRU update
        with self._cache_lock:
            if key in self.memory_cache:
                # Move to end (most recently used)
                value = self.memory_cache.pop(key)
                self.memory_cache[key] = value
                return value
        
        # Check disk cache
        with self._cache_lock:
            entry = self.cache.get(key)
            
        if entry and isinstance(entry, dict) and 'classification' in entry:
            try:
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if datetime.now() - entry_date < timedelta(days=self.duration_days):
                    classification = entry['classification']
                    
                    # Fixed: Add to memory cache with LRU eviction
                    self._add_to_memory_cache(key, classification)
                    return classification
            except (ValueError, TypeError, KeyError):
                pass
        
        return None
    
    def set(self, text: str, categories: List[str], model: str, prompt: str, classification: str):
        """Fixed: Cache result with reliable persistence."""
        if not text or not text.strip() or not classification or not categories:
            return
        
        key = self._generate_key(text, categories, model, prompt)
        if not key:
            return
        
        # Fixed: Add to memory cache with LRU eviction
        self._add_to_memory_cache(key, classification)
        
        # Add to disk cache
        with self._cache_lock:
            self.cache[key] = {
                'classification': classification,
                'timestamp': datetime.now().isoformat()
            }
            self._unsaved_changes += 1
            
            # Fixed: More frequent and reliable saving
            if (self._unsaved_changes >= 10 or 
                time.time() - self._last_save > 30):  # Save every 10 entries or 30 seconds
                self._save_cache()
    
    def _add_to_memory_cache(self, key: str, value: str):
        """Fixed: Add to memory cache with LRU eviction."""
        with self._cache_lock:
            # Remove if exists (to update position)
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            # Add new entry
            self.memory_cache[key] = value
            
            # Fixed: LRU eviction when full
            while len(self.memory_cache) > self.max_memory_cache:
                # Remove oldest (first) item
                self.memory_cache.popitem(last=False)
    
    def _save_cache(self):
        """Fixed: Reliable cache persistence."""
        if not self.cache:
            return
        
        try:
            with self._cache_lock:
                cache_copy = self.cache.copy()
                self._unsaved_changes = 0
                self._last_save = time.time()
            
            # Atomic write with backup
            temp_file = f"{self.cache_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_copy, f, ensure_ascii=False, separators=(',', ':'))
            
            # Atomic move
            os.replace(temp_file, self.cache_file)
            self.logger.debug(f"Cache saved: {len(cache_copy)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def save(self):
        """Explicitly save cache."""
        self._save_cache()
    
    def _load_cache(self) -> Dict:
        """Load existing cache with cleanup."""
        if not os.path.exists(self.cache_file):
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Clean expired entries
            cutoff_date = datetime.now() - timedelta(days=self.duration_days)
            cleaned_cache = {}
            
            for key, value in cache_data.items():
                if isinstance(value, dict) and 'timestamp' in value:
                    try:
                        entry_date = datetime.fromisoformat(value['timestamp'])
                        if entry_date > cutoff_date:
                            cleaned_cache[key] = value
                    except (ValueError, TypeError):
                        continue
            
            return cleaned_cache
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return {}
    
    
class OptimizedLLMClassificationManager:
    """Enhanced LLM Classification Manager with unique value processing."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classifiers = {}
        self.performance_stats = {}
    
    def create_classifier(self, perspective_name: str, perspective_config: Dict) -> OptimizedOpenAIClassifier:
        """Create an optimized classifier for a perspective."""
        provider = perspective_config.get('llm_config', {}).get('provider', 'openai')
        
        if provider == 'openai':
            classifier = OptimizedOpenAIClassifier(self.config, self.logger, perspective_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        self.classifiers[perspective_name] = classifier
        return classifier
    
    def classify_perspective(self, dataframe: pd.DataFrame, perspective_name: str, perspective_config: Dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fixed: Efficient DataFrame processing without iterrows."""
        self.logger.info(f"Applying OPTIMIZED LLM classification perspective: {perspective_name}")
        start_time = time.time()
        
        # Get or create classifier
        if perspective_name not in self.classifiers:
            classifier = self.create_classifier(perspective_name, perspective_config)
        else:
            classifier = self.classifiers[perspective_name]
        
        columns = perspective_config.get('columns', [])
        output_column = perspective_config.get('output_column', f"{perspective_name}_classification")
        
        # Fixed: Vectorized text combination instead of iterrows
        def combine_row_texts(row):
            text_parts = []
            for col in columns:
                if col in dataframe.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            return " | ".join(text_parts) if text_parts else ""
        
        # Fixed: Use apply instead of iterrows for better performance
        combined_texts = dataframe[columns].apply(combine_row_texts, axis=1).tolist()
        
        # Use optimized classification
        classifications, metadata = classifier.classify_texts_with_unique_processing(combined_texts)
        
        # Add results to dataframe
        result_df = dataframe.copy()
        result_df[output_column] = classifications
        
        # Enhanced metadata
        processing_time = time.time() - start_time
        enhanced_metadata = {
            **metadata,
            'perspective_name': perspective_name,
            'total_processing_time': processing_time,
            'columns_used': columns,
            'output_column': output_column
        }
        
        # Store performance stats
        self.performance_stats[perspective_name] = enhanced_metadata
        
        # Log performance summary
        self.logger.info(f"✅ Classification completed for {perspective_name}")
        self.logger.info(f"   📊 {metadata.get('original_count', 0):,} → {metadata.get('unique_count', 0):,} unique")
        self.logger.info(f"   ⚡ Reduction: {metadata.get('reduction_ratio', 0):.1%}")
        self.logger.info(f"   💰 Cost: ${metadata.get('total_cost', 0):.4f}")
        self.logger.info(f"   ⏱️  Time: {processing_time:.2f}s")
        
        return result_df, enhanced_metadata
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive statistics for all classifiers."""
        stats = {}
        for name, classifier in self.classifiers.items():
            classifier_stats = classifier.get_stats()
            performance_stats = self.performance_stats.get(name, {})
            
            stats[name] = {
                **classifier_stats,
                **performance_stats,
                'optimization_enabled': True,
                'unique_processing': True
            }
        
        return stats
    
    def generate_performance_report(self) -> str:
        """Generate a detailed performance report."""
        if not self.performance_stats:
            return "No performance data available."
        
        report_lines = [
            "🚀 OPTIMIZED AI CLASSIFICATION PERFORMANCE REPORT",
            "=" * 60,
            ""
        ]
        
        total_cost = 0
        total_original = 0
        total_unique = 0
        total_time = 0
        
        for perspective_name, stats in self.performance_stats.items():
            report_lines.extend([
                f"📋 Perspective: {perspective_name}",
                f"   • Original records: {stats.get('original_count', 0):,}",
                f"   • Unique values: {stats.get('unique_count', 0):,}",
                f"   • Reduction ratio: {stats.get('reduction_ratio', 0):.1%}",
                f"   • Cost: ${stats.get('total_cost', 0):.4f}",
                f"   • Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}",
                f"   • Processing time: {stats.get('total_processing_time', 0):.2f}s",
                ""
            ])
            
            total_cost += stats.get('total_cost', 0)
            total_original += stats.get('original_count', 0)
            total_unique += stats.get('unique_count', 0)
            total_time += stats.get('total_processing_time', 0)
        
        overall_reduction = (total_original - total_unique) / total_original if total_original > 0 else 0
        
        report_lines.extend([
            "📊 OVERALL SUMMARY:",
            f"   • Total cost: ${total_cost:.4f}",
            f"   • Total records: {total_original:,}",
            f"   • Unique processed: {total_unique:,}",
            f"   • Overall reduction: {overall_reduction:.1%}",
            f"   • Total time: {total_time:.2f}s",
            f"   • Records per second: {total_original/total_time:.1f}" if total_time > 0 else "",
            ""
        ])
        
        return "\n".join(report_lines)

    
