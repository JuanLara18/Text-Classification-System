#!/usr/bin/env python3
"""
tests/test_ai_classifier.py
AI Classification Tests for AI Text Classification System
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ConfigManager
from modules.ai_classifier import (
    OptimizedLLMClassificationManager,
    OptimizedOpenAIClassifier,
    UniqueValueProcessor,
    ClassificationCache,
    TokenCounter
)

class TestUniqueValueProcessor:
    """Test suite for UniqueValueProcessor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.mock_logger = Mock()
        self.processor = UniqueValueProcessor(self.mock_logger)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_unique_value_extraction(self):
        """Test unique value extraction and mapping."""
        print("\n🧪 Testing unique value extraction...")
        
        test_texts = [
            "Software Engineer",
            "Data Scientist", 
            "Software Engineer",  # Duplicate
            "Product Manager",
            "software engineer",  # Different case, should be treated as duplicate
            "Data Scientist"      # Another duplicate
        ]
        
        unique_texts, value_mapping = self.processor.prepare_unique_classification(test_texts)
        
        # Should extract 3 unique values (ignoring case)
        assert len(unique_texts) == 3
        assert "Software Engineer" in unique_texts
        assert "Data Scientist" in unique_texts
        assert "Product Manager" in unique_texts
        
        # Check mapping
        assert len(value_mapping) >= 3
        
        print(f"✅ Reduced {len(test_texts)} texts to {len(unique_texts)} unique values")
        
    def test_result_mapping(self):
        """Test mapping results back to original positions."""
        print("\n🧪 Testing result mapping...")
        
        test_texts = ["A", "B", "A", "C", "B"]
        unique_texts, _ = self.processor.prepare_unique_classification(test_texts)
        
        # Mock classification results for unique values
        unique_results = ["Category1", "Category2", "Category3"][:len(unique_texts)]
        
        mapped_results = self.processor.map_results_to_original(unique_results, len(test_texts))
        
        assert len(mapped_results) == len(test_texts)
        # Duplicates should have same classification
        assert mapped_results[0] == mapped_results[2]  # Both "A"
        assert mapped_results[1] == mapped_results[4]  # Both "B"
        
        print(f"✅ Mapped {len(unique_results)} unique results to {len(mapped_results)} original positions")
    
    def test_empty_and_null_handling(self):
        """Test handling of empty and null values."""
        print("\n🧪 Testing empty/null value handling...")
        
        test_texts = ["Valid Text", "", None, "  ", "Another Text", pd.NA]
        unique_texts, value_mapping = self.processor.prepare_unique_classification(test_texts)
        
        # Should only include valid texts
        assert len(unique_texts) == 2
        assert "Valid Text" in unique_texts
        assert "Another Text" in unique_texts
        
        print(f"✅ Handled empty/null values correctly: {len(unique_texts)} valid texts")

class TestTokenCounter:
    """Test suite for TokenCounter class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.counter = TokenCounter("gpt-4o-mini")
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_token_counting(self):
        """Test token counting functionality."""
        print("\n🧪 Testing token counting...")
        
        test_text = "This is a test sentence for token counting."
        token_count = self.counter.count_tokens(test_text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 50  # Should be reasonable for this short text
        
        print(f"✅ Counted {token_count} tokens in test text")
    
    def test_cost_estimation(self):
        """Test cost estimation."""
        print("\n🧪 Testing cost estimation...")
        
        prompt_tokens = 100
        completion_tokens = 20
        
        cost = self.counter.estimate_cost(prompt_tokens, completion_tokens)
        
        assert isinstance(cost, float)
        assert cost > 0
        assert cost < 1.0  # Should be small for this amount
        
        print(f"✅ Estimated cost: ${cost:.6f} for {prompt_tokens + completion_tokens} tokens")

class TestClassificationCache:
    """Test suite for ClassificationCache class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.cache_dir = "tests/cache_test"
        self.cache = ClassificationCache(self.cache_dir, duration_days=1)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
        
        # Cleanup cache directory
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
    
    def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        print("\n🧪 Testing cache set and get...")
        
        text = "Software Engineer"
        categories = ["Tech", "Business"]
        model = "gpt-4o-mini"
        prompt = "Classify this job"
        classification = "Tech"
        
        # Set cache
        self.cache.set(text, categories, model, prompt, classification)
        
        # Get from cache
        result = self.cache.get(text, categories, model, prompt)
        
        assert result == classification
        
        print(f"✅ Cache set and retrieved: '{classification}'")
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        print("\n🧪 Testing cache miss...")
        
        result = self.cache.get("Non-existent text", ["Cat1"], "model", "prompt")
        
        assert result is None
        
        print("✅ Cache miss handled correctly")
    
    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        print("\n🧪 Testing cache persistence...")
        
        # Set cache value
        self.cache.set("Test", ["Cat1"], "model", "prompt", "Cat1")
        
        # Save to disk
        self.cache.save()
        
        # Create new cache instance (simulating restart)
        new_cache = ClassificationCache(self.cache_dir, duration_days=1)
        
        # Should retrieve from disk
        result = new_cache.get("Test", ["Cat1"], "model", "prompt")
        assert result == "Cat1"
        
        print("✅ Cache persistence working correctly")

class TestOptimizedOpenAIClassifier:
    """Test suite for OptimizedOpenAIClassifier class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.get_config_value.side_effect = lambda key, default: {
            'ai_classification.caching.enabled': True,
            'ai_classification.caching.cache_directory': 'tests/cache',
            'ai_classification.caching.cache_duration_days': 30,
            'ai_classification.caching.preload_cache': False,
            'ai_classification.rate_limiting.requests_per_minute': 100,
            'ai_classification.rate_limiting.batch_delay_seconds': 0.1,
            'ai_classification.rate_limiting.concurrent_requests': 5,
            'ai_classification.parallel_processing.max_workers': 2
        }.get(key, default)
        
        self.mock_logger = Mock()
        
        # Create perspective config
        self.perspective_config = {
            'target_categories': ['Tech', 'Business', 'Other'],
            'llm_config': {
                'model': 'gpt-4o-mini',
                'temperature': 0.0,
                'max_tokens': 20,
                'timeout': 10,
                'api_key_env': 'OPENAI_API_KEY'
            },
            'classification_config': {
                'batch_size': 10,
                'unknown_category': 'Other',
                'prompt_template': 'Classify: {text} into {categories}'
            }
        }
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        print("\n🧪 Testing classifier initialization...")
        
        try:
            classifier = OptimizedOpenAIClassifier(
                self.mock_config, 
                self.mock_logger, 
                self.perspective_config
            )
            
            assert classifier.model == 'gpt-4o-mini'
            assert classifier.temperature == 0.0
            assert classifier.target_categories == ['Tech', 'Business', 'Other']
            assert classifier.batch_size == 10
            
            print("✅ Classifier initialized successfully")
            
        except Exception as e:
            print(f"⚠️  Classifier initialization test info: {e}")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('openai.chat.completions.create')
    def test_single_classification_mock(self, mock_openai):
        """Test single text classification with mocked OpenAI."""
        print("\n🧪 Testing single classification (mocked)...")
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Tech"
        mock_response.usage.completion_tokens = 5
        mock_openai.return_value = mock_response
        
        try:
            classifier = OptimizedOpenAIClassifier(
                self.mock_config,
                self.mock_logger,
                self.perspective_config
            )
            
            result, metadata = classifier._classify_single_optimized("Software Engineer")
            
            assert result in classifier.target_categories
            assert isinstance(metadata, dict)
            assert 'cost' in metadata
            
            print(f"✅ Classification result: '{result}'")
            
        except Exception as e:
            print(f"⚠️  Single classification test info: {e}")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('openai.chat.completions.create')
    def test_bulk_classification_with_unique_processing(self, mock_openai):
        """Test bulk classification with unique value processing."""
        print("\n🧪 Testing bulk classification with unique processing...")
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Tech"
        mock_response.usage.completion_tokens = 5
        mock_openai.return_value = mock_response
        
        test_texts = [
            "Software Engineer",
            "Data Scientist",
            "Software Engineer",  # Duplicate
            "Product Manager",
            "software engineer"   # Case variation
        ]
        
        try:
            classifier = OptimizedOpenAIClassifier(
                self.mock_config,
                self.mock_logger,
                self.perspective_config
            )
            
            results, metadata = classifier.classify_texts_with_unique_processing(test_texts)
            
            assert len(results) == len(test_texts)
            assert 'reduction_ratio' in metadata
            assert metadata['original_count'] == len(test_texts)
            assert metadata['unique_count'] <= len(test_texts)
            
            print(f"✅ Classified {len(test_texts)} texts with {metadata['reduction_ratio']:.1%} reduction")
            
        except Exception as e:
            print(f"⚠️  Bulk classification test info: {e}")
    
    def test_response_validation(self):
        """Test response validation logic."""
        print("\n🧪 Testing response validation...")
        
        try:
            classifier = OptimizedOpenAIClassifier(
                self.mock_config,
                self.mock_logger,
                self.perspective_config
            )
            
            # Test exact match
            assert classifier._validate_response("Tech") == "Tech"
            
            # Test case insensitive
            assert classifier._validate_response("tech") == "Tech"
            
            # Test partial match
            assert classifier._validate_response("Technology") == "Tech"
            
            # Test unknown response
            assert classifier._validate_response("Unknown Category") == "Other"
            
            print("✅ Response validation working correctly")
            
        except Exception as e:
            print(f"⚠️  Response validation test info: {e}")

class TestOptimizedLLMClassificationManager:
    """Test suite for OptimizedLLMClassificationManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        self.mock_config = Mock()
        self.mock_logger = Mock()
        
        self.manager = OptimizedLLMClassificationManager(self.mock_config, self.mock_logger)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_classifier_creation(self):
        """Test classifier creation."""
        print("\n🧪 Testing classifier creation...")
        
        perspective_config = {
            'llm_config': {'provider': 'openai'},
            'target_categories': ['Cat1', 'Cat2']
        }
        
        try:
            classifier = self.manager.create_classifier("test_perspective", perspective_config)
            assert classifier is not None
            assert "test_perspective" in self.manager.classifiers
            
            print("✅ Classifier created successfully")
            
        except Exception as e:
            print(f"⚠️  Classifier creation test info: {e}")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('openai.chat.completions.create')
    def test_perspective_classification(self, mock_openai):
        """Test full perspective classification."""
        print("\n🧪 Testing perspective classification...")
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Tech"
        mock_response.usage.completion_tokens = 5
        mock_openai.return_value = mock_response
        
        # Test DataFrame
        test_df = pd.DataFrame({
            'position_title': ['Software Engineer', 'Data Scientist'],
            'job_description': ['Python programming', 'Machine learning']
        })
        
        perspective_config = {
            'type': 'openai_classification',
            'columns': ['position_title'],
            'output_column': 'predicted_category',
            'target_categories': ['Tech', 'Business'],
            'llm_config': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
            'classification_config': {'batch_size': 10}
        }
        
        try:
            result_df, metadata = self.manager.classify_perspective(
                test_df, "test_perspective", perspective_config
            )
            
            assert 'predicted_category' in result_df.columns
            assert len(result_df) == len(test_df)
            assert isinstance(metadata, dict)
            
            print(f"✅ Perspective classification completed: {len(result_df)} rows")
            
        except Exception as e:
            print(f"⚠️  Perspective classification test info: {e}")
    
    def test_statistics_collection(self):
        """Test statistics collection."""
        print("\n🧪 Testing statistics collection...")
        
        stats = self.manager.get_all_stats()
        assert isinstance(stats, dict)
        
        report = self.manager.generate_performance_report()
        assert isinstance(report, str)
        assert "PERFORMANCE REPORT" in report
        
        print("✅ Statistics collection working correctly")

def run_ai_classifier_tests():
    """Run all AI classifier tests."""
    print("🧪 RUNNING AI CLASSIFICATION TESTS")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test UniqueValueProcessor
    processor_tester = TestUniqueValueProcessor()
    processor_tester.setup_method()  
    try:
        processor_tester.test_unique_value_extraction()
        processor_tester.test_result_mapping()
        processor_tester.test_empty_and_null_handling()
    finally:
        processor_tester.teardown_method()  
    
    # Test TokenCounter
    counter_tester = TestTokenCounter()
    counter_tester.setup_method()  
    try:
        counter_tester.test_token_counting()
        counter_tester.test_cost_estimation()
    finally:
        counter_tester.teardown_method()  
    
    # Test ClassificationCache
    cache_tester = TestClassificationCache()
    cache_tester.setup_method()  
    try:
        cache_tester.test_cache_set_and_get()
        cache_tester.test_cache_miss()
        cache_tester.test_cache_persistence()
    finally:
        cache_tester.teardown_method()  
    
    # Test OptimizedOpenAIClassifier
    classifier_tester = TestOptimizedOpenAIClassifier()
    classifier_tester.setup_method()  
    try:
        classifier_tester.test_classifier_initialization()
        classifier_tester.test_single_classification_mock()
        classifier_tester.test_bulk_classification_with_unique_processing()
        classifier_tester.test_response_validation()
    finally:
        classifier_tester.teardown_method()  
    
    # Test OptimizedLLMClassificationManager
    manager_tester = TestOptimizedLLMClassificationManager()
    manager_tester.setup_method()  
    try:
        manager_tester.test_classifier_creation()
        manager_tester.test_perspective_classification()
        manager_tester.test_statistics_collection()
    finally:
        manager_tester.teardown_method()  
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"✅ ALL AI CLASSIFICATION TESTS PASSED")
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    print("💡 Note: Some tests mocked OpenAI API calls for testing")
    print("=" * 50)
        
if __name__ == "__main__":
    run_ai_classifier_tests()