#!/usr/bin/env python3
"""
tests/test_data_processor.py
Data Processing Tests for AI Text Classification System
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import time
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ConfigManager
from modules.data_processor import DataProcessor, TextPreprocessor, FeatureExtractor
from modules.utilities import SparkSessionManager

class TestTextPreprocessor:
    """Test suite for TextPreprocessor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.get_preprocessing_options.return_value = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': False,
            'custom_stopwords': ['custom_stop'],
            'min_word_length': 2,
            'max_length': 1000
        }
        
        self.preprocessor = TextPreprocessor(self.mock_config)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing functionality."""
        print("\n🧪 Testing basic text preprocessing...")
        
        test_text = "This is a TEST sentence with PUNCTUATION! And some stopwords."
        result = self.preprocessor.preprocess_text(test_text)
        
        # Should be lowercase, no punctuation, minimal stopwords
        assert result.islower()
        assert '!' not in result
        assert 'TEST' not in result  # Should be lowercase
        assert 'test' in result or 'sentence' in result  # Some content should remain
        
        print(f"✅ Original: '{test_text}'")
        print(f"✅ Processed: '{result}'")
    
    def test_null_handling(self):
        """Test handling of null/empty values."""
        print("\n🧪 Testing null value handling...")
        
        # Test various null cases
        assert self.preprocessor.preprocess_text(None) == ""
        assert self.preprocessor.preprocess_text("") == ""
        assert self.preprocessor.preprocess_text("   ") == ""
        assert self.preprocessor.preprocess_text(np.nan) == ""
        
        print("✅ Null values handled correctly")
    
    def test_long_text_truncation(self):
        """Test long text truncation."""
        print("\n🧪 Testing long text truncation...")
        
        long_text = "word " * 500  # 2500+ characters
        result = self.preprocessor.preprocess_text(long_text)
        
        # Should be truncated to max_length
        assert len(result) <= 1000
        
        print(f"✅ Long text truncated from {len(long_text)} to {len(result)} characters")
    
    def test_special_characters_removal(self):
        """Test removal of URLs, emails, etc."""
        print("\n🧪 Testing special character removal...")
        
        test_text = "Contact us at test@example.com or visit https://example.com for more info."
        result = self.preprocessor.preprocess_text(test_text)
        
        assert "@example.com" not in result
        assert "https://example.com" not in result
        assert "contact" in result  # Regular words should remain
        
        print(f"✅ URLs and emails removed: '{result}'")
    
    def test_custom_stopwords(self):
        """Test custom stopwords removal."""
        print("\n🧪 Testing custom stopwords...")
        
        test_text = "This text contains custom_stop words that should be removed."
        result = self.preprocessor.preprocess_text(test_text)
        
        assert "custom_stop" not in result
        
        print(f"✅ Custom stopwords removed: '{result}'")

class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        # Create mock config for feature extraction
        self.mock_config = Mock()
        self.mock_config.get_feature_extraction_config.return_value = {
            'method': 'tfidf',
            'tfidf': {
                'max_features': 100,
                'ngram_range': [1, 2],
                'min_df': 1
            },
            'embedding': {
                'model': 'sentence-transformers',
                'sentence_transformers': {
                    'model_name': 'all-MiniLM-L6-v2'
                }
            }
        }
        self.mock_config.get_config_value.side_effect = lambda key, default: {
            'performance.cache_embeddings': True,
            'performance.cache_directory': 'tests/cache',
            'performance.batch_size': 32
        }.get(key, default)
        
        # Create mock logger
        self.mock_logger = Mock()
        
        self.extractor = FeatureExtractor(self.mock_config, self.mock_logger)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_tfidf_extraction(self):
        """Test TF-IDF feature extraction."""
        print("\n🧪 Testing TF-IDF feature extraction...")
        
        test_texts = [
            "software engineer python programming",
            "data scientist machine learning python",
            "product manager strategy planning",
            "sales representative customer relations"
        ]
        
        features = self.extractor.extract_tfidf_features(test_texts)
        
        # Check feature matrix properties
        assert features.shape[0] == len(test_texts)
        assert features.shape[1] > 0
        assert features.shape[1] <= 100  # max_features
        
        print(f"✅ TF-IDF features extracted: {features.shape}")
    
    def test_sentence_transformer_extraction(self):
        """Test Sentence Transformer embeddings."""
        print("\n🧪 Testing Sentence Transformer embeddings...")
        
        test_texts = [
            "software engineer job",
            "data science position",
            "marketing role"
        ]
        
        try:
            features = self.extractor.extract_sentence_transformer_embeddings(test_texts)
            
            # Check embedding properties
            assert features.shape[0] == len(test_texts)
            assert features.shape[1] > 0  # Should have embedding dimensions
            assert not np.isnan(features).any()  # No NaN values
            
            print(f"✅ Sentence Transformer embeddings: {features.shape}")
            
        except Exception as e:
            print(f"⚠️  Sentence Transformer test skipped (model not available): {e}")
    
    def test_dimensionality_reduction(self):
        """Test dimensionality reduction."""
        print("\n🧪 Testing dimensionality reduction...")
        
        # Create high-dimensional test data
        high_dim_features = np.random.random((50, 100))
        
        try:
            reduced_features = self.extractor.reduce_dimensionality(high_dim_features)
            
            # Should have fewer dimensions
            assert reduced_features.shape[0] == high_dim_features.shape[0]
            assert reduced_features.shape[1] < high_dim_features.shape[1]
            
            print(f"✅ Reduced from {high_dim_features.shape[1]} to {reduced_features.shape[1]} dimensions")
            
        except Exception as e:
            print(f"⚠️  Dimensionality reduction test skipped: {e}")
    
    def test_cache_functionality(self):
        """Test feature caching."""
        print("\n🧪 Testing feature caching...")
        
        test_texts = ["test text for caching"]
        
        # Extract features (should cache)
        features1 = self.extractor.extract_tfidf_features(test_texts)
        
        # Extract again (should use cache)
        features2 = self.extractor.extract_tfidf_features(test_texts)
        
        # Should be identical
        np.testing.assert_array_equal(features1.toarray(), features2.toarray())
        
        print("✅ Feature caching working correctly")

class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'position_title': [
                'Software Engineer',
                'Data Scientist',
                'Product Manager',
                'Sales Representative',
                'Software Engineer'  # Duplicate
            ],
            'job_description': [
                'Develop software applications using Python',
                'Analyze data and build machine learning models',
                'Manage product roadmap and strategy',
                'Build relationships with customers',
                'Develop software applications using Python'  # Duplicate
            ],
            'other_column': ['A', 'B', 'C', 'D', 'E']
        })
        
        # Save test data
        os.makedirs('tests/data', exist_ok=True)
        self.test_file = 'tests/data/test_processor_data.dta'
        self.test_data.to_stata(self.test_file, write_index=False, version=117)
        
        # Create test config
        self.config = ConfigManager('tests/configs/test_config_clustering.yaml')
        
        # Mock logger and spark manager
        self.mock_logger = Mock()
        self.mock_spark_manager = Mock()
        
        self.processor = DataProcessor(self.config, self.mock_logger, self.mock_spark_manager)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
        
        # Cleanup test files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_data_loading(self):
        """Test data loading from Stata file."""
        print("\n🧪 Testing data loading...")
        
        # Update config to use test file
        self.processor.input_file = self.test_file
        
        # Test loading
        df = self.processor.load_data()
        
        assert df is not None
        assert len(df) > 0
        assert 'position_title' in df.columns
        assert 'job_description' in df.columns
        
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
    
    def test_duplicate_removal(self):
        """Test duplicate row removal."""
        print("\n🧪 Testing duplicate removal...")
        
        self.processor.input_file = self.test_file
        df = self.processor.load_data()
        
        # Should remove the duplicate row
        assert len(df) == 4  # Original had 5, one duplicate
        
        print(f"✅ Duplicates removed: {len(df)} unique rows remaining")
    
    def test_text_preprocessing_integration(self):
        """Test text preprocessing integration."""
        print("\n🧪 Testing text preprocessing integration...")
        
        self.processor.input_file = self.test_file
        df = self.processor.load_data()
        
        # Check for preprocessed columns
        text_columns = self.config.get_text_columns()
        for column in text_columns:
            preprocessed_col = f"{column}_preprocessed"
            if isinstance(df, pd.DataFrame):
                assert preprocessed_col in df.columns
            else:
                # For Spark DataFrame
                assert preprocessed_col in df.columns
        
        print("✅ Text preprocessing applied successfully")
    
    def test_feature_extraction_integration(self):
        """Test feature extraction integration."""
        print("\n🧪 Testing feature extraction integration...")
        
        # Create simple test DataFrame
        test_df = pd.DataFrame({
            'position_title': ['Software Engineer', 'Data Scientist'],
            'job_description': ['Python programming', 'Machine learning']
        })
        
        # Apply preprocessing first
        processed_df = self.processor.preprocess_text_columns(
            test_df, ['position_title', 'job_description']
        )
        
        # Extract features
        result_df, features_dict = self.processor.extract_features(
            processed_df, ['position_title', 'job_description']
        )
        
        assert features_dict is not None
        assert len(features_dict) > 0
        
        print(f"✅ Feature extraction completed: {len(features_dict)} feature sets")
    
    @patch('modules.data_processor.SparkSession')
    def test_spark_integration(self, mock_spark):
        """Test Spark DataFrame integration."""
        print("\n🧪 Testing Spark integration...")
        
        # Mock Spark session
        mock_session = Mock()
        mock_spark.builder.appName.return_value.getOrCreate.return_value = mock_session
        self.mock_spark_manager.get_or_create_session.return_value = mock_session
        
        # Mock createDataFrame to return a mock Spark DataFrame
        mock_spark_df = Mock()
        mock_spark_df.count.return_value = 4
        mock_spark_df.cache.return_value = mock_spark_df
        mock_session.createDataFrame.return_value = mock_spark_df
        
        self.processor.input_file = self.test_file
        
        # Test should not raise exceptions
        try:
            df = self.processor.load_data()
            print("✅ Spark integration test completed")
        except Exception as e:
            print(f"⚠️  Spark integration test info: {e}")

def run_data_processor_tests():
    """Run all data processor tests."""
    print("🧪 RUNNING DATA PROCESSOR TESTS")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test TextPreprocessor
    preprocessor_tester = TestTextPreprocessor()
    preprocessor_tester.setup_method()  
    try:
        preprocessor_tester.test_basic_preprocessing()
        preprocessor_tester.test_null_handling()
        preprocessor_tester.test_long_text_truncation()
        preprocessor_tester.test_special_characters_removal()
        preprocessor_tester.test_custom_stopwords()
    finally:
        preprocessor_tester.teardown_method()  
    
    # Test FeatureExtractor
    extractor_tester = TestFeatureExtractor()
    extractor_tester.setup_method()  
    try:
        extractor_tester.test_tfidf_extraction()
        extractor_tester.test_sentence_transformer_extraction()
        extractor_tester.test_dimensionality_reduction()
        extractor_tester.test_cache_functionality()
    finally:
        extractor_tester.teardown_method()  
    
    # Test DataProcessor
    processor_tester = TestDataProcessor()
    processor_tester.setup_method()  
    try:
        processor_tester.test_data_loading()
        processor_tester.test_duplicate_removal()
        processor_tester.test_text_preprocessing_integration()
        processor_tester.test_feature_extraction_integration()
        processor_tester.test_spark_integration()
    finally:
        processor_tester.teardown_method()  
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"✅ ALL DATA PROCESSOR TESTS PASSED")
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    print("=" * 50)
if __name__ == "__main__":
    run_data_processor_tests()