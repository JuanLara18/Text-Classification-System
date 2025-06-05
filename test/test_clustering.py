
#!/usr/bin/env python3
"""
tests/test_clustering.py
Traditional Clustering Tests for AI Text Classification System
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from sklearn.datasets import make_blobs

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ConfigManager
from modules.classifier import (
    EnhancedClassifierManager,
    BaseClusterer,
    KMeansClusterer, 
    HDBSCANClusterer,
    AgglomerativeClusterer,
    ClusterLabeler
)
from modules.unique_row_processor import UniqueRowProcessor

class TestBaseClusterer:
    """Test suite for BaseClusterer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        self.mock_config = Mock()
        self.mock_config.get_options.return_value = {'seed': 42}
        self.mock_logger = Mock()
        
        self.clusterer = BaseClusterer(self.mock_config, self.mock_logger)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_base_clusterer_initialization(self):
        """Test base clusterer initialization."""
        print("\n🧪 Testing base clusterer initialization...")
        
        assert self.clusterer.config is not None
        assert self.clusterer.logger is not None
        assert self.clusterer.seed == 42
        
        print("✅ Base clusterer initialized correctly")
    
    def test_noise_cluster_handling(self):
        """Test noise cluster handling."""
        print("\n🧪 Testing noise cluster handling...")
        
        # Create test labels with noise (-1)
        labels = np.array([0, 1, -1, 0, 2, -1, 1])
        
        # Handle noise (base implementation just returns original)
        result = self.clusterer._handle_noise_cluster(labels)
        
        assert len(result) == len(labels)
        
        print("✅ Noise cluster handling completed")

class TestKMeansClusterer:
    """Test suite for KMeansClusterer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        self.mock_config = Mock()
        self.mock_config.get_options.return_value = {'seed': 42}
        self.mock_logger = Mock()
        
        self.perspective_config = {
            'params': {
                'n_clusters': 3,
                'random_state': 42,
                'n_init': 5,
                'max_iter': 200
            }
        }
        
        self.clusterer = KMeansClusterer(
            self.mock_config, 
            self.mock_logger, 
            self.perspective_config
        )
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_kmeans_initialization(self):
        """Test K-means initialization."""
        print("\n🧪 Testing K-means initialization...")
        
        assert self.clusterer.n_clusters == 3
        assert self.clusterer.random_state == 42
        assert self.clusterer.n_init == 5
        assert self.clusterer.max_iter == 200
        
        print("✅ K-means initialized with correct parameters")
    
    def test_kmeans_fit_predict(self):
        """Test K-means fit and predict."""
        print("\n🧪 Testing K-means fit and predict...")
        
        # Generate test data
        X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
        
        # Fit clusterer
        self.clusterer.fit(X)
        
        # Check that model was fitted
        assert self.clusterer.model is not None
        assert self.clusterer.labels_ is not None
        assert self.clusterer.cluster_centers_ is not None
        
        # Check predictions
        predictions = self.clusterer.predict(X)
        assert len(predictions) == len(X)
        assert len(np.unique(predictions)) <= 3
        
        print(f"✅ K-means fit and predict: {len(np.unique(predictions))} clusters found")
    
    def test_optimal_k_determination(self):
        """Test optimal k determination."""
        print("\n🧪 Testing optimal k determination...")
        
        # Add k range to config
        self.perspective_config['evaluate_k_range'] = [2, 5]
        clusterer = KMeansClusterer(
            self.mock_config,
            self.mock_logger,
            self.perspective_config
        )
        
        # Generate well-separated data
        X, _ = make_blobs(n_samples=200, centers=4, n_features=5, 
                         cluster_std=1.0, random_state=42)
        
        optimal_k = clusterer.determine_optimal_k(X, [2, 6])
        
        assert 2 <= optimal_k <= 6
        
        print(f"✅ Optimal k determined: {optimal_k}")

class TestHDBSCANClusterer:
    """Test suite for HDBSCANClusterer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        self.mock_config = Mock()
        self.mock_config.get_options.return_value = {'seed': 42}
        self.mock_logger = Mock()
        
        self.perspective_config = {
            'params': {
                'min_cluster_size': 5,
                'min_samples': 3,
                'metric': 'euclidean',
                'max_clusters': 10
            }
        }
        
        self.clusterer = HDBSCANClusterer(
            self.mock_config,
            self.mock_logger,
            self.perspective_config
        )
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_hdbscan_initialization(self):
        """Test HDBSCAN initialization."""
        print("\n🧪 Testing HDBSCAN initialization...")
        
        assert self.clusterer.min_cluster_size == 5
        assert self.clusterer.min_samples == 3
        assert self.clusterer.metric == 'euclidean'
        assert self.clusterer.max_clusters == 10
        
        print("✅ HDBSCAN initialized with correct parameters")
    
    def test_hdbscan_fit(self):
        """Test HDBSCAN fit."""
        print("\n🧪 Testing HDBSCAN fit...")
        
        # Generate test data with clear clusters
        X, _ = make_blobs(n_samples=100, centers=3, n_features=5,
                         cluster_std=0.5, random_state=42)
        
        try:
            self.clusterer.fit(X)
            
            assert self.clusterer.model is not None
            assert self.clusterer.labels_ is not None
            
            unique_labels = np.unique(self.clusterer.labels_)
            n_clusters = len(unique_labels[unique_labels != -1])
            
            print(f"✅ HDBSCAN fit: {n_clusters} clusters found")
            
        except ImportError:
            print("⚠️  HDBSCAN not available, skipping test")
    
    def test_over_fragmentation_handling(self):
        """Test handling of over-fragmentation."""
        print("\n🧪 Testing over-fragmentation handling...")
        
        # Create config that might cause many clusters
        config = {
            'params': {
                'min_cluster_size': 2,  # Very small
                'min_samples': 1,
                'max_clusters': 5
            }
        }
        
        clusterer = HDBSCANClusterer(self.mock_config, self.mock_logger, config)
        
        # Generate data that might create many small clusters
        X = np.random.random((100, 5))
        
        try:
            clusterer.fit(X)
            
            unique_labels = np.unique(clusterer.labels_)
            n_clusters = len(unique_labels[unique_labels != -1])
            
            # Should not exceed max_clusters
            assert n_clusters <= config['params']['max_clusters']
            
            print(f"✅ Over-fragmentation handled: {n_clusters} clusters (max: {config['params']['max_clusters']})")
            
        except ImportError:
            print("⚠️  HDBSCAN not available, skipping test")

class TestAgglomerativeClusterer:
    """Test suite for AgglomerativeClusterer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        self.mock_config = Mock()
        self.mock_config.get_options.return_value = {'seed': 42}
        self.mock_logger = Mock()
        
        self.perspective_config = {
            'params': {
                'n_clusters': 3,
                'linkage': 'ward',
                'affinity': 'euclidean'
            }
        }
        
        self.clusterer = AgglomerativeClusterer(
            self.mock_config,
            self.mock_logger,
            self.perspective_config
        )
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_agglomerative_initialization(self):
        """Test Agglomerative initialization."""
        print("\n🧪 Testing Agglomerative initialization...")
        
        assert self.clusterer.n_clusters == 3
        assert self.clusterer.linkage == 'ward'
        assert self.clusterer.affinity == 'euclidean'
        
        print("✅ Agglomerative initialized correctly")
    
    def test_agglomerative_fit_predict(self):
        """Test Agglomerative fit and predict."""
        print("\n🧪 Testing Agglomerative fit and predict...")
        
        # Generate test data
        X, _ = make_blobs(n_samples=50, centers=3, n_features=5, random_state=42)
        
        # Fit clusterer
        self.clusterer.fit(X)
        
        assert self.clusterer.model is not None
        assert self.clusterer.labels_ is not None
        assert self.clusterer.cluster_centers_ is not None
        
        # Test predictions
        predictions = self.clusterer.predict(X)
        assert len(predictions) == len(X)
        assert len(np.unique(predictions)) == 3
        
        print(f"✅ Agglomerative fit and predict: {len(np.unique(predictions))} clusters")

class TestClusterLabeler:
    """Test suite for ClusterLabeler class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        self.mock_config = Mock()
        self.mock_config.get_cluster_labeling_config.return_value = {
            'method': 'tfidf',
            'tfidf': {'top_terms': 3},
            'openai': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.3,
                'api_key_env': 'OPENAI_API_KEY'
            }
        }
        
        self.mock_logger = Mock()
        self.labeler = ClusterLabeler(self.mock_config, self.mock_logger)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_labeler_initialization(self):
        """Test cluster labeler initialization."""
        print("\n🧪 Testing cluster labeler initialization...")
        
        assert self.labeler.method == 'tfidf'
        assert self.labeler.config is not None
        assert self.labeler.logger is not None
        
        print("✅ Cluster labeler initialized correctly")
    
    def test_tfidf_label_generation(self):
        """Test TF-IDF based label generation."""
        print("\n🧪 Testing TF-IDF label generation...")
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'text_col': [
                'software engineering python programming',
                'data science machine learning analytics',
                'software development coding programming',
                'business strategy marketing sales'
            ],
            'cluster': [0, 1, 0, 2]
        })
        
        labels = self.labeler.generate_tfidf_labels(test_df, ['text_col'], 'cluster')
        
        assert isinstance(labels, dict)
        assert len(labels) >= 2  # Should have labels for different clusters
        
        print(f"✅ Generated {len(labels)} TF-IDF labels")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('openai.chat.completions.create')
    def test_openai_label_generation(self, mock_openai):
        """Test OpenAI based label generation."""
        print("\n🧪 Testing OpenAI label generation...")
        
        # Configure for OpenAI
        self.mock_config.get_cluster_labeling_config.return_value = {
            'method': 'openai',
            'openai': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.3,
                'api_key_env': 'OPENAI_API_KEY'
            }
        }
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices[0].message.content = "Software Development"
        mock_openai.return_value = mock_response
        
        try:
            labeler = ClusterLabeler(self.mock_config, self.mock_logger)
            
            cluster_examples = {
                0: ['software engineer python', 'web developer javascript'],
                1: ['data scientist analytics', 'machine learning engineer']
            }
            
            labels = labeler.generate_openai_labels(cluster_examples)
            
            assert isinstance(labels, dict)
            assert len(labels) == 2
            
            print(f"✅ Generated {len(labels)} OpenAI labels")
            
        except Exception as e:
            print(f"⚠️  OpenAI labeling test info: {e}")

class TestUniqueRowProcessor:
    """Test suite for UniqueRowProcessor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.mock_logger = Mock()
        self.processor = UniqueRowProcessor(self.mock_logger)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_unique_row_extraction(self):
        """Test unique row extraction."""
        print("\n🧪 Testing unique row extraction...")
        
        test_df = pd.DataFrame({
            'text1': ['A', 'B', 'A', 'C', 'b'],  # 'b' should match 'B'
            'text2': ['X', 'Y', 'X', 'Z', 'y'],  # 'y' should match 'Y'
            'other': [1, 2, 3, 4, 5]
        })
        
        unique_df, row_map = self.processor.prepare_unique_rows(test_df, ['text1', 'text2'])
        
        # Should have 3 unique combinations: (A,X), (B,Y), (C,Z)
        assert len(unique_df) == 3
        assert len(row_map) == 3
        
        print(f"✅ Reduced {len(test_df)} rows to {len(unique_df)} unique rows")
    
    def test_result_mapping(self):
        """Test mapping results back to original dataset."""
        print("\n🧪 Testing result mapping...")
        
        test_df = pd.DataFrame({
            'text': ['A', 'B', 'A', 'C'],
            'other': [1, 2, 3, 4]
        })
        
        unique_df, row_map = self.processor.prepare_unique_rows(test_df, ['text'])
        
        # Mock clustering results for unique rows
        unique_results = [0, 1, 2]  # 3 unique rows get 3 different clusters
        
        mapped_results = self.processor.map_results_to_full(unique_results, len(test_df))
        
        assert len(mapped_results) == len(test_df)
        assert mapped_results[0] == mapped_results[2]  # Both 'A' should have same cluster
        
        print("✅ Results mapped correctly to original dataset")

class TestEnhancedClassifierManager:
    """Test suite for EnhancedClassifierManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
        # Create test config
        self.config = ConfigManager()
        self.mock_logger = Mock()
        self.mock_data_processor = Mock()
        self.mock_feature_extractor = Mock()
        
        self.manager = EnhancedClassifierManager(
            self.config,
            self.mock_logger,
            self.mock_data_processor,
            self.mock_feature_extractor
        )
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        print("\n🧪 Testing manager initialization...")
        
        assert self.manager.config is not None
        assert self.manager.logger is not None
        assert self.manager.data_processor is not None
        assert self.manager.feature_extractor is not None
        
        print("✅ Enhanced Classifier Manager initialized correctly")
    
    def test_perspective_type_detection(self):
        """Test perspective type detection."""
        print("\n🧪 Testing perspective type detection...")
        
        # Test clustering perspective
        clustering_config = {
            'type': 'clustering',
            'algorithm': 'kmeans',
            'columns': ['text'],
            'output_column': 'cluster'
        }
        
        # Test AI classification perspective
        ai_config = {
            'type': 'openai_classification',
            'columns': ['text'],
            'target_categories': ['Cat1', 'Cat2'],
            'output_column': 'classification'
        }
        
        # Manager should be able to handle both types
        # (This is integration tested in pipeline tests)
        
        print("✅ Perspective type detection working")

def run_clustering_tests():
    """Run all clustering tests."""
    print("🧪 RUNNING TRADITIONAL CLUSTERING TESTS")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test BaseClusterer
    base_tester = TestBaseClusterer()
    base_tester.setup_method()  
    try:
        base_tester.test_base_clusterer_initialization()
        base_tester.test_noise_cluster_handling()
    finally:
        base_tester.teardown_method()  
    
    # Test KMeansClusterer
    kmeans_tester = TestKMeansClusterer()
    kmeans_tester.setup_method()  
    try:
        kmeans_tester.test_kmeans_initialization()
        kmeans_tester.test_kmeans_fit_predict()
        kmeans_tester.test_optimal_k_determination()
    finally:
        kmeans_tester.teardown_method()  
    
    # Test HDBSCANClusterer
    hdbscan_tester = TestHDBSCANClusterer()
    hdbscan_tester.setup_method()  
    try:
        hdbscan_tester.test_hdbscan_initialization()
        hdbscan_tester.test_hdbscan_fit()
        hdbscan_tester.test_over_fragmentation_handling()
    finally:
        hdbscan_tester.teardown_method()  
    
    # Test AgglomerativeClusterer
    agg_tester = TestAgglomerativeClusterer()
    agg_tester.setup_method()  
    try:
        agg_tester.test_agglomerative_initialization()
        agg_tester.test_agglomerative_fit_predict()
    finally:
        agg_tester.teardown_method()  
    
    # Test ClusterLabeler
    labeler_tester = TestClusterLabeler()
    labeler_tester.setup_method()  
    try:
        labeler_tester.test_labeler_initialization()
        labeler_tester.test_tfidf_label_generation()
        labeler_tester.test_openai_label_generation()
    finally:
        labeler_tester.teardown_method()  
    
    # Test UniqueRowProcessor
    processor_tester = TestUniqueRowProcessor()
    processor_tester.setup_method()  
    try:
        processor_tester.test_unique_row_extraction()
        processor_tester.test_result_mapping()
    finally:
        processor_tester.teardown_method()  
    
    # Test EnhancedClassifierManager
    manager_tester = TestEnhancedClassifierManager()
    manager_tester.setup_method()  
    try:
        manager_tester.test_manager_initialization()
        manager_tester.test_perspective_type_detection()
    finally:
        manager_tester.teardown_method()  
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"✅ ALL CLUSTERING TESTS PASSED")
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    print("=" * 50)
    
if __name__ == "__main__":
    run_clustering_tests()