#!/usr/bin/env python3
"""
tests/test_utilities.py
Utilities Module Tests for AI Text Classification System
"""

import pytest
import os
import sys
import time
import tempfile
import shutil
import pickle
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ConfigManager
from modules.utilities import (
    Logger,
    SparkSessionManager,
    FileOperationUtilities,
    PerformanceMonitor,
    CheckpointManager
)

class TestLogger:
    """Test suite for Logger class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
        
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_logger_initialization_console_only(self):
        """Test logger initialization with console output only."""
        print("\n🧪 Testing logger initialization (console only)...")
        
        mock_config = Mock()
        mock_config.get_logging_config.return_value = {
            'level': 'INFO',
            'console_output': True,
            'log_file': None
        }
        
        logger = Logger(mock_config)
        
        assert logger.logger is not None
        assert len(logger.logger.handlers) >= 1  # At least console handler
        
        print("✅ Logger initialized with console output")
    
    def test_logger_initialization_with_file(self):
        """Test logger initialization with file output."""
        print("\n🧪 Testing logger initialization (with file)...")
        
        log_file = os.path.join(self.test_dir, 'test.log')
        
        mock_config = Mock()
        mock_config.get_logging_config.return_value = {
            'level': 'DEBUG',
            'console_output': True,
            'log_file': log_file
        }
        
        logger = Logger(mock_config)
        
        assert logger.logger is not None
        assert len(logger.logger.handlers) >= 2  # Console and file handlers
        
        # Test logging
        logger.info("Test message")
        
        # Check that log file was created and contains message
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content
        
        print(f"✅ Logger initialized with file output: {log_file}")
    
    def test_logger_methods(self):
        """Test all logger methods."""
        print("\n🧪 Testing logger methods...")
        
        mock_config = Mock()
        mock_config.get_logging_config.return_value = {
            'level': 'DEBUG',
            'console_output': False,  # Disable console for this test
            'log_file': os.path.join(self.test_dir, 'methods_test.log')
        }
        
        logger = Logger(mock_config)
        
        # Test all logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check log file content
        with open(mock_config.get_logging_config()['log_file'], 'r') as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
        
        print("✅ All logger methods working correctly")
    
    def test_logger_fallback(self):
        """Test logger fallback on configuration error."""
        print("\n🧪 Testing logger fallback...")
        
        # Create config that will cause an error
        mock_config = Mock()
        mock_config.get_logging_config.side_effect = Exception("Config error")
        
        # Should not raise exception, should fallback to basic logging
        logger = Logger(mock_config)
        
        assert logger.logger is not None
        logger.info("Fallback test message")
        
        print("✅ Logger fallback working correctly")

class TestSparkSessionManager:
    """Test suite for SparkSessionManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_spark_manager_initialization(self):
        """Test Spark manager initialization."""
        print("\n🧪 Testing Spark manager initialization...")
        
        mock_config = Mock()
        mock_config.get_spark_config.return_value = {
            'driver_memory': '2g',
            'executor_memory': '2g',
            'executor_cores': 2,
            'default_parallelism': 4
        }
        
        manager = SparkSessionManager(mock_config)
        
        assert manager.spark_config is not None
        assert manager.app_name == "Text Classification System"
        assert manager.session is None  # Not created yet
        
        print("✅ Spark manager initialized correctly")
    
    @patch('modules.utilities.SparkSession')
    def test_spark_session_creation(self, mock_spark_session):
        """Test Spark session creation."""
        print("\n🧪 Testing Spark session creation...")
        
        # Mock SparkSession
        mock_session = Mock()
        mock_session._jsc.sc.return_value.isStopped.return_value = False
        
        mock_builder = Mock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session
        
        mock_spark_session.builder = mock_builder
        
        mock_config = Mock()
        mock_config.get_spark_config.return_value = {
            'driver_memory': '2g',
            'executor_memory': '2g'
        }
        
        manager = SparkSessionManager(mock_config)
        session = manager.get_or_create_session()
        
        assert session is not None
        assert manager.session == session
        
        # Second call should return the same session
        session2 = manager.get_or_create_session()
        assert session2 == session
        
        print("✅ Spark session creation working correctly")
    
    @patch('modules.utilities.SparkSession')
    def test_spark_session_stop(self, mock_spark_session):
        """Test Spark session stopping."""
        print("\n🧪 Testing Spark session stopping...")
        
        # Mock SparkSession
        mock_session = Mock()
        mock_session._jsc.sc.return_value.isStopped.return_value = False
        
        mock_builder = Mock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session
        
        mock_spark_session.builder = mock_builder
        
        mock_config = Mock()
        mock_config.get_spark_config.return_value = {}
        
        manager = SparkSessionManager(mock_config)
        manager.get_or_create_session()
        
        # Stop session
        manager.stop_session()
        
        mock_session.stop.assert_called_once()
        assert manager.session is None
        
        print("✅ Spark session stopping working correctly")

class TestFileOperationUtilities:
    """Test suite for FileOperationUtilities class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
        
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_directory_if_not_exists(self):
        """Test directory creation."""
        print("\n🧪 Testing directory creation...")
        
        new_dir = os.path.join(self.test_dir, 'new_directory')
        
        # Directory shouldn't exist initially
        assert not os.path.exists(new_dir)
        
        # Create directory
        result = FileOperationUtilities.create_directory_if_not_exists(new_dir)
        
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)
        assert result == new_dir
        
        # Calling again should not cause error
        result2 = FileOperationUtilities.create_directory_if_not_exists(new_dir)
        assert result2 == new_dir
        
        print(f"✅ Directory created: {new_dir}")
    
    def test_create_directory_none_path(self):
        """Test directory creation with None path."""
        print("\n🧪 Testing directory creation (None path)...")
        
        result = FileOperationUtilities.create_directory_if_not_exists(None)
        assert result is None
        
        result = FileOperationUtilities.create_directory_if_not_exists("")
        assert result == ""
        
        print("✅ None/empty path handled correctly")
    
    def test_clean_directory(self):
        """Test directory cleaning."""
        print("\n🧪 Testing directory cleaning...")
        
        # Create test directory with files and subdirectories
        test_clean_dir = os.path.join(self.test_dir, 'clean_test')
        os.makedirs(test_clean_dir, exist_ok=True)
        
        # Create test files
        test_file1 = os.path.join(test_clean_dir, 'file1.txt')
        test_file2 = os.path.join(test_clean_dir, 'file2.txt')
        with open(test_file1, 'w') as f:
            f.write("test content 1")
        with open(test_file2, 'w') as f:
            f.write("test content 2")
        
        # Create test subdirectory
        test_subdir = os.path.join(test_clean_dir, 'subdir')
        os.makedirs(test_subdir, exist_ok=True)
        with open(os.path.join(test_subdir, 'subfile.txt'), 'w') as f:
            f.write("sub content")
        
        # Verify files exist
        assert os.path.exists(test_file1)
        assert os.path.exists(test_file2)
        assert os.path.exists(test_subdir)
        
        # Clean directory
        success = FileOperationUtilities.clean_directory(test_clean_dir)
        
        assert success == True
        assert os.path.exists(test_clean_dir)  # Directory itself should remain
        assert not os.path.exists(test_file1)  # Files should be gone
        assert not os.path.exists(test_file2)
        assert not os.path.exists(test_subdir)  # Subdirectory should be gone
        
        print(f"✅ Directory cleaned: {test_clean_dir}")
    
    def test_clean_nonexistent_directory(self):
        """Test cleaning non-existent directory."""
        print("\n🧪 Testing cleaning non-existent directory...")
        
        nonexistent_dir = os.path.join(self.test_dir, 'nonexistent')
        
        success = FileOperationUtilities.clean_directory(nonexistent_dir)
        assert success == False
        
        success = FileOperationUtilities.clean_directory(None)
        assert success == False
        
        print("✅ Non-existent directory handled correctly")

class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.monitor = PerformanceMonitor()
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        print("\n🧪 Testing performance monitor initialization...")
        
        assert self.monitor.timers == {}
        assert self.monitor.memory_usage_records == []
        assert self.monitor.operation_durations is not None
        assert self.monitor.start_time is not None
        
        print("✅ Performance monitor initialized correctly")
    
    def test_timer_functionality(self):
        """Test timer start/stop functionality."""
        print("\n🧪 Testing timer functionality...")
        
        operation_name = "test_operation"
        
        # Start timer
        self.monitor.start_timer(operation_name)
        assert operation_name in self.monitor.timers
        
        # Simulate some work
        time.sleep(0.1)
        
        # Stop timer
        duration = self.monitor.stop_timer(operation_name)
        
        assert duration is not None
        assert duration >= 0.1  # Should be at least 0.1 seconds
        assert operation_name not in self.monitor.timers  # Should be removed after stop
        assert len(self.monitor.operation_durations[operation_name]) == 1
        
        print(f"✅ Timer test completed: {duration:.3f}s measured")
    
    def test_timer_nonexistent_operation(self):
        """Test stopping timer for non-existent operation."""
        print("\n🧪 Testing timer for non-existent operation...")
        
        duration = self.monitor.stop_timer("nonexistent_operation")
        assert duration is None
        
        print("✅ Non-existent timer handled correctly")
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        print("\n🧪 Testing memory usage tracking...")
        
        # Get current memory usage
        memory_info = self.monitor.memory_usage()
        
        assert 'rss_mb' in memory_info
        assert 'vms_mb' in memory_info
        assert 'percent' in memory_info
        assert 'timestamp' in memory_info
        
        assert memory_info['rss_mb'] > 0
        assert memory_info['vms_mb'] > 0
        assert memory_info['percent'] >= 0
        
        print(f"✅ Memory usage: {memory_info['rss_mb']:.1f}MB RSS, {memory_info['percent']:.1f}%")
    
    def test_memory_recording_during_operations(self):
        """Test memory recording during timed operations."""
        print("\n🧪 Testing memory recording during operations...")
        
        initial_records = len(self.monitor.memory_usage_records)
        
        # Start and stop an operation (should record memory twice)
        self.monitor.start_timer("memory_test")
        self.monitor.stop_timer("memory_test")
        
        final_records = len(self.monitor.memory_usage_records)
        
        # Should have added 2 records (start and end)
        assert final_records >= initial_records + 2
        
        print(f"✅ Memory recorded during operations: {final_records - initial_records} new records")
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        print("\n🧪 Testing performance report generation...")
        
        # Perform some timed operations
        self.monitor.start_timer("op1")
        time.sleep(0.05)
        self.monitor.stop_timer("op1")
        
        self.monitor.start_timer("op2")
        time.sleep(0.1)
        self.monitor.stop_timer("op2")
        
        # Generate report
        report = self.monitor.report_performance()
        
        assert isinstance(report, dict)
        assert 'operations' in report
        assert 'memory' in report
        assert 'overall_duration_seconds' in report
        assert 'generated_at' in report
        
        # Check operations section
        operations = report['operations']
        assert 'op1' in operations
        assert 'op2' in operations
        
        # Check operation stats
        op1_stats = operations['op1']
        assert 'min_seconds' in op1_stats
        assert 'max_seconds' in op1_stats
        assert 'avg_seconds' in op1_stats
        assert 'total_seconds' in op1_stats
        assert 'count' in op1_stats
        
        print(f"✅ Performance report generated with {len(operations)} operations")

class TestCheckpointManager:
    """Test suite for CheckpointManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.get_checkpoint_config.return_value = {
            'enabled': True,
            'directory': os.path.join(self.test_dir, 'checkpoints'),
            'max_checkpoints': 3,
            'interval': 1
        }
        
        self.checkpoint_manager = CheckpointManager(self.mock_config)
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
        
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization."""
        print("\n🧪 Testing checkpoint manager initialization...")
        
        assert self.checkpoint_manager.enabled == True
        assert self.checkpoint_manager.max_checkpoints == 3
        assert os.path.exists(self.checkpoint_manager.directory)
        
        print("✅ Checkpoint manager initialized correctly")
    
    def test_checkpoint_save_and_load_pandas(self):
        """Test saving and loading pandas DataFrame checkpoints."""
        print("\n🧪 Testing checkpoint save/load (pandas)...")
        
        # Create test data
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        step_name = "test_step"
        
        # Save checkpoint
        success = self.checkpoint_manager.save_checkpoint(test_df, step_name)
        assert success == True
        
        # Check that checkpoint exists
        assert self.checkpoint_manager.checkpoint_exists(step_name)
        
        # Load checkpoint
        loaded_df = self.checkpoint_manager.load_checkpoint(step_name)
        
        assert loaded_df is not None
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['col1', 'col2']
        
        # Compare data
        pd.testing.assert_frame_equal(test_df, loaded_df)
        
        print(f"✅ Pandas checkpoint save/load: {len(loaded_df)} rows")
    
    def test_checkpoint_save_and_load_tuple(self):
        """Test saving and loading tuple checkpoints."""
        print("\n🧪 Testing checkpoint save/load (tuple)...")
        
        test_data = (
            pd.DataFrame({'col': [1, 2]}),
            {'key': 'value'},
            [1, 2, 3]
        )
        
        step_name = "tuple_step"
        
        # Save checkpoint
        success = self.checkpoint_manager.save_checkpoint(test_data, step_name)
        assert success == True
        
        # Load checkpoint
        loaded_data = self.checkpoint_manager.load_checkpoint(step_name)
        
        assert loaded_data is not None
        assert isinstance(loaded_data, tuple)
        assert len(loaded_data) == 3
        
        # Check first element (DataFrame)
        pd.testing.assert_frame_equal(test_data[0], loaded_data[0])
        
        # Check other elements
        assert loaded_data[1] == test_data[1]
        assert loaded_data[2] == test_data[2]
        
        print("✅ Tuple checkpoint save/load working correctly")
    
    def test_checkpoint_nonexistent(self):
        """Test loading non-existent checkpoint."""
        print("\n🧪 Testing non-existent checkpoint...")
        
        loaded_data = self.checkpoint_manager.load_checkpoint("nonexistent_step")
        assert loaded_data is None
        
        exists = self.checkpoint_manager.checkpoint_exists("nonexistent_step")
        assert exists == False
        
        print("✅ Non-existent checkpoint handled correctly")
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup functionality."""
        print("\n🧪 Testing checkpoint cleanup...")
        
        step_name = "cleanup_test"
        
        # Create more checkpoints than max_checkpoints
        for i in range(5):
            test_data = pd.DataFrame({'id': [i]})
            self.checkpoint_manager.save_checkpoint(test_data, step_name)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Should only keep max_checkpoints (3) files
        import glob
        pattern = os.path.join(self.checkpoint_manager.directory, f"{step_name}_*.pkl")
        checkpoint_files = glob.glob(pattern)
        
        assert len(checkpoint_files) <= 3
        
        print(f"✅ Checkpoint cleanup: {len(checkpoint_files)} files remaining (max: 3)")
    
    def test_checkpoint_disabled(self):
        """Test checkpoint manager when disabled."""
        print("\n🧪 Testing checkpoint manager (disabled)...")
        
        # Create disabled config
        disabled_config = Mock()
        disabled_config.get_checkpoint_config.return_value = {
            'enabled': False,
            'directory': self.test_dir,
            'max_checkpoints': 3
        }
        
        disabled_manager = CheckpointManager(disabled_config)
        
        # Operations should return False/None when disabled
        assert disabled_manager.enabled == False
        
        save_result = disabled_manager.save_checkpoint(pd.DataFrame(), "test")
        assert save_result == False
        
        load_result = disabled_manager.load_checkpoint("test")
        assert load_result is None
        
        exists_result = disabled_manager.checkpoint_exists("test")
        assert exists_result == False
        
        print("✅ Disabled checkpoint manager working correctly")

def run_utilities_tests():
    """Run all utilities tests."""
    print("🧪 RUNNING UTILITIES TESTS")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test Logger
    logger_tester = TestLogger()
    logger_tester.setup_method()  
    try:
        logger_tester.test_logger_initialization_console_only()
        logger_tester.test_logger_initialization_with_file()
        logger_tester.test_logger_methods()
        logger_tester.test_logger_fallback()
    finally:
        logger_tester.teardown_method()  
    
    # Test SparkSessionManager
    spark_tester = TestSparkSessionManager()
    spark_tester.setup_method()  
    try:
        spark_tester.test_spark_manager_initialization()
        spark_tester.test_spark_session_creation()
        spark_tester.test_spark_session_stop()
    finally:
        spark_tester.teardown_method()  
    
    # Test FileOperationUtilities
    file_tester = TestFileOperationUtilities()
    file_tester.setup_method()  
    try:
        file_tester.test_create_directory_if_not_exists()
        file_tester.test_create_directory_none_path()
        file_tester.test_clean_directory()
        file_tester.test_clean_nonexistent_directory()
    finally:
        file_tester.teardown_method()  
    
    # Test PerformanceMonitor
    perf_tester = TestPerformanceMonitor()
    perf_tester.setup_method()  
    try:
        perf_tester.test_performance_monitor_initialization()
        perf_tester.test_timer_functionality()
        perf_tester.test_timer_nonexistent_operation()
        perf_tester.test_memory_usage_tracking()
        perf_tester.test_memory_recording_during_operations()
        perf_tester.test_performance_report_generation()
    finally:
        perf_tester.teardown_method()  
    
    # Test CheckpointManager
    checkpoint_tester = TestCheckpointManager()
    checkpoint_tester.setup_method()  
    try:
        checkpoint_tester.test_checkpoint_manager_initialization()
        checkpoint_tester.test_checkpoint_save_and_load_pandas()
        checkpoint_tester.test_checkpoint_save_and_load_tuple()
        checkpoint_tester.test_checkpoint_nonexistent()
        checkpoint_tester.test_checkpoint_cleanup()
        checkpoint_tester.test_checkpoint_disabled()
    finally:
        checkpoint_tester.teardown_method()  
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"✅ ALL UTILITIES TESTS PASSED")
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    print("=" * 50)
    
if __name__ == "__main__":
    run_utilities_tests()