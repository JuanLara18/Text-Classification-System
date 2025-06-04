#!/usr/bin/env python3
"""
tests/test_config.py
Configuration Management Tests for AI Text Classification System
"""

import pytest
import os
import sys
import tempfile
import yaml
from unittest.mock import patch
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ConfigManager, configure_argument_parser, ConfigurationError

class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.start_time = time.time()
        
    def teardown_method(self):
        """Cleanup and timing."""
        execution_time = time.time() - self.start_time
        print(f"\n⏱️  Test execution time: {execution_time:.3f}s")
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        print("\n🧪 Testing default configuration loading...")
        
        config = ConfigManager()
        
        # Check that default values are loaded
        assert config.config is not None
        assert 'preprocessing' in config.config
        assert 'feature_extraction' in config.config
        assert 'evaluation' in config.config
        
        print("✅ Default configuration loaded successfully")
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        print("\n🧪 Testing YAML configuration loading...")
        
        # Create temporary YAML config
        test_config = {
            'input_file': 'tests/data/test_job_dataset.dta',
            'output_file': 'tests/output/test_output.dta',
            'text_columns': ['position_title', 'job_description'],
            'clustering_perspectives': {
                'test_perspective': {
                    'type': 'openai_classification',
                    'columns': ['position_title'],
                    'target_categories': ['Category1', 'Category2'],
                    'output_column': 'test_output'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            config = ConfigManager(temp_config_path)
            
            # Verify loaded values
            assert config.get_input_file_path() == 'tests/data/test_job_dataset.dta'
            assert config.get_output_file_path() == 'tests/output/test_output.dta'
            assert config.get_text_columns() == ['position_title', 'job_description']
            
            perspectives = config.get_clustering_perspectives()
            assert 'test_perspective' in perspectives
            assert perspectives['test_perspective']['type'] == 'openai_classification'
            
            print("✅ YAML configuration loaded and validated successfully")
            
        finally:
            os.unlink(temp_config_path)
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        print("\n🧪 Testing configuration validation (success case)...")
        
        test_config = {
            'input_file': 'tests/data/test_job_dataset.dta',
            'output_file': 'tests/output/test_output.dta',
            'text_columns': ['position_title'],
            'clustering_perspectives': {
                'valid_clustering': {
                    'type': 'clustering',
                    'algorithm': 'kmeans',
                    'columns': ['position_title'],
                    'output_column': 'cluster_result',
                    'params': {'n_clusters': 5}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            config = ConfigManager(temp_config_path)
            is_valid = config.validate_config()
            assert is_valid == True
            
            print("✅ Configuration validation passed successfully")
            
        finally:
            os.unlink(temp_config_path)
    
    def test_config_validation_failure(self):
        """Test configuration validation with missing required parameters."""
        print("\n🧪 Testing configuration validation (failure case)...")
        
        # Missing required parameters
        invalid_config = {
            'input_file': 'test_input.dta',
            # Missing output_file, text_columns, clustering_perspectives
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError):
                config = ConfigManager(temp_config_path)
                config.validate_config()
            
            print("✅ Configuration validation correctly failed for invalid config")
            
        finally:
            os.unlink(temp_config_path)
    
    def test_ai_classification_validation(self):
        """Test AI classification perspective validation."""
        print("\n🧪 Testing AI classification perspective validation...")
        
        test_config = {
            'input_file': 'tests/data/test_job_dataset.dta',
            'output_file': 'tests/output/test_output.dta',
            'text_columns': ['position_title'],
            'clustering_perspectives': {
                'ai_classifier': {
                    'type': 'openai_classification',
                    'columns': ['position_title'],
                    'target_categories': ['Cat1', 'Cat2', 'Cat3'],
                    'output_column': 'ai_result',
                    'llm_config': {
                        'provider': 'openai',
                        'model': 'gpt-4o-mini',
                        'api_key_env': 'OPENAI_API_KEY'
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            # Mock environment variable
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}):
                config = ConfigManager(temp_config_path)
                is_valid = config.validate_config()
                assert is_valid == True
                
                # Test AI-specific methods
                ai_perspectives = config.get_ai_classification_perspectives()
                assert 'ai_classifier' in ai_perspectives
                assert config.has_ai_classification_perspectives() == True
            
            print("✅ AI classification validation passed successfully")
            
        finally:
            os.unlink(temp_config_path)
    
    def test_config_value_access(self):
        """Test accessing configuration values with dotted paths."""
        print("\n🧪 Testing configuration value access...")
        
        config = ConfigManager()
        
        # Test dotted path access
        preprocessing = config.get_config_value('preprocessing.lowercase', True)
        assert isinstance(preprocessing, bool)
        
        # Test default value
        non_existent = config.get_config_value('non.existent.path', 'default')
        assert non_existent == 'default'
        
        # Test updating config values
        config.update_config_value('test.new.value', 'test_value')
        assert config.get_config_value('test.new.value') == 'test_value'
        
        print("✅ Configuration value access working correctly")
    
    def test_cli_args_integration(self):
        """Test CLI arguments integration."""
        print("\n🧪 Testing CLI arguments integration...")
        
        # Create mock CLI args
        class MockArgs:
            def __init__(self):
                self.input_file = 'cli_input.dta'
                self.output_file = 'cli_output.dta'
                self.log_level = 'DEBUG'
                self.seed = 123
                self.no_checkpoints = True
        
        cli_args = MockArgs()
        config = ConfigManager(cli_args=cli_args)
        
        # Check CLI overrides
        assert config.get_input_file_path() == 'cli_input.dta'
        assert config.get_output_file_path() == 'cli_output.dta'
        assert config.get_config_value('logging.level') == 'DEBUG'
        assert config.get_config_value('options.seed') == 123
        assert config.get_config_value('checkpoint.enabled') == False
        
        print("✅ CLI arguments integration working correctly")
    
    def test_directory_creation(self):
        """Test automatic directory creation."""
        print("\n🧪 Testing directory creation...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = {
                'input_file': os.path.join(temp_dir, 'input', 'test.dta'),
                'output_file': os.path.join(temp_dir, 'output', 'result.dta'),
                'results_dir': os.path.join(temp_dir, 'results'),
                'text_columns': ['text'],
                'clustering_perspectives': {
                    'test': {
                        'type': 'clustering',
                        'algorithm': 'kmeans',
                        'columns': ['text'],
                        'output_column': 'cluster'
                    }
                },
                'checkpoint': {
                    'enabled': True,
                    'directory': os.path.join(temp_dir, 'checkpoints')
                },
                'logging': {
                    'log_file': os.path.join(temp_dir, 'logs', 'test.log')
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(test_config, f)
                temp_config_path = f.name
            
            try:
                config = ConfigManager(temp_config_path)
                config.validate_config()
                
                # Check that directories were created
                assert os.path.exists(os.path.join(temp_dir, 'output'))
                assert os.path.exists(os.path.join(temp_dir, 'results'))
                assert os.path.exists(os.path.join(temp_dir, 'checkpoints'))
                assert os.path.exists(os.path.join(temp_dir, 'logs'))
                
                print("✅ Directories created successfully")
                
            finally:
                os.unlink(temp_config_path)

class TestConfigurationParser:
    """Test suite for configuration argument parser."""
    
    def test_argument_parser_creation(self):
        """Test argument parser configuration."""
        print("\n🧪 Testing argument parser creation...")
        
        parser = configure_argument_parser()
        
        # Test that parser has expected arguments
        args = parser.parse_args(['--config', 'test.yaml', '--input', 'test.dta'])
        
        assert hasattr(args, 'config_file')
        assert hasattr(args, 'input_file')
        assert hasattr(args, 'output_file')
        assert hasattr(args, 'log_level')
        assert hasattr(args, 'seed')
        
        assert args.config_file == 'test.yaml'
        assert args.input_file == 'test.dta'
        
        print("✅ Argument parser configured correctly")

def run_config_tests():
    """Run all configuration tests."""
    print("🧪 RUNNING CONFIGURATION TESTS")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test basic functionality
    test_manager = TestConfigManager()
    test_manager.test_default_config_loading()
    test_manager.test_yaml_config_loading()
    test_manager.test_config_validation_success()
    test_manager.test_config_validation_failure()
    test_manager.test_ai_classification_validation()
    test_manager.test_config_value_access()
    test_manager.test_cli_args_integration()
    test_manager.test_directory_creation()
    
    # Test parser
    parser_tester = TestConfigurationParser()
    parser_tester.test_argument_parser_creation()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"✅ ALL CONFIGURATION TESTS PASSED")
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    print("=" * 50)

if __name__ == "__main__":
    run_config_tests()