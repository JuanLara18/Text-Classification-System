import os
import yaml
import argparse
import copy
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass


class ConfigManager:
    """
    Configuration manager for the entire classification system.
    
    This class is responsible for loading, validating, and providing access to
    the application configuration. It supports loading from YAML files,
    overriding with command line arguments, and provides convenient access
    methods for different parts of the configuration.
    
    Attributes:
        config_file (str): Path to the configuration file
        config (dict): The loaded configuration
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        'preprocessing': {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': False,
            'custom_stopwords': [],
            'min_word_length': 2,
            'max_length': 10000
        },
        'feature_extraction': {
            'method': 'hybrid',
            'tfidf': {
                'max_features': 5000,
                'ngram_range': [1, 2],
                'min_df': 5
            },
            'embedding': {
                'model': 'sentence-transformers',
                'sentence_transformers': {
                    'model_name': 'all-MiniLM-L6-v2'
                },
                'dimensionality_reduction': {
                    'method': 'umap',
                    'n_components': 50,
                    'random_state': 42
                }
            }
        },
        'evaluation': {
            'metrics': ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score'],
            'visualizations': ['embeddings_plot', 'silhouette_plot', 'distribution_plot'],
            'output_format': ['html', 'json', 'csv']
        },
        'cluster_analysis': {
        'enabled': True,
        'top_terms_count': 15,
        'examples_count': 5,
        'create_detailed_reports': True,
        'cross_perspective_analysis': True,
        'enhanced_naming': True
        },
        'spark': {
            'executor_memory': '4g',
            'driver_memory': '4g',
            'executor_cores': 2,
            'default_parallelism': 4
        },
        'checkpoint': {
            'enabled': True,
            'interval': 1,
            'directory': 'checkpoints',
            'max_checkpoints': 5
        },
        'logging': {
            'level': 'INFO',
            'console_output': True,
            'log_file': 'classification_process.log'
        },
        'options': {
            'seed': 42,
            'save_intermediate': True,
            'clean_intermediate_on_success': False
        }
    }
    
    # Required parameters that must be present in the configuration
    REQUIRED_PARAMS = [
        'input_file',
        'output_file',
        'text_columns',
        'clustering_perspectives'
    ]

    def __init__(self, config_file=None, cli_args=None):
        """
        Initializes the configuration manager.

        Args:
            config_file (str, optional): Path to the YAML configuration file.
            cli_args (argparse.Namespace or dict, optional): Command line arguments.
        """
        self.config_file = config_file
        self.cli_args = cli_args
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.logger = logging.getLogger(__name__)
        
        try:
            if config_file:
                self.load_config()
                self.validate_config()
        except Exception as e:
            # Re-raise as ConfigurationError with clear message
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(f"Configuration error: {str(e)}")
            raise
    
    def load_config(self):
        """
        Loads the configuration from the file and merges it with CLI arguments.
        
        The loading process follows these steps:
        1. Load configuration from the YAML file
        2. Merge with default configuration
        3. Override with command line arguments
        4. Resolve relative paths to absolute paths
        
        Returns:
            dict: The loaded and merged configuration
        
        Raises:
            FileNotFoundError: If the configuration file is not found
            ConfigurationError: If the configuration file cannot be loaded or parsed
        """
        try:
            if not self.config_file:
                self.logger.warning("No configuration file provided, using default configuration")
                return self.config
                
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
            
            with open(self.config_file, 'r', encoding='utf-8') as file:
                file_config = yaml.safe_load(file)
                
            if not file_config:
                raise ConfigurationError("Configuration file is empty")
            
            # Deep merge file configuration with default configuration
            self._deep_merge(self.config, file_config)
            
            # Merge with CLI arguments if provided
            if self.cli_args:
                self._merge_cli_args()
            
            # Convert relative paths to absolute paths
            self._resolve_paths()
            
            self.logger.debug(f"Configuration loaded from {self.config_file}")
            return self.config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {str(e)}")
        except Exception as e:
            if isinstance(e, FileNotFoundError) or isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def _deep_merge(self, base_dict, override_dict):
        """
        Deep merges override_dict into base_dict recursively.
        
        This method modifies the base_dict in place by adding or overriding
        values from override_dict. For nested dictionaries, it merges them
        recursively.
        
        Args:
            base_dict (dict): Base dictionary to merge into
            override_dict (dict): Dictionary with values to override
        """
        if not isinstance(override_dict, dict):
            return
            
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = copy.deepcopy(value)
    
    def _merge_cli_args(self):
        """
        Merges command line arguments into the configuration.
        
        This method handles overriding configuration values with command line arguments.
        It supports both flat arguments and nested arguments using dot notation.
        """
        if not self.cli_args:
            return
        
        # Convert Namespace to dictionary if needed
        cli_dict = vars(self.cli_args) if hasattr(self.cli_args, '__dict__') else self.cli_args
        
        # Process each argument
        for key, value in cli_dict.items():
            if value is None:  # Skip if the value is not provided
                continue
                
            # Handle special cases with direct mappings
            if key == 'log_level':
                self._set_nested_config(['logging', 'level'], value.upper())
            elif key == 'no_checkpoints':
                if value:  # If --no-checkpoints flag is present
                    self._set_nested_config(['checkpoint', 'enabled'], False)
            elif key == 'seed':
                self._set_nested_config(['options', 'seed'], value)
            elif key == 'config_file':
                # Already handled in __init__, skip
                continue
            else:
                # Handle nested keys with dot notation (e.g., 'feature_extraction.method')
                if '.' in key:
                    parts = key.split('.')
                    self._set_nested_config(parts, value)
                else:
                    # Direct key in the root configuration
                    self.config[key] = value
    
    def _set_nested_config(self, key_parts, value):
        """
        Sets a value in the nested configuration structure.
        
        Args:
            key_parts (list): List of keys representing the path in the configuration
            value: Value to set
        """
        current = self.config
        for i, part in enumerate(key_parts):
            if i == len(key_parts) - 1:
                # Last part - set the value
                current[part] = value
            else:
                # Create nested dictionaries if they don't exist
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
    
    def _resolve_paths(self):
        """
        Resolves relative paths in the configuration to absolute paths.
        
        This ensures that all file paths in the configuration are absolute,
        making them more reliable when the working directory changes.
        """
        base_dir = os.path.dirname(os.path.abspath(self.config_file)) if self.config_file else os.getcwd()
        
        # Resolve file paths
        for path_key in ['input_file', 'output_file', 'results_dir']:
            if path_key in self.config and self.config[path_key]:
                path = self.config[path_key]
                if not os.path.isabs(path):
                    self.config[path_key] = os.path.normpath(os.path.join(base_dir, path))
        
        # Resolve nested paths
        if 'checkpoint' in self.config and 'directory' in self.config['checkpoint']:
            path = self.config['checkpoint']['directory']
            if not os.path.isabs(path):
                self.config['checkpoint']['directory'] = os.path.normpath(os.path.join(base_dir, path))
        
        if 'logging' in self.config and 'log_file' in self.config['logging']:
            path = self.config['logging']['log_file']
            if path and not os.path.isabs(path):
                self.config['logging']['log_file'] = os.path.normpath(os.path.join(base_dir, path))
    
    def _create_directories(self):
        """
        Creates directories for output files if they don't exist.
        
        This ensures that all necessary directories exist before writing files.
        """
        # Create directories for output files
        for path_key in ['output_file', 'results_dir']:
            if path_key in self.config and self.config[path_key]:
                directory = os.path.dirname(self.config[path_key])
                if not os.path.exists(directory):
                    self.logger.info(f"Creating directory: {directory}")
                    os.makedirs(directory, exist_ok=True)
        
        # Create nested directories
        if 'checkpoint' in self.config and 'directory' in self.config['checkpoint'] and self.config['checkpoint'].get('enabled', True):
            directory = self.config['checkpoint']['directory']
            if not os.path.exists(directory):
                self.logger.info(f"Creating checkpoint directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        
        if 'logging' in self.config and 'log_file' in self.config['logging']:
            log_file = self.config['logging']['log_file']
            if log_file:
                directory = os.path.dirname(log_file)
                if directory and not os.path.exists(directory):
                    self.logger.info(f"Creating log directory: {directory}")
                    os.makedirs(directory, exist_ok=True)
    
    
    def get_input_file_path(self):
        """
        Gets the path of the input file.
        
        Returns:
            str: Path to the input file
        """
        return self.config.get('input_file')
    
    def get_output_file_path(self):
        """
        Gets the path of the output file.
        
        Returns:
            str: Path to the output file
        """
        return self.config.get('output_file')
    
    def get_results_dir(self):
        """
        Gets the path of the results directory.
        
        Returns:
            str: Path to the results directory
        """
        return self.config.get('results_dir')
    
    def get_text_columns(self):
        """
        Gets the text columns to classify.
        
        Returns:
            list: List of text column names
        """
        return self.config.get('text_columns', [])
    
    def get_preprocessing_options(self):
        """
        Gets text preprocessing options.
        
        Returns:
            dict: Preprocessing configuration
        """
        return self.config.get('preprocessing', {})
    
    def get_feature_extraction_config(self):
        """
        Gets the feature extraction configuration.
        
        Returns:
            dict: Feature extraction configuration
        """
        return self.config.get('feature_extraction', {})
    
    def get_clustering_perspectives(self):
        """
        Gets clustering perspectives configurations.
        
        Returns:
            dict: Clustering perspectives configuration
        """
        return self.config.get('clustering_perspectives', {})
    
    def get_evaluation_config(self):
        """
        Gets evaluation configuration.
        
        Returns:
            dict: Evaluation configuration
        """
        return self.config.get('evaluation', {})
    
    def get_cluster_labeling_config(self):
        """
        Gets cluster labeling configuration.
        
        Returns:
            dict: Cluster labeling configuration
        """
        return self.config.get('cluster_labeling', {})
    
    def get_spark_config(self):
        """
        Gets PySpark configuration.
        
        Returns:
            dict: PySpark configuration
        """
        return self.config.get('spark', {})
    
    def get_checkpoint_config(self):
        """
        Gets checkpoint configuration.
        
        Returns:
            dict: Checkpoint configuration
        """
        return self.config.get('checkpoint', {})
    
    def get_logging_config(self):
        """
        Gets logging configuration.
        
        Returns:
            dict: Logging configuration
        """
        return self.config.get('logging', {})
    
    def get_options(self):
        """
        Gets miscellaneous options.
        
        Returns:
            dict: Miscellaneous options
        """
        return self.config.get('options', {})
    
    def get_config_value(self, key_path, default=None):
        """
        Gets a configuration value using a dotted key path.
        
        This method allows accessing nested configuration values using a dotted
        path notation, making it easy to access deeply nested values without
        multiple dictionary lookups.
        
        Args:
            key_path (str): Dotted path to the configuration value (e.g., 'feature_extraction.method')
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update_config(self, updates):
        """
        Updates the configuration with new values.
        
        This method allows updating multiple configuration values at once by
        providing a dictionary with the updates. The updates are deep merged
        into the current configuration.
        
        Args:
            updates (dict): Dictionary with updates to apply
            
        Returns:
            ConfigManager: Self for method chaining
        """
        self._deep_merge(self.config, updates)
        # Re-validate after updates
        self.validate_config()
        return self
    
    def update_config_value(self, key_path, value):
        """
        Updates a specific configuration value using a dotted key path.
        
        This method allows updating a single configuration value using a dotted
        path notation, similar to get_config_value.
        
        Args:
            key_path (str): Dotted path to the configuration value (e.g., 'feature_extraction.method')
            value: New value to set
            
        Returns:
            ConfigManager: Self for method chaining
        """
        keys = key_path.split('.')
        self._set_nested_config(keys, value)
        return self
    
    def as_dict(self):
        """
        Gets the complete configuration as a dictionary.
        
        Returns:
            dict: The complete configuration
        """
        return copy.deepcopy(self.config)

    def get_cluster_analysis_config(self):
        """
        Gets cluster analysis configuration.
        
        Returns:
            dict: Cluster analysis configuration
        """
        return self.config.get('cluster_analysis', {})

    def validate_config(self):
        """
        Validates the configuration for required parameters and consistency.
        UPDATED to handle AI classification perspectives.
        """
        if not self.config:
            raise ConfigurationError("Configuration is empty or not loaded")
        
        # Check required parameters
        missing_params = []
        for param in self.REQUIRED_PARAMS:
            if param not in self.config or self.config[param] is None:
                missing_params.append(param)
        
        if missing_params:
            raise ConfigurationError(f"Required parameters missing in configuration: {', '.join(missing_params)}")
        
        # Validate text columns
        if not isinstance(self.config['text_columns'], list) or not self.config['text_columns']:
            raise ConfigurationError("text_columns must be a non-empty list")
        
        # Validate clustering perspectives
        perspectives = self.config.get('clustering_perspectives', {})
        if not perspectives or not isinstance(perspectives, dict):
            raise ConfigurationError("clustering_perspectives must be a non-empty dictionary")
        
        perspective_errors = []
        for name, perspective in perspectives.items():
            if not isinstance(perspective, dict):
                perspective_errors.append(f"Perspective '{name}' must be a dictionary")
                continue
            
            # Get perspective type (default to clustering for backward compatibility)
            perspective_type = perspective.get('type', 'clustering')
            
            if perspective_type == 'clustering':
                # Validate clustering perspective
                self._validate_clustering_perspective(name, perspective, perspective_errors)
            elif perspective_type == 'openai_classification':
                # Validate AI classification perspective
                self._validate_ai_classification_perspective(name, perspective, perspective_errors)
            else:
                perspective_errors.append(f"Perspective '{name}' has unknown type: {perspective_type}")
        
        if perspective_errors:
            raise ConfigurationError("Configuration validation failed:\n" + "\n".join(perspective_errors))
        
        # Validate AI classification global settings if any AI perspectives exist
        self._validate_ai_classification_global_config(perspectives)
        
        # Create necessary directories
        self._create_directories()
        
        return True
    
    def _validate_clustering_perspective(self, name: str, perspective: dict, errors: list):
        """Validate a clustering perspective configuration."""
        # Check required fields for clustering
        required_fields = ['columns', 'algorithm', 'output_column']
        missing_fields = []
        for field in required_fields:
            if field not in perspective:
                missing_fields.append(field)
        
        if missing_fields:
            errors.append(f"Clustering perspective '{name}' is missing required fields: {', '.join(missing_fields)}")
            return
        
        # Validate columns
        if not isinstance(perspective['columns'], list) or not perspective['columns']:
            errors.append(f"Clustering perspective '{name}' 'columns' must be a non-empty list")
            return
        
        # Check if all columns exist in text_columns
        invalid_columns = []
        for column in perspective['columns']:
            if column not in self.config['text_columns']:
                invalid_columns.append(column)
        
        if invalid_columns:
            errors.append(
                f"Clustering perspective '{name}' references columns not defined in text_columns: {', '.join(invalid_columns)}"
            )
        
        # Validate algorithm-specific parameters
        algorithm = perspective.get('algorithm', '').lower()
        if algorithm == 'hdbscan':
            params = perspective.get('params', {})
            min_cluster_size = params.get('min_cluster_size', 0)
            if min_cluster_size < 10:
                errors.append(f"Clustering perspective '{name}': HDBSCAN min_cluster_size should be at least 10 to avoid over-fragmentation")
    
    def _validate_ai_classification_perspective(self, name: str, perspective: dict, errors: list):
        """Validate an AI classification perspective configuration."""
        # Check required fields for AI classification
        required_fields = ['columns', 'target_categories', 'output_column']
        missing_fields = []
        for field in required_fields:
            if field not in perspective:
                missing_fields.append(field)
        
        if missing_fields:
            errors.append(f"AI classification perspective '{name}' is missing required fields: {', '.join(missing_fields)}")
            return
        
        # Validate columns
        if not isinstance(perspective['columns'], list) or not perspective['columns']:
            errors.append(f"AI classification perspective '{name}' 'columns' must be a non-empty list")
            return
        
        # Check if all columns exist in text_columns
        invalid_columns = []
        for column in perspective['columns']:
            if column not in self.config['text_columns']:
                invalid_columns.append(column)
        
        if invalid_columns:
            errors.append(
                f"AI classification perspective '{name}' references columns not defined in text_columns: {', '.join(invalid_columns)}"
            )
        
        # Validate target categories
        target_categories = perspective.get('target_categories', [])
        if not isinstance(target_categories, list) or len(target_categories) < 2:
            errors.append(f"AI classification perspective '{name}' must have at least 2 target_categories")
        
        # Check for duplicate categories
        if len(target_categories) != len(set(target_categories)):
            errors.append(f"AI classification perspective '{name}' has duplicate target_categories")
        
        # Validate LLM configuration
        llm_config = perspective.get('llm_config', {})
        provider = llm_config.get('provider', 'openai')
        
        if provider == 'openai':
            # Check for API key environment variable
            api_key_env = llm_config.get('api_key_env', 'OPENAI_API_KEY')
            if not os.environ.get(api_key_env):
                errors.append(f"AI classification perspective '{name}': OpenAI API key not found in environment variable '{api_key_env}'")
            
            # Validate model
            model = llm_config.get('model', 'gpt-3.5-turbo-0125')
            valid_models = [
                'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106',
                'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4'
            ]
            if model not in valid_models:
                errors.append(f"AI classification perspective '{name}': Unknown OpenAI model '{model}'. Consider using one of: {', '.join(valid_models[:3])}")
        else:
            errors.append(f"AI classification perspective '{name}': Unsupported LLM provider '{provider}'. Currently only 'openai' is supported.")
        
        # Validate temperature range
        temperature = llm_config.get('temperature', 0.0)
        if not (0.0 <= temperature <= 2.0):
            errors.append(f"AI classification perspective '{name}': Temperature must be between 0.0 and 2.0")
        
        # Validate classification configuration
        classification_config = perspective.get('classification_config', {})
        batch_size = classification_config.get('batch_size', 10)
        if not (1 <= batch_size <= 100):
            errors.append(f"AI classification perspective '{name}': batch_size must be between 1 and 100")
    
    def _validate_ai_classification_global_config(self, perspectives: dict):
        """Validate global AI classification configuration."""
        # Check if any AI classification perspectives exist
        has_ai_perspectives = any(
            p.get('type') == 'openai_classification' 
            for p in perspectives.values()
        )
        
        if not has_ai_perspectives:
            return  # No validation needed
        
        # Validate global AI classification config
        ai_config = self.config.get('ai_classification', {})
        
        # Validate cost management
        cost_config = ai_config.get('cost_management', {})
        max_cost = cost_config.get('max_cost_per_run', 50.0)
        if max_cost <= 0:
            raise ConfigurationError("ai_classification.cost_management.max_cost_per_run must be positive")
        
        # Validate rate limiting
        rate_config = ai_config.get('rate_limiting', {})
        requests_per_minute = rate_config.get('requests_per_minute', 100)
        if not (1 <= requests_per_minute <= 1000):
            raise ConfigurationError("ai_classification.rate_limiting.requests_per_minute must be between 1 and 1000")
        
        # Validate caching directory
        cache_config = ai_config.get('caching', {})
        if cache_config.get('enabled', True):
            cache_dir = cache_config.get('cache_directory', 'ai_cache')
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(f"Cannot create AI classification cache directory '{cache_dir}': {e}")
    
    def get_ai_classification_config(self):
        """
        Gets AI classification global configuration.
        
        Returns:
            dict: AI classification configuration
        """
        return self.config.get('ai_classification', {})
    
    def get_ai_classification_perspectives(self):
        """
        Gets only the AI classification perspectives.
        
        Returns:
            dict: AI classification perspectives
        """
        perspectives = self.get_clustering_perspectives()
        return {
            name: config for name, config in perspectives.items()
            if config.get('type') == 'openai_classification'
        }
    
    def get_clustering_only_perspectives(self):
        """
        Gets only the traditional clustering perspectives.
        
        Returns:
            dict: Clustering perspectives
        """
        perspectives = self.get_clustering_perspectives()
        return {
            name: config for name, config in perspectives.items()
            if config.get('type', 'clustering') == 'clustering'
        }
    
    def has_ai_classification_perspectives(self):
        """
        Checks if the configuration has any AI classification perspectives.
        
        Returns:
            bool: True if AI classification perspectives exist
        """
        return len(self.get_ai_classification_perspectives()) > 0

    # Update the existing _create_directories method to include AI cache directory
    def _create_directories(self):
        """
        Creates directories for output files if they don't exist.
        UPDATED to include AI classification cache directory.
        """
        # Create directories for output files
        for path_key in ['output_file', 'results_dir']:
            if path_key in self.config and self.config[path_key]:
                directory = os.path.dirname(self.config[path_key])
                if not os.path.exists(directory):
                    self.logger.info(f"Creating directory: {directory}")
                    os.makedirs(directory, exist_ok=True)
        
        # Create nested directories
        if 'checkpoint' in self.config and 'directory' in self.config['checkpoint'] and self.config['checkpoint'].get('enabled', True):
            directory = self.config['checkpoint']['directory']
            if not os.path.exists(directory):
                self.logger.info(f"Creating checkpoint directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        
        if 'logging' in self.config and 'log_file' in self.config['logging']:
            log_file = self.config['logging']['log_file']
            if log_file:
                directory = os.path.dirname(log_file)
                if directory and not os.path.exists(directory):
                    self.logger.info(f"Creating log directory: {directory}")
                    os.makedirs(directory, exist_ok=True)
        
        # NEW: Create AI classification cache directory
        ai_config = self.config.get('ai_classification', {})
        cache_config = ai_config.get('caching', {})
        if cache_config.get('enabled', True):
            cache_dir = cache_config.get('cache_directory', 'ai_cache')
            if not os.path.exists(cache_dir):
                self.logger.info(f"Creating AI classification cache directory: {cache_dir}")
                os.makedirs(cache_dir, exist_ok=True)

def configure_argument_parser():
    """
    Configures an argument parser for the application.
    
    This function creates an ArgumentParser with options that align with
    the configuration system, making it easy to override configuration
    values from the command line.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Text Classification System')
    
    parser.add_argument('--config', dest='config_file', 
                        help='Path to the configuration file')
    
    parser.add_argument('--input', dest='input_file',
                        help='Path to the input file')
    
    parser.add_argument('--output', dest='output_file',
                        help='Path to the output file')
    
    parser.add_argument('--results-dir', dest='results_dir',
                        help='Directory for results and visualizations')
    
    parser.add_argument('--log-level', dest='log_level',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Logging level')
    
    parser.add_argument('--no-checkpoints', dest='no_checkpoints',
                        action='store_true',
                        help='Disable checkpointing')
    
    parser.add_argument('--seed', dest='seed', type=int,
                        help='Random seed for reproducibility')
    
    return parser