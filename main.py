#!/usr/bin/env python3
"""
Main entry point for the classification system.

This module provides the main pipeline for classifying text columns in data files,
implementing a modular workflow for preprocessing, clustering, evaluation, and reporting.
"""

import os
import sys
import traceback
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json
import re

from config import ConfigManager, configure_argument_parser
from modules.utilities import (
    Logger,
    SparkSessionManager,
    FileOperationUtilities,
    PerformanceMonitor,
    CheckpointManager
)
from modules.data_processor import DataProcessor
from modules.classifier import ClassifierManager
from modules.evaluation import ClusterAnalyzer, ClusteringEvaluator, ClusteringVisualizer, EvaluationReporter
from modules.evaluation import ClassificationEvaluator, ClassificationVisualizer

from pyspark.sql import DataFrame as SparkDataFrame

class ClassificationPipeline:
    """Pipeline for the classification process."""

    def __init__(self, config_file=None, cli_args=None):
        """
        Initializes the classification pipeline.

        Args:
            config_file: Path to the configuration file
            cli_args: Command line arguments
        """
        self.config_file = config_file
        self.cli_args = cli_args
        self.config = None
        self.logger = None
        self.spark_manager = None
        self.performance_monitor = None
        self.checkpoint_manager = None
        self.data_processor = None
        self.classifier_manager = None
        self.evaluator = None
        self.visualizer = None
        self.reporter = None
        self.initialized = False
        self.start_time = datetime.now()
        self.perspectives = None

    def setup(self):
        """Sets up the pipeline components."""
        try:
            # Initialize configuration manager
            self.config = ConfigManager(self.config_file, self.cli_args)
            
            # Initialize the logger
            logger_instance = Logger(self.config)
            self.logger = logger_instance.logger
            self.logger.info(f"Classification pipeline setup started at {self.start_time}")
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_timer('setup')
            
            # Initialize Spark session manager
            self.spark_manager = SparkSessionManager(self.config)
            
            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(self.config)
            
            # Configurar el atributo perspectives
            self.perspectives = self.config.get_clustering_perspectives()
            
            # Ensure directories exist
            results_dir = self.config.get_results_dir()
            FileOperationUtilities.create_directory_if_not_exists(results_dir)
            
            # Log configuration summary
            self.logger.info(f"Configuration loaded from: {self.config_file}")
            self.logger.info(f"Input file: {self.config.get_input_file_path()}")
            self.logger.info(f"Output file: {self.config.get_output_file_path()}")
            self.logger.info(f"Results directory: {results_dir}")
            
            # Inicializar los componentes
            self.initialize_components()
            
            self.initialized = True
            self.performance_monitor.stop_timer('setup')
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during setup: {str(e)}")
                self.logger.error(traceback.format_exc())
            else:
                print(f"Error during setup: {str(e)}")
                print(traceback.format_exc())
            return False
        
    def verify_environment(self):
        """
        Verifies all necessary conditions are met before starting the main processing.
        This method checks:
        - Input/output file paths
        - API credentials (if required)
        - Clustering perspective configurations
        - Critical dependencies
        - Spark configuration
        - Data loading capability
        
        Returns:
            bool: True if all critical checks pass, False otherwise
        """
        self.logger.info("Starting environment verification...")
        all_checks_passed = True
        verification_results = {}
        
        # 1. Verify input/output files and directories
        try:
            # Check input file existence
            input_file = self.config.get_input_file_path()
            if not os.path.exists(input_file):
                verification_results["input_file"] = f"ERROR: Input file not found: {input_file}"
                all_checks_passed = False
            else:
                verification_results["input_file"] = f"SUCCESS: Input file verified: {input_file}"
            
            # Check output directory (create if doesn't exist)
            output_dir = os.path.dirname(self.config.get_output_file_path())
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    verification_results["output_directory"] = f"INFO: Output directory created: {output_dir}"
                except Exception as dir_error:
                    verification_results["output_directory"] = f"ERROR: Cannot create output directory: {output_dir} - {str(dir_error)}"
                    all_checks_passed = False
            else:
                verification_results["output_directory"] = f"SUCCESS: Output directory verified: {output_dir}"
            
            # Check results directory (create if doesn't exist)
            results_dir = self.config.get_results_dir()
            if not os.path.exists(results_dir):
                try:
                    os.makedirs(results_dir, exist_ok=True)
                    verification_results["results_directory"] = f"INFO: Results directory created: {results_dir}"
                except Exception as dir_error:
                    verification_results["results_directory"] = f"ERROR: Cannot create results directory: {results_dir} - {str(dir_error)}"
                    all_checks_passed = False
            else:
                verification_results["results_directory"] = f"SUCCESS: Results directory verified: {results_dir}"
        except Exception as e:
            verification_results["file_paths"] = f"ERROR: Failed to verify file paths: {str(e)}"
            all_checks_passed = False
        
        # 2. Verify API credentials (if OpenAI is used for cluster labeling)
        try:
            labeling_method = self.config.get_config_value('cluster_labeling.method', 'tfidf')
            if labeling_method == 'openai':
                api_key_env = self.config.get_config_value('cluster_labeling.openai.api_key_env', 'OPENAI_API_KEY')
                api_key = os.environ.get(api_key_env, '')
                
                if not api_key:
                    verification_results["openai_api"] = f"WARNING: OpenAI API key not found in environment variable: {api_key_env}"
                    verification_results["openai_fallback"] = "INFO: Will fall back to TF-IDF for cluster labeling"
                elif not api_key.startswith('sk-'):
                    verification_results["openai_api"] = f"WARNING: Invalid OpenAI API key format in {api_key_env}"
                    verification_results["openai_fallback"] = "INFO: Will fall back to TF-IDF for cluster labeling"
                else:
                    # Light API connection test
                    try:
                        import openai
                        openai.api_key = api_key
                        response = openai.models.list(limit=1)
                        verification_results["openai_api"] = "SUCCESS: OpenAI API key verified and working"
                    except Exception as api_error:
                        verification_results["openai_api"] = f"WARNING: OpenAI API key verification failed: {str(api_error)}"
                        verification_results["openai_fallback"] = "INFO: Will fall back to TF-IDF for cluster labeling"
            else:
                verification_results["openai_api"] = f"INFO: OpenAI API not required (using {labeling_method} method)"
        except Exception as e:
            verification_results["openai_api"] = f"WARNING: Failed to verify OpenAI API: {str(e)}"
            # Not considered critical, so all_checks_passed remains unchanged
        
        # 3. Verify clustering perspective configurations
        try:
            perspectives = self.config.get_clustering_perspectives()
            if not perspectives:
                verification_results["clustering_perspectives"] = "ERROR: No clustering perspectives found in configuration"
                all_checks_passed = False
            else:
                perspective_issues = []
                for name, config in perspectives.items():
                    # Check for required algorithm field
                    if 'algorithm' not in config:
                        perspective_issues.append(f"Missing 'algorithm' in '{name}'")
                    # Check HDBSCAN-specific parameters
                    elif config['algorithm'].lower() == 'hdbscan':
                        params = config.get('params', {})
                        min_cluster_size = params.get('min_cluster_size', 0)
                        if min_cluster_size < 25:
                            perspective_issues.append(f"Low min_cluster_size ({min_cluster_size}) in '{name}' may cause excessive fragmentation")
                    # Check required fields
                    if 'columns' not in config:
                        perspective_issues.append(f"Missing 'columns' in '{name}'")
                    if 'output_column' not in config:
                        perspective_issues.append(f"Missing 'output_column' in '{name}'")
                
                if perspective_issues:
                    verification_results["clustering_perspectives"] = f"WARNING: Issues with clustering perspectives: {', '.join(perspective_issues)}"
                else:
                    verification_results["clustering_perspectives"] = f"SUCCESS: {len(perspectives)} clustering perspectives configured correctly"
        except Exception as e:
            verification_results["clustering_perspectives"] = f"ERROR: Failed to verify clustering perspectives: {str(e)}"
            all_checks_passed = False
        
        # 4. Verify critical dependencies
        try:
            # Check for required packages
            import hdbscan
            import umap
            import sklearn
            import numpy as np
            import pandas as pd
            verification_results["dependencies"] = "SUCCESS: All critical dependencies available"
        except ImportError as e:
            verification_results["dependencies"] = f"ERROR: Missing critical dependency: {str(e)}"
            all_checks_passed = False
        
        # 5. Verify Spark configuration
        try:
            # Attempt to create a Spark session
            spark = self.spark_manager.get_or_create_session()
            spark_version = spark.version
            verification_results["spark"] = f"SUCCESS: Spark session created successfully (version {spark_version})"
        except Exception as e:
            verification_results["spark"] = f"ERROR: Failed to create Spark session: {str(e)}"
            all_checks_passed = False
        
        # 6. Test data loading to verify the format is supported
        try:
            input_file = self.config.get_input_file_path()
            # Test loading a small sample only
            # Note: pandas read_stata doesn't support 'nrows' but can use 'iterator'
            # to read just a few rows
            try:
                with pd.read_stata(input_file, convert_categoricals=False, iterator=True) as reader:
                    pd_sample = reader.read(10)  # Read just 10 rows
                verification_results["data_loading"] = f"SUCCESS: Successfully tested data loading from {input_file}"
            except Exception:
                # Fall back to loading the entire file if iterator approach fails
                pd_sample = pd.read_stata(input_file, convert_categoricals=False)
                verification_results["data_loading"] = f"SUCCESS: Successfully loaded data from {input_file} (full file)"
            
            # Verify required text columns exist
            text_columns = self.config.get_text_columns()
            missing_columns = [col for col in text_columns if col not in pd_sample.columns]
            if missing_columns:
                verification_results["text_columns"] = f"ERROR: Missing required text columns: {', '.join(missing_columns)}"
                all_checks_passed = False
            else:
                verification_results["text_columns"] = f"SUCCESS: All required text columns present in dataset"
            
            # Check for empty text columns
            empty_columns = []
            for col in text_columns:
                if col in pd_sample.columns and pd_sample[col].notna().sum() == 0:
                    empty_columns.append(col)
            
            if empty_columns:
                verification_results["empty_columns"] = f"WARNING: The following text columns contain no data: {', '.join(empty_columns)}"
        
        except Exception as e:
            verification_results["data_loading"] = f"ERROR: Failed to test data loading: {str(e)}"
            all_checks_passed = False
        
        # 7. Verify other configurations
        try:
            # Check for consistent parameters between perspectives
            perspectives = self.config.get_clustering_perspectives()
            num_perspectives = len(perspectives)
            if num_perspectives > 1:
                # Check for column overlap
                all_columns = set()
                duplicate_columns = set()
                for name, config in perspectives.items():
                    output_col = config.get('output_column', '')
                    if output_col in all_columns:
                        duplicate_columns.add(output_col)
                    all_columns.add(output_col)
                
                if duplicate_columns:
                    verification_results["perspective_output_columns"] = f"WARNING: Duplicate output columns detected: {', '.join(duplicate_columns)}"
            
            # Check memory settings (warn if too low)
            spark_config = self.config.get_spark_config()
            driver_memory = spark_config.get('driver_memory', '4g')
            if driver_memory.endswith('g') and int(driver_memory[:-1]) < 4:
                verification_results["memory_settings"] = f"WARNING: Driver memory ({driver_memory}) may be too low for large datasets"
                
        except Exception as e:
            verification_results["config_checks"] = f"WARNING: Additional configuration checks failed: {str(e)}"
            # Not considered critical
        
        # Display verification results
        self.logger.info("=== ENVIRONMENT VERIFICATION RESULTS ===")
        for check, result in verification_results.items():
            if result.startswith("ERROR"):
                self.logger.error(result)
            elif result.startswith("WARNING"):
                self.logger.warning(result)
            else:
                self.logger.info(result)
        self.logger.info("=======================================")
        
        # Final result
        if all_checks_passed:
            self.logger.info("All critical checks PASSED. Environment is ready for processing.")
        else:
            self.logger.error("Some critical checks FAILED. Please resolve the issues before proceeding.")
        
        return all_checks_passed

    def run(self):
        """
        Executes the complete classification pipeline with improved error handling and robustness.
        
        Returns:
            bool: True if the pipeline ran successfully, False otherwise
        """
        if not self.initialized:
            success = self.setup()
            if not success:
                if self.logger:
                    self.logger.error("Pipeline setup failed, aborting")
                else:
                    print("Pipeline setup failed, aborting")
                return False
        
        try:
            self.logger.info("Starting classification pipeline execution")
            self.performance_monitor.start_timer('total_pipeline')
            
            # Verification with fallback
            self.logger.info("Performing initial environment verification")
            environment_valid = self.verify_environment()
            if not environment_valid:
                self.logger.warning("Environment verification failed, attempting to continue with degraded functionality")
                # Continue anyway - be permissive
            
            # Enhanced retry logic for Spark/data processing issues
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # Step 1: Load and preprocess data with validation
                    self.performance_monitor.start_timer('data_processing')
                    dataframe = self.load_and_preprocess_data()
                    if dataframe is None:
                        raise RuntimeError("Data processing returned None - invalid input data")
                    
                    # Basic validation of loaded data
                    if hasattr(dataframe, 'shape'):
                        if dataframe.shape[0] == 0:
                            raise RuntimeError("No data loaded - empty dataset")
                        self.logger.info(f"Successfully loaded {dataframe.shape[0]} rows")
                    elif hasattr(dataframe, 'count'):
                        row_count = dataframe.count()
                        if row_count == 0:
                            raise RuntimeError("No data loaded - empty dataset")
                        self.logger.info(f"Successfully loaded {row_count} rows")
                    
                    self.performance_monitor.stop_timer('data_processing')
                    
                    # Step 2: Apply clustering perspectives with error isolation
                    self.performance_monitor.start_timer('classification')
                    result_dataframe, features_dict, cluster_assignments_dict = self.apply_clustering_perspectives(dataframe)
                    if result_dataframe is None:
                        raise RuntimeError("Classification perspectives failed completely")
                    
                    # Validate we have some results
                    perspectives_applied = len([k for k in result_dataframe.columns if any(p.get('output_column', '') == k for p in self.perspectives.values())])
                    if perspectives_applied == 0:
                        self.logger.warning("No perspectives were applied successfully, but continuing")
                    else:
                        self.logger.info(f"Successfully applied {perspectives_applied} perspectives")
                    
                    self.performance_monitor.stop_timer('classification')
                    
                    # If we get here, processing succeeded - break retry loop
                    break
                    
                except (EOFError, ConnectionError, OSError) as comm_error:
                    retry_count += 1
                    last_error = comm_error
                    self.logger.warning(f"Communication/IO error (attempt {retry_count}/{max_retries}): {str(comm_error)}")
                    
                    if retry_count >= max_retries:
                        self.logger.error("Maximum retries reached for communication errors")
                        raise RuntimeError(f"Pipeline failed after {max_retries} attempts: {str(last_error)}")
                    
                    # Enhanced recovery for Spark issues
                    self.logger.info("Attempting recovery...")
                    try:
                        if self.spark_manager:
                            self.spark_manager.stop_session()
                        time.sleep(min(5 * retry_count, 30))  # Progressive backoff
                        if self.spark_manager:
                            self.spark_manager.get_or_create_session()
                    except Exception as recovery_error:
                        self.logger.warning(f"Recovery attempt failed: {recovery_error}")
                    
                    continue
                    
                except Exception as processing_error:
                    # For non-communication errors, fail fast but with good error info
                    self.logger.error(f"Processing error: {str(processing_error)}")
                    self.logger.error(traceback.format_exc())
                    raise RuntimeError(f"Pipeline processing failed: {str(processing_error)}")
            
            # Step 3: Evaluation and reporting (optional, don't fail if this breaks)
            try:
                self.performance_monitor.start_timer('evaluation')
                evaluation_results = self.evaluate_and_report(result_dataframe, features_dict, cluster_assignments_dict)
                if evaluation_results is None:
                    self.logger.warning("Evaluation produced no results, but continuing with saving")
                self.performance_monitor.stop_timer('evaluation')
            except Exception as eval_error:
                self.logger.warning(f"Evaluation failed but continuing: {eval_error}")
                evaluation_results = None
            
            # Step 4: Cross-perspective analysis (optional)
            try:
                if result_dataframe is not None and len(self.perspectives) > 1:
                    self.performance_monitor.start_timer('cross_perspective_analysis')
                    cross_analysis_results = self.perform_cross_perspective_analysis(result_dataframe, evaluation_results)
                    self.performance_monitor.stop_timer('cross_perspective_analysis')
            except Exception as cross_error:
                self.logger.warning(f"Cross-perspective analysis failed but continuing: {cross_error}")
            
            # Step 5: Save results (critical - try multiple approaches)
            self.performance_monitor.start_timer('saving_results')
            success = self.save_results(result_dataframe)
            if not success:
                self.logger.error("Failed to save results in preferred format")
                return False
            self.performance_monitor.stop_timer('saving_results')
            
            # Final cleanup
            try:
                self.cleanup()
            except Exception as cleanup_error:
                self.logger.warning(f"Cleanup had issues but pipeline completed: {cleanup_error}")
            
            self.performance_monitor.stop_timer('total_pipeline')
            
            # Display summary
            total_time = self.performance_monitor.operation_durations['total_pipeline'][-1]
            self.logger.info(f"Classification pipeline completed successfully in {total_time:.2f} seconds")
            
            # Log performance metrics
            try:
                performance_report = self.performance_monitor.report_performance()
                self.logger.info("Performance summary:")
                for op, stats in performance_report['operations'].items():
                    self.logger.info(f"  - {op}: {stats['total_seconds']:.2f}s ({stats['avg_seconds']:.2f}s avg)")
            except Exception as perf_error:
                self.logger.warning(f"Performance reporting failed: {perf_error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error during pipeline execution: {str(e)}")
            self.logger.error(traceback.format_exc())
            try:
                self.cleanup(error=True)
            except Exception as cleanup_error:
                self.logger.error(f"Cleanup after error also failed: {cleanup_error}")
            return False

    def initialize_components(self):
        """Initializes all necessary components for the pipeline."""
        self.logger.info("Initializing pipeline components")
        
        try:
            # Initialize data processor
            self.data_processor = DataProcessor(
                config=self.config, 
                logger=self.logger, 
                spark_manager=self.spark_manager
            )
            
            # Extract feature extractor for classifier manager
            feature_extractor = self.data_processor.feature_extractor
            
            # Initialize classifier manager
            self.classifier_manager = ClassifierManager(
                config=self.config,
                logger=self.logger,
                data_processor=self.data_processor,
                feature_extractor=feature_extractor
            )
            
            # Initialize evaluation components
            results_dir = self.config.get_results_dir()
            self.evaluator = ClusteringEvaluator(self.config, self.logger)
            self.visualizer = ClusteringVisualizer(self.config, self.logger, results_dir)
            self.reporter = EvaluationReporter(self.config, self.logger, results_dir)
            
            self.logger.info("Pipeline components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline components: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the input data with enhanced validation and error handling.
        Supports both AI classification and traditional clustering workflows.
        
        Returns:
            Preprocessed DataFrame or None if failed
        """
        try:
            self.logger.info("Loading and preprocessing data")
            
            # Check for checkpoint first
            if self.checkpoint_manager and self.checkpoint_manager.checkpoint_exists('preprocessed_data'):
                self.logger.info("Found checkpoint for preprocessed data, attempting to load")
                try:
                    dataframe = self.checkpoint_manager.load_checkpoint('preprocessed_data')
                    if dataframe is not None:
                        # Validate checkpoint data
                        if hasattr(dataframe, 'shape') and dataframe.shape[0] > 0:
                            self.logger.info("Successfully loaded preprocessed data from checkpoint")
                            return dataframe
                        elif hasattr(dataframe, 'count') and dataframe.count() > 0:
                            self.logger.info("Successfully loaded preprocessed data from checkpoint")
                            return dataframe
                    self.logger.warning("Checkpoint data was invalid, proceeding with full processing")
                except Exception as checkpoint_error:
                    self.logger.warning(f"Failed to load checkpoint: {checkpoint_error}, proceeding with full processing")
            
            # Validate input file with detailed error messages
            input_file = self.config.get_input_file_path()
            if not input_file:
                self.logger.error("No input file specified in configuration")
                return None
            
            if not os.path.exists(input_file):
                self.logger.error(f"Input file not found: {input_file}")
                return None
            
            # Check file size and permissions
            try:
                file_size = os.path.getsize(input_file)
                if file_size == 0:
                    self.logger.error(f"Input file is empty: {input_file}")
                    return None
                self.logger.info(f"Input file size: {file_size / (1024*1024):.2f} MB")
            except Exception as file_check_error:
                self.logger.warning(f"Could not check file properties: {file_check_error}")
            
            # Determine processing approach based on perspectives
            perspectives = self.config.get_clustering_perspectives()
            has_ai_classification = any(
                p.get('type') == 'openai_classification' 
                for p in perspectives.values()
            )
            has_clustering = any(
                p.get('type', 'clustering') == 'clustering'
                for p in perspectives.values()
            )
            
            self.logger.info(f"Processing approach: AI Classification={has_ai_classification}, Clustering={has_clustering}")
            
            # Load data with enhanced error handling
            self.logger.info(f"Loading data from {input_file}")
            try:
                # Try to load Stata file with multiple approaches
                pd_df = None
                
                # Primary approach
                try:
                    pd_df = pd.read_stata(input_file, convert_categoricals=False)
                    self.logger.info("Successfully loaded Stata file with primary method")
                except Exception as primary_error:
                    self.logger.warning(f"Primary Stata loading failed: {primary_error}")
                    
                    # Fallback approach - try with iterator
                    try:
                        with pd.read_stata(input_file, convert_categoricals=False, iterator=True) as reader:
                            pd_df = reader.read()
                        self.logger.info("Successfully loaded Stata file with iterator method")
                    except Exception as iterator_error:
                        self.logger.warning(f"Iterator Stata loading failed: {iterator_error}")
                        
                        # Final fallback - try CSV if extension is wrong
                        try:
                            pd_df = pd.read_csv(input_file)
                            self.logger.info("Successfully loaded as CSV file")
                        except Exception as csv_error:
                            raise RuntimeError(f"All loading methods failed. Stata: {primary_error}, Iterator: {iterator_error}, CSV: {csv_error}")
                
                if pd_df is None or pd_df.empty:
                    raise RuntimeError("Loaded DataFrame is None or empty")
                
                self.logger.info(f"Loaded dataset with {pd_df.shape[0]} rows and {pd_df.shape[1]} columns")
                
            except Exception as load_error:
                self.logger.error(f"Failed to load data: {str(load_error)}")
                return None
            
            # Validate required text columns exist
            text_columns = self.config.get_text_columns()
            if not text_columns:
                self.logger.error("No text columns specified in configuration")
                return None
            
            missing_columns = [col for col in text_columns if col not in pd_df.columns]
            if missing_columns:
                self.logger.error(f"Required text columns missing: {missing_columns}")
                self.logger.info(f"Available columns: {list(pd_df.columns)}")
                return None
            
            # Check data quality in text columns
            valid_text_columns = []
            for col in text_columns:
                non_null_count = pd_df[col].notna().sum()
                total_count = len(pd_df)
                if non_null_count > 0:
                    valid_text_columns.append(col)
                    self.logger.info(f"Column '{col}': {non_null_count}/{total_count} ({non_null_count/total_count:.1%}) non-null values")
                else:
                    self.logger.warning(f"Column '{col}' has no valid data - will be processed but may cause issues")
            
            if not valid_text_columns:
                self.logger.error("No text columns contain valid data")
                return None
            
            # Remove duplicates with enhanced logging
            initial_rows = pd_df.shape[0]
            try:
                pd_df = pd_df.drop_duplicates()
                deduped_rows = pd_df.shape[0]
                if initial_rows > deduped_rows:
                    self.logger.info(f"Removed {initial_rows - deduped_rows} exact duplicate rows")
            except Exception as dedup_error:
                self.logger.warning(f"Deduplication failed, continuing with original data: {dedup_error}")
            
            # Preprocess text columns with error isolation
            self.logger.info(f"Preprocessing {len(text_columns)} text columns")
            preprocessing_errors = []
            
            for column in text_columns:
                try:
                    self.logger.info(f"Preprocessing column: {column}")
                    processed_col = f"{column}_preprocessed"
                    
                    # Apply preprocessing with error handling for each row
                    pd_df[processed_col] = pd_df[column].apply(
                        lambda x: self._safe_preprocess_text(x, column)
                    )
                    
                    # Validate preprocessing results
                    processed_count = pd_df[processed_col].notna().sum()
                    self.logger.info(f"Preprocessing '{column}': {processed_count} valid results")
                    
                except Exception as preprocess_error:
                    self.logger.error(f"Failed to preprocess column {column}: {preprocess_error}")
                    preprocessing_errors.append(column)
                    # Continue with other columns
            
            if len(preprocessing_errors) == len(text_columns):
                self.logger.error("All text preprocessing failed - cannot continue")
                return None
            
            # Decision point: AI classification vs traditional clustering vs both
            if has_ai_classification and not has_clustering:
                # Pure AI classification - return pandas DataFrame
                self.logger.info("Pure AI classification workflow - returning pandas DataFrame")
                dataframe = pd_df
                
            elif has_clustering and not has_ai_classification:
                # Pure traditional clustering - convert to Spark
                self.logger.info("Pure clustering workflow - converting to Spark DataFrame")
                try:
                    dataframe = self._safe_convert_to_spark(pd_df)
                    if dataframe is None:
                        self.logger.error("Failed to convert to Spark - falling back to pandas")
                        dataframe = pd_df
                except Exception as spark_error:
                    self.logger.warning(f"Spark conversion failed: {spark_error}, using pandas DataFrame")
                    dataframe = pd_df
                    
            else:
                # Mixed workflow - prefer pandas for flexibility
                self.logger.info("Mixed AI/clustering workflow - using pandas DataFrame")
                dataframe = pd_df
            
            # Save checkpoint if successful
            try:
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(dataframe, 'preprocessed_data')
            except Exception as checkpoint_error:
                self.logger.warning(f"Failed to save checkpoint: {checkpoint_error}")
            
            self.logger.info("Data loading and preprocessing completed successfully")
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Critical error during data loading and preprocessing: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _identify_missing_rows(self, dataframe, columns):
        """
        Identify rows that have truly missing values (None, NaN, pd.NA) in any of the specified columns.
        
        Empty strings are not considered missing values, as they are often the result
        of valid text preprocessing after removing stopwords, punctuation, URLs, etc.
        Only truly null/undefined values are considered missing.
        
        Args:
            dataframe: DataFrame to check
            columns: List of column names to check for missing values
            
        Returns:
            Boolean mask indicating which rows have truly missing values
        """
        missing_mask = pd.Series(False, index=dataframe.index)
        
        for col in columns:
            if col in dataframe.columns:
                col_missing = (
                    dataframe[col].isna() |           
                    dataframe[col].isnull() |         
                    (dataframe[col] is None) |        
                    (dataframe[col] == pd.NA)         
                )
                missing_mask = missing_mask | col_missing
        
        return missing_mask

    def _identify_completely_empty_rows(self, dataframe, columns):
        """
        Identify rows where all specified columns are either missing or empty strings.
        This is useful for identifying rows that have no processable content at all.
        
        Args:
            dataframe: DataFrame to check
            columns: List of column names to check
            
        Returns:
            Boolean mask indicating which rows have no processable content
        """
        empty_mask = pd.Series(True, index=dataframe.index)
        
        for col in columns:
            if col in dataframe.columns:
                col_has_content = (
                    dataframe[col].notna() & 
                    dataframe[col].notnull() & 
                    (dataframe[col] != '') &
                    (dataframe[col] != 'nan') &
                    (dataframe[col] != 'None') &
                    (dataframe[col] != pd.NA)
                )
                empty_mask = empty_mask & ~col_has_content
        
        return empty_mask

    def _get_processable_rows_info(self, dataframe, columns):
        """
        Get comprehensive information about row processing status.
        
        Args:
            dataframe: DataFrame to analyze
            columns: List of column names to check
            
        Returns:
            Dict with processing statistics and masks
        """
        truly_missing_mask = self._identify_missing_rows(dataframe, columns)
        completely_empty_mask = self._identify_completely_empty_rows(dataframe, columns)
        
        processable_mask = ~completely_empty_mask
        
        stats = {
            'total_rows': len(dataframe),
            'truly_missing_count': truly_missing_mask.sum(),
            'completely_empty_count': completely_empty_mask.sum(),
            'processable_count': processable_mask.sum(),
            'processable_percentage': (processable_mask.sum() / len(dataframe)) * 100 if len(dataframe) > 0 else 0,
            'truly_missing_mask': truly_missing_mask,
            'completely_empty_mask': completely_empty_mask,
            'processable_mask': processable_mask
        }
        
        return stats

    def _validate_preprocessing_results(self, dataframe, text_columns):
        """
        Validate preprocessing results and provide insights about content distribution.
        
        Args:
            dataframe: DataFrame with preprocessing results
            text_columns: List of text column names to validate
            
        Returns:
            Boolean indicating successful validation
        """
        self.logger.info("Validating preprocessing results...")
        
        for col in text_columns:
            original_col = col
            processed_col = f"{col}_preprocessed"
            
            if processed_col in dataframe.columns:
                total_rows = len(dataframe)
                original_nulls = dataframe[original_col].isna().sum()
                processed_nulls = dataframe[processed_col].isna().sum()
                processed_empty = (dataframe[processed_col] == '').sum()
                processed_valid = total_rows - processed_nulls - processed_empty
                
                self.logger.info(f"Column '{col}' preprocessing results:")
                self.logger.info(f"  Original nulls: {original_nulls:,} ({original_nulls/total_rows*100:.1f}%)")
                self.logger.info(f"  Processed nulls: {processed_nulls:,} ({processed_nulls/total_rows*100:.1f}%)")
                self.logger.info(f"  Processed empty strings: {processed_empty:,} ({processed_empty/total_rows*100:.1f}%)")
                self.logger.info(f"  Processed valid content: {processed_valid:,} ({processed_valid/total_rows*100:.1f}%)")
                
                if processed_empty > 0:
                    empty_examples = dataframe[dataframe[processed_col] == ''][original_col].head(3).tolist()
                    self.logger.debug(f"Examples of texts that became empty strings in '{col}': {empty_examples}")
        
        return True

    def apply_clustering_perspectives(self, dataframe):
        """
        Applies all clustering perspectives with enhanced error handling and isolation.
        Handles both AI classification and traditional clustering perspectives safely.
        Uses improved missing value detection that distinguishes between truly missing data
        and empty strings from preprocessing.

        Args:
            dataframe: DataFrame with preprocessed data

        Returns:
            Tuple of (DataFrame with added columns, features_dict, cluster_assignments_dict)
            or (None, None, None) if critical failure
        """
        try:
            self.logger.info("Applying clustering and classification perspectives")
            
            perspectives = self.config.get_clustering_perspectives()
            if not perspectives:
                self.logger.error("No clustering perspectives found in configuration")
                return None, None, None
            
            self.logger.info(f"Found {len(perspectives)} perspectives: {', '.join(perspectives.keys())}")
            
            if dataframe is None:
                self.logger.error("Input dataframe is None")
                return None, None, None
            
            if self.checkpoint_manager and self.checkpoint_manager.checkpoint_exists('clustering_results'):
                self.logger.info("Found checkpoint for clustering results, attempting to load")
                try:
                    result = self.checkpoint_manager.load_checkpoint('clustering_results')
                    if result is not None and len(result) == 3:
                        checkpoint_df, features_dict, assignments_dict = result
                        if checkpoint_df is not None and len(assignments_dict) > 0:
                            self.logger.info("Successfully loaded clustering results from checkpoint")
                            return result
                    self.logger.warning("Checkpoint data was invalid, proceeding with full processing")
                except Exception as checkpoint_error:
                    self.logger.warning(f"Failed to load checkpoint: {checkpoint_error}, proceeding with full processing")
            
            features_dict = {}
            cluster_assignments_dict = {}
            successful_perspectives = []
            failed_perspectives = []
            
            is_spark_df = isinstance(dataframe, SparkDataFrame)
            if is_spark_df:
                self.logger.info("Converting Spark DataFrame to pandas for perspective processing")
                try:
                    pandas_df = dataframe.toPandas()
                except Exception as conversion_error:
                    self.logger.error(f"Failed to convert Spark DataFrame to pandas: {conversion_error}")
                    return None, None, None
            else:
                pandas_df = dataframe.copy()
            
            if pandas_df.empty:
                self.logger.error("DataFrame is empty after conversion")
                return None, None, None
            
            for perspective_name, perspective_config in perspectives.items():
                self.logger.info(f"Processing perspective: {perspective_name}")
                self.performance_monitor.start_timer(f'perspective_{perspective_name}')
                
                try:
                    perspective_type = perspective_config.get('type', 'clustering')
                    columns = perspective_config.get('columns', [])
                    output_column = perspective_config.get('output_column')
                    
                    if not columns:
                        raise ValueError(f"No columns specified for perspective {perspective_name}")
                    if not output_column:
                        raise ValueError(f"No output column specified for perspective {perspective_name}")
                    
                    missing_columns = [col for col in columns if col not in pandas_df.columns]
                    if missing_columns:
                        preprocessed_columns = []
                        still_missing = []
                        for col in columns:
                            preprocessed_col = f"{col}_preprocessed"
                            if preprocessed_col in pandas_df.columns:
                                preprocessed_columns.append(preprocessed_col)
                            else:
                                still_missing.append(col)
                        
                        if still_missing:
                            raise ValueError(f"Missing columns for perspective {perspective_name}: {still_missing}")
                        else:
                            self.logger.info(f"Using preprocessed columns for {perspective_name}: {preprocessed_columns}")
                    
                    perspective_columns = [col if col in pandas_df.columns else f"{col}_preprocessed" for col in columns]
                    
                    self.logger.info(f"Analyzing data availability for perspective: {perspective_name}")
                    processing_info = self._get_processable_rows_info(pandas_df, perspective_columns)
                    
                    self.logger.info(f"Perspective {perspective_name} data analysis:")
                    self.logger.info(f"  Total rows: {processing_info['total_rows']:,}")
                    self.logger.info(f"  Truly missing data: {processing_info['truly_missing_count']:,} rows")
                    self.logger.info(f"  Completely empty rows: {processing_info['completely_empty_count']:,} rows")
                    self.logger.info(f"  Processable rows: {processing_info['processable_count']:,} rows ({processing_info['processable_percentage']:.1f}%)")
                    
                    truly_missing_mask = processing_info['truly_missing_mask']
                    
                    if processing_info['processable_count'] == 0:
                        self.logger.warning(f"No processable rows found for perspective {perspective_name}, skipping")
                        continue
                    
                    perspective_checkpoint_key = f'perspective_{perspective_name}'
                    if self.checkpoint_manager and self.checkpoint_manager.checkpoint_exists(perspective_checkpoint_key):
                        self.logger.info(f"Loading checkpoint for perspective {perspective_name}")
                        try:
                            checkpoint_data = self.checkpoint_manager.load_checkpoint(perspective_checkpoint_key)
                            if checkpoint_data and len(checkpoint_data) == 3:
                                perspective_df, perspective_features, perspective_assignments = checkpoint_data
                                
                                if (perspective_df is not None and output_column in perspective_df.columns):
                                    pandas_df[output_column] = perspective_df[output_column]
                                    pandas_df.loc[truly_missing_mask, output_column] = pd.NA
                                    
                                    label_column = f"{output_column}_label"
                                    if label_column in perspective_df.columns:
                                        pandas_df[label_column] = perspective_df[label_column]
                                        pandas_df.loc[truly_missing_mask, label_column] = pd.NA
                                    
                                    features_dict[f"{perspective_name}_combined"] = perspective_features
                                    cluster_assignments_dict[perspective_name] = perspective_assignments
                                    successful_perspectives.append(perspective_name)
                                    
                                    self.logger.info(f"Successfully loaded perspective {perspective_name} from checkpoint")
                                    self.performance_monitor.stop_timer(f'perspective_{perspective_name}')
                                    continue
                        except Exception as checkpoint_load_error:
                            self.logger.warning(f"Failed to load checkpoint for {perspective_name}: {checkpoint_load_error}")
                    
                    try:
                        perspective_df, perspective_features, perspective_assignments = self.classifier_manager.classify_perspective(
                            pandas_df, perspective_name, perspective_config
                        )
                        
                        if perspective_df is None:
                            raise RuntimeError(f"Classifier returned None for perspective {perspective_name}")
                        
                        if output_column not in perspective_df.columns:
                            raise RuntimeError(f"Output column {output_column} not found in results")
                        
                        pandas_df[output_column] = perspective_df[output_column]
                        pandas_df.loc[truly_missing_mask, output_column] = pd.NA
                        
                        label_column = f"{output_column}_label"
                        if label_column in perspective_df.columns:
                            pandas_df[label_column] = perspective_df[label_column]
                            pandas_df.loc[truly_missing_mask, label_column] = pd.NA
                        
                        features_dict[f"{perspective_name}_combined"] = perspective_features
                        cluster_assignments_dict[perspective_name] = perspective_assignments
                        
                        try:
                            if self.checkpoint_manager:
                                self.checkpoint_manager.save_checkpoint(
                                    (perspective_df, perspective_features, perspective_assignments),
                                    perspective_checkpoint_key
                                )
                        except Exception as checkpoint_save_error:
                            self.logger.warning(f"Failed to save checkpoint for {perspective_name}: {checkpoint_save_error}")
                        
                        successful_perspectives.append(perspective_name)
                        
                        final_classifications = pandas_df[output_column].notna().sum()
                        final_missing = pandas_df[output_column].isna().sum()
                        
                        self.logger.info(f"Perspective {perspective_name} completed successfully:")
                        self.logger.info(f"  Classifications assigned: {final_classifications:,} rows")
                        self.logger.info(f"  Missing assignments: {final_missing:,} rows")
                        self.logger.info(f"  Processing efficiency: {(final_classifications/processing_info['total_rows'])*100:.1f}%")
                    
                    except Exception as perspective_error:
                        self.logger.error(f"Failed to apply perspective {perspective_name}: {str(perspective_error)}")
                        self.logger.error(traceback.format_exc())
                        failed_perspectives.append((perspective_name, str(perspective_error)))
                    
                except Exception as config_error:
                    self.logger.error(f"Configuration error for perspective {perspective_name}: {str(config_error)}")
                    failed_perspectives.append((perspective_name, str(config_error)))
                
                finally:
                    self.performance_monitor.stop_timer(f'perspective_{perspective_name}')
            
            self.logger.info(f"Perspective processing completed:")
            self.logger.info(f"  - Successful: {len(successful_perspectives)} perspectives")
            self.logger.info(f"  - Failed: {len(failed_perspectives)} perspectives")
            
            if failed_perspectives:
                self.logger.warning("Failed perspectives:")
                for name, error in failed_perspectives:
                    self.logger.warning(f"  - {name}: {error}")
            
            if not successful_perspectives:
                self.logger.error("No perspectives were applied successfully")
                return None, None, None
            
            if len(successful_perspectives) < len(perspectives):
                self.logger.warning(f"Only {len(successful_perspectives)}/{len(perspectives)} perspectives succeeded - continuing with partial results")
            
            try:
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        (pandas_df, features_dict, cluster_assignments_dict),
                        'clustering_results'
                    )
            except Exception as checkpoint_error:
                self.logger.warning(f"Failed to save overall checkpoint: {checkpoint_error}")
            
            self.logger.info("All clustering perspectives processing completed")
            return pandas_df, features_dict, cluster_assignments_dict
            
        except Exception as e:
            self.logger.error(f"Critical error applying clustering perspectives: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None, None

    def evaluate_and_report(self, dataframe, features_dict, cluster_assignments_dict):
        """
        Evaluates clustering AND classification results and generates reports.
        UPDATED to handle both clustering and AI classification perspectives.
        """
        try:
            self.logger.info("Evaluating clustering and classification results")
            
            # Check for existing evaluation checkpoint
            if self.checkpoint_manager.checkpoint_exists('evaluation_results'):
                self.logger.info("Found checkpoint for evaluation results, attempting to load")
                evaluation_results = self.checkpoint_manager.load_checkpoint('evaluation_results')
                if evaluation_results is not None:
                    self.logger.info("Successfully loaded evaluation results from checkpoint")
                    return evaluation_results
                self.logger.warning("Failed to load checkpoint, proceeding with full evaluation")
            
            # Get visualization types from configuration
            visualization_types = self.config.get_config_value(
                'evaluation.visualizations', 
                ['embeddings_plot', 'silhouette_plot', 'distribution_plot']
            )
            
            # Retrieve all perspectives (clustering + classification)
            all_perspectives = self.config.get_clustering_perspectives()
            
            # Initialize visualizers for both types
            results_dir = self.config.get_results_dir()
            clustering_visualizer = ClusteringVisualizer(self.config, self.logger, results_dir)
            classification_visualizer = ClassificationVisualizer(self.config, self.logger, results_dir)
            
            # Dictionary to store all evaluation outputs
            evaluation_results = {}
            
            # Initialize ClusterAnalyzer for clustering perspectives
            cluster_analyzer = ClusterAnalyzer(self.config, self.logger)
            
            # Evaluate each perspective
            for perspective_name, perspective_config in all_perspectives.items():
                perspective_type = perspective_config.get('type', 'clustering')
                
                self.logger.info(f"Evaluating {perspective_type} perspective: {perspective_name}")
                self.performance_monitor.start_timer(f'evaluate_{perspective_name}')
                
                # Initialize entry in evaluation results
                evaluation_results[perspective_name] = {
                    'type': perspective_type,
                    'metrics': {},
                    'visualization_paths': {},
                    'report_paths': {}
                }
                
                try:
                    if perspective_type == 'clustering':
                        # Handle clustering perspective (existing logic)
                        self._evaluate_clustering_perspective(
                            perspective_name, perspective_config, dataframe,
                            features_dict, cluster_assignments_dict, 
                            evaluation_results, clustering_visualizer,
                            cluster_analyzer, visualization_types
                        )
                    
                    elif perspective_type == 'openai_classification':
                        # Handle AI classification perspective (new logic)
                        self._evaluate_classification_perspective(
                            perspective_name, perspective_config, dataframe,
                            evaluation_results, classification_visualizer
                        )
                    
                    self.logger.info(f"Evaluation completed for perspective {perspective_name}")
                
                except Exception as e:
                    self.logger.error(f"Error evaluating perspective {perspective_name}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                
                self.performance_monitor.stop_timer(f'evaluate_{perspective_name}')
            
            # Create cross-perspective comparisons if we have multiple classification perspectives
            classification_perspectives = {
                name: config for name, config in all_perspectives.items()
                if config.get('type') == 'openai_classification'
            }
            
            if len(classification_perspectives) > 1:
                self._create_classification_comparisons(
                    dataframe, classification_perspectives, 
                    evaluation_results, classification_visualizer
                )
            
            # Generate AI classification cost report if applicable
            if classification_perspectives:
                self._generate_ai_cost_report(evaluation_results, classification_perspectives)
            
            if not evaluation_results:
                self.logger.error("No perspectives were evaluated successfully")
                return None

            self.checkpoint_manager.save_checkpoint(evaluation_results, 'evaluation_results')
            self.logger.info("Evaluation and reporting completed for all perspectives")
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error during evaluation and reporting: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _evaluate_clustering_perspective(self, perspective_name, perspective_config, dataframe,
                                       features_dict, cluster_assignments_dict, evaluation_results,
                                       clustering_visualizer, cluster_analyzer, visualization_types):
        """Evaluate a clustering perspective (existing logic extracted)."""
        combined_key = f"{perspective_name}_combined"
        if combined_key not in features_dict or perspective_name not in cluster_assignments_dict:
            self.logger.warning(f"Missing data for clustering perspective {perspective_name}, skipping evaluation")
            return
        
        features = features_dict[combined_key]
        assignments = cluster_assignments_dict[perspective_name]
        
        # Compute evaluation metrics
        metrics = self.evaluator.evaluate_clustering(features, assignments)
        evaluation_results[perspective_name]['metrics'] = metrics
        
        visualization_paths = {}
        
        # Perform enhanced cluster analysis if enabled
        if self.config.get_config_value('cluster_analysis.enabled', True):
            self.logger.info(f"Performing enhanced cluster analysis for {perspective_name}")
            
            if hasattr(dataframe, 'attrs') and 'cluster_characteristics' in dataframe.attrs:
                characteristics = dataframe.attrs['cluster_characteristics']
                self.logger.info(f"Using {len(characteristics)} stored cluster characteristics")
            else:
                characteristics = cluster_analyzer.analyze_cluster_content(
                    features, vectorizer=None, cluster_assignments=assignments, k=len(np.unique(assignments))
                )

            cluster_summary = cluster_analyzer.generate_cluster_summary(characteristics)
            
            # Create clustering visualizations
            try:
                output_column = perspective_config.get('output_column', f"{perspective_name}_cluster")
                label_column = f"{output_column}_label"
                cluster_names = {}

                if label_column in dataframe.columns:
                    for cluster_id in np.unique(assignments):
                        mask = dataframe[output_column] == cluster_id
                        if any(mask):
                            names = dataframe.loc[mask, label_column].dropna()
                            if len(names) > 0:
                                cluster_names[cluster_id] = names.iloc[0]

                viz_path = clustering_visualizer.create_cluster_size_distribution_plot(
                    assignments, cluster_names, perspective_name
                )
                visualization_paths['cluster_size_distribution'] = viz_path
            except Exception as e:
                self.logger.error(f"Error creating size distribution plot: {str(e)}")
            
            # Additional clustering visualizations...
            try:
                if 'embeddings_plot' in visualization_types:
                    viz_path = clustering_visualizer.create_embeddings_plot(features, assignments, perspective_name)
                    visualization_paths['embeddings_plot'] = viz_path
            except Exception as e:
                self.logger.error(f"Error creating embeddings plot: {str(e)}")

            try:
                if 'silhouette_plot' in visualization_types:
                    viz_path = clustering_visualizer.create_silhouette_plot(features, assignments, perspective_name)
                    visualization_paths['silhouette_plot'] = viz_path
            except Exception as e:
                self.logger.error(f"Error creating silhouette plot: {str(e)}")

        evaluation_results[perspective_name]['visualization_paths'] = visualization_paths

        # Generate reports
        try:
            report_paths = self.reporter.generate_report(perspective_name, metrics, visualization_paths)
            evaluation_results[perspective_name]['report_paths'].update(report_paths)
        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")
    
    def _evaluate_classification_perspective(self, perspective_name, perspective_config, dataframe,
                                           evaluation_results, classification_visualizer):
        """Evaluate an AI classification perspective (new logic)."""
        output_column = perspective_config.get('output_column', f"{perspective_name}_classification")
        
        if output_column not in dataframe.columns:
            self.logger.warning(f"Classification column {output_column} not found for perspective {perspective_name}")
            return
        
        # Get classification results
        classifications = dataframe[output_column].tolist()
        target_categories = perspective_config.get('target_categories', [])
        
        # Get metadata if available (from classifier manager)
        metadata = None
        if hasattr(self, 'classifier_manager'):
            stats = self.classifier_manager.get_ai_classification_stats()
            if perspective_name in stats:
                metadata = stats[perspective_name]
        
        # Evaluate classification
        metrics = self.evaluator.classification_evaluator.evaluate_classification(
            classifications, target_categories, metadata
        )
        evaluation_results[perspective_name]['metrics'] = metrics
        
        # Create visualizations
        visualization_paths = {}
        try:
            viz_path = classification_visualizer.create_classification_distribution_plot(
                classifications, perspective_name, target_categories
            )
            visualization_paths['distribution_plot'] = viz_path
        except Exception as e:
            self.logger.error(f"Error creating classification distribution plot: {str(e)}")
        
        evaluation_results[perspective_name]['visualization_paths'] = visualization_paths
        
        # Generate classification report
        try:
            report_paths = self._generate_classification_report(perspective_name, metrics, visualization_paths)
            evaluation_results[perspective_name]['report_paths'].update(report_paths)
        except Exception as e:
            self.logger.error(f"Error generating classification reports: {str(e)}")
    
    def _create_classification_comparisons(self, dataframe, classification_perspectives, 
                                         evaluation_results, classification_visualizer):
        """Create comparison visualizations for multiple classification perspectives."""
        self.logger.info("Creating classification perspective comparisons")
        
        try:
            # Collect results for comparison
            perspective_results = {}
            perspective_names = list(classification_perspectives.keys())
            
            for name, config in classification_perspectives.items():
                output_column = config.get('output_column', f"{name}_classification")
                if output_column in dataframe.columns:
                    perspective_results[name] = dataframe[output_column].tolist()
            
            if len(perspective_results) >= 2:
                comparison_plot_path = classification_visualizer.create_classification_comparison_plot(
                    perspective_results, perspective_names
                )
                
                # Add to evaluation results
                for name in perspective_names:
                    if name in evaluation_results:
                        evaluation_results[name]['visualization_paths']['comparison_plot'] = comparison_plot_path
                        
        except Exception as e:
            self.logger.error(f"Error creating classification comparisons: {str(e)}")
    
    def _generate_ai_cost_report(self, evaluation_results, classification_perspectives):
        """Generate a cost summary report for AI classification."""
        self.logger.info("Generating AI classification cost report")
        
        try:
            total_cost = 0
            total_tokens = 0
            total_api_calls = 0
            perspective_costs = {}
            
            for perspective_name in classification_perspectives.keys():
                if perspective_name in evaluation_results:
                    metrics = evaluation_results[perspective_name].get('metrics', {})
                    cost_metrics = metrics.get('cost_metrics', {})
                    
                    perspective_cost = cost_metrics.get('total_cost', 0)
                    perspective_tokens = cost_metrics.get('total_tokens', 0)
                    perspective_calls = cost_metrics.get('api_calls', 0)
                    
                    total_cost += perspective_cost
                    total_tokens += perspective_tokens
                    total_api_calls += perspective_calls
                    
                    perspective_costs[perspective_name] = {
                        'cost': perspective_cost,
                        'tokens': perspective_tokens,
                        'api_calls': perspective_calls
                    }
            
            # Create cost report
            cost_report = {
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'total_api_calls': total_api_calls,
                'perspective_breakdown': perspective_costs,
                'generated_at': datetime.now().isoformat()
            }
            
            # Save cost report
            results_dir = self.config.get_results_dir()
            cost_report_path = os.path.join(results_dir, 'ai_classification_cost_report.json')
            with open(cost_report_path, 'w') as f:
                json.dump(cost_report, f, indent=2)
            
            self.logger.info(f"AI classification cost report saved to {cost_report_path}")
            self.logger.info(f"Total AI classification cost: ${total_cost:.4f} ({total_api_calls} API calls, {total_tokens} tokens)")
            
            # Check cost limits
            max_cost = self.config.get_config_value('ai_classification.cost_management.max_cost_per_run', 50.0)
            if total_cost > max_cost:
                self.logger.warning(f"AI classification cost ${total_cost:.4f} exceeds limit ${max_cost:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error generating AI cost report: {str(e)}")
    
    def _generate_classification_report(self, perspective_name, metrics, visualization_paths):
        """Generate HTML report for classification perspective."""
        try:
            results_dir = self.config.get_results_dir()
            file_path = os.path.join(results_dir, f"{perspective_name}_classification_report.html")
            
            # Build HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Classification Report - {perspective_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
                    .visualization {{ margin: 20px 0; text-align: center; }}
                    .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
                    .cost-alert {{ color: red; font-weight: bold; }}
                    .distribution-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .distribution-table th, .distribution-table td {{ 
                        border: 1px solid #ddd; padding: 8px; text-align: left; 
                    }}
                    .distribution-table th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>AI Classification Report: {perspective_name}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Classification Metrics</h2>
                <div class="metric">
                    <strong>Total Classified:</strong> {metrics.get('total_classified', 0)}
                </div>
                <div class="metric">
                    <strong>Categories Used:</strong> {metrics.get('categories_used', 0)} / {metrics.get('total_categories', 0)}
                    (Coverage: {metrics.get('coverage_ratio', 0):.1%})
                </div>
                <div class="metric">
                    <strong>Distribution Balance:</strong> {metrics.get('balance_ratio', 0):.3f}
                    (1.0 = perfectly balanced)
                </div>
                <div class="metric">
                    <strong>Entropy:</strong> {metrics.get('normalized_entropy', 0):.3f}
                    (1.0 = maximum diversity)
                </div>
            """
            
            # Add cost information if available
            cost_metrics = metrics.get('cost_metrics', {})
            if cost_metrics:
                total_cost = cost_metrics.get('total_cost', 0)
                cost_class = 'cost-alert' if total_cost > 10 else ''
                
                html_content += f"""
                <h2>Cost Analysis</h2>
                <div class="metric {cost_class}">
                    <strong>Total Cost:</strong> ${total_cost:.4f}
                </div>
                <div class="metric">
                    <strong>Cost per Classification:</strong> ${cost_metrics.get('cost_per_classification', 0):.4f}
                </div>
                <div class="metric">
                    <strong>Cache Hit Rate:</strong> {cost_metrics.get('cache_hit_rate', 0):.1%}
                </div>
                <div class="metric">
                    <strong>Error Rate:</strong> {cost_metrics.get('error_rate', 0):.1%}
                </div>
                """
            
            # Add distribution table
            distribution = metrics.get('distribution', {})
            if distribution:
                html_content += """
                <h2>Category Distribution</h2>
                <table class="distribution-table">
                    <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
                """
                
                total = sum(distribution.values())
                for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / total * 100 if total > 0 else 0
                    html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                    """
                
                html_content += "</table>"
            
            # Add visualizations
            if visualization_paths:
                html_content += "<h2>Visualizations</h2>"
                for viz_name, viz_path in visualization_paths.items():
                    if os.path.exists(viz_path):
                        rel_path = os.path.relpath(viz_path, results_dir)
                        html_content += f"""
                        <div class="visualization">
                            <h3>{viz_name.replace('_', ' ').title()}</h3>
                            <img src="{rel_path}" alt="{viz_name}" />
                        </div>
                        """
            
            # Add issues if any
            issues = metrics.get('issues', [])
            if issues:
                html_content += "<h2>Issues Detected</h2><ul>"
                for issue in issues:
                    html_content += f"<li>{issue}</li>"
                html_content += "</ul>"
            
            html_content += """
            </body>
            </html>
            """
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Classification report saved to {file_path}")
            return {'html': file_path}
            
        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            return {}

    def _safe_preprocess_text(self, text, column_name):
        """
        Safely preprocess text with fallback to original text on errors.
        
        Args:
            text: Text to preprocess
            column_name: Name of column for logging
            
        Returns:
            Preprocessed text or original text if preprocessing fails
        """
        try:
            return self.data_processor.text_preprocessor.preprocess_text(text)
        except Exception as preprocess_error:
            # Log error but don't fail - return original text
            if hasattr(self, '_preprocessing_error_count'):
                self._preprocessing_error_count += 1
            else:
                self._preprocessing_error_count = 1
            
            # Only log first few errors to avoid spam
            if self._preprocessing_error_count <= 5:
                self.logger.warning(f"Preprocessing error in column {column_name}: {preprocess_error}")
            elif self._preprocessing_error_count == 6:
                self.logger.warning("Suppressing further preprocessing error messages...")
            
            return str(text) if text is not None else ""

    def _safe_convert_to_spark(self, pd_df):
        """
        Safely convert pandas DataFrame to Spark DataFrame with multiple fallback approaches.
        
        Args:
            pd_df: Pandas DataFrame to convert
            
        Returns:
            Spark DataFrame or None if conversion fails
        """
        try:
            spark = self.spark_manager.get_or_create_session()
            
            # Primary conversion approach
            try:
                spark_df = spark.createDataFrame(pd_df)
                # Test the DataFrame
                row_count = spark_df.count()
                self.logger.info(f"Successfully converted to Spark DataFrame with {row_count} rows")
                return spark_df.cache()
                
            except Exception as primary_error:
                self.logger.warning(f"Primary Spark conversion failed: {primary_error}")
                
                # Fallback 1: Disable Arrow
                try:
                    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
                    spark_df = spark.createDataFrame(pd_df)
                    row_count = spark_df.count()
                    self.logger.info(f"Spark conversion succeeded with Arrow disabled: {row_count} rows")
                    return spark_df.cache()
                    
                except Exception as arrow_error:
                    self.logger.warning(f"Arrow-disabled conversion failed: {arrow_error}")
                    
                    # Fallback 2: Schema inference disabled
                    try:
                        # Convert to records and back
                        records = pd_df.to_dict('records')
                        spark_df = spark.createDataFrame(records)
                        row_count = spark_df.count()
                        self.logger.info(f"Spark conversion succeeded with records approach: {row_count} rows")
                        return spark_df.cache()
                        
                    except Exception as records_error:
                        self.logger.error(f"All Spark conversion methods failed: {records_error}")
                        return None
                        
        except Exception as spark_error:
            self.logger.error(f"Spark session error during conversion: {spark_error}")
            return None

    def perform_cross_perspective_analysis(self, dataframe, evaluation_results):
        """
        Performs analysis across different clustering perspectives.
        
        Args:
            dataframe: DataFrame with all clustering results
            evaluation_results: Dictionary of evaluation results by perspective
            
        Returns:
            Dictionary with cross-perspective analysis results
        """
        self.logger.info("Performing cross-perspective analysis")
        
        try:
            # Check if analysis is enabled
            if not self.config.get_config_value('cluster_analysis.cross_perspective_analysis', True):
                self.logger.info("Cross-perspective analysis is disabled in configuration")
                return {}
            
            # Get perspectives
            perspectives = self.config.get_clustering_perspectives()
            if len(perspectives) < 2:
                self.logger.info("Need at least 2 perspectives for cross-perspective analysis")
                return {}
            
            # Prepare for analysis
            perspective_names = list(perspectives.keys())
            perspective_columns = [config.get('output_column', f"{name}_cluster") for name, config in perspectives.items()]
            
            # Create cross-perspective visualizations and analysis
            cross_analysis_results = {}
            
            # Initialize the EvaluationReporter if not already available
            if not hasattr(self, 'reporter') or self.reporter is None:
                results_dir = self.config.get_results_dir()
                self.reporter = EvaluationReporter(self.config, self.logger, results_dir)
            
            # Initialize the ClusteringVisualizer if not already available
            if not hasattr(self, 'visualizer') or self.visualizer is None:
                results_dir = self.config.get_results_dir()
                self.visualizer = ClusteringVisualizer(self.config, self.logger, results_dir)
            
            # Generate correlation heatmaps between each pair of perspectives
            from itertools import combinations
            for i, j in combinations(range(len(perspective_names)), 2):
                name1 = perspective_names[i]
                name2 = perspective_names[j]
                col1 = perspective_columns[i]
                col2 = perspective_columns[j]
                
                # Create correlation heatmap
                heatmap_path = self.visualizer.create_cluster_correlation_heatmap(
                    dataframe, col1, col2, name1, name2
                )
                
                # Create cross-perspective analysis
                analysis = self.reporter.create_cross_perspective_analysis(
                    dataframe, col1, col2, name1, name2
                )
                
                # Store results
                key = f"{name1}_vs_{name2}"
                cross_analysis_results[key] = {
                    'heatmap_path': heatmap_path,
                    'analysis': analysis
                }
            
            # Generate combined perspectives report
            combined_report_path = self.reporter.generate_combined_perspectives_report(
                dataframe, perspective_columns, perspective_names
            )
            
            cross_analysis_results['combined_report'] = combined_report_path
            
            self.logger.info(f"Cross-perspective analysis completed with {len(cross_analysis_results)-1} perspective pairs")
            return cross_analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during cross-perspective analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}

    def save_results(self, dataframe):
        """
        Saves results to Stata format with enhanced error handling and multiple fallback strategies.
        Preserves data integrity while being permissive about format constraints.
        
        Args:
            dataframe: DataFrame with classification results
            
        Returns:
            bool: True if results were saved successfully, False otherwise
        """
        try:
            self.logger.info("Saving classification results")
            
            # Validate input
            if dataframe is None:
                self.logger.error("Cannot save: DataFrame is None")
                return False
            
            if hasattr(dataframe, 'empty') and dataframe.empty:
                self.logger.error("Cannot save: DataFrame is empty")
                return False
            
            output_file = self.config.get_output_file_path()
            if not output_file:
                self.logger.error("No output file specified in configuration")
                return False
            
            self.logger.info(f"Saving results to {output_file}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as dir_error:
                self.logger.error(f"Cannot create output directory {output_dir}: {dir_error}")
                return False
            
            # Create working copy for cleaning
            try:
                df_to_save = dataframe.copy()
                self.logger.info(f"Preparing DataFrame with {df_to_save.shape[0]} rows and {df_to_save.shape[1]} columns")
            except Exception as copy_error:
                self.logger.error(f"Failed to copy DataFrame: {copy_error}")
                return False
            
            # Identify new classification columns
            perspectives = self.config.get_clustering_perspectives()
            new_columns = []
            for name, config in perspectives.items():
                output_col = config.get('output_column')
                if output_col and output_col in df_to_save.columns:
                    new_columns.append(output_col)
                    label_col = f"{output_col}_label"
                    if label_col in df_to_save.columns:
                        new_columns.append(label_col)
            
            self.logger.info(f"New classification columns: {new_columns}")
            
            # Enhanced data cleaning for Stata compatibility
            try:
                df_to_save = self._clean_dataframe_for_stata(df_to_save)
            except Exception as clean_error:
                self.logger.warning(f"Data cleaning had issues: {clean_error}, attempting to continue")
            
            # Multiple save strategies
            save_strategies = [
                ('stata_primary', self._save_stata_primary),
                ('stata_essential', self._save_stata_essential_columns),
                ('stata_minimal', self._save_stata_minimal),
                ('csv_backup', self._save_csv_backup)
            ]
            
            for strategy_name, strategy_func in save_strategies:
                try:
                    self.logger.info(f"Attempting save strategy: {strategy_name}")
                    success = strategy_func(df_to_save, output_file, new_columns)
                    if success:
                        self.logger.info(f"✅ Successfully saved using {strategy_name} strategy")
                        break
                except Exception as strategy_error:
                    self.logger.warning(f"Save strategy {strategy_name} failed: {strategy_error}")
                    continue
            else:
                # All strategies failed
                self.logger.error("❌ All save strategies failed")
                return False
            
            # Create completion marker
            try:
                timestamp_file = os.path.join(output_dir, f"classification_completed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(timestamp_file, 'w') as f:
                    f.write(f"Classification completed at {datetime.now()}\n")
                    f.write(f"Output file: {output_file}\n")
                    f.write(f"Added columns: {', '.join(new_columns)}\n")
                    f.write(f"Total rows: {df_to_save.shape[0]}\n")
                    f.write(f"Final columns: {df_to_save.shape[1]}\n")
            except Exception as marker_error:
                self.logger.warning(f"Could not create completion marker: {marker_error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def cleanup(self, error=False):
        """
        Cleans up resources and temporary files.
        
        Args:
            error: Whether the cleanup is being called after an error
            
        Returns:
            None
        """
        try:
            self.logger.info("Cleaning up resources")
            
            # Stop Spark session if running
            if self.spark_manager:
                self.spark_manager.stop_session()
            
            # Clean intermediate files if configured and not in error state
            if not error and self.config and self.config.get_config_value('options.clean_intermediate_on_success', False):
                intermediate_dir = self.config.get_config_value('options.intermediate_directory', 'intermediate')
                if os.path.exists(intermediate_dir):
                    self.logger.info(f"Cleaning intermediate directory: {intermediate_dir}")
                    FileOperationUtilities.clean_directory(intermediate_dir)
            
            # Clean old checkpoints
            if self.checkpoint_manager:
                deleted_count = self.checkpoint_manager.clean_old_checkpoints()
                if deleted_count > 0:
                    self.logger.info(f"Cleaned {deleted_count} old checkpoint files")
            
            # Log memory usage
            if self.performance_monitor:
                mem_usage = self.performance_monitor.memory_usage()
                self.logger.info(f"Final memory usage: {mem_usage['rss_mb']:.2f} MB RSS, {mem_usage['vms_mb']:.2f} MB VMS")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _clean_dataframe_for_stata(self, df):
        """
        Clean DataFrame for Stata compatibility with data preservation focus.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Step 1: Handle problematic columns
        problematic_columns = []
        for col in df_clean.columns:
            try:
                # Check for columns with all null values
                if df_clean[col].isna().all():
                    problematic_columns.append(col)
                    continue
                
                # Clean object columns
                if df_clean[col].dtype == 'object':
                    # Convert to string and handle special values
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', '<NA>', 'nat', 'NaT'], '')
                
                # Handle datetime columns
                elif df_clean[col].dtype.name.startswith('datetime'):
                    df_clean[col] = df_clean[col].astype(str).replace(['NaT', 'nat'], '')
                
            except Exception as col_error:
                self.logger.warning(f"Column {col} cleaning failed: {col_error}")
                problematic_columns.append(col)
        
        # Remove truly problematic columns
        if problematic_columns:
            self.logger.info(f"Removing {len(problematic_columns)} problematic columns for Stata compatibility")
            df_clean = df_clean.drop(columns=problematic_columns, errors='ignore')
        
        # Step 2: Clean column names for Stata
        column_mapping = {}
        for col in df_clean.columns:
            clean_col = str(col)[:32]  # Stata column name limit
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', clean_col)
            clean_col = re.sub(r'^[0-9]', '_', clean_col)  # Cannot start with number
            if clean_col != col:
                column_mapping[col] = clean_col
        
        if column_mapping:
            df_clean = df_clean.rename(columns=column_mapping)
            self.logger.info(f"Renamed {len(column_mapping)} columns for Stata compatibility")
        
        # Step 3: Final data cleaning
        df_clean = df_clean.fillna('')
        
        return df_clean

    def _save_stata_primary(self, df, output_file, new_columns):
        """Primary Stata save strategy."""
        try:
            df.to_stata(output_file, write_index=False, version=117)
            return True
        except Exception as e:
            self.logger.warning(f"Primary Stata save failed: {e}")
            return False

    def _save_stata_essential_columns(self, df, output_file, new_columns):
        """Save only essential columns to Stata."""
        try:
            # Get essential columns
            text_columns = self.config.get_text_columns()
            essential_cols = []
            
            # Add original text columns
            for col in text_columns:
                if col in df.columns:
                    essential_cols.append(col)
            
            # Add new classification columns
            essential_cols.extend([col for col in new_columns if col in df.columns])
            
            # Add key identifier columns if they exist
            for key_col in ['id', 'ID', 'key', 'index']:
                if key_col in df.columns and key_col not in essential_cols:
                    essential_cols.append(key_col)
            
            if essential_cols:
                minimal_df = df[essential_cols].copy()
                minimal_df.to_stata(output_file, write_index=False, version=117)
                
                # Save complete version as backup
                backup_file = output_file.replace('.dta', '_complete.csv')
                df.to_csv(backup_file, index=False)
                self.logger.info(f"Complete dataset saved as CSV backup: {backup_file}")
                
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Essential columns Stata save failed: {e}")
            return False

    def _save_stata_minimal(self, df, output_file, new_columns):
        """Minimal Stata save with only new columns."""
        try:
            if new_columns:
                available_new_cols = [col for col in new_columns if col in df.columns]
                if available_new_cols:
                    minimal_df = df[available_new_cols].copy()
                    minimal_df.to_stata(output_file, write_index=False, version=117)
                    
                    # Save complete as CSV
                    backup_file = output_file.replace('.dta', '_complete.csv')
                    df.to_csv(backup_file, index=False)
                    self.logger.warning(f"Only saved classification columns to Stata. Complete data in: {backup_file}")
                    
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Minimal Stata save failed: {e}")
            return False

    def _save_csv_backup(self, df, output_file, new_columns):
        """Final fallback: save as CSV."""
        try:
            csv_file = output_file.replace('.dta', '.csv')
            df.to_csv(csv_file, index=False)
            self.logger.error(f"❌ STATA FILE COULD NOT BE CREATED")
            self.logger.error(f"💾 Results saved as CSV: {csv_file}")
            self.logger.error(f"📋 Added classification columns: {', '.join(new_columns)}")
            return True
        except Exception as e:
            self.logger.error(f"Even CSV backup failed: {e}")
            return False

def parse_arguments():
    """
    Parses command line arguments.
    
    Returns:
        Parsed arguments
    """
    # Use the argument parser configuration from config.py
    parser = configure_argument_parser()
    
    # Add additional pipeline-specific arguments
    parser.add_argument('--force-recalculate', action='store_true',
                        help='Force recalculation even if checkpoints exist')
    
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation and reporting steps')
    
    parser.add_argument('--export-config', dest='export_config',
                        help='Export the complete configuration to a file')
    
    parser.add_argument('--perspectives', dest='selected_perspectives', 
                        help='Comma-separated list of perspectives to process (default: all)')
    
    args = parser.parse_args()
    
    # Handle default config path if not specified
    if not args.config_file:
        default_path = "config.yaml"
        if os.path.exists(default_path):
            args.config_file = default_path
    
    return args


def main():
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # If no config file was specified or found
    if not args.config_file:
        print("Error: No configuration file specified and no default config.yaml found.")
        print("Please provide a configuration file with --config option.")
        return 1
    
    # Validate that config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file not found: {args.config_file}")
        return 1
    
    start_time = time.time()
    print(f"Starting classification process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using configuration file: {args.config_file}")
    
    try:
        # If force recalculate is specified, remove existing checkpoints
        if args.force_recalculate:
            # Get checkpoint directory
            config_manager = ConfigManager(args.config_file)
            checkpoint_dir = config_manager.get_config_value('checkpoint.directory', 'checkpoints')
            
            if os.path.exists(checkpoint_dir):
                print(f"Removing existing checkpoints from {checkpoint_dir}")
                FileOperationUtilities.clean_directory(checkpoint_dir)
        
        # Initialize and run the pipeline
        pipeline = ClassificationPipeline(args.config_file, args)
        success = pipeline.run()
        
        # Handle export_config if requested
        if args.export_config and pipeline.config:
            try:
                complete_config = pipeline.config.as_dict()
                with open(args.export_config, 'w') as f:
                    import yaml
                    yaml.dump(complete_config, f, default_flow_style=False)
                print(f"Complete configuration exported to {args.export_config}")
            except Exception as e:
                print(f"Error exporting configuration: {str(e)}")
        
        elapsed_time = time.time() - start_time
        print(f"Classification process completed in {elapsed_time:.2f} seconds")
        print(f"Status: {'Success' if success else 'Failed'}")
        
        # Return appropriate exit code
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
