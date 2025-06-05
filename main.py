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
        Executes the complete classification pipeline.
        
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
            
            # Verification
            self.logger.info("Performing initial environment verification")
            environment_valid = self.verify_environment()
            if not environment_valid:
                self.logger.error("Environment verification failed, aborting pipeline")
                return False
            
            # Agregar manejo de excepciones específicas para Spark
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Step 1: Load and preprocess data
                    self.performance_monitor.start_timer('data_processing')
                    dataframe = self.load_and_preprocess_data()
                    if dataframe is None:
                        self.logger.error("Data processing failed, aborting pipeline")
                        return False
                    self.performance_monitor.stop_timer('data_processing')
                    
                    # Step 2: Apply clustering perspectives
                    self.performance_monitor.start_timer('clustering')
                    result_dataframe, features_dict, cluster_assignments_dict = self.apply_clustering_perspectives(dataframe)
                    if result_dataframe is None:
                        self.logger.error("Clustering failed, aborting pipeline")
                        return False
                    self.performance_monitor.stop_timer('clustering')
                    
                    # Si llegamos aquí, no necesitamos reintentar
                    break
                    
                except EOFError as e:
                    retry_count += 1
                    self.logger.warning(f"Communication error with Spark (attempt {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count >= max_retries:
                        self.logger.error("Maximum retries reached, aborting pipeline")
                        return False
                    
                    # Reiniciar la sesión Spark
                    self.logger.info("Restarting Spark session")
                    self.spark_manager.stop_session()
                    time.sleep(5)  # Esperar un poco antes de reiniciar
                    self.spark_manager.get_or_create_session()
                    continue
            
            # Step 3: Evaluate and generate reports
            self.performance_monitor.start_timer('evaluation')
            evaluation_results = self.evaluate_and_report(result_dataframe, features_dict, cluster_assignments_dict)
            if evaluation_results is None:
                self.logger.warning("Evaluation produced no results, but continuing with saving")
            self.performance_monitor.stop_timer('evaluation')
            
            # Step 3.5: Perform cross-perspective analysis if we have multiple perspectives
            if result_dataframe is not None and len(self.perspectives) > 1:
                self.performance_monitor.start_timer('cross_perspective_analysis')
                cross_analysis_results = self.perform_cross_perspective_analysis(
                    result_dataframe, evaluation_results
                )
                self.performance_monitor.stop_timer('cross_perspective_analysis')            
            
            # Step 4: Save results
            self.performance_monitor.start_timer('saving_results')
            success = self.save_results(result_dataframe)
            if not success:
                self.logger.error("Failed to save results")
                return False
            self.performance_monitor.stop_timer('saving_results')
            
            # Final cleanup
            self.cleanup()
            
            self.performance_monitor.stop_timer('total_pipeline')
            
            # Display summary of pipeline performance
            total_time = self.performance_monitor.operation_durations['total_pipeline'][-1]
            self.logger.info(f"Classification pipeline completed successfully in {total_time:.2f} seconds")
            
            # Log detailed performance metrics
            performance_report = self.performance_monitor.report_performance()
            self.logger.info("Performance summary:")
            for op, stats in performance_report['operations'].items():
                self.logger.info(f"  - {op}: {stats['total_seconds']:.2f}s ({stats['avg_seconds']:.2f}s avg)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during pipeline execution: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.cleanup(error=True)
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
        Loads and preprocesses the input data.
        For AI classification, works directly with pandas.
        For traditional clustering, converts to Spark.
        
        Returns:
            Preprocessed DataFrame or None if failed
        """
        try:
            self.logger.info("Loading and preprocessing data")
            
            # Check if we have a checkpoint
            if self.checkpoint_manager.checkpoint_exists('preprocessed_data'):
                self.logger.info("Found checkpoint for preprocessed data, attempting to load")
                dataframe = self.checkpoint_manager.load_checkpoint('preprocessed_data')
                if dataframe is not None:
                    self.logger.info("Successfully loaded preprocessed data from checkpoint")
                    return dataframe
                self.logger.warning("Failed to load checkpoint, proceeding with full processing")
            
            # Validate input file
            input_file = self.config.get_input_file_path()
            if not os.path.exists(input_file):
                self.logger.error(f"Input file not found: {input_file}")
                return None
            
            # Check if we have AI classification perspectives
            perspectives = self.config.get_clustering_perspectives()
            has_ai_classification = any(
                p.get('type') == 'openai_classification' 
                for p in perspectives.values()
            )
            
            # Load and preprocess data
            self.logger.info(f"Loading and preprocessing data from {input_file}")
            
            if has_ai_classification:
                # For AI classification, work directly with pandas
                self.logger.info("AI classification detected - processing with pandas")
                
                # Load the Stata file into pandas
                pd_df = pd.read_stata(input_file, convert_categoricals=False)
                self.logger.info(f"Loaded dataset with {pd_df.shape[0]} rows and {pd_df.shape[1]} columns")
                
                # Remove duplicates
                initial_rows = pd_df.shape[0]
                pd_df = pd_df.drop_duplicates()
                if initial_rows > pd_df.shape[0]:
                    self.logger.info(f"Removed {initial_rows - pd_df.shape[0]} duplicate rows")
                
                # Preprocess text columns
                text_columns = self.config.get_text_columns()
                for column in text_columns:
                    if column in pd_df.columns:
                        self.logger.info(f"Preprocessing column: {column}")
                        pd_df[f"{column}_preprocessed"] = pd_df[column].apply(
                            self.data_processor.text_preprocessor.preprocess_text
                        )
                
                dataframe = pd_df
                
            else:
                # For traditional clustering, use the full data processor
                dataframe = self.data_processor.load_data()
                if dataframe is None:
                    self.logger.error("Failed to load data")
                    return None
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(dataframe, 'preprocessed_data')
            
            self.logger.info("Data loading and preprocessing completed")
            return dataframe
                
        except Exception as e:
            self.logger.error(f"Error during data loading and preprocessing: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _identify_missing_rows(self, dataframe, columns):
        """
        Identify rows that have missing values in ANY of the specified columns.
        
        Args:
            dataframe: DataFrame to check
            columns: List of column names to check for missing values
            
        Returns:
            Boolean mask indicating which rows have missing values
        """
        missing_mask = pd.Series(False, index=dataframe.index)
        
        for col in columns:
            if col in dataframe.columns:
                # Check for various types of missing values
                col_missing = (
                    dataframe[col].isna() | 
                    dataframe[col].isnull() |
                    (dataframe[col] == '') |
                    (dataframe[col] == 'nan') |
                    (dataframe[col] == 'None')
                )
                missing_mask = missing_mask | col_missing
        
        return missing_mask

    def apply_clustering_perspectives(self, dataframe):
        """
        Applies all clustering perspectives to the data.
        FIXED: Now properly handles missing values in input columns.

        Args:
            dataframe: DataFrame with preprocessed data

        Returns:
            Tuple of (DataFrame with added clustering columns, features_dict, cluster_assignments_dict)
            or (None, None, None) if failed
        """
        try:
            self.logger.info("Applying clustering perspectives")
            
            # Get clustering perspectives from configuration
            perspectives = self.config.get_clustering_perspectives()
            if not perspectives:
                self.logger.error("No clustering perspectives found in configuration")
                return None, None, None
            
            self.logger.info(f"Found {len(perspectives)} clustering perspectives: {', '.join(perspectives.keys())}")
            
            # Check for checkpoint
            if self.checkpoint_manager.checkpoint_exists('clustering_results'):
                self.logger.info("Found checkpoint for clustering results, attempting to load")
                result = self.checkpoint_manager.load_checkpoint('clustering_results')
                if result is not None and len(result) == 3:
                    self.logger.info("Successfully loaded clustering results from checkpoint")
                    return result
                self.logger.warning("Failed to load checkpoint or incomplete data, proceeding with full processing")
            
            # Initialize dictionaries to store results
            features_dict = {}
            cluster_assignments_dict = {}
            
            # Check if we're working with a Spark DataFrame
            is_spark_df = isinstance(dataframe, SparkDataFrame)
            
            # If it's a Spark DataFrame, convert to pandas for easier manipulation
            if is_spark_df:
                self.logger.info("Converting Spark DataFrame to pandas for classification processing")
                pandas_df = dataframe.toPandas()
            else:
                pandas_df = dataframe.copy()
            
            # Apply each perspective
            for perspective_name, perspective_config in perspectives.items():
                self.logger.info(f"Applying perspective: {perspective_name}")
                self.performance_monitor.start_timer(f'perspective_{perspective_name}')
                
                # FIXED: Track missing values for this perspective
                perspective_columns = perspective_config.get('columns', [])
                output_column = perspective_config.get('output_column')
                
                # Identify rows with missing values in input columns
                missing_rows_mask = self._identify_missing_rows(pandas_df, perspective_columns)
                missing_count = missing_rows_mask.sum()
                
                if missing_count > 0:
                    self.logger.info(f"Perspective {perspective_name}: Found {missing_count} rows with missing values in input columns")
                
                # Apply the perspective
                try:
                    # Check if the perspective has its own checkpoint
                    perspective_checkpoint_key = f'perspective_{perspective_name}'
                    if self.checkpoint_manager.checkpoint_exists(perspective_checkpoint_key):
                        self.logger.info(f"Loading checkpoint for perspective {perspective_name}")
                        checkpoint_data = self.checkpoint_manager.load_checkpoint(perspective_checkpoint_key)
                        if checkpoint_data and len(checkpoint_data) == 3:
                            perspective_df, perspective_features, perspective_assignments = checkpoint_data
                            
                            # Update the pandas DataFrame with the perspective's clustering column
                            if output_column in perspective_df.columns:
                                pandas_df[output_column] = perspective_df[output_column]
                                # FIXED: Apply missing values mask
                                pandas_df.loc[missing_rows_mask, output_column] = pd.NA
                                
                                # If labels were generated, add them too
                                label_column = f"{output_column}_label"
                                if label_column in perspective_df.columns:
                                    pandas_df[label_column] = perspective_df[label_column]
                                    # FIXED: Apply missing values mask to labels too
                                    pandas_df.loc[missing_rows_mask, label_column] = pd.NA
                                
                                # Store features and assignments
                                features_dict[f"{perspective_name}_combined"] = perspective_features
                                cluster_assignments_dict[perspective_name] = perspective_assignments
                                
                                self.logger.info(f"Successfully loaded perspective {perspective_name} from checkpoint")
                                self.performance_monitor.stop_timer(f'perspective_{perspective_name}')
                                continue
                    
                    # If no checkpoint or loading failed, process the perspective
                    perspective_df, perspective_features, perspective_assignments = self.classifier_manager.classify_perspective(
                        pandas_df, perspective_name, perspective_config
                    )
                    
                    # Update the pandas DataFrame with the perspective's output
                    pandas_df[output_column] = perspective_df[output_column]
                    
                    # FIXED: Apply missing values mask - set cluster assignments to missing for rows with missing input
                    pandas_df.loc[missing_rows_mask, output_column] = pd.NA
                    
                    # If labels were generated, add them too
                    label_column = f"{output_column}_label"
                    if label_column in perspective_df.columns:
                        pandas_df[label_column] = perspective_df[label_column]
                        # FIXED: Apply missing values mask to labels too
                        pandas_df.loc[missing_rows_mask, label_column] = pd.NA
                    
                    # Store features and assignments (keep original assignments for evaluation)
                    features_dict[f"{perspective_name}_combined"] = perspective_features
                    cluster_assignments_dict[perspective_name] = perspective_assignments
                    
                    # Log missing value handling
                    if missing_count > 0:
                        final_missing = pandas_df[output_column].isna().sum()
                        self.logger.info(f"Perspective {perspective_name}: Set {final_missing} cluster assignments to missing due to missing input values")
                    
                    # Save checkpoint for this perspective
                    self.checkpoint_manager.save_checkpoint(
                        (perspective_df, perspective_features, perspective_assignments),
                        perspective_checkpoint_key
                    )
                    
                    self.logger.info(f"Perspective {perspective_name} applied successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error applying perspective {perspective_name}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Continue with other perspectives if one fails
                
                self.performance_monitor.stop_timer(f'perspective_{perspective_name}')
            
            # Check if any perspectives were applied successfully
            if not cluster_assignments_dict:
                self.logger.error("No perspectives were applied successfully")
                return None, None, None
            
            # Save overall checkpoint with pandas DataFrame
            self.checkpoint_manager.save_checkpoint(
                (pandas_df, features_dict, cluster_assignments_dict),
                'clustering_results'
            )
            
            self.logger.info("All clustering perspectives applied")
            return pandas_df, features_dict, cluster_assignments_dict
            
        except Exception as e:
            self.logger.error(f"Error applying clustering perspectives: {str(e)}")
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
        Saves the COMPLETE results with ALL original variables to output files.
        FIXED: Now saves the entire dataset, not just essential columns.
        """
        try:
            self.logger.info("Saving classification results")
            
            # Get output file path
            output_file = self.config.get_output_file_path()
            self.logger.info(f"Saving results to {output_file}")
            
            # Ensure the directory exists
            output_dir = os.path.dirname(output_file)
            FileOperationUtilities.create_directory_if_not_exists(output_dir)
            
            self.logger.info(f"Preparing COMPLETE DataFrame with {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")
            
            # FIXED: Save ALL columns from the original dataset
            df_to_save = dataframe.copy()
            
            # Fix column names for Stata compatibility
            column_mapping = {}
            for col in df_to_save.columns:
                # Stata column names: max 32 chars, no special chars except underscore
                clean_col = str(col)[:32]
                clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', clean_col)
                if clean_col != col:
                    column_mapping[col] = clean_col
            
            if column_mapping:
                df_to_save = df_to_save.rename(columns=column_mapping)
                self.logger.info(f"Renamed {len(column_mapping)} columns for Stata compatibility")
            
            # Fix data types for Stata compatibility - but keep ALL columns
            problematic_columns = []
            for col in df_to_save.columns:
                try:
                    # Convert object columns that look numeric
                    if df_to_save[col].dtype == 'object':
                        # Try to convert to numeric if possible
                        numeric_col = pd.to_numeric(df_to_save[col], errors='coerce')
                        if not numeric_col.isna().all():
                            df_to_save[col] = numeric_col
                        else:
                            # Keep as string but ensure it's clean
                            df_to_save[col] = df_to_save[col].astype(str)
                            # Replace problematic values
                            df_to_save[col] = df_to_save[col].replace(['nan', 'None', 'NaN'], '')
                except Exception as e:
                    problematic_columns.append(col)
                    self.logger.warning(f"Could not process column {col}: {e}")
            
            if problematic_columns:
                self.logger.warning(f"Found {len(problematic_columns)} problematic columns that may cause Stata issues")
            
            # Handle missing values - Stata doesn't like certain NaN representations
            df_to_save = df_to_save.fillna('')
            
            # Try to save to Stata format with complete dataset
            try:
                self.logger.info("Attempting to save COMPLETE dataset in Stata format...")
                df_to_save.to_stata(output_file, write_index=False, version=117)
                self.logger.info(f"Successfully saved COMPLETE dataset to Stata format: {output_file}")
                
            except Exception as stata_error:
                self.logger.warning(f"Stata format save failed: {stata_error}")
                
                # Fallback 1: Try with version 118
                try:
                    self.logger.info("Trying Stata version 118...")
                    df_to_save.to_stata(output_file, write_index=False, version=118)
                    self.logger.info(f"Successfully saved COMPLETE dataset to Stata format (v118): {output_file}")
                    
                except Exception as stata_error2:
                    self.logger.warning(f"Stata v118 save failed: {stata_error2}")
                    
                    # Fallback 2: Save as CSV and pickle for complete preservation
                    csv_file = output_file.replace('.dta', '_complete.csv')
                    self.logger.warning(f"Saving COMPLETE dataset as CSV: {csv_file}")
                    df_to_save.to_csv(csv_file, index=False)
                    
                    # Also save as pickle for complete data preservation
                    pickle_file = output_file.replace('.dta', '_complete.pkl')
                    df_to_save.to_pickle(pickle_file)
                    self.logger.info(f"Complete dataset saved as pickle: {pickle_file}")
                    
                    # Try one more time with only string conversion for Stata
                    try:
                        self.logger.info("Trying final Stata save with string conversion...")
                        
                        # Convert problematic columns to strings
                        df_stata = df_to_save.copy()
                        for col in problematic_columns:
                            if col in df_stata.columns:
                                df_stata[col] = df_stata[col].astype(str)
                                df_stata[col] = df_stata[col].replace(['nan', 'None'], '')
                        
                        # Remove any remaining problematic columns for Stata
                        stata_safe_df = df_stata.copy()
                        for col in df_stata.columns:
                            try:
                                # Test if column can be saved to Stata
                                test_df = pd.DataFrame({col: df_stata[col].iloc[:10]})
                                test_df.to_stata('test_temp.dta', write_index=False, version=117)
                                os.remove('test_temp.dta')
                            except:
                                # Remove problematic column
                                self.logger.warning(f"Removing column {col} for Stata compatibility")
                                stata_safe_df = stata_safe_df.drop(columns=[col])
                        
                        stata_safe_df.to_stata(output_file, write_index=False, version=117)
                        self.logger.info(f"Successfully saved dataset to Stata (some columns removed): {output_file}")
                        
                    except Exception as final_error:
                        self.logger.error(f"All Stata save attempts failed: {final_error}")
                        # Final fallback - save as CSV with .dta extension
                        df_to_save.to_csv(output_file, index=False)
                        self.logger.warning(f"Saved as CSV with .dta extension: {output_file}")
            
            # Log summary of added columns
            perspectives = self.config.get_clustering_perspectives()
            added_columns = []
            for name, config in perspectives.items():
                output_column = config.get('output_column')
                if output_column in dataframe.columns:
                    added_columns.append(output_column)
                    label_column = f"{output_column}_label"
                    if label_column in dataframe.columns:
                        added_columns.append(label_column)
            
            self.logger.info(f"COMPLETE dataset saved with ALL {dataframe.shape[1]} original columns")
            self.logger.info(f"NEW classification columns added: {', '.join(added_columns)}")
            
            # Save timestamp file to mark successful completion
            timestamp_file = os.path.join(output_dir, f"classification_completed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(timestamp_file, 'w') as f:
                f.write(f"Classification completed at {datetime.now()}\n")
                f.write(f"Output file: {output_file}\n")
                f.write(f"Total columns saved: {dataframe.shape[1]}\n")
                f.write(f"New classification columns: {', '.join(added_columns)}\n")
                f.write(f"Total rows: {dataframe.shape[0]}\n")
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
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
