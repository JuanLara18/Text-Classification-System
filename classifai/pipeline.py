"""
Classification pipeline orchestrator.
"""

import os
import sys
import traceback
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import ConfigManager
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

from classifai.loader import DataLoader
from classifai.evaluator import PipelineEvaluator
from classifai.saver import ResultSaver


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

            # Set the perspectives attribute
            self.perspectives = self.config.get_clustering_perspectives()

            # Ensure directories exist
            results_dir = self.config.get_results_dir()
            FileOperationUtilities.create_directory_if_not_exists(results_dir)

            # Log configuration summary
            self.logger.info(f"Configuration loaded from: {self.config_file}")
            self.logger.info(f"Input file: {self.config.get_input_file_path()}")
            self.logger.info(f"Output file: {self.config.get_output_file_path()}")
            self.logger.info(f"Results directory: {results_dir}")

            # Initialize the components
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
        Loads and preprocesses the input data. Delegates to DataLoader.

        Returns:
            Preprocessed DataFrame or None if failed
        """
        loader = DataLoader(
            config=self.config,
            logger=self.logger,
            spark_manager=self.spark_manager,
            checkpoint_manager=self.checkpoint_manager,
            data_processor=self.data_processor,
            performance_monitor=self.performance_monitor
        )
        return loader.load_and_preprocess()

    def _identify_missing_rows(self, dataframe, columns):
        """Delegate to DataLoader."""
        loader = DataLoader(
            config=self.config,
            logger=self.logger,
            spark_manager=self.spark_manager,
            checkpoint_manager=self.checkpoint_manager,
            data_processor=self.data_processor,
            performance_monitor=self.performance_monitor
        )
        return loader._identify_missing_rows(dataframe, columns)

    def _identify_completely_empty_rows(self, dataframe, columns):
        """Delegate to DataLoader."""
        loader = DataLoader(
            config=self.config,
            logger=self.logger,
            spark_manager=self.spark_manager,
            checkpoint_manager=self.checkpoint_manager,
            data_processor=self.data_processor,
            performance_monitor=self.performance_monitor
        )
        return loader._identify_completely_empty_rows(dataframe, columns)

    def _get_processable_rows_info(self, dataframe, columns):
        """Delegate to DataLoader."""
        loader = DataLoader(
            config=self.config,
            logger=self.logger,
            spark_manager=self.spark_manager,
            checkpoint_manager=self.checkpoint_manager,
            data_processor=self.data_processor,
            performance_monitor=self.performance_monitor
        )
        return loader._get_processable_rows_info(dataframe, columns)

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

            # Build a transient loader for row-info helpers
            _loader = DataLoader(
                config=self.config,
                logger=self.logger,
                spark_manager=self.spark_manager,
                checkpoint_manager=self.checkpoint_manager,
                data_processor=self.data_processor,
                performance_monitor=self.performance_monitor
            )

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
                    processing_info = _loader._get_processable_rows_info(pandas_df, perspective_columns)

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
        Delegates to PipelineEvaluator.
        """
        # Check for existing evaluation checkpoint
        if self.checkpoint_manager.checkpoint_exists('evaluation_results'):
            self.logger.info("Found checkpoint for evaluation results, attempting to load")
            evaluation_results = self.checkpoint_manager.load_checkpoint('evaluation_results')
            if evaluation_results is not None:
                self.logger.info("Successfully loaded evaluation results from checkpoint")
                return evaluation_results
            self.logger.warning("Failed to load checkpoint, proceeding with full evaluation")

        pipeline_evaluator = PipelineEvaluator(
            config=self.config,
            logger=self.logger,
            perspectives=self.perspectives
        )
        evaluation_results = pipeline_evaluator.evaluate_and_report(dataframe, features_dict, cluster_assignments_dict)

        if evaluation_results is not None:
            self.checkpoint_manager.save_checkpoint(evaluation_results, 'evaluation_results')

        return evaluation_results

    def perform_cross_perspective_analysis(self, dataframe, evaluation_results):
        """
        Performs analysis across different clustering perspectives.
        Delegates to PipelineEvaluator.

        Args:
            dataframe: DataFrame with all clustering results
            evaluation_results: Dictionary of evaluation results by perspective

        Returns:
            Dictionary with cross-perspective analysis results
        """
        pipeline_evaluator = PipelineEvaluator(
            config=self.config,
            logger=self.logger,
            perspectives=self.perspectives
        )
        return pipeline_evaluator.perform_cross_perspective_analysis(dataframe, evaluation_results)

    def save_results(self, dataframe):
        """
        Saves results to Stata format. Delegates to ResultSaver.

        Args:
            dataframe: DataFrame with classification results

        Returns:
            bool: True if results were saved successfully, False otherwise
        """
        saver = ResultSaver(config=self.config, logger=self.logger)
        return saver.save_results(dataframe)

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
