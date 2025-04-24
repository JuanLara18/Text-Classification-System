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
        Verifica que todas las condiciones necesarias se cumplan antes 
        de iniciar el procesamiento principal.
        
        Returns:
            bool: True si todas las verificaciones pasan, False en caso contrario
        """
        self.logger.info("Starting environment verification...")
        all_checks_passed = True
        verification_results = {}
        
        # 1. Verificar archivos de entrada/salida
        try:
            input_file = self.config.get_input_file_path()
            if not os.path.exists(input_file):
                verification_results["input_file"] = f"ERROR: Input file not found: {input_file}"
                all_checks_passed = False
            else:
                verification_results["input_file"] = f"SUCCESS: Input file verified: {input_file}"
                
            output_dir = os.path.dirname(self.config.get_output_file_path())
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    verification_results["output_directory"] = f"INFO: Output directory created: {output_dir}"
                except:
                    verification_results["output_directory"] = f"ERROR: Cannot create output directory: {output_dir}"
                    all_checks_passed = False
            else:
                verification_results["output_directory"] = f"SUCCESS: Output directory verified: {output_dir}"
                
            results_dir = self.config.get_results_dir()
            if not os.path.exists(results_dir):
                try:
                    os.makedirs(results_dir, exist_ok=True)
                    verification_results["results_directory"] = f"INFO: Results directory created: {results_dir}"
                except:
                    verification_results["results_directory"] = f"ERROR: Cannot create results directory: {results_dir}"
                    all_checks_passed = False
            else:
                verification_results["results_directory"] = f"SUCCESS: Results directory verified: {results_dir}"
        except Exception as e:
            verification_results["file_paths"] = f"ERROR: Failed to verify file paths: {str(e)}"
            all_checks_passed = False
        
        # 2. Verificar credenciales API (solo si se usa OpenAI)
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
                    # Verificar conexión a la API
                    import openai
                    openai.api_key = api_key
                    try:
                        # Solo hacer una llamada liviana para verificar
                        response = openai.models.list(limit=1)
                        verification_results["openai_api"] = "SUCCESS: OpenAI API key verified and working"
                    except Exception as e:
                        verification_results["openai_api"] = f"WARNING: OpenAI API key verification failed: {str(e)}"
                        verification_results["openai_fallback"] = "INFO: Will fall back to TF-IDF for cluster labeling"
            else:
                verification_results["openai_api"] = f"INFO: OpenAI API not required (using {labeling_method} method)"
        except Exception as e:
            verification_results["openai_api"] = f"ERROR: Failed to verify OpenAI API: {str(e)}"
            # No se considera crítico, por lo que no cambiamos all_checks_passed
        
        # 3. Verificar configuración de clustering perspectivas
        try:
            perspectives = self.config.get_clustering_perspectives()
            if not perspectives:
                verification_results["clustering_perspectives"] = "ERROR: No clustering perspectives found in configuration"
                all_checks_passed = False
            else:
                perspective_issues = []
                for name, config in perspectives.items():
                    if 'algorithm' not in config:
                        perspective_issues.append(f"Missing 'algorithm' in '{name}'")
                    elif config['algorithm'] == 'hdbscan':
                        # Verificar parámetros de HDBSCAN
                        params = config.get('params', {})
                        min_cluster_size = params.get('min_cluster_size', 0)
                        if min_cluster_size < 25:
                            perspective_issues.append(f"Low min_cluster_size ({min_cluster_size}) in '{name}' may cause excessive fragmentation")
                            
                if perspective_issues:
                    verification_results["clustering_perspectives"] = f"WARNING: Issues with clustering perspectives: {', '.join(perspective_issues)}"
                else:
                    verification_results["clustering_perspectives"] = f"SUCCESS: {len(perspectives)} clustering perspectives configured correctly"
        except Exception as e:
            verification_results["clustering_perspectives"] = f"ERROR: Failed to verify clustering perspectives: {str(e)}"
            all_checks_passed = False
        
        # 4. Verificar dependencias críticas
        try:
            import hdbscan
            import umap
            import sklearn
            verification_results["dependencies"] = "SUCCESS: All critical dependencies available"
        except ImportError as e:
            verification_results["dependencies"] = f"ERROR: Missing critical dependency: {str(e)}"
            all_checks_passed = False
        
        # 5. Verificar configuración de Spark
        try:
            # Intentar crear una sesión de Spark para verificar
            spark = self.spark_manager.get_or_create_session()
            spark_version = spark.version
            verification_results["spark"] = f"SUCCESS: Spark session created successfully (version {spark_version})"
        except Exception as e:
            verification_results["spark"] = f"ERROR: Failed to create Spark session: {str(e)}"
            all_checks_passed = False
        
        # 6. Probar carga básica de datos para verificar que el formato está soportado
        try:
            input_file = self.config.get_input_file_path()
            # Solo verificar que podemos leer el archivo, sin cargarlo completamente
            sample_size = 10  # Solo intentar leer algunas filas
            pd_sample = pd.read_stata(input_file, convert_categoricals=False, nrows=sample_size)
            verification_results["data_loading"] = f"SUCCESS: Successfully tested data loading from {input_file}"
            
            # Verificar que las columnas de texto necesarias existen
            text_columns = self.config.get_text_columns()
            missing_columns = [col for col in text_columns if col not in pd_sample.columns]
            if missing_columns:
                verification_results["text_columns"] = f"ERROR: Missing required text columns: {', '.join(missing_columns)}"
                all_checks_passed = False
            else:
                verification_results["text_columns"] = f"SUCCESS: All required text columns present in dataset"
        except Exception as e:
            verification_results["data_loading"] = f"ERROR: Failed to test data loading: {str(e)}"
            all_checks_passed = False
        
        # Mostrar resultados de verificación
        self.logger.info("=== ENVIRONMENT VERIFICATION RESULTS ===")
        for check, result in verification_results.items():
            if result.startswith("ERROR"):
                self.logger.error(result)
            elif result.startswith("WARNING"):
                self.logger.warning(result)
            else:
                self.logger.info(result)
        self.logger.info("=======================================")
        
        # Resultado final
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
                
            # Load data from input file
            self.logger.info(f"Loading data from {input_file}")
            dataframe = self.data_processor.load_data()
            if dataframe is None:
                self.logger.error("Failed to load data")
                return None
            
            # List columns to be preprocessed
            text_columns = self.config.get_text_columns()
            self.logger.info(f"Preprocessing columns: {text_columns}")
            
            # Verify that required columns exist in the dataframe
            missing_columns = [col for col in text_columns if col not in dataframe.columns]
            if missing_columns:
                self.logger.error(f"The following columns specified in the configuration are missing from the dataset: {missing_columns}")
                return None
            
            # Preprocess text columns
            dataframe = self.data_processor.preprocess_text_columns(dataframe, text_columns)
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(dataframe, 'preprocessed_data')
            
            self.logger.info("Data loading and preprocessing completed")
            return dataframe
            
        except Exception as e:
            self.logger.error(f"Error during data loading and preprocessing: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def apply_clustering_perspectives(self, dataframe):
        """
        Applies all clustering perspectives to the data.

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
                            output_column = perspective_config.get('output_column')
                            if output_column in perspective_df.columns:
                                pandas_df[output_column] = perspective_df[output_column]
                                # If labels were generated, add them too
                                label_column = f"{output_column}_label"
                                if label_column in perspective_df.columns:
                                    pandas_df[label_column] = perspective_df[label_column]
                                
                                # Store features and assignments
                                features_dict[f"{perspective_name}_combined"] = perspective_features
                                cluster_assignments_dict[perspective_name] = perspective_assignments
                                
                                self.logger.info(f"Successfully loaded perspective {perspective_name} from checkpoint")
                                self.performance_monitor.stop_timer(f'perspective_{perspective_name}')
                                continue
                    
                    # If no checkpoint or loading failed, process the perspective
                    # Pass the pandas DataFrame to classify_perspective
                    perspective_df, perspective_features, perspective_assignments = self.classifier_manager.classify_perspective(
                        pandas_df, perspective_name, perspective_config
                    )
                    
                    # Update the pandas DataFrame with the perspective's output
                    output_column = perspective_config.get('output_column')
                    pandas_df[output_column] = perspective_df[output_column]
                    
                    # If labels were generated, add them too
                    label_column = f"{output_column}_label"
                    if label_column in perspective_df.columns:
                        pandas_df[label_column] = perspective_df[label_column]
                    
                    # Store features and assignments
                    features_dict[f"{perspective_name}_combined"] = perspective_features
                    cluster_assignments_dict[perspective_name] = perspective_assignments
                    
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
        Evaluates clustering results for each perspective and generates associated reports and visualizations.

        Args:
            dataframe: The original DataFrame containing metadata and results.
            features_dict: Dictionary mapping perspective names to their feature matrices.
            cluster_assignments_dict: Dictionary mapping perspective names to cluster labels.

        Returns:
            Dictionary containing evaluation metrics, visualization paths, and report paths for each perspective.
        """
        try:
            self.logger.info("Evaluating clustering results and generating reports")
            
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
            
            self.logger.info(f"Generating visualizations: {', '.join(visualization_types)}")
            
            # Retrieve clustering perspectives
            perspectives = self.config.get_clustering_perspectives()
            
            # Dictionary to store all evaluation outputs
            evaluation_results = {}
            
            # Initialize ClusterAnalyzer
            cluster_analyzer = ClusterAnalyzer(self.config, self.logger)
            
            # Evaluate each clustering perspective
            for perspective_name in cluster_assignments_dict.keys():
                self.logger.info(f"Evaluating perspective: {perspective_name}")
                self.performance_monitor.start_timer(f'evaluate_{perspective_name}')
                
                # Initialize entry in evaluation results
                evaluation_results[perspective_name] = {
                    'metrics': {},
                    'visualization_paths': {},
                    'report_paths': {}
                }
                
                combined_key = f"{perspective_name}_combined"
                if combined_key not in features_dict or perspective_name not in cluster_assignments_dict:
                    self.logger.warning(f"Missing data for perspective {perspective_name}, skipping evaluation")
                    continue
                
                features = features_dict[combined_key]
                assignments = cluster_assignments_dict[perspective_name]
                
                try:
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
                        
                        try:
                            output_column = perspectives[perspective_name].get('output_column', f"{perspective_name}_cluster")
                            label_column = f"{output_column}_label"
                            cluster_names = {}

                            if label_column in dataframe.columns:
                                for cluster_id in np.unique(assignments):
                                    mask = dataframe[output_column] == cluster_id
                                    if any(mask):
                                        names = dataframe.loc[mask, label_column].dropna()
                                        if len(names) > 0:
                                            cluster_names[cluster_id] = names.iloc[0]

                            viz_path = self.visualizer.create_cluster_size_distribution_plot(
                                assignments, cluster_names, perspective_name
                            )
                            visualization_paths['cluster_size_distribution'] = viz_path
                        except Exception as e:
                            self.logger.error(f"Error creating size distribution plot: {str(e)}")
                        
                        try:
                            if characteristics and len(characteristics) > 0 and 'top_terms' in characteristics[0]:
                                viz_path = self.visualizer.create_cluster_term_importance_plot(
                                    characteristics, perspective_name
                                )
                                visualization_paths['term_importance'] = viz_path
                        except Exception as e:
                            self.logger.error(f"Error creating term importance plot: {str(e)}")
                        
                        try:
                            if self.config.get_config_value('cluster_analysis.create_detailed_reports', True):
                                report_path = self.evaluator.generate_detailed_cluster_report(
                                    perspective_name, characteristics, visualization_paths
                                )
                                evaluation_results[perspective_name]['report_paths']['detailed_report'] = report_path
                        except Exception as e:
                            self.logger.error(f"Error generating detailed cluster report: {str(e)}")
                    
                    try:
                        if 'embeddings_plot' in visualization_types:
                            viz_path = self.visualizer.create_embeddings_plot(features, assignments, perspective_name)
                            visualization_paths['embeddings_plot'] = viz_path
                    except Exception as e:
                        self.logger.error(f"Error creating embeddings plot: {str(e)}")

                    try:
                        if 'silhouette_plot' in visualization_types:
                            viz_path = self.visualizer.create_silhouette_plot(features, assignments, perspective_name)
                            visualization_paths['silhouette_plot'] = viz_path
                    except Exception as e:
                        self.logger.error(f"Error creating silhouette plot: {str(e)}")

                    try:
                        if 'distribution_plot' in visualization_types:
                            viz_path = self.visualizer.create_distribution_plot(assignments, perspective_name)
                            visualization_paths['distribution_plot'] = viz_path
                    except Exception as e:
                        self.logger.error(f"Error creating distribution plot: {str(e)}")

                    evaluation_results[perspective_name]['visualization_paths'] = visualization_paths

                    try:
                        report_paths = self.reporter.generate_report(perspective_name, metrics, visualization_paths)
                        evaluation_results[perspective_name]['report_paths'].update(report_paths)
                    except Exception as e:
                        self.logger.error(f"Error generating reports: {str(e)}")

                    self.logger.info(f"Evaluation and reporting completed for perspective {perspective_name}")
                
                except Exception as e:
                    self.logger.error(f"Error evaluating perspective {perspective_name}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                
                self.performance_monitor.stop_timer(f'evaluate_{perspective_name}')
            
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
        Saves the results to output files.

        Args:
            dataframe: DataFrame with all results (pandas DataFrame)
                
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Saving classification results")
            
            # Get output file path
            output_file = self.config.get_output_file_path()
            self.logger.info(f"Saving results to {output_file}")
            
            # Ensure the directory exists
            output_dir = os.path.dirname(output_file)
            FileOperationUtilities.create_directory_if_not_exists(output_dir)
            
            # No need to check if it's a Spark DataFrame, we're ensuring it's pandas now
            self.logger.info(f"Saving pandas DataFrame with {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")
            
            # Save to Stata format directly
            dataframe.to_stata(output_file, write_index=False, version=117)
            
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
            
            self.logger.info(f"Added classification columns: {', '.join(added_columns)}")
            self.logger.info(f"Results saved to {output_file}")
            
            # Save timestamp file to mark successful completion
            timestamp_file = os.path.join(output_dir, f"classification_completed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(timestamp_file, 'w') as f:
                f.write(f"Classification completed at {datetime.now()}\n")
                f.write(f"Output file: {output_file}\n")
                f.write(f"Added columns: {', '.join(added_columns)}\n")
            
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