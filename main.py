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
from modules.evaluation import ClusteringEvaluator, ClusteringVisualizer, EvaluationReporter


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
            
            # Ensure directories exist
            results_dir = self.config.get_results_dir()
            FileOperationUtilities.create_directory_if_not_exists(results_dir)
            
            # Log configuration summary
            self.logger.info(f"Configuration loaded from: {self.config_file}")
            self.logger.info(f"Input file: {self.config.get_input_file_path()}")
            self.logger.info(f"Output file: {self.config.get_output_file_path()}")
            self.logger.info(f"Results directory: {results_dir}")
            
            # Initialize other components
            self.initialize_components()
            
            # Mark as initialized
            self.initialized = True
            
            self.performance_monitor.stop_timer('setup')
            self.logger.info(f"Pipeline setup completed in {self.performance_monitor.operation_durations['setup'][-1]:.2f} seconds")
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error during pipeline setup: {str(e)}")
                self.logger.error(traceback.format_exc())
            else:
                print(f"Error during pipeline setup: {str(e)}")
                print(traceback.format_exc())
            
            return False

    def run(self):
        """
        Executes the complete classification pipeline.
        
        Returns:
            bool: True if the pipeline ran successfully, False otherwise
        """
        if not self.initialized:
            if not self.setup():
                return False
        
        try:
            self.logger.info("Starting classification pipeline execution")
            self.performance_monitor.start_timer('total_pipeline')
            
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
            
            # Step 3: Evaluate and generate reports
            self.performance_monitor.start_timer('evaluation')
            evaluation_results = self.evaluate_and_report(result_dataframe, features_dict, cluster_assignments_dict)
            if evaluation_results is None:
                self.logger.warning("Evaluation produced no results, but continuing with saving")
            self.performance_monitor.stop_timer('evaluation')
            
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
            
            result_dataframe = dataframe
            
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
                            # Update the result dataframe with the perspective's clustering column
                            output_column = perspective_config.get('output_column')
                            if output_column in perspective_df.columns:
                                result_dataframe[output_column] = perspective_df[output_column]
                                # If labels were generated, add them too
                                label_column = f"{output_column}_label"
                                if label_column in perspective_df.columns:
                                    result_dataframe[label_column] = perspective_df[label_column]
                                
                                # Store features and assignments
                                features_dict[f"{perspective_name}_combined"] = perspective_features
                                cluster_assignments_dict[perspective_name] = perspective_assignments
                                
                                self.logger.info(f"Successfully loaded perspective {perspective_name} from checkpoint")
                                self.performance_monitor.stop_timer(f'perspective_{perspective_name}')
                                continue
                    
                    # If no checkpoint or loading failed, process the perspective
                    perspective_df, perspective_features, perspective_assignments = self.classifier_manager.classify_perspective(
                        result_dataframe, perspective_name, perspective_config
                    )
                    
                    # Update the result dataframe with the perspective's output
                    output_column = perspective_config.get('output_column')
                    result_dataframe[output_column] = perspective_df[output_column]
                    
                    # If labels were generated, add them too
                    label_column = f"{output_column}_label"
                    if label_column in perspective_df.columns:
                        result_dataframe[label_column] = perspective_df[label_column]
                    
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
            
            # Save overall checkpoint
            self.checkpoint_manager.save_checkpoint(
                (result_dataframe, features_dict, cluster_assignments_dict),
                'clustering_results'
            )
            
            self.logger.info("All clustering perspectives applied")
            return result_dataframe, features_dict, cluster_assignments_dict
            
        except Exception as e:
            self.logger.error(f"Error applying clustering perspectives: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None, None

    def evaluate_and_report(self, dataframe, features_dict, cluster_assignments_dict):
        """
        Evaluates clustering results and generates reports.

        Args:
            dataframe: DataFrame with data and cluster assignments
            features_dict: Dictionary mapping perspective names to feature matrices
            cluster_assignments_dict: Dictionary mapping perspective names to cluster assignments
            
        Returns:
            Dictionary of evaluation results by perspective or None if failed
        """
        try:
            self.logger.info("Evaluating clustering results and generating reports")
            
            # Check if we have a checkpoint
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
            
            # Get clustering perspectives from configuration
            perspectives = self.config.get_clustering_perspectives()
            
            # Dictionary to store evaluation results
            evaluation_results = {}
            
            # Evaluate each perspective
            for perspective_name in cluster_assignments_dict.keys():
                self.logger.info(f"Evaluating perspective: {perspective_name}")
                self.performance_monitor.start_timer(f'evaluate_{perspective_name}')
                
                # Skip if features or assignments are missing
                combined_key = f"{perspective_name}_combined"
                if combined_key not in features_dict or perspective_name not in cluster_assignments_dict:
                    self.logger.warning(f"Missing data for perspective {perspective_name}, skipping evaluation")
                    continue
                
                features = features_dict[combined_key]
                assignments = cluster_assignments_dict[perspective_name]
                
                try:
                    # Evaluate clustering
                    metrics = self.evaluator.evaluate_clustering(features, assignments)
                    
                    # Generate visualizations
                    visualization_paths = {}
                    
                    if 'embeddings_plot' in visualization_types:
                        viz_path = self.visualizer.create_embeddings_plot(features, assignments, perspective_name)
                        visualization_paths['embeddings_plot'] = viz_path
                    
                    if 'silhouette_plot' in visualization_types:
                        viz_path = self.visualizer.create_silhouette_plot(features, assignments, perspective_name)
                        visualization_paths['silhouette_plot'] = viz_path
                    
                    if 'distribution_plot' in visualization_types:
                        viz_path = self.visualizer.create_distribution_plot(assignments, perspective_name)
                        visualization_paths['distribution_plot'] = viz_path
                    
                    # Generate report
                    report_paths = self.reporter.generate_report(perspective_name, metrics, visualization_paths)
                    
                    # Store results
                    evaluation_results[perspective_name] = {
                        'metrics': metrics,
                        'visualization_paths': visualization_paths,
                        'report_paths': report_paths
                    }
                    
                    self.logger.info(f"Evaluation and reporting completed for perspective {perspective_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating perspective {perspective_name}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Continue with other perspectives if one fails
                
                self.performance_monitor.stop_timer(f'evaluate_{perspective_name}')
            
            # Check if any perspectives were evaluated successfully
            if not evaluation_results:
                self.logger.error("No perspectives were evaluated successfully")
                return None
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(evaluation_results, 'evaluation_results')
            
            self.logger.info("Evaluation and reporting completed for all perspectives")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation and reporting: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def save_results(self, dataframe):
        """
        Saves the results to output files.

        Args:
            dataframe: DataFrame with all results
            
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
            
            # Save to Stata format
            self.data_processor.save_data(dataframe, output_file)
            
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