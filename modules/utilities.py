import os
import sys
import time
import logging
import shutil
import datetime
import pickle
from pathlib import Path
import psutil
from pyspark.sql import SparkSession
from collections import defaultdict
import glob
import nltk


class Logger:
    """Logging system for the entire application."""

    def __init__(self, config):
        """
        Initializes the logger with the specified configuration.
        
        Args:
            config: Configuration manager containing logging configuration
        """
        try:
            # Extract logging configuration
            log_config = config.get_logging_config()
            
            # Set up logging level
            level_str = log_config.get('level', 'INFO')
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR
            }
            level = level_map.get(level_str, logging.INFO)
            
            # Create logger instance
            self.logger = logging.getLogger('classification_system')
            self.logger.setLevel(level)
            self.logger.handlers = []  # Clear any existing handlers
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Add console handler if configured
            if log_config.get('console_output', True):
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # Add file handler if log file is specified
            log_file = log_config.get('log_file')
            if log_file:
                # Ensure directory exists for log file
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            self.logger.info("Logger initialized successfully")
            
        except Exception as e:
            # Fallback to basic configuration in case of failures
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
            self.logger = logging.getLogger('classification_system')
            self.logger.error(f"Failed to initialize logger with config: {str(e)}")
    
    def info(self, message):
        """
        Logs an informational message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message):
        """
        Logs a warning message.
        
        Args:
            message: Warning message to log
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        Logs an error message.
        
        Args:
            message: Error message to log
        """
        self.logger.error(message)
    
    def debug(self, message):
        """
        Logs a debug message.
        
        Args:
            message: Debug message to log
        """
        self.logger.debug(message)


class SparkSessionManager:
    """Manager for PySpark sessions."""
    
    def __init__(self, config):
        """
        Initializes the manager with the Spark configuration.
        
        Args:
            config: Configuration manager containing Spark configuration
        """
        try:
            # Extract Spark configuration
            self.spark_config = config.get_spark_config()
            self.app_name = "Text Classification System"
            self.session = None
            
        except Exception as e:
            print(f"Failed to initialize SparkSessionManager: {str(e)}")
            self.spark_config = {}
            self.app_name = "Text Classification System"
    
    def get_or_create_session(self):
        """
        Gets an existing session or creates a new one.
        
        Returns:
            Active SparkSession
        """
        if self.session is not None and not self.session._jsc.sc().isStopped():
            return self.session
        
        try:
            # Start building the session
            builder = SparkSession.builder.appName(self.app_name)

            # Apply configuration settings
            builder = builder.config("spark.driver.memory", 
                                    self.spark_config.get('driver_memory', '4g'))

            builder = builder.config("spark.executor.memory", 
                                    self.spark_config.get('executor_memory', '4g'))

            builder = builder.config("spark.executor.cores", 
                                    self.spark_config.get('executor_cores', 2))

            builder = builder.config("spark.default.parallelism", 
                                    self.spark_config.get('default_parallelism', 4))

            # Configuración para NLTK y recursos externos
            nltk_data_dir = os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data"))

            # Fixed Arrow configuration for pandas compatibility
            builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Changed to false
            builder = builder.config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")  # Added fallback
            builder = builder.config("spark.sql.adaptive.enabled", "true")
            builder = builder.config("spark.driver.extraPythonPath", nltk_data_dir)
            builder = builder.config("spark.executor.extraPythonPath", nltk_data_dir)

            # Minimize shuffling for text processing workloads
            builder = builder.config("spark.sql.shuffle.partitions", "10")

            # Create and cache the session
            self.session = builder.getOrCreate()

            return self.session
        
        except Exception as e:
            print(f"Failed to create or get Spark session: {str(e)}")
            # Return a simpler session as fallback
            return SparkSession.builder.appName(self.app_name).getOrCreate()
    
    def stop_session(self):
        """Stops the current Spark session."""
        if self.session is not None and not self.session._jsc.sc().isStopped():
            try:
                self.session.stop()
                self.session = None
            except Exception as e:
                print(f"Error stopping Spark session: {str(e)}")


class FileOperationUtilities:
    """Utilities for file operations."""
    
    @staticmethod
    def validate_file_path(file_path):
        """
        Validates that a file path exists.
        
        Args:
            file_path: Path to validate
            
        Returns:
            bool: True if the file exists
            
        Raises:
            ValueError: If the file does not exist
        """
        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"File does not exist or is not a valid file: {file_path}")
        return True
    
    @staticmethod
    def create_directory_if_not_exists(directory_path):
        """
        Creates a directory if it does not exist.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            str: Path to the directory
        """
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path
    
    @staticmethod
    def clean_directory(directory_path):
        """
        Cleans a directory by deleting all files.
        
        Args:
            directory_path: Path to the directory to clean
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not directory_path or not os.path.exists(directory_path):
                return False
                
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            return True
        except Exception as e:
            print(f"Error cleaning directory {directory_path}: {str(e)}")
            return False


class PerformanceMonitor:
    """Performance monitor for the application."""
    
    def __init__(self):
        """Initializes the performance monitor."""
        self.timers = {}
        self.memory_usage_records = []
        self.operation_durations = defaultdict(list)
        self.start_time = datetime.datetime.now()
    
    def start_timer(self, operation_name):
        """
        Starts a timer for an operation.
        
        Args:
            operation_name: Name of the operation to time
        """
        self.timers[operation_name] = time.time()
        
        # Record memory usage at the start
        memory_info = psutil.Process(os.getpid()).memory_info()
        self.memory_usage_records.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'operation': f"start_{operation_name}",
            'rss_mb': memory_info.rss / (1024 * 1024),  # Convert to MB
            'vms_mb': memory_info.vms / (1024 * 1024)   # Convert to MB
        })
    
    def stop_timer(self, operation_name):
        """
        Stops a timer and logs the duration.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            float: Duration in seconds or None if timer not found
        """
        if operation_name not in self.timers:
            return None
        
        duration = time.time() - self.timers[operation_name]
        self.operation_durations[operation_name].append(duration)
        
        # Record memory usage at the end
        memory_info = psutil.Process(os.getpid()).memory_info()
        self.memory_usage_records.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'operation': f"end_{operation_name}",
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024)
        })
        
        # Remove the timer
        del self.timers[operation_name]
        
        return duration
    
    def memory_usage(self):
        """
        Gets the current memory usage.
        
        Returns:
            dict: Memory usage information
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            'percent': process.memory_percent(),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def report_performance(self):
        """
        Generates a report of performance metrics.
        
        Returns:
            dict: Performance report
        """
        # Calculate statistics for each operation
        operation_stats = {}
        for operation, durations in self.operation_durations.items():
            if durations:
                operation_stats[operation] = {
                    'min_seconds': min(durations),
                    'max_seconds': max(durations),
                    'avg_seconds': sum(durations) / len(durations),
                    'total_seconds': sum(durations),
                    'count': len(durations)
                }
        
        # Get current memory usage
        current_memory = self.memory_usage()
        
        # Calculate overall execution time
        overall_duration = (datetime.datetime.now() - self.start_time).total_seconds()
        
        return {
            'operations': operation_stats,
            'memory': {
                'current': current_memory,
                'history': self.memory_usage_records[-10:],  # Last 10 records to avoid too much data
                'record_count': len(self.memory_usage_records)
            },
            'overall_duration_seconds': overall_duration,
            'generated_at': datetime.datetime.now().isoformat()
        }


class CheckpointManager:
    """Manager for checkpoints for robust processing."""
    
    def __init__(self, config):
        """
        Initializes the manager with the specified configuration.
        
        Args:
            config: Configuration manager containing checkpoint configuration
        """
        try:
            # Extract checkpoint configuration
            checkpoint_config = config.get_checkpoint_config()
            
            self.enabled = checkpoint_config.get('enabled', True)
            self.directory = checkpoint_config.get('directory', 'checkpoints')
            self.max_checkpoints = checkpoint_config.get('max_checkpoints', 5)
            self.interval = checkpoint_config.get('interval', 1)
            
            # Create checkpoint directory if it doesn't exist and checkpointing is enabled
            if self.enabled:
                os.makedirs(self.directory, exist_ok=True)
        
        except Exception as e:
            print(f"Error initializing CheckpointManager: {str(e)}")
            self.enabled = False
            self.directory = 'checkpoints'
            self.max_checkpoints = 5
            self.interval = 1
        
    def save_checkpoint(self, data, step_name):
        """
        Saves a checkpoint for a processing step.
        
        Args:
            data: Data to save
            step_name: Name of the processing step
            
        Returns:
            bool: True if checkpoint was saved successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Handle Spark DataFrame by converting to pandas
            from pyspark.sql import DataFrame as SparkDataFrame
            
            # If data is a tuple/list containing SparkDataFrame, process each item
            if isinstance(data, (tuple, list)):
                processed_data = []
                for item in data:
                    if isinstance(item, SparkDataFrame):
                        # Convert Spark DataFrame to pandas
                        processed_item = item.toPandas()
                        processed_data.append(processed_item)
                    else:
                        processed_data.append(item)
                data_to_save = tuple(processed_data) if isinstance(data, tuple) else processed_data
            # If data is a single SparkDataFrame
            elif isinstance(data, SparkDataFrame):
                data_to_save = data.toPandas()
            else:
                data_to_save = data
            
            # Create a timestamped filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{step_name}_{timestamp}.pkl"
            filepath = os.path.join(self.directory, filename)
            
            # Create the directory if it doesn't exist (in case it was deleted)
            os.makedirs(self.directory, exist_ok=True)
            
            # Save the data
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            # Clean old checkpoints for this step
            self._clean_step_checkpoints(step_name)
            
            return True
        
        except Exception as e:
            print(f"Error saving checkpoint for {step_name}: {str(e)}")
            return False

    def load_checkpoint(self, step_name):
        """
        Loads the latest checkpoint if it exists.
        
        Args:
            step_name: Name of the processing step
            
        Returns:
            The loaded data or None if no checkpoint exists
        """
        if not self.enabled:
            return None
        
        try:
            # Find all checkpoints for this step
            pattern = os.path.join(self.directory, f"{step_name}_*.pkl")
            checkpoint_files = sorted(glob.glob(pattern))
            
            if not checkpoint_files:
                return None
            
            # Load the latest checkpoint
            latest_checkpoint = checkpoint_files[-1]
            with open(latest_checkpoint, 'rb') as f:
                data = pickle.load(f)
            
            return data
        
        except Exception as e:
            print(f"Error loading checkpoint for {step_name}: {str(e)}")
            return None
    
    def checkpoint_exists(self, step_name):
        """
        Checks if a checkpoint exists for a step.
        
        Args:
            step_name: Name of the processing step
            
        Returns:
            bool: True if a checkpoint exists, False otherwise
        """
        if not self.enabled or not os.path.exists(self.directory):
            return False
        
        pattern = os.path.join(self.directory, f"{step_name}_*.pkl")
        checkpoint_files = glob.glob(pattern)
        return len(checkpoint_files) > 0
    
    def clean_old_checkpoints(self):
        """
        Cleans old checkpoints according to the configuration.
        
        Returns:
            int: Number of files deleted
        """
        if not self.enabled or not os.path.exists(self.directory):
            return 0
        
        try:
            # Group checkpoints by step
            step_files = defaultdict(list)
            for filename in os.listdir(self.directory):
                if filename.endswith('.pkl'):
                    parts = filename.split('_')
                    if len(parts) > 1:
                        step_name = parts[0]
                        step_files[step_name].append(os.path.join(self.directory, filename))
            
            # Clean old checkpoints for each step
            deleted_count = 0
            for step_name in step_files:
                deleted_count += self._clean_step_checkpoints(step_name)
            
            return deleted_count
        
        except Exception as e:
            print(f"Error cleaning old checkpoints: {str(e)}")
            return 0
    
    def _clean_step_checkpoints(self, step_name):
        """
        Cleans old checkpoints for a specific step.
        
        Args:
            step_name: Name of the processing step
            
        Returns:
            int: Number of files deleted
        """
        if not self.enabled:
            return 0
        
        try:
            pattern = os.path.join(self.directory, f"{step_name}_*.pkl")
            checkpoint_files = sorted(
                [(f, os.path.getmtime(f)) for f in glob.glob(pattern)],
                key=lambda x: x[1]
            )
            
            # Delete older files if we have more than max_checkpoints
            deleted_count = 0
            if len(checkpoint_files) > self.max_checkpoints:
                for filepath, _ in checkpoint_files[:-self.max_checkpoints]:
                    os.remove(filepath)
                    deleted_count += 1
            
            return deleted_count
        
        except Exception as e:
            print(f"Error cleaning checkpoints for step {step_name}: {str(e)}")
            return 0
