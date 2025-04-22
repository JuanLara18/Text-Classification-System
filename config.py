class ConfigManager:
    """Configuration manager for the entire application."""

    def __init__(self, config_file=None, cli_args=None):
        """
        Initializes the configuration manager.

        Args:
            config_file: Path to the YAML configuration file.
            cli_args: Command line arguments.
        """

    def load_config(self):
        """Loads the configuration from the file and merges it with CLI arguments."""

    def validate_config(self):
        """Validates the configuration for required parameters and consistency."""

    def get_input_file_path(self):
        """Gets the path of the input file."""

    def get_output_file_path(self):
        """Gets the path of the output file."""

    def get_text_columns(self):
        """Gets the text columns to classify."""

    def get_preprocessing_options(self):
        """Gets text preprocessing options."""

    def get_feature_extraction_config(self):
        """Gets the feature extraction configuration."""

    def get_clustering_perspectives(self):
        """Gets clustering perspectives configurations."""

    def get_evaluation_config(self):
        """Gets evaluation configuration."""

    def get_spark_config(self):
        """Gets PySpark configuration."""

    def get_checkpoint_config(self):
        """Gets checkpoint configuration."""

    def get_logging_config(self):
        """Gets logging configuration."""

    def as_dict(self):
        """Gets the complete configuration as a dictionary."""
