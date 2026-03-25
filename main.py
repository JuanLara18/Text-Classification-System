#!/usr/bin/env python3
"""
Main entry point for the classification system.

This module provides the main entry point for classifying text columns in data files,
implementing a modular workflow for preprocessing, clustering, evaluation, and reporting.
"""

import os
import sys
import traceback
import time
from datetime import datetime

from config import ConfigManager, configure_argument_parser
from modules.utilities import FileOperationUtilities
from classifai.pipeline import ClassificationPipeline


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
