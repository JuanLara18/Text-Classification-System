"""
Result saving module for the classification pipeline.
"""

import os
import re
import traceback
from datetime import datetime


class ResultSaver:
    """Handles saving classification results to disk."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def save(self, dataframe):
        """Save results. Delegates to save_results."""
        return self.save_results(dataframe)

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
                        self.logger.info(f"Successfully saved using {strategy_name} strategy")
                        break
                except Exception as strategy_error:
                    self.logger.warning(f"Save strategy {strategy_name} failed: {strategy_error}")
                    continue
            else:
                # All strategies failed
                self.logger.error("All save strategies failed")
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
            self.logger.error(f"STATA FILE COULD NOT BE CREATED")
            self.logger.error(f"Results saved as CSV: {csv_file}")
            self.logger.error(f"Added classification columns: {', '.join(new_columns)}")
            return True
        except Exception as e:
            self.logger.error(f"Even CSV backup failed: {e}")
            return False
