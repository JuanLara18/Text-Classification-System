"""
Data loading and preprocessing module for the classification pipeline.
"""

import os
import traceback
import pandas as pd


class DataLoader:
    """Handles data loading and preprocessing for the classification pipeline."""

    def __init__(self, config, logger, spark_manager, checkpoint_manager, data_processor, performance_monitor):
        self.config = config
        self.logger = logger
        self.spark_manager = spark_manager
        self.checkpoint_manager = checkpoint_manager
        self.data_processor = data_processor
        self.performance_monitor = performance_monitor
        self._preprocessing_error_count = 0

    def load(self):
        """Load and preprocess the input data. Delegates to load_and_preprocess."""
        return self.load_and_preprocess()

    def load_and_preprocess(self):
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
            self._preprocessing_error_count += 1

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
