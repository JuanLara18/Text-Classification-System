class UniqueRowProcessor:
    """Utility to deduplicate DataFrame rows based on selected text columns."""

    def __init__(self, logger):
        self.logger = logger
        self.row_map = None
        self.unique_df = None

    def prepare_unique_rows(self, dataframe, columns):
        """Return DataFrame with unique rows and mapping to originals."""
        from collections import defaultdict

        if not columns:
            self.logger.warning("No columns provided for deduplication")
            self.row_map = {i: [i] for i in range(len(dataframe))}
            self.unique_df = dataframe.copy()
            return self.unique_df, self.row_map

        self.logger.info(
            f"Finding unique rows using columns: {columns}")

        # Normalize specified columns for comparison
        normalized = dataframe[columns].fillna("").astype(str)
        normalized = normalized.apply(lambda row: tuple(row.str.strip().str.lower()), axis=1)

        row_to_indices = defaultdict(list)
        for idx, key in enumerate(normalized):
            row_to_indices[key].append(idx)

        unique_indices = [indices[0] for indices in row_to_indices.values()]
        self.unique_df = dataframe.iloc[unique_indices].reset_index(drop=True)

        self.row_map = {}
        for new_idx, key in enumerate(row_to_indices.keys()):
            self.row_map[new_idx] = row_to_indices[key]

        reduction_ratio = len(self.unique_df) / len(dataframe) if len(dataframe) else 0
        self.logger.info(
            f"Reduced {len(dataframe)} rows to {len(self.unique_df)} unique rows ({reduction_ratio:.2%} reduction)")

        return self.unique_df, self.row_map

    def map_results_to_full(self, unique_results, original_length):
        """Map results from unique rows back to full dataset."""
        if self.row_map is None:
            raise ValueError("Must call prepare_unique_rows first")

        results = [None] * original_length
        for unique_idx, value in enumerate(unique_results):
            indices = self.row_map.get(unique_idx, [])
            for idx in indices:
                results[idx] = value
        return results
