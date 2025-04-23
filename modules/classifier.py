import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
import logging
import pickle
import traceback 
from pathlib import Path
import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import warnings
import random
import openai
from collections import defaultdict, Counter
from pyspark.sql import DataFrame as SparkDataFrame
import time
import json


class BaseClusterer:
    """Base class for clustering algorithms."""

    def __init__(self, config, logger):
        """
        Initializes the base clusterer.

        Args:
            config: Configuration manager
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.options = config.get_options()
        self.seed = self.options.get('seed', 42)
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

    def fit(self, features):
        """
        Fits the clustering algorithm to the features.

        Args:
            features: Feature matrix

        Returns:
            Fitted clusterer
        """
        self.logger.warning("fit() method called on BaseClusterer. This method should be implemented by subclasses.")
        return self

    def predict(self, features):
        """
        Predicts cluster assignments for features.

        Args:
            features: Feature matrix

        Returns:
            Cluster assignments
        """
        if self.model is None:
            raise RuntimeError("Clusterer must be fitted before calling predict()")
        
        self.logger.warning("predict() method called on BaseClusterer. This method should be implemented by subclasses.")
        return np.zeros(features.shape[0], dtype=int)
    
    def get_labels(self):
        """
        Gets the cluster labels from the fitted model.

        Returns:
            Cluster assignments
        """
        if self.labels_ is None:
            raise RuntimeError("Clusterer must be fitted before getting labels")
        return self.labels_
    
    def get_cluster_centers(self):
        """
        Gets the cluster centers from the fitted model.

        Returns:
            Cluster centers or None if not available
        """
        return self.cluster_centers_
    
    def _handle_noise_cluster(self, labels):
        """
        Handles noise clusters (labeled as -1) by assigning them to the nearest cluster.
        
        Args:
            labels: Cluster labels
            
        Returns:
            New labels with noise points assigned to clusters
        """
        # If there are no noise points, return original labels
        if -1 not in labels:
            return labels
        
        # Count noise points
        noise_count = np.sum(labels == -1)
        self.logger.info(f"Found {noise_count} noise points ({noise_count/len(labels):.2%} of data)")
        
        return labels


class KMeansClusterer(BaseClusterer):
    """K-Means clustering algorithm."""

    def __init__(self, config, logger, perspective_config):
        """
        Initializes the K-Means clusterer.

        Args:
            config: Configuration manager
            logger: Logger instance
            perspective_config: Configuration for this perspective
        """
        super().__init__(config, logger)
        self.perspective_config = perspective_config
        
        # Get K-Means specific parameters
        self.params = perspective_config.get('params', {})
        self.n_clusters = self.params.get('n_clusters', 8)
        self.random_state = self.params.get('random_state', self.seed)
        self.n_init = self.params.get('n_init', 10)
        self.max_iter = self.params.get('max_iter', 300)
        
        # Get evaluation range if present
        self.k_range = perspective_config.get('evaluate_k_range', None)
        
        self.logger.info(f"Initialized KMeansClusterer with {self.n_clusters} clusters")

    def fit(self, features):
        """
        Fits the K-Means clustering algorithm to the features.

        Args:
            features: Feature matrix

        Returns:
            Fitted clusterer
        """
        # Check if we need to determine optimal k
        if self.k_range is not None:
            self.n_clusters = self.determine_optimal_k(features, self.k_range)
            self.logger.info(f"Using optimal number of clusters: {self.n_clusters}")
        
        try:
            start_time = time.time()
            self.logger.info(f"Fitting K-Means with {self.n_clusters} clusters to {features.shape} feature matrix")
            
            # Initialize and fit the model
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter
            )
            
            self.model.fit(features)
            
            # Store the cluster centers and labels
            self.cluster_centers_ = self.model.cluster_centers_
            self.labels_ = self.model.labels_
            
            # Log number of samples in each cluster
            unique, counts = np.unique(self.labels_, return_counts=True)
            cluster_counts = dict(zip(unique, counts))
            self.logger.info(f"K-Means cluster distribution: {cluster_counts}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"K-Means clustering completed in {elapsed_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error during K-Means clustering: {str(e)}")
            raise RuntimeError(f"K-Means clustering failed: {str(e)}")

    def predict(self, features):
        """
        Predicts cluster assignments for features.

        Args:
            features: Feature matrix

        Returns:
            Cluster assignments
        """
        if self.model is None:
            raise RuntimeError("K-Means model must be fitted before calling predict()")
        
        try:
            self.logger.info(f"Predicting clusters for {features.shape} feature matrix")
            predictions = self.model.predict(features)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting K-Means clusters: {str(e)}")
            raise RuntimeError(f"K-Means prediction failed: {str(e)}")

    def determine_optimal_k(self, features, k_range):
        """
        Determines the optimal number of clusters using the silhouette score.

        Args:
            features: Feature matrix
            k_range: Range of k values to evaluate

        Returns:
            Optimal k value
        """
        if isinstance(k_range, list) and len(k_range) == 2:
            k_min, k_max = k_range
            k_values = range(k_min, k_max + 1)
        else:
            self.logger.warning(f"Invalid k_range: {k_range}. Using default range [2, 10]")
            k_values = range(2, 11)
        
        self.logger.info(f"Determining optimal k in range {min(k_values)} to {max(k_values)}")
        
        # Initialize variables to track best k
        best_k = -1
        best_score = -1
        scores = {}
        
        for k in k_values:
            try:
                # Use subset of data for efficiency if dataset is large
                sample_size = min(10000, features.shape[0])
                if features.shape[0] > sample_size:
                    indices = np.random.choice(features.shape[0], sample_size, replace=False)
                    feature_subset = features[indices]
                else:
                    feature_subset = features
                
                # Fit KMeans with current k
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=min(5, self.n_init),  # Use fewer initializations for speed
                    max_iter=min(100, self.max_iter)  # Use fewer iterations for speed
                )
                
                cluster_labels = kmeans.fit_predict(feature_subset)
                
                # Skip if only one cluster was found (can happen with outliers)
                if len(np.unique(cluster_labels)) < 2:
                    self.logger.warning(f"Only one cluster found for k={k}, skipping")
                    scores[k] = 0
                    continue
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(feature_subset, cluster_labels)
                scores[k] = silhouette_avg
                
                self.logger.info(f"K={k}, Silhouette Score: {silhouette_avg:.4f}")
                
                # Update best k if current is better
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_k = k
            
            except Exception as e:
                self.logger.warning(f"Error evaluating k={k}: {str(e)}")
                scores[k] = 0
        
        # If no valid k was found, use the middle of the range
        if best_k == -1:
            best_k = (min(k_values) + max(k_values)) // 2
            self.logger.warning(f"Could not determine optimal k. Using default: {best_k}")
        else:
            self.logger.info(f"Optimal number of clusters: {best_k} (Silhouette Score: {best_score:.4f})")
        
        return best_k


class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN clustering algorithm."""

    def __init__(self, config, logger, perspective_config):
        """
        Initializes the HDBSCAN clusterer.

        Args:
            config: Configuration manager
            logger: Logger instance
            perspective_config: Configuration for this perspective
        """
        super().__init__(config, logger)
        self.perspective_config = perspective_config
        
        # Get HDBSCAN specific parameters
        self.params = perspective_config.get('params', {})
        self.min_cluster_size = self.params.get('min_cluster_size', 15)
        self.min_samples = self.params.get('min_samples', 5)
        metric = self.params.get('metric', 'euclidean')
        if metric == 'cosine':
            self.logger.warning(
                "HDBSCAN doesn't support 'cosine'. Changed to 'euclidean'."
            )
            self.metric = 'euclidean'
        else:
            self.metric = metric
        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.0)
        self.alpha = self.params.get('alpha', 1.0)
        self.cluster_selection_method = self.params.get('cluster_selection_method', 'eom')  # 'eom' or 'leaf'
        
        self.logger.info(
            f"Initialized HDBSCANClusterer with min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, metric={self.metric}"
        )

    def fit(self, features):
        """
        Fits the HDBSCAN clustering algorithm to the features.

        Args:
            features: Feature matrix

        Returns:
            Fitted clusterer
        """
        try:
            start_time = time.time()
            self.logger.info(
                f"Fitting HDBSCAN with min_cluster_size={self.min_cluster_size} to "
                f"{features.shape} feature matrix"
            )
            
            # Initialize HDBSCAN
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                alpha=self.alpha,
                cluster_selection_method=self.cluster_selection_method,
                gen_min_span_tree=True,  # Needed for visualization
                core_dist_n_jobs=-1  # Use all available processors
            )
            
            # Fit the model
            self.model.fit(features)
            
            # Store the labels
            self.labels_ = self.model.labels_
            
            # HDBSCAN doesn't provide cluster centers, but we can compute them
            unique_labels = np.unique(self.labels_)
            centers = []
            for label in unique_labels:
                if label != -1:  # Ignore noise cluster
                    mask = self.labels_ == label
                    center = features[mask].mean(axis=0)
                    centers.append(center)
            
            if centers:
                self.cluster_centers_ = np.vstack(centers)
            
            # Process noise points if requested
            if self.params.get('handle_noise_points', True) and -1 in self.labels_:
                self.labels_ = self._handle_noise_cluster(self.labels_)
            
            # Log number of samples in each cluster
            unique, counts = np.unique(self.labels_, return_counts=True)
            cluster_counts = dict(zip(unique, counts))
            self.logger.info(f"HDBSCAN cluster distribution: {cluster_counts}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"HDBSCAN clustering completed in {elapsed_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error during HDBSCAN clustering: {str(e)}")
            raise RuntimeError(f"HDBSCAN clustering failed: {str(e)}")

    def predict(self, features):
        """
        Predicts cluster assignments for features.

        Args:
            features: Feature matrix

        Returns:
            Cluster assignments
        """
        if self.model is None:
            raise RuntimeError("HDBSCAN model must be fitted before calling predict()")
        
        try:
            self.logger.info(f"Approximating cluster predictions for {features.shape} feature matrix")
            
            # HDBSCAN doesn't have a native predict method for new samples
            # We'll use the approximate_predict method instead
            predictions, strengths = hdbscan.approximate_predict(self.model, features)
            
            # Handle noise points if requested
            if self.params.get('handle_noise_points', True) and -1 in predictions:
                predictions = self._assign_noise_to_nearest(features, predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting HDBSCAN clusters: {str(e)}")
            raise RuntimeError(f"HDBSCAN prediction failed: {str(e)}")
    
    def _handle_noise_cluster(self, labels):
        """
        Handles noise points in HDBSCAN clustering.
        
        Args:
            labels: Cluster labels
            
        Returns:
            New labels with noise points assigned to clusters
        """
        # Count noise points
        noise_count = np.sum(labels == -1)
        self.logger.info(f"Found {noise_count} noise points ({noise_count/len(labels):.2%} of data)")
        
        # If more than 50% are noise, something is wrong with parameters
        if noise_count > len(labels) * 0.5:
            self.logger.warning(
                f"More than 50% of points classified as noise. "
                f"Consider adjusting min_cluster_size or min_samples."
            )
        
        # For now, we'll create a separate cluster for noise points
        # A more sophisticated approach would assign them to the nearest cluster
        new_labels = labels.copy()
        if noise_count > 0:
            noise_cluster_id = np.max(labels) + 1
            new_labels[labels == -1] = noise_cluster_id
            self.logger.info(f"Assigned noise points to new cluster {noise_cluster_id}")
        
        return new_labels
    
    def _assign_noise_to_nearest(self, features, predictions):
        """
        Assigns noise points to the nearest cluster.
        
        Args:
            features: Feature matrix
            predictions: Cluster predictions with noise (-1)
            
        Returns:
            New predictions with noise assigned to clusters
        """
        new_predictions = predictions.copy()
        noise_mask = predictions == -1
        
        # If there are no noise points or no valid clusters, return original
        if not np.any(noise_mask) or np.max(predictions) < 0:
            return predictions
        
        # Get unique cluster labels (excluding noise)
        valid_clusters = np.unique(predictions[~noise_mask])
        
        # Get noise points
        noise_features = features[noise_mask]
        
        # For each noise point, find the nearest cluster center
        for i, noise_idx in enumerate(np.where(noise_mask)[0]):
            min_dist = float('inf')
            best_cluster = -1
            
            for cluster in valid_clusters:
                # Get all points in this cluster
                cluster_features = features[predictions == cluster]
                
                # Compute distance to cluster center
                center = np.mean(cluster_features, axis=0)
                dist = np.linalg.norm(noise_features[i] - center)
                
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster
            
            if best_cluster != -1:
                new_predictions[noise_idx] = best_cluster
        
        return new_predictions


class AgglomerativeClusterer(BaseClusterer):
    """Agglomerative clustering algorithm."""

    def __init__(self, config, logger, perspective_config):
        """
        Initializes the Agglomerative clusterer.

        Args:
            config: Configuration manager
            logger: Logger instance
            perspective_config: Configuration for this perspective
        """
        super().__init__(config, logger)
        self.perspective_config = perspective_config
        
        # Get Agglomerative specific parameters
        self.params = perspective_config.get('params', {})
        self.n_clusters = self.params.get('n_clusters', 8)
        self.linkage = self.params.get('linkage', 'ward')
        self.affinity = self.params.get('affinity', 'euclidean')
        
        # Validate parameters
        if self.linkage == 'ward' and self.affinity != 'euclidean':
            self.logger.warning(
                f"Ward linkage requires euclidean affinity. "
                f"Changing affinity from '{self.affinity}' to 'euclidean'"
            )
            self.affinity = 'euclidean'
        
        self.compute_distances = self.params.get('compute_distances', False)
        
        self.logger.info(
            f"Initialized AgglomerativeClusterer with n_clusters={self.n_clusters}, "
            f"linkage={self.linkage}, affinity={self.affinity}"
        )

    def fit(self, features):
        """
        Fits the Agglomerative clustering algorithm to the features.

        Args:
            features: Feature matrix

        Returns:
            Fitted clusterer
        """
        try:
            start_time = time.time()
            self.logger.info(
                f"Fitting Agglomerative clustering with {self.n_clusters} clusters "
                f"to {features.shape} feature matrix"
            )
            
            # Initialize and fit the model
            if self.linkage == 'ward':
                self.model = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage=self.linkage,
                )
            else:
                self.model = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage=self.linkage,
                    metric=self.affinity 
                )
            
            self.model.fit(features)
            
            # Store the labels
            self.labels_ = self.model.labels_
            
            # Agglomerative clustering doesn't provide cluster centers directly
            # Calculate them manually
            unique_labels = np.unique(self.labels_)
            centers = []
            for label in unique_labels:
                mask = self.labels_ == label
                center = features[mask].mean(axis=0)
                centers.append(center)
            
            self.cluster_centers_ = np.vstack(centers)
            
            # Log number of samples in each cluster
            unique, counts = np.unique(self.labels_, return_counts=True)
            cluster_counts = dict(zip(unique, counts))
            self.logger.info(f"Agglomerative cluster distribution: {cluster_counts}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Agglomerative clustering completed in {elapsed_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error during Agglomerative clustering: {str(e)}")
            raise RuntimeError(f"Agglomerative clustering failed: {str(e)}")

    def predict(self, features):
        """
        Predicts cluster assignments for features.

        Args:
            features: Feature matrix

        Returns:
            Cluster assignments
        """
        if self.model is None or self.cluster_centers_ is None:
            raise RuntimeError("Agglomerative model must be fitted before calling predict()")
        
        try:
            self.logger.info(f"Predicting clusters for {features.shape} feature matrix")
            
            # Agglomerative clustering doesn't have a native predict method
            # Assign each sample to the nearest cluster center
            predictions = np.zeros(features.shape[0], dtype=int)
            
            for i in range(features.shape[0]):
                # Calculate distances to all cluster centers
                distances = np.linalg.norm(self.cluster_centers_ - features[i], axis=1)
                # Assign to the closest cluster
                predictions[i] = np.argmin(distances)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting Agglomerative clusters: {str(e)}")
            raise RuntimeError(f"Agglomerative prediction failed: {str(e)}")


class ClusterLabeler:
    """Labeler for clusters."""

    def __init__(self, config, logger):
        """
        Initializes the cluster labeler.

        Args:
            config: Configuration manager
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.labeling_config = config.get_cluster_labeling_config()
        self.method = self.labeling_config.get('method', 'tfidf')
        
        # Initialize OpenAI API if needed
        if self.method == 'openai':
            openai_config = self.labeling_config.get('openai', {})
            api_key_env = openai_config.get('api_key_env', 'OPENAI_API_KEY')
            api_key = os.environ.get(api_key_env)
            
            if not api_key:
                self.logger.warning(
                    f"OpenAI API key not found in environment variable: {api_key_env}. "
                    f"Falling back to TF-IDF labeling."
                )
                self.method = 'tfidf'
            else:
                openai.api_key = api_key
        
        self.logger.info(f"Initialized ClusterLabeler using {self.method} method")

    def generate_labels(self, dataframe, text_columns, cluster_column):
        """
        Generates labels for clusters.

        Args:
            dataframe: DataFrame with the data
            text_columns: Text columns used for clustering
            cluster_column: Column with cluster assignments

        Returns:
            Dictionary mapping cluster IDs to labels
        """
        self.logger.info(f"Generating cluster labels for {cluster_column} using {self.method} method")
        
        try:
            # Validate inputs
            if cluster_column not in dataframe.columns:
                raise ValueError(f"Cluster column '{cluster_column}' not found in DataFrame")
            
            for col in text_columns:
                if col not in dataframe.columns:
                    self.logger.warning(f"Text column '{col}' not found in DataFrame")
            
            # Filter to only use available columns
            valid_text_columns = [col for col in text_columns if col in dataframe.columns]
            
            if not valid_text_columns:
                raise ValueError("No valid text columns found for labeling")
            
            # Get cluster examples for labeling
            cluster_examples = self._get_cluster_examples(dataframe, valid_text_columns, cluster_column)
            
            # Generate labels based on method
            if self.method == 'openai':
                labels = self.generate_openai_labels(cluster_examples)
            elif self.method == 'tfidf':
                labels = self.generate_tfidf_labels(dataframe, valid_text_columns, cluster_column)
            elif self.method == 'manual':
                # Placeholder for manual labeling - use cluster IDs as labels
                unique_clusters = dataframe[cluster_column].unique()
                labels = {cluster_id: f"Cluster {cluster_id}" for cluster_id in unique_clusters}
            else:
                self.logger.warning(f"Unknown labeling method: {self.method}. Using TF-IDF method instead.")
                labels = self.generate_tfidf_labels(dataframe, valid_text_columns, cluster_column)
            
            self.logger.info(f"Generated {len(labels)} cluster labels")
            return labels
            
        except Exception as e:
            self.logger.error(f"Error generating cluster labels: {str(e)}")
            # Return generic labels as fallback
            unique_clusters = dataframe[cluster_column].unique()
            return {cluster_id: f"Cluster {cluster_id}" for cluster_id in unique_clusters}
    
    def _get_cluster_examples(self, dataframe, text_columns, cluster_column):
        """
        Gets representative examples from each cluster for labeling.
        
        Args:
            dataframe: DataFrame with the data
            text_columns: Text columns to use
            cluster_column: Column with cluster assignments
            
        Returns:
            Dictionary mapping cluster IDs to lists of example texts
        """
        examples_per_cluster = self.labeling_config.get('openai', {}).get('examples_per_cluster', 10)
        cluster_examples = defaultdict(list)
        
        # Get unique cluster IDs
        unique_clusters = dataframe[cluster_column].unique()
        
        # For each cluster, get a sample of texts
        for cluster_id in unique_clusters:
            cluster_mask = dataframe[cluster_column] == cluster_id
            cluster_data = dataframe[cluster_mask]
            
            # If cluster is small, use all examples
            sample_size = min(examples_per_cluster, len(cluster_data))
            
            # Sample from the cluster
            if len(cluster_data) > sample_size:
                sample_indices = np.random.choice(len(cluster_data), sample_size, replace=False)
                cluster_sample = cluster_data.iloc[sample_indices]
            else:
                cluster_sample = cluster_data
            
            # Combine text from all columns for each example
            for _, row in cluster_sample.iterrows():
                combined_text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                if combined_text.strip():
                    cluster_examples[cluster_id].append(combined_text)
        
        return cluster_examples

    def generate_openai_labels(self, cluster_examples):
        """
        Generates labels using the OpenAI API.

        Args:
            cluster_examples: Dictionary mapping cluster IDs to example texts

        Returns:
            Dictionary mapping cluster IDs to labels
        """
        openai_config = self.labeling_config.get('openai', {})
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0.3)
        max_tokens = openai_config.get('max_tokens', 30)
        prompt_template = openai_config.get(
            'prompt_template',
            "Based on these examples from a cluster, provide a short and descriptive label for the cluster (max 5 words): {examples}"
        )
        
        labels = {}
        
        self.logger.info(f"Generating labels for {len(cluster_examples)} clusters using OpenAI")
        
        for cluster_id, examples in cluster_examples.items():
            try:
                # Limit number of examples to avoid exceeding token limits
                sample_size = min(5, len(examples))
                selected_examples = examples[:sample_size]
                
                # Create the prompt
                examples_text = " | ".join(selected_examples)
                
                # Truncate examples if too long (avoid token limits)
                if len(examples_text) > 2000:
                    examples_text = examples_text[:2000] + "..."
                
                prompt = prompt_template.replace("{examples}", examples_text)
                
                # Call OpenAI API
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates short, descriptive cluster labels."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract and clean the label
                label = response.choices[0].message.content.strip()
                
                # Remove quotes if present
                label = label.strip('"\'')
                
                # Truncate if too long
                if len(label) > 50:
                    label = label[:47] + "..."
                
                labels[cluster_id] = label
                self.logger.debug(f"Generated label for cluster {cluster_id}: {label}")
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"Error generating OpenAI label for cluster {cluster_id}: {str(e)}")
                labels[cluster_id] = f"Cluster {cluster_id}"
        
        return labels

    def generate_tfidf_labels(self, dataframe, text_columns, cluster_column):
        """
        Generates labels based on top TF-IDF terms.

        Args:
            dataframe: DataFrame with the data
            text_columns: Text columns used for clustering
            cluster_column: Column with cluster assignments

        Returns:
            Dictionary mapping cluster IDs to labels
        """
        tfidf_config = self.labeling_config.get('tfidf', {})
        top_terms = tfidf_config.get('top_terms', 5)
        
        labels = {}
        
        self.logger.info(f"Generating TF-IDF based labels for cluster column: {cluster_column}")
        
        try:
            # Combine all text columns into a single series
            combined_texts = dataframe.apply(
                lambda row: ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])]),
                axis=1
            )
            
            # Get unique cluster IDs
            unique_clusters = dataframe[cluster_column].unique()
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Fit vectorizer on all texts
            text_matrix = vectorizer.fit_transform(combined_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # For each cluster, find top terms
            for cluster_id in unique_clusters:
                cluster_mask = dataframe[cluster_column] == cluster_id

                # Get the TF-IDF values for documents in this cluster
                cluster_indices = np.where(cluster_mask)[0]
                cluster_tfidf = text_matrix[cluster_indices]
                
                # Calculate the average TF-IDF score for each term in this cluster
                cluster_avg = cluster_tfidf.mean(axis=0).A1
                
                # Find indices of top terms for this cluster
                top_indices = cluster_avg.argsort()[-top_terms:][::-1]
                
                # Get the actual terms
                top_cluster_terms = [feature_names[i] for i in top_indices]
                
                # Create a label from top terms
                label = ", ".join(top_cluster_terms)
                
                # Limit label length
                if len(label) > 50:
                    label = label[:47] + "..."
                
                labels[cluster_id] = label
                self.logger.debug(f"Generated TF-IDF label for cluster {cluster_id}: {label}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error generating TF-IDF labels: {str(e)}")
            # Return generic labels as fallback
            return {cluster_id: f"Cluster {cluster_id}" for cluster_id in unique_clusters}

    def extract_cluster_characteristics(self, dataframe, text_columns, cluster_column, cluster_id, vectorizer=None):
        """
        Extracts detailed characteristics for a specific cluster.
        
        This method analyzes a cluster and extracts key information including size,
        percentage of total data, top terms based on TF-IDF scores, and representative
        examples that illustrate the cluster's content.
        
        Args:
            dataframe: DataFrame containing the data
            text_columns: List of text column names to analyze
            cluster_column: Column name containing cluster assignments
            cluster_id: ID of the cluster to analyze
            vectorizer: Optional pre-fitted TF-IDF vectorizer
            
        Returns:
            Dictionary containing cluster characteristics:
                - id: Cluster identifier
                - size: Number of records in the cluster
                - percentage: Percentage of total records
                - top_terms: List of (term, score) tuples for most representative terms
                - examples: List of representative example texts
        """
        self.logger.info(f"Extracting characteristics for cluster {cluster_id} in column {cluster_column}")
        
        try:
            # Create cluster mask
            cluster_mask = dataframe[cluster_column] == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size == 0:
                self.logger.warning(f"No records found for cluster {cluster_id}")
                return {
                    'id': cluster_id,
                    'size': 0,
                    'percentage': 0,
                    'top_terms': [],
                    'examples': []
                }
            
            # Calculate percentage
            total_records = len(dataframe[dataframe[cluster_column].notna()])
            percentage = (cluster_size / total_records) * 100
            
            # Select examples from the cluster
            examples = self.select_representative_examples(
                dataframe, text_columns, cluster_column, cluster_id
            )
            
            # Extract top terms if vectorizer is provided
            top_terms = []
            if vectorizer:
                # Combine text from columns for records in this cluster
                cluster_texts = []
                for _, row in dataframe[cluster_mask].iterrows():
                    combined_text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                    if combined_text.strip():
                        cluster_texts.append(combined_text)
                
                if cluster_texts:
                    # Transform texts to TF-IDF
                    cluster_vectors = vectorizer.transform(cluster_texts)
                    
                    # Calculate average vector (centroid)
                    centroid = cluster_vectors.mean(axis=0).A1
                    
                    # Get feature names
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Get top terms
                    top_indices = centroid.argsort()[-15:][::-1]
                    top_terms = [(feature_names[idx], float(centroid[idx])) for idx in top_indices]
            
            # Create and return characteristics dictionary
            characteristics = {
                'id': cluster_id,
                'size': int(cluster_size),
                'percentage': round(percentage, 2),
                'top_terms': top_terms,
                'examples': examples
            }
            
            self.logger.info(f"Extracted characteristics for cluster {cluster_id}: {cluster_size} records, {len(top_terms)} terms, {len(examples)} examples")
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error extracting cluster characteristics: {str(e)}")
            # Return basic structure with empty values
            return {
                'id': cluster_id,
                'size': 0,
                'percentage': 0,
                'top_terms': [],
                'examples': []
            }
        
    def create_detailed_naming_prompt(self, characteristics, perspective_name, domain_context=None):
        """
        Creates a detailed prompt for OpenAI to generate high-quality cluster labels.
        
        This method constructs a structured prompt containing rich context about each 
        cluster, providing clear guidelines for naming and including representative
        examples and key terms.
        
        Args:
            characteristics: List of cluster characteristic dictionaries 
                            (from extract_cluster_characteristics)
            perspective_name: Name of the clustering perspective (e.g., "Content Categories")
            domain_context: Optional domain-specific context to guide naming (default: None)
            
        Returns:
            String containing the formatted prompt for OpenAI
        """
        self.logger.info(f"Creating detailed naming prompt for {perspective_name} clusters")
        
        # Default domain context if none provided
        if not domain_context:
            domain_context = "educational materials and learning assets"
        
        # Create main header and guidelines
        prompt = [
            f"# Cluster Naming Task: {perspective_name}",
            f"\nObjective: Create precise, descriptive names for {len(characteristics)} clusters of {domain_context}.",
            "\n## Guidelines:",
            "- Generate concise yet descriptive names (3-5 words) that capture the core theme of each cluster",
            "- Use domain-appropriate terminology related to learning materials and educational content",
            "- Ensure names are distinctive and non-overlapping between clusters",
            "- Prioritize subject matter and content theme over format or metadata characteristics",
            "- Names should be immediately understandable to educators and content designers",
            "- Avoid generic names like 'General Information' or 'Miscellaneous Content'",
            f"\n## {perspective_name} Clusters Analysis"
        ]
        
        # Add detailed information for each cluster
        for cluster in characteristics:
            if 'id' not in cluster or 'size' not in cluster:
                continue
                
            # Calculate percentage if not provided
            percentage = cluster.get('percentage', 0)
            
            # Add cluster header and stats
            prompt.extend([
                f"\n### Cluster {cluster['id']} ({percentage:.1f}% of total, {cluster['size']} records)"
            ])
            
            # Add key terms section if available
            if 'top_terms' in cluster and cluster['top_terms']:
                prompt.append("\n#### Key Terms (with weights):")
                term_strings = []
                for term, score in cluster['top_terms'][:10]:  # Limit to top 10 terms
                    term_strings.append(f"{term} ({score:.3f})")
                prompt.append(", ".join(term_strings))
            
            # Add examples section if available
            if 'examples' in cluster and cluster['examples']:
                prompt.append("\n#### Representative Examples:")
                for i, example in enumerate(cluster['examples'][:3]):  # Limit to 3 examples
                    # Truncate long examples
                    short_example = example[:200] + "..." if len(example) > 200 else example
                    prompt.append(f"{i+1}. {short_example}")
        
        # Add response format instructions
        prompt.extend([
            "\n## Response Format",
            "Provide a JSON array with exactly one name for each cluster:",
            '{"cluster_names": [',
            '  "Descriptive name for cluster 0",',
            '  "Descriptive name for cluster 1",',
            '  ...',
            ']}',
            "\nEnsure each name reflects the distinctive content of its cluster and is relevant to the domain."
        ])
        
        # Join all parts into a single string
        complete_prompt = "\n".join(prompt)
        
        self.logger.debug(f"Created naming prompt of {len(complete_prompt)} characters")
        return complete_prompt

    def select_representative_examples(self, dataframe, text_columns, cluster_column, cluster_id, n_samples=5):
        """
        Selects diverse, representative examples from a cluster.
        
        This method attempts to select a diverse set of examples that represent
        different aspects of a cluster, not just random samples. It prioritizes
        longer, more content-rich examples when available.
        
        Args:
            dataframe: DataFrame containing the data
            text_columns: List of text column names to consider
            cluster_column: Column name containing cluster assignments
            cluster_id: ID of the cluster to analyze
            n_samples: Number of examples to select (default: 5)
            
        Returns:
            List of example texts from the cluster
        """
        self.logger.info(f"Selecting representative examples for cluster {cluster_id}")
        
        try:
            # Create cluster mask
            cluster_mask = dataframe[cluster_column] == cluster_id
            cluster_data = dataframe[cluster_mask]
            
            if len(cluster_data) == 0:
                self.logger.warning(f"No records found for cluster {cluster_id}")
                return []
                
            # Combine text fields for each example
            combined_texts = []
            for _, row in cluster_data.iterrows():
                combined_text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                if combined_text.strip():  # Only add non-empty texts
                    combined_texts.append(combined_text)
            
            if not combined_texts:
                self.logger.warning(f"No text content found in cluster {cluster_id}")
                return []
            
            # Determine text lengths
            text_lengths = [len(text) for text in combined_texts]
            
            # If we have more texts than requested samples, select strategically
            if len(combined_texts) > n_samples:
                # Get indices sorted by text length (descending)
                sorted_indices = sorted(range(len(text_lengths)), key=lambda i: text_lengths[i], reverse=True)
                
                # Take some from the longest and some diversity 
                # (here taking top 60% longest and 40% from different parts)
                top_count = int(n_samples * 0.6)
                diversity_count = n_samples - top_count
                
                # Get longest texts
                selected_indices = sorted_indices[:top_count]
                
                # Add some diverse selections from different parts of the length distribution
                if diversity_count > 0 and len(sorted_indices) > top_count:
                    remaining = sorted_indices[top_count:]
                    
                    # Get evenly spaced indices from remaining
                    step = max(1, len(remaining) // diversity_count)
                    for i in range(0, len(remaining), step):
                        if len(selected_indices) < n_samples:
                            selected_indices.append(remaining[i])
                        else:
                            break
                
                # Get the selected examples
                examples = [combined_texts[i] for i in selected_indices]
            else:
                # If we have fewer texts than requested, use all of them
                examples = combined_texts
            
            self.logger.info(f"Selected {len(examples)} examples from cluster {cluster_id}")
            return examples
            
        except Exception as e:
            self.logger.error(f"Error selecting representative examples: {str(e)}")
            return []
        
    def generate_enhanced_openai_labels(self, cluster_characteristics, perspective_name):
        """
        Generates high-quality cluster labels using OpenAI with enhanced prompts.
        
        This method creates detailed prompts that provide rich context about each cluster,
        then uses OpenAI's API to generate meaningful, descriptive labels based on
        cluster content and characteristics.
        
        Args:
            cluster_characteristics: List of cluster characteristic dictionaries
                                (from extract_cluster_characteristics)
            perspective_name: Name of the clustering perspective (e.g., "Content Categories")
            
        Returns:
            Dictionary mapping cluster IDs to generated labels
        """
        self.logger.info(f"Generating enhanced OpenAI labels for {perspective_name} clusters")
        
        # Check method configuration
        if self.method != 'openai':
            self.logger.warning(
                f"Method is set to '{self.method}', not 'openai'. "
                f"Switching to OpenAI for enhanced labeling."
            )
        
        # Get OpenAI configuration
        openai_config = self.labeling_config.get('openai', {})
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0.3)
        max_tokens = openai_config.get('max_tokens', 500)
        api_key_env = openai_config.get('api_key_env', 'OPENAI_API_KEY')
        
        # Check API key
        api_key = os.environ.get(api_key_env)
        if not api_key:
            self.logger.error(
                f"OpenAI API key not found in environment variable: {api_key_env}. "
                f"Falling back to TF-IDF labeling."
            )
            # Get cluster IDs and return generic labels
            cluster_ids = [char['id'] for char in cluster_characteristics if 'id' in char]
            return {cluster_id: f"Cluster {cluster_id}" for cluster_id in cluster_ids}
        
        try:
            # Create detailed prompt
            prompt = self.create_detailed_naming_prompt(cluster_characteristics, perspective_name)
            
            # Save prompt for reference
            try:
                prompt_dir = os.path.join(os.getcwd(), "prompts")
                os.makedirs(prompt_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                prompt_path = os.path.join(prompt_dir, f"{perspective_name}_naming_prompt_{timestamp}.txt")
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                self.logger.info(f"Saved naming prompt to: {prompt_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save prompt: {str(e)}")
            
            # Call OpenAI API
            import openai
            openai.api_key = api_key
            
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in educational content classification with deep knowledge of learning materials and instructional design."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Save response for reference
            try:
                response_dir = os.path.join(os.getcwd(), "responses")
                os.makedirs(response_dir, exist_ok=True)
                response_path = os.path.join(response_dir, f"{perspective_name}_naming_response_{timestamp}.txt")
                with open(response_path, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                self.logger.info(f"Saved naming response to: {response_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save response: {str(e)}")
            
            # Parse response to extract cluster names
            labels = {}
            try:
                # First try JSON parsing
                import json
                
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    names_data = json.loads(json_str)
                    names = names_data.get('cluster_names', [])
                    
                    # Map names to cluster IDs
                    for i, char in enumerate(cluster_characteristics):
                        if 'id' in char and i < len(names):
                            labels[char['id']] = names[i]
                    
                    if labels:
                        self.logger.info(f"Successfully parsed {len(labels)} cluster names from JSON response")
                        return labels
                
                # Fallback: Try line-by-line parsing
                self.logger.warning("JSON parsing failed, trying line-by-line extraction")
                names = []
                lines = response_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('"') or line.startswith('-') or line.startswith('*')):
                        # Extract name from line
                        name = line.lstrip('"-* ').strip().strip('"').strip(',').strip()
                        if name and len(name) > 0:
                            names.append(name)
                
                # Map names to cluster IDs
                for i, char in enumerate(cluster_characteristics):
                    if 'id' in char and i < len(names):
                        labels[char['id']] = names[i]
                
                if labels:
                    self.logger.info(f"Extracted {len(labels)} cluster names via line parsing")
                    return labels
                
                # If we still can't parse, use generic labels
                raise ValueError("Failed to parse cluster names from response")
                
            except Exception as e:
                self.logger.error(f"Error parsing cluster names: {str(e)}")
                # Fallback to generic names
                for char in cluster_characteristics:
                    if 'id' in char:
                        labels[char['id']] = f"{perspective_name} Cluster {char['id']}"
                return labels
                
        except Exception as e:
            self.logger.error(f"Error generating enhanced OpenAI labels: {str(e)}")
            # Fallback to generic names
            labels = {}
            for char in cluster_characteristics:
                if 'id' in char:
                    labels[char['id']] = f"{perspective_name} Cluster {char['id']}"
            return labels         
        

class ClassifierManager:
    """Manager for the classification process."""

    def __init__(self, config, logger, data_processor, feature_extractor):
        """
        Initializes the classification manager.

        Args:
            config: Configuration manager
            logger: Logger instance
            data_processor: DataProcessor instance
            feature_extractor: FeatureExtractor instance
        """
        self.config = config
        self.logger = logger
        self.data_processor = data_processor
        self.feature_extractor = feature_extractor
        self.cluster_labeler = ClusterLabeler(config, logger)
        self.perspectives = config.get_clustering_perspectives()
        
        # Dictionary to store clusterers for each perspective
        self.clusterers = {}
        
        # Dictionary to store features for each perspective
        self.features_dict = {}
        
        # Dictionary to store cluster assignments for each perspective
        self.cluster_assignments_dict = {}
        
        self.logger.info(f"ClassifierManager initialized with {len(self.perspectives)} perspectives")

    def classify_perspective(self, dataframe, perspective_name, perspective_config):
        """
        Applies a clustering perspective to the data.

        Args:
            dataframe: DataFrame with the data (either pandas DataFrame or PySpark DataFrame)
            perspective_name: Name of the perspective
            perspective_config: Configuration for this perspective

        Returns:
            Tuple of (DataFrame with added cluster assignments, features, cluster assignments)
        """
        self.logger.info(f"Applying {perspective_name} classification perspective")
        
        try:
            # Get the columns to use for this perspective
            columns = perspective_config.get('columns', [])
            if not columns:
                raise ValueError(f"No columns specified for perspective {perspective_name}")
            
            # Get output column name
            output_column = perspective_config.get('output_column', f"{perspective_name}_cluster")
            
            # Check if columns exist in dataframe
            missing_columns = [col for col in columns if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Columns not found in dataframe: {missing_columns}")
            
            # Get preprocessed columns
            preprocessed_columns = [f"{col}_preprocessed" if f"{col}_preprocessed" in dataframe.columns else col for col in columns]
            
            # Extract features for each column
            features_list = []
            weights = perspective_config.get('weight', [1.0] * len(columns))
            
            # Normalize weights if necessary
            if len(weights) != len(columns):
                self.logger.warning(
                    f"Number of weights ({len(weights)}) doesn't match number of columns ({len(columns)}). "
                    f"Using equal weights."
                )
                weights = [1.0] * len(columns)
            
            weights = np.array(weights) / sum(weights)
            
            # Check if we're working with a PySpark DataFrame
            is_spark_df = isinstance(dataframe, SparkDataFrame)
            
            # Extract or load features for each column
            for i, column in enumerate(columns):
                # Check if features are already extracted for this column
                feature_key = f"{column}_embedding"
                if feature_key in self.features_dict:
                    features = self.features_dict[feature_key]
                    self.logger.info(f"Using existing features for column {column}")
                else:
                    # Extract features
                    self.logger.info(f"Extracting features for column {column}")
                    
                    # Get texts for feature extraction
                    preprocessed_column = preprocessed_columns[i]
                    
                    # Handle PySpark DataFrame differently from pandas DataFrame
                    if is_spark_df:
                        # Convert to pandas for feature extraction
                        pandas_df = dataframe.toPandas()
                        texts = pandas_df[preprocessed_column].dropna().tolist()
                    else:
                        # Para pandas DataFrame, procesar directamente
                        texts = dataframe[preprocessed_column].dropna().tolist()
                    
                    if not texts:
                        self.logger.warning(f"No texts found in column {preprocessed_column}")
                        continue
                    
                    # Extract features based on configuration
                    features = self.feature_extractor.extract_embeddings(texts)
                    self.features_dict[feature_key] = features
                
                # Apply weight to this column's features
                weighted_features = features * weights[i]
                features_list.append(weighted_features)
            
            if not features_list:
                raise ValueError("No features could be extracted for any column")
            
            # Combine features from all columns (weighted average)
            combined_features = np.zeros(features_list[0].shape)
            for features in features_list:
                combined_features += features
            
            # Store the combined features
            self.features_dict[f"{perspective_name}_combined"] = combined_features
            
            # Create and fit the clusterer
            algorithm = perspective_config.get('algorithm', 'kmeans')
            clusterer = self.create_clusterer(algorithm, perspective_config)
            
            # Fit the clusterer
            clusterer.fit(combined_features)
            
            # Get cluster assignments
            cluster_assignments = clusterer.get_labels()
            
            # Store the clusterer and assignments
            self.clusterers[perspective_name] = clusterer
            self.cluster_assignments_dict[perspective_name] = cluster_assignments
            
            # Si estamos usando PySpark, asegurarnos de trabajar con pandas
            if is_spark_df:
                result_df = dataframe.toPandas()
            else:
                # Para pandas DataFrame, usar una copia
                result_df = dataframe.copy()
            
            # Añadir asignaciones de clusters
            result_df[output_column] = cluster_assignments
            
            # Add cluster labels if configured
            if self.config.get_config_value('cluster_analysis.enabled', True):
                # Extract characteristics for enhanced analysis
                cluster_characteristics = []
                for cluster_id in np.unique(cluster_assignments):
                    # Skip noise points (-1) if present
                    if cluster_id == -1:
                        continue
                    
                    # Get characteristics for this cluster
                    characteristics = self.cluster_labeler.extract_cluster_characteristics(
                        result_df, 
                        text_columns, 
                        output_column, 
                        cluster_id,
                        vectorizer=None  # We don't have the vectorizer here
                    )
                    cluster_characteristics.append(characteristics)
                
                # Store characteristics for this perspective
                self.logger.info(f"Extracted characteristics for {len(cluster_characteristics)} clusters")
                
                # Add to DataFrame for tracking
                result_df.attrs['cluster_characteristics'] = cluster_characteristics
                
                # Use enhanced OpenAI naming if configured
                if (self.config.get_config_value('cluster_analysis.enhanced_naming', True) and 
                    self.config.get_config_value('cluster_labeling.method', 'tfidf') == 'openai'):
                    self.logger.info("Using enhanced OpenAI cluster naming")
                    cluster_labels = self.cluster_labeler.generate_enhanced_openai_labels(
                        cluster_characteristics,
                        perspective_name
                    )
                    # Add labels to result DataFrame
                    label_column = f"{output_column}_label"
                    result_df[label_column] = result_df[output_column].map(cluster_labels)
                else:
                    # Use standard labeling approach
                    result_df = self.add_cluster_labels(result_df, perspective_config)
            else:
                # Use standard labeling approach
                result_df = self.add_cluster_labels(result_df, perspective_config)
            
            self.logger.info(f"Completed {perspective_name} classification perspective")
            return result_df, combined_features, cluster_assignments
            
        except Exception as e:
            self.logger.error(f"Error applying perspective {perspective_name}: {str(e)}")
            try:
                self.logger.error(traceback.format_exc())
            except Exception:
                self.logger.error("Error getting traceback")
            raise RuntimeError(f"Failed to apply perspective {perspective_name}: {str(e)}")
    
    def create_clusterer(self, algorithm, perspective_config):
        """
        Creates a clusterer based on the algorithm name.

        Args:
            algorithm: Name of the clustering algorithm
            perspective_config: Configuration for this perspective

        Returns:
            Clusterer instance
        """
        self.logger.info(f"Creating clusterer using algorithm: {algorithm}")
        
        if algorithm.lower() == 'kmeans':
            return KMeansClusterer(self.config, self.logger, perspective_config)
        elif algorithm.lower() in ['hdbscan', 'dbscan']:
            return HDBSCANClusterer(self.config, self.logger, perspective_config)
        elif algorithm.lower() in ['agglomerative', 'hierarchical']:
            return AgglomerativeClusterer(self.config, self.logger, perspective_config)
        else:
            self.logger.warning(f"Unknown algorithm: {algorithm}. Falling back to KMeans.")
            return KMeansClusterer(self.config, self.logger, perspective_config)

    def add_cluster_labels(self, dataframe, perspective_config):
        """
        Adds cluster labels to the DataFrame.

        Args:
            dataframe: DataFrame with the data
            perspective_config: Configuration for this perspective

        Returns:
            DataFrame with added cluster labels
        """
        output_column = perspective_config.get('output_column')
        text_columns = perspective_config.get('columns')
        label_column = f"{output_column}_label"
        
        self.logger.info(f"Adding cluster labels for {output_column}")
        
        try:
            # Generate labels for clusters
            labels = self.cluster_labeler.generate_labels(dataframe, text_columns, output_column)
            
            # Map cluster IDs to labels
            result_df = dataframe.copy()
            result_df[label_column] = result_df[output_column].map(labels)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error adding cluster labels: {str(e)}")
            # Return original dataframe if labeling fails
            return dataframe