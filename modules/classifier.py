import os
import random
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .ai_classifier import OptimizedLLMClassificationManager


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
    """K-Means clustering algorithm implementation.
    
    This class provides an implementation of the K-Means clustering algorithm
    for text classification. It supports automatic determination of the optimal
    number of clusters using silhouette score.
    
    Attributes:
        perspective_config (dict): Configuration for this clustering perspective.
        n_clusters (int): Number of clusters to generate.
        random_state (int): Random seed for reproducibility.
        n_init (int): Number of times to run the algorithm with different centroid seeds.
        max_iter (int): Maximum number of iterations for a single run.
        k_range (list, optional): Range of k values to evaluate for optimal k.
        model (KMeans): The scikit-learn KMeans model after fitting.
        cluster_centers_ (numpy.ndarray): Coordinates of cluster centers.
        labels_ (numpy.ndarray): Cluster labels for each data point.
    """
    
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
        """Determines the optimal number of clusters using silhouette score.
        
        This method evaluates different numbers of clusters within the specified
        range and selects the value that maximizes the silhouette score, which
        measures how well-separated the resulting clusters are.
        
        Args:
            features (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            k_range (list): Range of k values to evaluate, specified as [min_k, max_k].
        
        Returns:
            int: The optimal number of clusters.
            
        Raises:
            ValueError: If k_range is not a list of length 2.
            
        Note:
            For large datasets, this method uses a sample of the data to improve performance.
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
    """HDBSCAN clustering algorithm with enhanced configuration."""

    def __init__(self, config, logger, perspective_config):
        """
        Initializes the HDBSCAN clusterer.

        Args:
            config: Configuration manager
            logger: Logger instance
            perspective_config: Perspective-specific configuration
        """
        super().__init__(config, logger)
        self.perspective_config = perspective_config

        # Retrieve HDBSCAN-specific parameters
        self.params = perspective_config.get('params', {})
        self.min_cluster_size = self.params.get('min_cluster_size', 250)  # Increased default value
        self.min_samples = self.params.get('min_samples', 25)  # Increased default value

        # Set a maximum number of clusters to prevent over-fragmentation
        self.max_clusters = self.params.get('max_clusters', 50)

        metric = self.params.get('metric', 'euclidean')
        if metric == 'cosine':
            self.logger.warning(
                "HDBSCAN doesn't support 'cosine'. Changed to 'euclidean'."
            )
            self.metric = 'euclidean'
        else:
            self.metric = metric

        self.cluster_selection_epsilon = self.params.get('cluster_selection_epsilon', 0.5)  # Increased
        self.alpha = self.params.get('alpha', 1.0)
        self.cluster_selection_method = self.params.get('cluster_selection_method', 'leaf')

        self.logger.info(
            f"Initialized HDBSCANClusterer with min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples}, metric={self.metric}, "
            f"max_clusters={self.max_clusters}"
        )

    def fit(self, features):
        """
        Fits the HDBSCAN clustering algorithm to the given feature matrix with safeguards.

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
                gen_min_span_tree=True,
                core_dist_n_jobs=-1
            )

            # Fit the model
            self.model.fit(features)

            # Store labels
            self.labels_ = self.model.labels_

            # Count unique clusters (excluding noise)
            unique_clusters = set(label for label in np.unique(self.labels_) if label != -1)
            n_clusters = len(unique_clusters)

            # If too many clusters, try increasing min_cluster_size and refit
            original_min_cluster_size = self.min_cluster_size
            refits = 0
            max_refits = 3

            while n_clusters > self.max_clusters and refits < max_refits:
                refits += 1
                self.min_cluster_size = int(self.min_cluster_size * 1.5)
                self.logger.warning(
                    f"Too many clusters ({n_clusters}). Increasing min_cluster_size to "
                    f"{self.min_cluster_size} and refitting (attempt {refits}/{max_refits})"
                )

                self.model = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    alpha=self.alpha,
                    cluster_selection_method=self.cluster_selection_method,
                    gen_min_span_tree=True,
                    core_dist_n_jobs=-1
                )

                self.model.fit(features)
                self.labels_ = self.model.labels_
                unique_clusters = set(label for label in np.unique(self.labels_) if label != -1)
                n_clusters = len(unique_clusters)

            if refits > 0:
                self.logger.info(
                    f"After {refits} refits, final min_cluster_size={self.min_cluster_size}, "
                    f"resulting in {n_clusters} clusters"
                )

            # If still too many clusters, fallback to KMeans
            if n_clusters > self.max_clusters:
                self.logger.warning(
                    f"HDBSCAN produced too many clusters ({n_clusters}). "
                    f"Falling back to KMeans with {self.max_clusters} clusters"
                )

                from sklearn.cluster import KMeans
                kmeans = KMeans(
                    n_clusters=self.max_clusters,
                    random_state=self.seed,
                    n_init=10
                )

                kmeans.fit(features)
                self.labels_ = kmeans.labels_
                self.cluster_centers_ = kmeans.cluster_centers_
                self.model = kmeans

                self.logger.info(f"Fallback to KMeans completed with {self.max_clusters} clusters")
            else:
                # HDBSCAN doesn't provide cluster centers — compute manually
                unique_labels = np.unique(self.labels_)
                centers = []

                for label in unique_labels:
                    if label != -1:  # Ignore noise cluster
                        mask = self.labels_ == label
                        center = features[mask].mean(axis=0)
                        centers.append(center)

                if centers:
                    self.cluster_centers_ = np.vstack(centers)
                else:
                    self.logger.warning("No valid clusters formed. Creating a single artificial cluster.")
                    self.cluster_centers_ = np.mean(features, axis=0).reshape(1, -1)
                    self.labels_ = np.zeros(features.shape[0], dtype=int)

            # Handle noise points if required
            if self.params.get('handle_noise_points', True) and -1 in self.labels_:
                self.labels_ = self._handle_noise_cluster(self.labels_)

            # Log number of samples per cluster
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
        
        # Initialize OpenAI API if specified
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
            elif not api_key.startswith('sk-'):
                self.logger.warning(
                    f"Invalid OpenAI API key format in {api_key_env}. "
                    f"Falling back to TF-IDF labeling."
                )
                self.method = 'tfidf'
            else:
                # Configure API key without validation
                import openai
                openai.api_key = api_key

                # Attempt a basic validation call
                try:
                    response = openai.models.list(limit=1)
                    self.logger.info("Successfully validated OpenAI API key")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to validate OpenAI API key: {str(e)}. "
                        f"Falling back to TF-IDF labeling."
                    )
                    self.method = 'tfidf'
        
        self.logger.info(f"Initialized ClusterLabeler using '{self.method}' method")

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
        
        Args:
            dataframe: DataFrame containing the data
            text_columns: List of text column names to analyze
            cluster_column: Column name containing cluster assignments
            cluster_id: ID of the cluster to analyze
            vectorizer: Optional pre-fitted TF-IDF vectorizer
                
        Returns:
            Dictionary containing cluster characteristics
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
            
            # Initialize vectorizer if not provided
            if vectorizer is None:
                # Create a fresh TF-IDF vectorizer for this specific cluster analysis
                vectorizer = TfidfVectorizer(
                    max_features=3000,
                    stop_words='english',
                    ngram_range=(1, 3),  # Include up to trigrams
                    min_df=2,
                    max_df=0.85  # Ignore terms that appear in >85% of documents
                )
            
            # Combine text from columns for records in this cluster
            cluster_data = dataframe[cluster_mask]
            combined_texts = []
            
            for _, row in cluster_data.iterrows():
                combined_text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                if combined_text.strip():  # Only add non-empty texts
                    combined_texts.append(combined_text)
            
            top_terms = []
            if combined_texts:
                # Fit and transform texts to TF-IDF
                vectors = vectorizer.fit_transform(combined_texts)
                
                # Calculate average vector (centroid)
                centroid = vectors.mean(axis=0).A1
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top terms
                top_indices = centroid.argsort()[-20:][::-1]  # Get more terms
                top_terms = [(feature_names[idx], float(centroid[idx])) for idx in top_indices]
            
            # Select diverse, representative examples
            examples = self.select_representative_examples(
                dataframe, text_columns, cluster_column, cluster_id, n_samples=5
            )
            
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
        Creates a comprehensive and effective prompt for OpenAI to generate high-quality cluster labels.
        
        This method constructs a carefully engineered prompt with clear instructions,
        cluster data, and formatting requirements to obtain meaningful and distinctive
        cluster labels regardless of the data domain.
        
        Args:
            characteristics: List of cluster characteristic dictionaries 
                            (from extract_cluster_characteristics)
            perspective_name: Name of the clustering perspective (e.g., "Content Categories")
            domain_context: Optional domain-specific context to guide naming (default: None)
                
        Returns:
            String containing the formatted prompt for OpenAI
        """
        self.logger.info(f"Creating detailed naming prompt for {perspective_name} clusters")
        
        # Infer domain context if possible when none is provided
        if not domain_context and characteristics and len(characteristics) > 0:
            # Try to infer domain from common terms across clusters
            all_terms = []
            for cluster in characteristics:
                if 'top_terms' in cluster and cluster['top_terms']:
                    all_terms.extend([term for term, _ in cluster['top_terms'][:5]])
            
            # Get most common terms that might indicate domain
            if all_terms:
                term_counter = Counter(all_terms)
                common_terms = [term for term, count in term_counter.most_common(10)]
                inferred_domain = ", ".join(common_terms)
                domain_context = f"data related to {inferred_domain}"
            else:
                domain_context = "the provided data"
        elif not domain_context:
            domain_context = "the provided data"
        
        # Create main header and guidelines
        prompt = [
            f"# Cluster Naming Task: {perspective_name}",
            f"\nYou are an expert in data analysis tasked with creating descriptive, meaningful names for {len(characteristics)} clusters of {domain_context}.",
            "\n## Objective",
            f"Create precise, descriptive names for each cluster that clearly communicate the distinct theme or pattern represented by that cluster.",
            
            "\n## Guidelines:",
            "- Generate concise (3-5 words) yet descriptive names that capture the core concept of each cluster",
            "- Focus on what makes each cluster DISTINCTIVE from other clusters",
            "- Names should be specific and meaningful, not generic or vague",
            "- Base names on key terminology, topics, or themes evident in the cluster data",
            "- Prioritize nouns and specialized terminology that reveal the subject matter",
            "- Avoid generic labels like 'Group A', 'Basic Information', or 'Miscellaneous Content'",
            "- If clusters represent clear categories, use category names",
            "- If clusters represent different aspects of the same topic, highlight the distinguishing aspect",
            "- Use consistent naming style and specificity level across all clusters",
            
            "\n## DO NOT:",
            "- Do not use generic descriptors that could apply to multiple clusters",
            "- Do not use cluster numbers in the names",
            "- Do not create overly long names (keep to 5 words maximum)",
            "- Do not use terms like 'miscellaneous' or 'other' unless truly appropriate",
            
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
                prompt.append("\n#### Top Terms (with importance weights):")
                term_strings = []
                for term, score in cluster['top_terms'][:15]:  # Include up to 15 terms for context
                    term_strings.append(f"{term} ({score:.3f})")
                prompt.append(", ".join(term_strings))
            
            # Add examples section if available
            if 'examples' in cluster and cluster['examples']:
                prompt.append("\n#### Representative Examples:")
                # Limit number of examples based on their length to manage token usage
                examples_to_show = min(3, len(cluster['examples']))
                for i in range(examples_to_show):
                    example = cluster['examples'][i]
                    # Truncate long examples to manage token usage
                    display_length = min(250, len(example))
                    if display_length < len(example):
                        display_example = example[:display_length] + "..."
                    else:
                        display_example = example
                    prompt.append(f"{i+1}. {display_example}")
        
        # Add response format instructions with examples
        prompt.extend([
            "\n## Response Format",
            "Provide a JSON array with one descriptive name for each cluster:",
            '```json',
            '{"cluster_names": [',
            '  "Name for Cluster 0",',
            '  "Name for Cluster 1",',
            '  "Name for Cluster 2",',
            '  ...',
            ']}',
            '```',
            "\n## Examples of Good vs Poor Naming",
            "✓ GOOD: \"Network Security Protocols\" (specific, descriptive, focused)",
            "✓ GOOD: \"Mobile UI Design\" (clear domain with specialization)",
            "✓ GOOD: \"Financial Risk Analysis\" (subject matter focused)",
            "✗ POOR: \"Important Information\" (vague, non-specific)",
            "✗ POOR: \"Group 3 Content\" (uses cluster number, non-descriptive)",
            "✗ POOR: \"Basic Material with Some Advanced Topics\" (too long and vague)",
            
            "\nEnsure each cluster name clearly communicates the distinctive content or pattern in that specific cluster."
        ])
        
        # Join all parts into a single string
        complete_prompt = "\n".join(prompt)
        
        self.logger.debug(f"Created naming prompt of {len(complete_prompt)} characters")
        return complete_prompt

    def select_representative_examples(self, dataframe, text_columns, cluster_column, cluster_id, n_samples=5):
        """
        Selects diverse representative examples from a cluster, avoiding duplicates.

        Args:
            dataframe: DataFrame containing the data
            text_columns: List of text column names to consider
            cluster_column: Name of the column with cluster assignments
            cluster_id: Cluster ID to analyze
            n_samples: Number of examples to select (default: 5)

        Returns:
            List of representative text examples from the cluster
        """
        self.logger.info(f"Selecting representative examples for cluster {cluster_id}")

        try:
            # Create cluster mask
            cluster_mask = dataframe[cluster_column] == cluster_id
            cluster_data = dataframe[cluster_mask]

            if len(cluster_data) == 0:
                self.logger.warning(f"No records found for cluster {cluster_id}")
                return []

            # Combine text fields for each row
            combined_texts = []
            indices = []

            for idx, row in cluster_data.iterrows():
                combined_text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                if combined_text.strip():  # Only include non-empty texts
                    combined_texts.append(combined_text)
                    indices.append(idx)

            if not combined_texts:
                self.logger.warning(f"No text content found in cluster {cluster_id}")
                return []

            # Compute text lengths
            text_lengths = np.array([len(text) for text in combined_texts])
            selected_examples = []

            # Strategic selection if there are more texts than required
            if len(combined_texts) > n_samples:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

                    if len(combined_texts) == 1:
                        return combined_texts

                    text_vectors = vectorizer.fit_transform(combined_texts)

                    # Normalize lengths for scoring
                    length_scores = text_lengths / np.max(text_lengths) if np.max(text_lengths) > 0 else np.ones_like(text_lengths)

                    # Start with the longest text
                    selected_indices = [np.argmax(text_lengths)]
                    selected_examples = [combined_texts[selected_indices[0]]]

                    # Compute similarity matrix
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_matrix = cosine_similarity(text_vectors)

                    # Select diverse examples based on minimal similarity
                    while len(selected_indices) < min(n_samples, len(combined_texts)):
                        avg_similarities = np.zeros(len(combined_texts))
                        for i in range(len(combined_texts)):
                            if i in selected_indices:
                                avg_similarities[i] = float('inf')
                            else:
                                similarities = [similarity_matrix[i, j] for j in selected_indices]
                                avg_similarities[i] = np.mean(similarities)

                        # Scoring: 70% diversity (1 - similarity), 30% length
                        scores = 0.7 * (1 - avg_similarities) + 0.3 * length_scores
                        next_idx = np.argmax(scores)

                        if scores[next_idx] <= 0.3:
                            break

                        selected_indices.append(next_idx)
                        selected_examples.append(combined_texts[next_idx])

                except Exception as e:
                    self.logger.warning(f"Error in similarity-based selection: {str(e)}. Falling back to length-based selection.")

                    sorted_indices = sorted(range(len(text_lengths)), key=lambda i: text_lengths[i], reverse=True)
                    top_count = int(n_samples * 0.6)
                    diversity_count = n_samples - top_count

                    selected_indices = sorted_indices[:top_count]
                    if diversity_count > 0 and len(sorted_indices) > top_count:
                        remaining = sorted_indices[top_count:]
                        step = max(1, len(remaining) // diversity_count)
                        for i in range(0, len(remaining), step):
                            if len(selected_indices) < n_samples:
                                selected_indices.append(remaining[i])
                            else:
                                break

                    selected_examples = [combined_texts[i] for i in selected_indices]
            else:
                selected_examples = combined_texts

            # Ensure all examples are unique
            unique_examples = []
            seen_signatures = set()

            for ex in selected_examples:
                signature = ex[:100].strip()
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_examples.append(ex)

            self.logger.info(f"Selected {len(unique_examples)} unique examples from cluster {cluster_id}")
            return unique_examples

        except Exception as e:
            self.logger.error(f"Error selecting representative examples: {str(e)}")
            self.logger.error(traceback.format_exc())
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
        

class EnhancedClassifierManager:
    """Enhanced ClassifierManager that uses optimized AI processing."""
    
    def __init__(self, config, logger, data_processor, feature_extractor):
        """Initialize with optimized components."""
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
        
        # Initialize OPTIMIZED LLM classification manager
        self.llm_manager = OptimizedLLMClassificationManager(config, logger)
        
        self.logger.info(f"Enhanced ClassifierManager initialized with {len(self.perspectives)} perspectives")
    
    def classify_perspective(self, dataframe, perspective_name, perspective_config):
        """
        Enhanced perspective classification with optimization detection.
        Automatically chooses between clustering and AI classification with optimizations.
        """
        self.logger.info(f"Applying enhanced {perspective_name} perspective")
        
        try:
            # Check perspective type
            perspective_type = perspective_config.get('type', 'clustering')
            
            if perspective_type == 'openai_classification':
                # Use optimized AI classification
                return self._apply_optimized_ai_classification_perspective(dataframe, perspective_name, perspective_config)
            elif perspective_type == 'clustering':
                # Use traditional clustering
                return self._apply_clustering_perspective(dataframe, perspective_name, perspective_config)
            else:
                raise ValueError(f"Unknown perspective type: {perspective_type}")
                
        except Exception as e:
            self.logger.error(f"Error applying perspective {perspective_name}: {str(e)}")
            raise RuntimeError(f"Failed to apply perspective {perspective_name}: {str(e)}")
    
    def _apply_optimized_ai_classification_perspective(self, dataframe, perspective_name, perspective_config):
        """Apply optimized AI classification perspective."""
        self.logger.info(f"Applying OPTIMIZED AI classification perspective: {perspective_name}")
        
        # Check if we're working with a PySpark DataFrame
        from pyspark.sql import DataFrame as SparkDataFrame
        is_spark_df = isinstance(dataframe, SparkDataFrame)
        
        # Convert to pandas if needed
        if is_spark_df:
            self.logger.info("Converting Spark DataFrame to pandas for optimized AI classification")
            pandas_df = dataframe.toPandas()
        else:
            pandas_df = dataframe.copy()
        
        # Apply optimized LLM classification
        result_df, metadata = self.llm_manager.classify_perspective(
            pandas_df, perspective_name, perspective_config
        )
        
        # Get output column and classifications
        output_column = perspective_config.get('output_column', f"{perspective_name}_classification")
        classifications = result_df[output_column].values
        
        self.logger.info(f"Optimized AI classification perspective {perspective_name} completed")
        
        return result_df, metadata, classifications
    
    def _apply_clustering_perspective(self, dataframe, perspective_name, perspective_config):
        """Apply traditional clustering perspective."""
        self.logger.info(f"Applying clustering perspective: {perspective_name}")
        
        try:
            # Get perspective configuration
            columns = perspective_config.get('columns', [])
            algorithm = perspective_config.get('algorithm', 'hdbscan').lower()
            output_column = perspective_config.get('output_column', f"{perspective_name}_cluster")
            
            # Check if we're working with a PySpark DataFrame
            from pyspark.sql import DataFrame as SparkDataFrame
            is_spark_df = isinstance(dataframe, SparkDataFrame)
            
            # Convert to pandas if needed for feature extraction
            if is_spark_df:
                pandas_df = dataframe.toPandas()
            else:
                pandas_df = dataframe.copy()
            
            # Preprocess text columns if needed
            pandas_df = self.data_processor.preprocess_text_columns(pandas_df, columns)
            
            # Extract features for the specified columns
            _, features_dict = self.data_processor.extract_features(
                pandas_df, text_columns=columns
            )
            
            # Combine features from all columns for this perspective
            combined_features = None
            feature_extraction_method = self.config.get_config_value('feature_extraction.method', 'hybrid')
            
            if feature_extraction_method == 'hybrid':
                # Combine TF-IDF and embedding features
                all_features = []
                for column in columns:
                    if f"{column}_tfidf" in features_dict:
                        all_features.append(features_dict[f"{column}_tfidf"])
                    if f"{column}_embedding" in features_dict:
                        all_features.append(features_dict[f"{column}_embedding"])
                
                if all_features:
                    import scipy.sparse
                    if any(scipy.sparse.issparse(f) for f in all_features):
                        # Handle sparse matrices
                        combined_features = scipy.sparse.hstack(all_features)
                    else:
                        # Handle dense matrices
                        combined_features = np.hstack(all_features)
            else:
                # Use features from the first column or combined
                for column in columns:
                    if column in features_dict:
                        combined_features = features_dict[column]
                        break
            
            if combined_features is None:
                raise ValueError(f"No features could be extracted for perspective {perspective_name}")
            
            # Initialize the appropriate clusterer
            if algorithm == 'kmeans':
                clusterer = KMeansClusterer(self.config, self.logger, perspective_config)
            elif algorithm == 'hdbscan':
                clusterer = HDBSCANClusterer(self.config, self.logger, perspective_config)
            elif algorithm == 'agglomerative':
                clusterer = AgglomerativeClusterer(self.config, self.logger, perspective_config)
            else:
                raise ValueError(f"Unknown clustering algorithm: {algorithm}")
            
            # Fit the clusterer
            clusterer.fit(combined_features)
            cluster_assignments = clusterer.get_labels()
            
            # Generate cluster labels
            cluster_labels = self.cluster_labeler.generate_labels(
                pandas_df, columns, output_column
            )
            
            # Add clustering results to dataframe
            pandas_df[output_column] = cluster_assignments
            
            # Add cluster labels if generated
            if cluster_labels:
                label_column = f"{output_column}_label"
                pandas_df[label_column] = pandas_df[output_column].map(cluster_labels)
            
            # Store results
            self.clusterers[perspective_name] = clusterer
            self.features_dict[f"{perspective_name}_combined"] = combined_features
            self.cluster_assignments_dict[perspective_name] = cluster_assignments
            
            self.logger.info(f"Clustering perspective {perspective_name} completed successfully")
            
            return pandas_df, combined_features, cluster_assignments
            
        except Exception as e:
            self.logger.error(f"Error applying clustering perspective {perspective_name}: {str(e)}")
            raise RuntimeError(f"Failed to apply clustering perspective {perspective_name}: {str(e)}")
    
    def get_ai_classification_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics for all AI classification perspectives."""
        return self.llm_manager.get_all_stats()
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive performance report."""
        return self.llm_manager.generate_performance_report()


# Agregar esta línea al final para mantener compatibilidad
ClassifierManager = EnhancedClassifierManager
