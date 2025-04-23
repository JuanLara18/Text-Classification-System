import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import csv
from pathlib import Path
import time
from datetime import datetime
import warnings


class ClusteringEvaluator:
    """Evaluator for clustering results."""

    def __init__(self, config, logger):
        """
        Initializes the clustering evaluator.

        Args:
            config: Configuration manager
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.evaluation_config = config.get_evaluation_config()
        self.metrics = self.evaluation_config.get('metrics', [
            'silhouette_score', 
            'davies_bouldin_score', 
            'calinski_harabasz_score'
        ])
        self.logger.info(f"ClusteringEvaluator initialized with metrics: {self.metrics}")

    def evaluate_clustering(self, features, cluster_assignments):
        """
        Evaluates a clustering result with multiple metrics.

        Args:
            features: Feature matrix used for clustering
            cluster_assignments: Cluster assignments

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating clustering results")
        metrics_results = {}
        
        try:
            # Check if we have at least 2 clusters (required for evaluation metrics)
            unique_clusters = np.unique(cluster_assignments)
            if len(unique_clusters) < 2:
                self.logger.warning("Cannot evaluate clustering with less than 2 clusters")
                return {"error": "Insufficient clusters for evaluation"}
            
            # Calculate requested metrics
            for metric in self.metrics:
                if metric == 'silhouette_score':
                    try:
                        score = self.calculate_silhouette_score(features, cluster_assignments)
                        metrics_results['silhouette_score'] = score
                    except Exception as e:
                        self.logger.error(f"Error calculating silhouette score: {str(e)}")
                        metrics_results['silhouette_score'] = None
                
                elif metric == 'davies_bouldin_score':
                    try:
                        score = self.calculate_davies_bouldin_score(features, cluster_assignments)
                        metrics_results['davies_bouldin_score'] = score
                    except Exception as e:
                        self.logger.error(f"Error calculating Davies-Bouldin score: {str(e)}")
                        metrics_results['davies_bouldin_score'] = None
                
                elif metric == 'calinski_harabasz_score':
                    try:
                        score = self.calculate_calinski_harabasz_score(features, cluster_assignments)
                        metrics_results['calinski_harabasz_score'] = score
                    except Exception as e:
                        self.logger.error(f"Error calculating Calinski-Harabasz score: {str(e)}")
                        metrics_results['calinski_harabasz_score'] = None
                
                else:
                    self.logger.warning(f"Unknown metric: {metric}")
            
            # Calculate basic statistics
            cluster_sizes = {int(cluster): np.sum(cluster_assignments == cluster) for cluster in unique_clusters}
            metrics_results['cluster_sizes'] = cluster_sizes
            metrics_results['num_clusters'] = len(unique_clusters)
            metrics_results['samples_total'] = len(cluster_assignments)
            
            # Add timestamp
            metrics_results['timestamp'] = datetime.now().isoformat()
            
            self.logger.info(f"Completed clustering evaluation: {metrics_results}")
            return metrics_results
            
        except Exception as e:
            self.logger.error(f"Error during clustering evaluation: {str(e)}")
            return {"error": str(e)}

    def calculate_silhouette_score(self, features, cluster_assignments):
        """
        Calculates the silhouette score for a clustering.

        Args:
            features: Feature matrix
            cluster_assignments: Cluster assignments

        Returns:
            Silhouette score
        """
        # Get unique clusters
        unique_clusters = np.unique(cluster_assignments)
        
        # Check if there are enough clusters for silhouette score
        if len(unique_clusters) < 2:
            self.logger.warning("Silhouette score requires at least 2 clusters")
            return None
        
        # Check if any cluster is too small
        for cluster in unique_clusters:
            if np.sum(cluster_assignments == cluster) <= 1:
                self.logger.warning(f"Cluster {cluster} has only one sample, using alternate method")
                # For very small clusters, return approximated score
                return np.nan
        
        # For large datasets, use a sample to speed up computation
        sample_size_limit = 10000  # Limit for calculation speed
        if features.shape[0] > sample_size_limit:
            self.logger.info(f"Using a sample of {sample_size_limit} for silhouette calculation")
            indices = np.random.choice(features.shape[0], sample_size_limit, replace=False)
            features_sample = features[indices]
            cluster_sample = cluster_assignments[indices]
            
            # Ensure we have at least 2 clusters in the sample
            unique_in_sample = np.unique(cluster_sample)
            if len(unique_in_sample) < 2:
                self.logger.warning("Sample doesn't contain at least 2 clusters, using full dataset")
                features_sample = features
                cluster_sample = cluster_assignments
            else:
                features = features_sample
                cluster_assignments = cluster_sample
        
        # Calculate silhouette score with appropriate metric
        try:
            score = silhouette_score(features, cluster_assignments, metric='euclidean')
            self.logger.info(f"Calculated silhouette score: {score:.4f}")
            return score
        except Exception as e:
            self.logger.error(f"Error calculating silhouette score: {str(e)}")
            return None

    def calculate_davies_bouldin_score(self, features, cluster_assignments):
        """
        Calculates the Davies-Bouldin index for a clustering.

        Args:
            features: Feature matrix
            cluster_assignments: Cluster assignments

        Returns:
            Davies-Bouldin index
        """
        # Get unique clusters
        unique_clusters = np.unique(cluster_assignments)
        
        # Check if there are enough clusters for Davies-Bouldin score
        if len(unique_clusters) < 2:
            self.logger.warning("Davies-Bouldin score requires at least 2 clusters")
            return None
        
        # Check if any cluster is too small
        for cluster in unique_clusters:
            if np.sum(cluster_assignments == cluster) <= 1:
                self.logger.warning(f"Cluster {cluster} has only one sample, using alternate method")
                # For very small clusters, return approximated score
                return np.nan
        
        # Calculate Davies-Bouldin index
        try:
            score = davies_bouldin_score(features, cluster_assignments)
            self.logger.info(f"Calculated Davies-Bouldin score: {score:.4f}")
            return score
        except Exception as e:
            self.logger.error(f"Error calculating Davies-Bouldin score: {str(e)}")
            return None

    def calculate_calinski_harabasz_score(self, features, cluster_assignments):
        """
        Calculates the Calinski-Harabasz index for a clustering.

        Args:
            features: Feature matrix
            cluster_assignments: Cluster assignments

        Returns:
            Calinski-Harabasz index
        """
        # Get unique clusters
        unique_clusters = np.unique(cluster_assignments)
        
        # Check if there are enough clusters for Calinski-Harabasz score
        if len(unique_clusters) < 2:
            self.logger.warning("Calinski-Harabasz score requires at least 2 clusters")
            return None
        
        # Check if any cluster is too small
        for cluster in unique_clusters:
            if np.sum(cluster_assignments == cluster) <= 1:
                self.logger.warning(f"Cluster {cluster} has only one sample, using alternate method")
                # For very small clusters, return approximated score
                return np.nan
        
        # Calculate Calinski-Harabasz index
        try:
            score = calinski_harabasz_score(features, cluster_assignments)
            self.logger.info(f"Calculated Calinski-Harabasz score: {score:.4f}")
            return score
        except Exception as e:
            self.logger.error(f"Error calculating Calinski-Harabasz score: {str(e)}")
            return None


class ClusteringVisualizer:
    """Visualizer for clustering results."""

    def __init__(self, config, logger, results_dir):
        """
        Initializes the clustering visualizer.

        Args:
            config: Configuration manager
            logger: Logger instance
            results_dir: Directory to save visualizations
        """
        self.config = config
        self.logger = logger
        self.results_dir = results_dir
        self.visualization_config = config.get_evaluation_config().get('visualizations', [
            'embeddings_plot',
            'silhouette_plot',
            'distribution_plot'
        ])
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up styles for consistent, appealing visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # Custom color palette for better visual distinction
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        self.logger.info(f"ClusteringVisualizer initialized for directory: {self.results_dir}")

    def create_embeddings_plot(self, features, cluster_assignments, perspective_name):
        """
        Creates a 2D/3D projection of embeddings colored by cluster.

        Args:
            features: Feature matrix
            cluster_assignments: Cluster assignments
            perspective_name: Name of the clustering perspective

        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Creating embeddings plot for perspective: {perspective_name}")
        
        try:
            # Determine if we need to reduce dimensionality
            if features.shape[1] > 2:
                self.logger.info(f"Reducing dimensionality from {features.shape[1]} to 2D for visualization")
                
                # For large datasets, use a sample to speed up projection
                sample_size_limit = 5000
                if features.shape[0] > sample_size_limit:
                    self.logger.info(f"Using a sample of {sample_size_limit} points for projection")
                    indices = np.random.choice(features.shape[0], sample_size_limit, replace=False)
                    features_sample = features[indices]
                    cluster_sample = cluster_assignments[indices]
                else:
                    features_sample = features
                    cluster_sample = cluster_assignments
                
                # Use UMAP for dimensionality reduction
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=15,
                    min_dist=0.1
                )
                
                embedding = reducer.fit_transform(features_sample)
            else:
                # Already 2D, use directly
                embedding = features
                cluster_sample = cluster_assignments
            
            # Create a DataFrame for plotting
            df_plot = pd.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'cluster': cluster_sample
            })
            
            # Get unique clusters for coloring
            unique_clusters = np.unique(cluster_sample)
            num_clusters = len(unique_clusters)
            
            # Create an interactive plot with Plotly
            fig = px.scatter(
                df_plot, 
                x='x', 
                y='y', 
                color='cluster',
                color_discrete_sequence=px.colors.qualitative.Plotly[:num_clusters],
                title=f'Cluster Visualization for {perspective_name}',
                labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'cluster': 'Cluster'},
                hover_data=['cluster']
            )
            
            # Improve layout
            fig.update_layout(
                template='plotly_white',
                legend_title_text='Cluster',
                font=dict(size=14),
                height=800,
                width=1000,
                title={
                    'text': f'Cluster Visualization for {perspective_name}',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            # Add annotations for cluster centers
            for cluster in unique_clusters:
                cluster_points = df_plot[df_plot['cluster'] == cluster]
                centroid_x = cluster_points['x'].mean()
                centroid_y = cluster_points['y'].mean()
                
                fig.add_annotation(
                    x=centroid_x,
                    y=centroid_y,
                    text=f"Cluster {cluster}",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(
                        size=14,
                        color="black"
                    ),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            
            # Save the plot
            file_path = os.path.join(self.results_dir, f"{perspective_name}_embeddings_plot.html")
            fig.write_html(file_path)
            
            # Also save a static image version
            img_path = os.path.join(self.results_dir, f"{perspective_name}_embeddings_plot.png")
            fig.write_image(img_path)
            
            self.logger.info(f"Embeddings plot saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating embeddings plot: {str(e)}")
            # Create a simple error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                    horizontalalignment='center', fontsize=14, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective_name}_embeddings_plot_error.png")
            plt.savefig(error_path)
            plt.close()
            
            return error_path

    def create_silhouette_plot(self, features, cluster_assignments, perspective_name):
        """
        Creates a silhouette plot for a clustering.

        Args:
            features: Feature matrix
            cluster_assignments: Cluster assignments
            perspective_name: Name of the clustering perspective

        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Creating silhouette plot for perspective: {perspective_name}")
        
        try:
            # Get unique clusters
            unique_clusters = np.unique(cluster_assignments)
            
            # Check if there are enough clusters for silhouette plot
            if len(unique_clusters) < 2:
                self.logger.warning("Silhouette plot requires at least 2 clusters")
                # Create a simple error plot
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "Need at least 2 clusters for silhouette plot", 
                        horizontalalignment='center', fontsize=14, color='red')
                plt.axis('off')
                
                error_path = os.path.join(self.results_dir, f"{perspective_name}_silhouette_plot_error.png")
                plt.savefig(error_path)
                plt.close()
                
                return error_path
            
            # Sample data if it's too large
            sample_size_limit = 5000
            if features.shape[0] > sample_size_limit:
                self.logger.info(f"Using a sample of {sample_size_limit} points for silhouette plot")
                indices = np.random.choice(features.shape[0], sample_size_limit, replace=False)
                features_sample = features[indices]
                cluster_sample = cluster_assignments[indices]
                
                # Make sure we have at least one sample for each cluster
                for cluster in unique_clusters:
                    if np.sum(cluster_sample == cluster) == 0:
                        # Add a sample from this cluster
                        cluster_indices = np.where(cluster_assignments == cluster)[0]
                        if len(cluster_indices) > 0:
                            idx_to_add = cluster_indices[0]
                            indices = np.append(indices, idx_to_add)
                            features_sample = features[indices]
                            cluster_sample = cluster_assignments[indices]
            else:
                features_sample = features
                cluster_sample = cluster_assignments
            
            # Calculate silhouette scores for each sample
            try:
                silhouette_values = silhouette_samples(features_sample, cluster_sample)
            except Exception as e:
                self.logger.error(f"Error calculating silhouette samples: {str(e)}")
                # Create a simple error plot
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error calculating silhouette: {str(e)}", 
                        horizontalalignment='center', fontsize=14, color='red')
                plt.axis('off')
                
                error_path = os.path.join(self.results_dir, f"{perspective_name}_silhouette_plot_error.png")
                plt.savefig(error_path)
                plt.close()
                
                return error_path
            
            # Create a new figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # The vertical line for average silhouette score
            avg_score = np.mean(silhouette_values)
            ax.axvline(x=avg_score, color='red', linestyle='--', label=f'Average: {avg_score:.3f}')
            
            # Compute and sort silhouette scores by cluster
            y_lower = 10
            for i, cluster in enumerate(sorted(unique_clusters)):
                # Get silhouette scores for this cluster
                cluster_silhouette_values = silhouette_values[cluster_sample == cluster]
                cluster_silhouette_values.sort()
                
                cluster_size = len(cluster_silhouette_values)
                y_upper = y_lower + cluster_size
                
                # Get a color from the colormap for this cluster
                color = plt.cm.tab20(i % 20)
                
                # Fill the silhouette
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7
                )
                
                # Label the silhouette with cluster number
                ax.text(-0.05, y_lower + 0.5 * cluster_size, f'Cluster {cluster}')
                
                # Update y_lower for next plot
                y_lower = y_upper + 10
            
            # Set axis labels and title
            ax.set_title(f'Silhouette Plot for {perspective_name}', fontsize=16)
            ax.set_xlabel('Silhouette Coefficient Values', fontsize=14)
            ax.set_ylabel('Cluster', fontsize=14)
            
            # Set x-axis limits
            ax.set_xlim([-0.1, 1])
            
            # Set y-axis limits
            ax.set_ylim([0, y_lower])
            
            # Add legend and grid
            ax.legend(loc='lower right')
            ax.grid(True)
            
            # Add text with interpretation
            if avg_score < 0.25:
                quality_text = "Poor structure"
            elif avg_score < 0.5:
                quality_text = "Medium structure"
            elif avg_score < 0.7:
                quality_text = "Good structure"
            else:
                quality_text = "Strong structure"
                
            ax.text(
                0.99, 0.01, 
                f"Average silhouette: {avg_score:.3f}\n{quality_text}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            )
            
            # Save the plot
            file_path = os.path.join(self.results_dir, f"{perspective_name}_silhouette_plot.png")
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Silhouette plot saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating silhouette plot: {str(e)}")
            # Create a simple error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                    horizontalalignment='center', fontsize=14, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective_name}_silhouette_plot_error.png")
            plt.savefig(error_path)
            plt.close()
            
            return error_path

    def create_distribution_plot(self, cluster_assignments, perspective_name):
        """
        Creates a distribution plot of cluster sizes.

        Args:
            cluster_assignments: Cluster assignments
            perspective_name: Name of the clustering perspective

        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Creating distribution plot for perspective: {perspective_name}")
        
        try:
            # Count samples in each cluster
            unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
            
            # Sort clusters by size
            sorted_indices = np.argsort(counts)[::-1]  # Descending order
            sorted_clusters = unique_clusters[sorted_indices]
            sorted_counts = counts[sorted_indices]
            
            # Create a DataFrame for plotting
            df_plot = pd.DataFrame({
                'Cluster': [f'Cluster {c}' for c in sorted_clusters],
                'Count': sorted_counts,
                'Percentage': sorted_counts / len(cluster_assignments) * 100
            })
            
            # Create a bar plot with plotly
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for counts
            fig.add_trace(
                go.Bar(
                    x=df_plot['Cluster'],
                    y=df_plot['Count'],
                    name='Sample Count',
                    marker_color='royalblue'
                ),
                secondary_y=False
            )
            
            # Add line chart for cumulative percentage
            cumulative_pct = np.cumsum(df_plot['Percentage'])
            df_plot['Cumulative'] = cumulative_pct
            
            fig.add_trace(
                go.Scatter(
                    x=df_plot['Cluster'],
                    y=df_plot['Cumulative'],
                    mode='lines+markers',
                    name='Cumulative %',
                    marker=dict(color='firebrick'),
                    line=dict(width=3)
                ),
                secondary_y=True
            )
            
            # Add percentage labels above bars
            for i, row in df_plot.iterrows():
                fig.add_annotation(
                    x=row['Cluster'],
                    y=row['Count'],
                    text=f"{row['Percentage']:.1f}%",
                    showarrow=False,
                    yshift=10
                )
            
            # Update layout
            fig.update_layout(
                title=f'Cluster Size Distribution for {perspective_name}',
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                font=dict(size=14),
                height=600,
                width=900
            )
            
            # Update axes
            fig.update_xaxes(title_text='Cluster')
            fig.update_yaxes(title_text='Number of Samples', secondary_y=False)
            fig.update_yaxes(
                title_text='Cumulative Percentage',
                secondary_y=True,
                range=[0, 105],
                ticksuffix='%'
            )
            
            # Save the plot
            file_path = os.path.join(self.results_dir, f"{perspective_name}_distribution_plot.html")
            fig.write_html(file_path)
            
            # Also save a static image version
            img_path = os.path.join(self.results_dir, f"{perspective_name}_distribution_plot.png")
            fig.write_image(img_path)
            
            self.logger.info(f"Distribution plot saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {str(e)}")
            # Create a simple error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                    horizontalalignment='center', fontsize=14, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective_name}_distribution_plot_error.png")
            plt.savefig(error_path)
            plt.close()
            
            return error_path


class EvaluationReporter:
    """Report generator for clustering evaluation."""

    def __init__(self, config, logger, results_dir):
        """
        Initializes the report generator.

        Args:
            config: Configuration manager
            logger: Logger instance
            results_dir: Directory to save reports
        """
        self.config = config
        self.logger = logger
        self.results_dir = results_dir
        self.evaluation_config = config.get_evaluation_config()
        self.output_formats = self.evaluation_config.get('output_format', ['html', 'json', 'csv'])
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"EvaluationReporter initialized for directory: {self.results_dir}")

    def generate_report(self, perspective_name, metrics, visualization_paths):
        """
        Generates a report for a clustering perspective.

        Args:
            perspective_name: Name of the perspective
            metrics: Dictionary of evaluation metrics
            visualization_paths: Paths to visualizations

        Returns:
            Path to the generated report
        """
        self.logger.info(f"Generating report for perspective: {perspective_name}")
        report_paths = {}
        
        try:
            # Save metrics in requested formats
            if 'json' in self.output_formats:
                json_path = self.save_metrics_to_json(metrics, f"{perspective_name}_metrics")
                report_paths['json'] = json_path
            
            if 'csv' in self.output_formats:
                csv_path = self.save_metrics_to_csv(metrics, f"{perspective_name}_metrics")
                report_paths['csv'] = csv_path
            
            if 'html' in self.output_formats:
                html_path = self.generate_html_report(perspective_name, metrics, visualization_paths)
                report_paths['html'] = html_path
            
            self.logger.info(f"Reports generated: {report_paths}")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {"error": str(e)}

    def save_metrics_to_csv(self, metrics, file_name):
        """
        Saves metrics to a CSV file.

        Args:
            metrics: Dictionary of metrics
            file_name: Name of the file to save

        Returns:
            Path to the saved file
        """
        self.logger.info(f"Saving metrics to CSV: {file_name}")
        
        try:
            file_path = os.path.join(self.results_dir, f"{file_name}.csv")
            
            # Flatten nested dictionaries for CSV format
            flat_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_metrics[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_metrics[key] = value
            
            # Convert to DataFrame for easy CSV export
            df = pd.DataFrame([flat_metrics])
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Metrics saved to CSV: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to CSV: {str(e)}")
            return None

    def save_metrics_to_json(self, metrics, file_name):
        """
        Saves metrics to a JSON file.

        Args:
            metrics: Dictionary of metrics
            file_name: Name of the file to save

        Returns:
            Path to the saved file
        """
        self.logger.info(f"Saving metrics to JSON: {file_name}")
        
        try:
            file_path = os.path.join(self.results_dir, f"{file_name}.json")
            
            # Ensure the metrics are JSON serializable
            serializable_metrics = {}
            for key, value in metrics.items():
                if key == 'cluster_sizes' and isinstance(value, dict):
                    # Convert dict with int keys to string keys
                    serializable_metrics[key] = {str(k): v for k, v in value.items()}
                elif isinstance(value, np.int64):
                    serializable_metrics[key] = int(value)
                elif isinstance(value, np.float64):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            
            with open(file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            
            self.logger.info(f"Metrics saved to JSON: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to JSON: {str(e)}")
            return None

    def generate_html_report(self, perspective_name, metrics, visualization_paths):
        """
        Generates an HTML report for a clustering perspective.

        Args:
            perspective_name: Name of the perspective
            metrics: Dictionary of evaluation metrics
            visualization_paths: Paths to visualizations

        Returns:
            Path to the generated HTML file
        """
        self.logger.info(f"Generating HTML report for: {perspective_name}")
        
        try:
            file_path = os.path.join(self.results_dir, f"{perspective_name}_report.html")
            
            # Start building HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Clustering Report - {perspective_name}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    h1 {{
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    .section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .metrics-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    .metrics-table th, .metrics-table td {{
                        padding: 12px 15px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }}
                    .metrics-table th {{
                        background-color: #3498db;
                        color: white;
                    }}
                    .metrics-table tr:nth-child(even) {{
                        background-color: #f2f2f2;
                    }}
                    .visualization {{
                        margin: 20px 0;
                        text-align: center;
                    }}
                    .visualization img {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        padding: 5px;
                    }}
                    .interpretation {{
                        background-color: #e8f4f8;
                        padding: 15px;
                        border-left: 4px solid #3498db;
                        margin: 20px 0;
                    }}
                    .chart-container {{
                        width: 100%;
                        height: 800px;
                        margin: 20px 0;
                    }}
                    .footer {{
                        margin-top: 30px;
                        padding-top: 10px;
                        border-top: 1px solid #ddd;
                        font-size: 0.9em;
                        color: #777;
                    }}
                </style>
            </head>
            <body>
                <h1>Clustering Report: {perspective_name}</h1>
                <div class="section">
                    <h2>Overview</h2>
                    <p>This report presents the evaluation of clustering results for the <strong>{perspective_name}</strong> perspective.</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # Add metrics section
            html_content += """
                <div class="section">
                    <h2>Evaluation Metrics</h2>
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Interpretation</th>
                        </tr>
            """
            
            # Add silhouette score
            if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
                score = metrics['silhouette_score']
                if score < 0.25:
                    interpretation = "Poor structure. Clusters are not well separated."
                elif score < 0.5:
                    interpretation = "Medium structure. Clusters have some separation."
                elif score < 0.7:
                    interpretation = "Good structure. Clusters are well separated."
                else:
                    interpretation = "Strong structure. Clusters are very well separated."
                
                html_content += f"""
                    <tr>
                        <td>Silhouette Score</td>
                        <td>{score:.4f}</td>
                        <td>{interpretation}</td>
                    </tr>
                """
            
            # Add Davies-Bouldin score
            if 'davies_bouldin_score' in metrics and metrics['davies_bouldin_score'] is not None:
                score = metrics['davies_bouldin_score']
                # Lower values are better for Davies-Bouldin
                if score > 1.0:
                    interpretation = "Poor separation. Clusters overlap significantly."
                elif score > 0.7:
                    interpretation = "Medium separation. Some cluster overlap."
                elif score > 0.4:
                    interpretation = "Good separation. Limited cluster overlap."
                else:
                    interpretation = "Excellent separation. Clusters are well defined."
                
                html_content += f"""
                    <tr>
                        <td>Davies-Bouldin Score</td>
                        <td>{score:.4f}</td>
                        <td>{interpretation} (Lower is better)</td>
                    </tr>
                """
            
            # Add Calinski-Harabasz score
            if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
                score = metrics['calinski_harabasz_score']
                # Higher values are better for Calinski-Harabasz
                if score < 50:
                    interpretation = "Poor separation. Clusters are not distinct."
                elif score < 100:
                    interpretation = "Medium separation. Clusters show some distinction."
                elif score < 200:
                    interpretation = "Good separation. Clusters are distinct."
                else:
                    interpretation = "Excellent separation. Clusters are very distinct."
                
                html_content += f"""
                    <tr>
                        <td>Calinski-Harabasz Score</td>
                        <td>{score:.4f}</td>
                        <td>{interpretation} (Higher is better)</td>
                    </tr>
                """
            
            # Add number of clusters
            if 'num_clusters' in metrics:
                html_content += f"""
                    <tr>
                        <td>Number of Clusters</td>
                        <td>{metrics['num_clusters']}</td>
                        <td>Total number of clusters identified.</td>
                    </tr>
                """
            
            # Add sample count
            if 'samples_total' in metrics:
                html_content += f"""
                    <tr>
                        <td>Total Samples</td>
                        <td>{metrics['samples_total']}</td>
                        <td>Total number of samples classified.</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
            
            # Add cluster distribution section
            if 'cluster_sizes' in metrics and metrics['cluster_sizes']:
                html_content += """
                <div class="section">
                    <h2>Cluster Distribution</h2>
                    <table class="metrics-table">
                        <tr>
                            <th>Cluster ID</th>
                            <th>Size</th>
                            <th>Percentage</th>
                        </tr>
                """
                
                total_samples = metrics['samples_total']
                cluster_sizes = metrics['cluster_sizes']
                
                # Sort clusters by size (descending)
                sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
                
                for cluster_id, size in sorted_clusters:
                    percentage = (size / total_samples) * 100
                    html_content += f"""
                        <tr>
                            <td>Cluster {cluster_id}</td>
                            <td>{size}</td>
                            <td>{percentage:.2f}%</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Add visualizations section
            if visualization_paths:
                html_content += """
                <div class="section">
                    <h2>Visualizations</h2>
                """
                
                # Add embeddings plot
                embeddings_html = None
                embeddings_png = None
                for path in visualization_paths.values():
                    if path.endswith('embeddings_plot.html'):
                        embeddings_html = path
                    elif path.endswith('embeddings_plot.png'):
                        embeddings_png = path
                
                if embeddings_html:
                    relative_path = os.path.relpath(embeddings_html, self.results_dir)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Cluster Embeddings Visualization</h3>
                        <p>This plot shows a 2D projection of the high-dimensional feature space, with points colored by cluster assignment. Points that are close together in this visualization are similar in the original feature space.</p>
                        <iframe src="{relative_path}" width="100%" height="800px" frameborder="0"></iframe>
                    </div>
                    """
                elif embeddings_png:
                    relative_path = os.path.basename(embeddings_png)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Cluster Embeddings Visualization</h3>
                        <p>This plot shows a 2D projection of the high-dimensional feature space, with points colored by cluster assignment. Points that are close together in this visualization are similar in the original feature space.</p>
                        <img src="{relative_path}" alt="Cluster Embeddings" />
                    </div>
                    """
                
                # Add silhouette plot
                silhouette_path = None
                for path in visualization_paths.values():
                    if 'silhouette_plot' in path and path.endswith('.png'):
                        silhouette_path = path
                
                if silhouette_path:
                    relative_path = os.path.basename(silhouette_path)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Silhouette Analysis</h3>
                        <p>This plot shows silhouette values for each sample. Higher values (closer to 1) indicate that samples are well-matched to their own cluster and separated from neighboring clusters. The red dashed line represents the average silhouette score across all samples.</p>
                        <img src="{relative_path}" alt="Silhouette Plot" />
                    </div>
                    """
                
                # Add distribution plot
                distribution_html = None
                distribution_png = None
                for path in visualization_paths.values():
                    if path.endswith('distribution_plot.html'):
                        distribution_html = path
                    elif path.endswith('distribution_plot.png'):
                        distribution_png = path
                
                if distribution_html:
                    relative_path = os.path.relpath(distribution_html, self.results_dir)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Cluster Size Distribution</h3>
                        <p>This plot shows the distribution of sample counts across clusters, with the cumulative percentage line indicating the proportion of data covered by the largest clusters.</p>
                        <iframe src="{relative_path}" width="100%" height="600px" frameborder="0"></iframe>
                    </div>
                    """
                elif distribution_png:
                    relative_path = os.path.basename(distribution_png)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Cluster Size Distribution</h3>
                        <p>This plot shows the distribution of sample counts across clusters, with the cumulative percentage line indicating the proportion of data covered by the largest clusters.</p>
                        <img src="{relative_path}" alt="Cluster Distribution" />
                    </div>
                    """
                
                html_content += """
                </div>
                """
            
            # Add interpretation section
            html_content += """
            <div class="section">
                <h2>Interpretation & Insights</h2>
            """
            
            # Add overall interpretation based on metrics
            silhouette = metrics.get('silhouette_score')
            davies_bouldin = metrics.get('davies_bouldin_score')
            calinski = metrics.get('calinski_harabasz_score')
            num_clusters = metrics.get('num_clusters')
            
            # Overall quality assessment
            quality = "undetermined"
            if silhouette is not None:
                if silhouette > 0.7:
                    quality = "excellent"
                elif silhouette > 0.5:
                    quality = "good"
                elif silhouette > 0.25:
                    quality = "moderate"
                else:
                    quality = "poor"
            
            html_content += f"""
                <div class="interpretation">
                    <h3>Clustering Quality Assessment</h3>
                    <p>The overall clustering quality appears to be <strong>{quality}</strong> based on the evaluation metrics.</p>
            """
            
            # Add specific insights
            if 'cluster_sizes' in metrics and metrics['cluster_sizes']:
                cluster_sizes = metrics['cluster_sizes']
                sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
                
                # Check for imbalance
                largest = sorted_clusters[0][1]
                smallest = sorted_clusters[-1][1]
                ratio = largest / max(smallest, 1)
                
                if ratio > 10:
                    html_content += f"""
                    <p><strong>Significant cluster imbalance detected:</strong> The largest cluster is {ratio:.1f}x the size of the smallest cluster.
                    This might indicate either natural data grouping or potential issues with the clustering parameters.</p>
                    """
                
                # Calculate percentage in top clusters
                total = sum(cluster_sizes.values())
                top_3_pct = sum([size for _, size in sorted_clusters[:min(3, len(sorted_clusters))]]) / total * 100
                
                html_content += f"""
                <p>The top 3 clusters contain {top_3_pct:.1f}% of all samples.</p>
                """
            
            # Add recommendations based on metrics
            html_content += """
                <h3>Recommendations</h3>
                <ul>
            """
            
            if silhouette is not None and silhouette < 0.3:
                html_content += """
                <li>Consider adjusting the number of clusters or feature extraction parameters to improve separation.</li>
                """
            
            if davies_bouldin is not None and davies_bouldin > 1.0:
                html_content += """
                <li>The high Davies-Bouldin index suggests overlapping clusters. Consider testing different clustering algorithms or parameters.</li>
                """
            
            if ratio > 10:
                html_content += """
                <li>The cluster size distribution is highly imbalanced. Consider investigating if this reflects natural data groupings or if different clustering parameters might provide more balanced results.</li>
                """
            
            html_content += """
                </ul>
                </div>
            </div>
            """
            
            # Close HTML
            html_content += """
            <div class="footer">
                <p>Generated by the Text Classification System</p>
            </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            # Create a simple error report
            error_path = os.path.join(self.results_dir, f"{perspective_name}_report_error.html")
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report</title>
                </head>
                <body>
                    <h1>Error Generating Report</h1>
                    <p>An error occurred while generating the report: {str(e)}</p>
                </body>
                </html>
                """)
            return error_path