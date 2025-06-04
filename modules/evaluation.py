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
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import logging
import seaborn as sns


class ClassificationEvaluator:
    """Evaluator for AI classification results."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.logger.info("ClassificationEvaluator initialized")
    
    def evaluate_classification(self, classifications: List[str], categories: List[str], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluates AI classification results.
        
        Args:
            classifications: List of classification results
            categories: List of target categories
            metadata: Optional metadata from classification process
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating AI classification results")
        
        try:
            # Basic statistics
            total_classified = len(classifications)
            distribution = Counter(classifications)
            
            # Calculate coverage and balance metrics
            unique_categories = set(classifications)
            categories_used = len(unique_categories)
            coverage_ratio = categories_used / len(categories) if categories else 0
            
            # Calculate balance metrics
            max_count = max(distribution.values()) if distribution else 0
            min_count = min(distribution.values()) if distribution else 0
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            # Calculate entropy (measure of distribution uniformity)
            total = sum(distribution.values())
            probabilities = [count / total for count in distribution.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(distribution)) if len(distribution) > 1 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Detect potential issues
            issues = []
            if coverage_ratio < 0.5:
                issues.append(f"Low category coverage: only {categories_used}/{len(categories)} categories used")
            if balance_ratio < 0.1:
                issues.append(f"Highly imbalanced distribution: ratio {balance_ratio:.3f}")
            if normalized_entropy < 0.5:
                issues.append("Low distribution entropy: classifications may be too concentrated")
            
            # Calculate cost metrics if available
            cost_metrics = {}
            if metadata:
                cost_metrics = {
                    'total_cost': metadata.get('total_cost', 0),
                    'cost_per_classification': metadata.get('total_cost', 0) / total_classified if total_classified > 0 else 0,
                    'total_tokens': metadata.get('total_tokens', 0),
                    'api_calls': metadata.get('api_calls', 0),
                    'cached_responses': metadata.get('cached_responses', 0),
                    'cache_hit_rate': metadata.get('cached_responses', 0) / total_classified if total_classified > 0 else 0,
                    'errors': metadata.get('errors', 0),
                    'error_rate': metadata.get('errors', 0) / total_classified if total_classified > 0 else 0
                }
            
            # Compile results
            evaluation_results = {
                'total_classified': total_classified,
                'categories_used': categories_used,
                'total_categories': len(categories),
                'coverage_ratio': coverage_ratio,
                'balance_ratio': balance_ratio,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'distribution': dict(distribution),
                'issues': issues,
                'cost_metrics': cost_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Classification evaluation completed: {total_classified} items, {categories_used}/{len(categories)} categories used")
            if issues:
                for issue in issues:
                    self.logger.warning(f"Classification issue detected: {issue}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during classification evaluation: {str(e)}")
            return {"error": str(e)}


class ClassificationVisualizer:
    """Visualizer for AI classification results."""
    
    def __init__(self, config, logger, results_dir):
        self.config = config
        self.logger = logger
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
    
    def create_classification_distribution_plot(self, classifications: List[str], perspective_name: str, target_categories: List[str] = None) -> str:
        """
        Creates a distribution plot for classification results.
        
        Args:
            classifications: List of classification results
            perspective_name: Name of the perspective
            target_categories: Optional list of expected categories for comparison
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Creating classification distribution plot for {perspective_name}")
        
        try:
            # Count classifications
            distribution = Counter(classifications)
            
            # Prepare data for plotting
            categories = list(distribution.keys())
            counts = list(distribution.values())
            percentages = [count / len(classifications) * 100 for count in counts]
            
            # Sort by count (descending)
            sorted_data = sorted(zip(categories, counts, percentages), key=lambda x: x[1], reverse=True)
            categories, counts, percentages = zip(*sorted_data)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Bar plot
            bars = ax1.bar(range(len(categories)), counts, color='skyblue', alpha=0.8)
            ax1.set_xlabel('Categories')
            ax1.set_ylabel('Number of Classifications')
            ax1.set_title(f'Classification Distribution - {perspective_name}')
            ax1.set_xticks(range(len(categories)))
            ax1.set_xticklabels(categories, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
            
            # Pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            ax2.set_title(f'Classification Distribution - {perspective_name}')
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            
            # Save plot
            file_path = os.path.join(self.results_dir, f"{perspective_name}_classification_distribution.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Classification distribution plot saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating classification distribution plot: {str(e)}")
            
            # Create error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                    horizontalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective_name}_classification_error.png")
            plt.savefig(error_path)
            plt.close()
            
            return error_path
    
    def create_classification_comparison_plot(self, perspective_results: Dict[str, List[str]], perspective_names: List[str]) -> str:
        """
        Creates a comparison plot across multiple classification perspectives.
        
        Args:
            perspective_results: Dictionary mapping perspective names to classification results
            perspective_names: List of perspective names for ordering
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info("Creating classification comparison plot")
        
        try:
            # Prepare data
            all_categories = set()
            for classifications in perspective_results.values():
                all_categories.update(classifications)
            
            all_categories = sorted(list(all_categories))
            
            # Create comparison matrix
            comparison_data = []
            for perspective in perspective_names:
                if perspective in perspective_results:
                    distribution = Counter(perspective_results[perspective])
                    total = len(perspective_results[perspective])
                    percentages = [distribution.get(cat, 0) / total * 100 for cat in all_categories]
                    comparison_data.append(percentages)
            
            comparison_data = np.array(comparison_data)
            
            # Create heatmap
            plt.figure(figsize=(max(12, len(all_categories) * 0.8), len(perspective_names) * 0.8 + 2))
            
            sns.heatmap(comparison_data, 
                       xticklabels=all_categories, 
                       yticklabels=perspective_names,
                       annot=True, 
                       fmt='.1f', 
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Percentage (%)'})
            
            plt.title('Classification Distribution Comparison Across Perspectives')
            plt.xlabel('Categories')
            plt.ylabel('Perspectives')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            file_path = os.path.join(self.results_dir, "classification_perspectives_comparison.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Classification comparison plot saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating classification comparison plot: {str(e)}")
            return None


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
        self.classification_evaluator = ClassificationEvaluator(config, logger)
        
        self.logger.info(f"ClusteringEvaluator initialized with metrics: {self.metrics}")

    def evaluate_perspective(self, perspective_type: str, perspective_name: str, **kwargs) -> Dict[str, Any]:
        """
        NEW METHOD: Evaluates either clustering or classification perspective.
        
        Args:
            perspective_type: Type of perspective ('clustering' or 'openai_classification')
            perspective_name: Name of the perspective
            **kwargs: Additional arguments specific to the perspective type
            
        Returns:
            Dictionary of evaluation metrics
        """
        if perspective_type == 'clustering':
            features = kwargs.get('features')
            cluster_assignments = kwargs.get('cluster_assignments')
            return self.evaluate_clustering(features, cluster_assignments)
        
        elif perspective_type == 'openai_classification':
            classifications = kwargs.get('classifications')
            categories = kwargs.get('categories')
            metadata = kwargs.get('metadata')
            return self.classification_evaluator.evaluate_classification(classifications, categories, metadata)
        
        else:
            self.logger.error(f"Unknown perspective type: {perspective_type}")
            return {"error": f"Unknown perspective type: {perspective_type}"}

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

    def generate_detailed_cluster_report(self, perspective_name, cluster_characteristics, visualization_paths):
        """
        Generates a detailed HTML report for a clustering perspective.
        
        This method creates a comprehensive report containing detailed information
        about each cluster, including top terms, representative examples, and
        visualizations to provide deeper insights.
        
        Args:
            perspective_name: Name of the clustering perspective
            cluster_characteristics: List of cluster characteristic dictionaries
            visualization_paths: Dictionary mapping visualization types to file paths
            
        Returns:
            Path to the generated HTML report
        """
        self.logger.info(f"Generating detailed cluster report for {perspective_name}")
        
        try:
            file_path = os.path.join(self.results_dir, f"{perspective_name}_detailed_report.html")
            
            # Start building HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Detailed Cluster Report - {perspective_name}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f9f9f9;
                    }}
                    h1, h2, h3, h4 {{
                        color: #2c3e50;
                    }}
                    h1 {{
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                        text-align: center;
                    }}
                    .section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .cluster-card {{
                        margin-bottom: 20px;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        overflow: hidden;
                    }}
                    .cluster-header {{
                        background-color: #edf2f7;
                        padding: 12px 15px;
                        border-bottom: 1px solid #ddd;
                    }}
                    .cluster-body {{
                        padding: 15px;
                    }}
                    .cluster-metrics {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 15px;
                        margin-bottom: 15px;
                    }}
                    .metric {{
                        background-color: #f8f9fa;
                        padding: 8px 12px;
                        border-radius: 4px;
                        font-size: 0.9em;
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
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .terms-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 8px;
                        margin-bottom: 15px;
                    }}
                    .term {{
                        background-color: #e9f7fe;
                        border: 1px solid #bde5f8;
                        border-radius: 15px;
                        padding: 4px 10px;
                        font-size: 0.9em;
                    }}
                    .examples {{
                        background-color: #f8f9fa;
                        border-left: 3px solid #3498db;
                        padding: 10px 15px;
                        margin-bottom: 15px;
                        font-style: italic;
                        color: #555;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 0.9em;
                        color: #777;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 10px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }}
                    th {{
                        background-color: #edf2f7;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                </style>
            </head>
            <body>
                <h1>Detailed Cluster Analysis Report: {perspective_name}</h1>
                
                <div class="section">
                    <h2>Overview</h2>
                    <p>This report presents a detailed analysis of the clustering results for the <strong>{perspective_name}</strong> perspective.</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h3>Cluster Distribution Summary</h3>
                    <table>
                        <tr>
                            <th>Cluster ID</th>
                            <th>Size</th>
                            <th>Percentage</th>
                            <th>Top Terms</th>
                        </tr>
            """
            
            # Add cluster distribution table
            sorted_chars = sorted(cluster_characteristics, key=lambda x: x.get('size', 0), reverse=True)
            for cluster in sorted_chars:
                top_terms_str = ", ".join([term for term, _ in cluster.get('top_terms', [])[:5]])
                html_content += f"""
                    <tr>
                        <td>Cluster {cluster.get('id', 'N/A')}</td>
                        <td>{cluster.get('size', 0)}</td>
                        <td>{cluster.get('percentage', 0):.2f}%</td>
                        <td>{top_terms_str}</td>
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
                
                for viz_type, path in visualization_paths.items():
                    if os.path.exists(path):
                        # Create a relative path
                        rel_path = os.path.relpath(path, self.results_dir)
                        title = viz_type.replace('_', ' ').title()
                        
                        html_content += f"""
                        <div class="visualization">
                            <h3>{title}</h3>
                            <img src="{rel_path}" alt="{title}" />
                        </div>
                        """
                
                html_content += """
                </div>
                """
            
            # Add detailed cluster analysis section
            html_content += """
                <div class="section">
                    <h2>Detailed Cluster Analysis</h2>
            """
            
            # Add cards for each cluster
            for cluster in sorted_chars:
                cluster_id = cluster.get('id', 'N/A')
                size = cluster.get('size', 0)
                percentage = cluster.get('percentage', 0)
                
                html_content += f"""
                    <div class="cluster-card">
                        <div class="cluster-header">
                            <h3>Cluster {cluster_id} ({percentage:.2f}% of data)</h3>
                        </div>
                        <div class="cluster-body">
                            <div class="cluster-metrics">
                                <div class="metric"><strong>Size:</strong> {size} records</div>
                """
                
                # Add dispersion and distinctiveness if available
                if 'dispersion' in cluster:
                    html_content += f"""
                                <div class="metric"><strong>Dispersion:</strong> {cluster['dispersion']:.4f}</div>
                    """
                
                if 'distinctiveness' in cluster:
                    html_content += f"""
                                <div class="metric"><strong>Distinctiveness:</strong> {cluster['distinctiveness']:.4f}</div>
                    """
                
                html_content += """
                            </div>
                """
                
                # Add top terms with weights
                if 'top_terms' in cluster and cluster['top_terms']:
                    html_content += """
                            <h4>Key Terms with Weights</h4>
                            <div class="terms-container">
                    """
                    
                    for term, weight in cluster['top_terms'][:15]:
                        html_content += f"""
                                <div class="term">{term} ({weight:.3f})</div>
                        """
                    
                    html_content += """
                            </div>
                    """
                
                # Add representative examples
                if 'examples' in cluster and cluster['examples']:
                    html_content += """
                            <h4>Representative Examples</h4>
                    """
                    
                    for i, example in enumerate(cluster['examples'][:3]):
                        # Truncate long examples
                        display_example = example
                        if len(example) > 300:
                            display_example = example[:300] + "..."
                        
                        html_content += f"""
                            <div class="examples">
                                <strong>Example {i+1}:</strong> {display_example}
                            </div>
                        """
                
                html_content += """
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div class="footer">
                    <p>Generated by the Text Classification System</p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Detailed cluster report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating detailed cluster report: {str(e)}")
            
            # Create a simple error report
            error_path = os.path.join(self.results_dir, f"{perspective_name}_detailed_report_error.html")
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report</title>
                </head>
                <body>
                    <h1>Error Generating Detailed Report</h1>
                    <p>An error occurred while generating the report: {str(e)}</p>
                </body>
                </html>
                """)
            
            return error_path

    def create_cross_perspective_analysis(self, dataframe, perspective1_col, perspective2_col, perspective1_name, perspective2_name):
        """
        Analyzes the relationship between different clustering perspectives.
        
        This method performs a detailed analysis of how two clustering perspectives
        relate to each other, providing insights into their similarities and differences.
        
        Args:
            dataframe: DataFrame containing the clustering results
            perspective1_col: Column name for the first clustering perspective
            perspective2_col: Column name for the second clustering perspective
            perspective1_name: Display name for the first perspective
            perspective2_name: Display name for the second perspective
            
        Returns:
            Dictionary containing analysis results:
                - crosstab: Cross-tabulation of cluster assignments
                - normalized_crosstab: Percentage-based cross-tabulation
                - mutual_information: Mutual information score
                - adjusted_rand_index: Adjusted Rand index
                - key_relationships: List of strongest relationships
                - markdown_summary: Markdown-formatted text summary
        """
        self.logger.info(f"Creating cross-perspective analysis between {perspective1_name} and {perspective2_name}")
        
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            # Check if columns exist
            if perspective1_col not in dataframe.columns or perspective2_col not in dataframe.columns:
                missing_cols = []
                if perspective1_col not in dataframe.columns:
                    missing_cols.append(perspective1_col)
                if perspective2_col not in dataframe.columns:
                    missing_cols.append(perspective2_col)
                raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_cols)}")
            
            # Get cluster assignments
            df_filtered = dataframe.dropna(subset=[perspective1_col, perspective2_col])
            
            if len(df_filtered) == 0:
                raise ValueError(f"No overlapping data points between {perspective1_name} and {perspective2_name}")
            
            # Create cross-tabulation (raw counts)
            crosstab = pd.crosstab(df_filtered[perspective1_col], df_filtered[perspective2_col])
            
            # Create normalized cross-tabulation (row percentages)
            norm_crosstab = pd.crosstab(
                df_filtered[perspective1_col], 
                df_filtered[perspective2_col],
                normalize='index'
            ) * 100
            
            # Calculate mutual information score
            try:
                mi_score = normalized_mutual_info_score(
                    df_filtered[perspective1_col],
                    df_filtered[perspective2_col]
                )
            except:
                mi_score = None
            
            # Calculate adjusted Rand index
            try:
                ari_score = adjusted_rand_score(
                    df_filtered[perspective1_col],
                    df_filtered[perspective2_col]
                )
            except:
                ari_score = None
            
            # Find strongest relationships between clusters
            key_relationships = []
            
            for idx in norm_crosstab.index:
                row = norm_crosstab.loc[idx]
                max_col = row.idxmax()
                max_value = row[max_col]
                
                if max_value >= 25:  # Only include strong relationships (25% or more)
                    # Get cluster names if available
                    p1_name_col = f"{perspective1_col}_label"
                    p2_name_col = f"{perspective2_col}_label"
                    
                    p1_name = f"Cluster {idx}"
                    p2_name = f"Cluster {max_col}"
                    
                    if p1_name_col in dataframe.columns:
                        names = dataframe.loc[dataframe[perspective1_col] == idx, p1_name_col].dropna()
                        if len(names) > 0:
                            p1_name = names.iloc[0]
                    
                    if p2_name_col in dataframe.columns:
                        names = dataframe.loc[dataframe[perspective2_col] == max_col, p2_name_col].dropna()
                        if len(names) > 0:
                            p2_name = names.iloc[0]
                    
                    key_relationships.append({
                        'p1_cluster': idx,
                        'p2_cluster': max_col,
                        'p1_name': p1_name,
                        'p2_name': p2_name,
                        'percentage': max_value,
                        'count': crosstab.loc[idx, max_col]
                    })
            
            # Sort by percentage (descending)
            key_relationships.sort(key=lambda x: x['percentage'], reverse=True)
            
            # Generate markdown summary
            summary = [
                f"# Cross-Perspective Analysis: {perspective1_name} vs {perspective2_name}",
                "",
                "## Overall Relationship Metrics",
                f"- **Records analyzed**: {len(df_filtered)}",
                f"- **Normalized Mutual Information**: {mi_score:.4f}" if mi_score is not None else "",
                f"- **Adjusted Rand Index**: {ari_score:.4f}" if ari_score is not None else "",
                "",
                "## Key Relationships Between Clusters",
                ""
            ]
            
            for rel in key_relationships[:10]:  # Limit to top 10
                summary.append(f"- **{rel['p1_name']}** → **{rel['p2_name']}**: {rel['percentage']:.1f}% ({rel['count']} records)")
            
            summary.append("")
            summary.append("## Interpretation")
            
            # Add interpretation based on scores
            if mi_score is not None:
                if mi_score > 0.7:
                    summary.append("- The clustering perspectives show **strong alignment** (high mutual information)")
                elif mi_score > 0.4:
                    summary.append("- The clustering perspectives show **moderate alignment** (medium mutual information)")
                else:
                    summary.append("- The clustering perspectives show **weak alignment** (low mutual information)")
            
            if ari_score is not None:
                if ari_score > 0.5:
                    summary.append("- The cluster assignments are **highly consistent** between perspectives (high adjusted Rand index)")
                elif ari_score > 0.2:
                    summary.append("- The cluster assignments show **some consistency** between perspectives (medium adjusted Rand index)")
                else:
                    summary.append("- The cluster assignments show **little consistency** between perspectives (low adjusted Rand index)")
            
            markdown_summary = "\n".join(summary)
            
            # Save the analysis to a markdown file
            save_path = os.path.join(self.results_dir, f"{perspective1_name}_{perspective2_name}_analysis.md")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(markdown_summary)
            
            # Create result dictionary
            result = {
                'crosstab': crosstab,
                'normalized_crosstab': norm_crosstab,
                'mutual_information': mi_score,
                'adjusted_rand_index': ari_score,
                'key_relationships': key_relationships,
                'markdown_summary': markdown_summary,
                'file_path': save_path
            }
            
            self.logger.info(f"Cross-perspective analysis saved to: {save_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating cross-perspective analysis: {str(e)}")
            
            # Create error markdown
            error_text = f"# Error in Cross-Perspective Analysis\n\nAn error occurred: {str(e)}"
            error_path = os.path.join(self.results_dir, f"{perspective1_name}_{perspective2_name}_analysis_error.md")
            
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(error_text)
            
            return {
                'error': str(e),
                'file_path': error_path,
                'markdown_summary': error_text
            }

    def generate_combined_perspectives_report(self, dataframe, perspective_columns, perspective_names):
        """
        Creates a comprehensive report comparing all clustering perspectives.
        
        This method generates a single HTML report that presents an overview of all
        clustering perspectives, their relationships, and key insights to provide
        a holistic view of the clustering results.
        
        Args:
            dataframe: DataFrame containing the clustering results
            perspective_columns: List of column names for each clustering perspective
            perspective_names: List of display names for each perspective
            
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating combined perspectives report")
        
        try:
            from sklearn.metrics import normalized_mutual_info_score
            from itertools import combinations
            
            if len(perspective_columns) != len(perspective_names):
                raise ValueError("Number of perspective columns and names must match")
            
            if len(perspective_columns) < 1:
                raise ValueError("At least one perspective is required")
            
            file_path = os.path.join(self.results_dir, "combined_perspectives_report.html")
            
            # Start building HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Combined Clustering Perspectives Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f9f9f9;
                    }}
                    h1, h2, h3, h4 {{
                        color: #2c3e50;
                    }}
                    h1 {{
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                        text-align: center;
                    }}
                    .section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .grid-container {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        margin-top: 20px;
                    }}
                    .grid-item {{
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                    }}
                    .heatmap {{
                        margin: 20px 0;
                        text-align: center;
                    }}
                    .heatmap img {{
                        max-width: 100%;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 10px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }}
                    th {{
                        background-color: #edf2f7;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                    .relationship-score {{
                        font-weight: bold;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }}
                    .high {{
                        background-color: #d4edda;
                        color: #155724;
                    }}
                    .medium {{
                        background-color: #fff3cd;
                        color: #856404;
                    }}
                    .low {{
                        background-color: #f8d7da;
                        color: #721c24;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 0.9em;
                        color: #777;
                    }}
                </style>
            </head>
            <body>
                <h1>Combined Clustering Perspectives Report</h1>
                
                <div class="section">
                    <h2>Overview</h2>
                    <p>This report provides a comprehensive view of all clustering perspectives and their relationships.</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h3>Clustering Perspectives Summary</h3>
                    <table>
                        <tr>
                            <th>Perspective</th>
                            <th>Column</th>
                            <th>Clusters</th>
                            <th>Records</th>
                        </tr>
            """
            
            # Add summary table for each perspective
            for i in range(len(perspective_columns)):
                col = perspective_columns[i]
                name = perspective_names[i]
                
                # Count non-null records
                valid_records = dataframe[col].notna().sum()
                
                # Count unique clusters
                unique_clusters = dataframe[col].dropna().nunique()
                
                html_content += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{col}</td>
                        <td>{unique_clusters}</td>
                        <td>{valid_records}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
            
            # Add cross-perspective relationships section
            if len(perspective_columns) > 1:
                html_content += """
                <div class="section">
                    <h2>Cross-Perspective Relationships</h2>
                    <p>This section analyzes how different clustering perspectives relate to each other.</p>
                    
                    <h3>Mutual Information Scores</h3>
                    <table>
                        <tr>
                            <th>Perspective A</th>
                            <th>Perspective B</th>
                            <th>Mutual Information</th>
                            <th>Interpretation</th>
                        </tr>
                """
                
                # Calculate mutual information for each pair of perspectives
                for (i, j) in combinations(range(len(perspective_columns)), 2):
                    col1 = perspective_columns[i]
                    col2 = perspective_columns[j]
                    name1 = perspective_names[i]
                    name2 = perspective_names[j]
                    
                    # Get filtered data where both columns have values
                    filtered_data = dataframe.dropna(subset=[col1, col2])
                    
                    if len(filtered_data) > 0:
                        try:
                            mi_score = normalized_mutual_info_score(
                                filtered_data[col1],
                                filtered_data[col2]
                            )
                            
                            # Determine interpretation and class
                            if mi_score > 0.7:
                                interpretation = "Strong alignment"
                                score_class = "high"
                            elif mi_score > 0.4:
                                interpretation = "Moderate alignment"
                                score_class = "medium"
                            else:
                                interpretation = "Weak alignment"
                                score_class = "low"
                            
                            html_content += f"""
                            <tr>
                                <td>{name1}</td>
                                <td>{name2}</td>
                                <td><span class="relationship-score {score_class}">{mi_score:.4f}</span></td>
                                <td>{interpretation}</td>
                            </tr>
                            """
                        except Exception as e:
                            html_content += f"""
                            <tr>
                                <td>{name1}</td>
                                <td>{name2}</td>
                                <td colspan="2">Error calculating score: {str(e)}</td>
                            </tr>
                            """
                
                html_content += """
                    </table>
                """
                
                # Add visualizations grid if available
                html_content += """
                    <h3>Relationship Visualizations</h3>
                    <div class="grid-container">
                """
                
                # Check for correlation heatmaps between perspectives
                for (i, j) in combinations(range(len(perspective_columns)), 2):
                    col1 = perspective_columns[i]
                    col2 = perspective_columns[j]
                    name1 = perspective_names[i]
                    name2 = perspective_names[j]
                    
                    heatmap_path = os.path.join(self.results_dir, f"{name1}_{name2}_correlation_heatmap.png")
                    
                    if os.path.exists(heatmap_path):
                        # Create a relative path
                        rel_path = os.path.relpath(heatmap_path, self.results_dir)
                        
                        html_content += f"""
                        <div class="grid-item">
                            <h4>{name1} vs {name2}</h4>
                            <div class="heatmap">
                                <img src="{rel_path}" alt="Correlation Heatmap" />
                            </div>
                        </div>
                        """
                
                html_content += """
                    </div>
                </div>
                """
            
            # Add individual perspectives section
            html_content += """
                <div class="section">
                    <h2>Individual Perspectives</h2>
                    <p>This section provides links to detailed reports for each clustering perspective.</p>
                    
                    <ul>
            """
            
            # Add links to individual reports
            for i in range(len(perspective_columns)):
                name = perspective_names[i]
                detailed_report_path = os.path.join(self.results_dir, f"{name}_detailed_report.html")
                
                if os.path.exists(detailed_report_path):
                    # Create a relative path
                    rel_path = os.path.relpath(detailed_report_path, self.results_dir)
                    
                    html_content += f"""
                    <li><a href="{rel_path}">{name} Detailed Report</a></li>
                    """
            
            html_content += """
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated by the Text Classification System</p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Combined perspectives report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating combined perspectives report: {str(e)}")
            
            # Create a simple error report
            error_path = os.path.join(self.results_dir, "combined_perspectives_report_error.html")
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report</title>
                </head>
                <body>
                    <h1>Error Generating Combined Perspectives Report</h1>
                    <p>An error occurred while generating the report: {str(e)}</p>
                </body>
                </html>
                """)
            
            return error_path


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

    def create_cluster_correlation_heatmap(self, df, perspective1_col, perspective2_col, perspective1_name, perspective2_name):
        """
        Creates a heatmap visualization showing correlation between two clustering perspectives.
        
        This method generates a heatmap that shows how clusters from two different 
        perspectives relate to each other, helping to understand relationships
        between different clustering approaches.
        
        Args:
            df: DataFrame containing the clustering results
            perspective1_col: Column name for the first clustering perspective
            perspective2_col: Column name for the second clustering perspective
            perspective1_name: Display name for the first perspective
            perspective2_name: Display name for the second perspective
            
        Returns:
            Path to the saved heatmap visualization
        """
        self.logger.info(f"Creating correlation heatmap between {perspective1_name} and {perspective2_name}")
        
        try:
            # Check if columns exist
            if perspective1_col not in df.columns or perspective2_col not in df.columns:
                missing_cols = []
                if perspective1_col not in df.columns:
                    missing_cols.append(perspective1_col)
                if perspective2_col not in df.columns:
                    missing_cols.append(perspective2_col)
                raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_cols)}")
            
            # Create cross-tabulation
            # Use normalize='index' to show percentage of perspective1 clusters in each perspective2 cluster
            crosstab = pd.crosstab(
                df[perspective1_col], 
                df[perspective2_col],
                normalize='index'
            ) * 100
            
            # Get cluster name columns if they exist
            perspective1_name_col = f"{perspective1_col}_label"
            perspective2_name_col = f"{perspective2_col}_label"
            
            # Create dictionaries to map cluster IDs to names
            perspective1_names = {}
            perspective2_names = {}
            
            if perspective1_name_col in df.columns:
                for cluster_id in crosstab.index:
                    if pd.notna(cluster_id):
                        # Get the first non-null name for this cluster
                        mask = df[perspective1_col] == cluster_id
                        names = df.loc[mask, perspective1_name_col].dropna()
                        if len(names) > 0:
                            name = names.iloc[0]
                            # Truncate long names
                            if len(name) > 20:
                                name = name[:17] + "..."
                            perspective1_names[cluster_id] = f"{int(cluster_id)}: {name}"
                        else:
                            perspective1_names[cluster_id] = f"{int(cluster_id)}"
            
            if perspective2_name_col in df.columns:
                for cluster_id in crosstab.columns:
                    if pd.notna(cluster_id):
                        # Get the first non-null name for this cluster
                        mask = df[perspective2_col] == cluster_id
                        names = df.loc[mask, perspective2_name_col].dropna()
                        if len(names) > 0:
                            name = names.iloc[0]
                            # Truncate long names
                            if len(name) > 20:
                                name = name[:17] + "..."
                            perspective2_names[cluster_id] = f"{int(cluster_id)}: {name}"
                        else:
                            perspective2_names[cluster_id] = f"{int(cluster_id)}"
            
            # Set custom figure size based on number of clusters
            fig_width = max(12, crosstab.shape[1] * 0.8)
            fig_height = max(10, crosstab.shape[0] * 0.8)
            plt.figure(figsize=(fig_width, fig_height))
            
            # Apply custom labels if available
            if perspective1_names:
                crosstab.index = [perspective1_names.get(idx, str(idx)) for idx in crosstab.index]
            if perspective2_names:
                crosstab.columns = [perspective2_names.get(col, str(col)) for col in crosstab.columns]
            
            # Create heatmap with custom colormap and annotations
            cmap = plt.cm.YlGnBu
            sns.heatmap(
                crosstab, 
                annot=True, 
                cmap=cmap, 
                fmt=".1f",
                linewidths=0.5,
                cbar_kws={'label': 'Percentage (%)'}
            )
            
            # Set title and labels
            plt.title(f'Correlation Between {perspective1_name} and {perspective2_name} Clusters', fontsize=14)
            plt.xlabel(f'{perspective2_name} Clusters', fontsize=12)
            plt.ylabel(f'{perspective1_name} Clusters', fontsize=12)
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Tight layout to ensure all labels are visible
            plt.tight_layout()
            
            # Save the figure
            file_path = os.path.join(self.results_dir, f"{perspective1_name}_{perspective2_name}_correlation_heatmap.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Correlation heatmap saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            
            # Create a simple error image
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating heatmap: {str(e)}",
                    horizontalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective1_name}_{perspective2_name}_correlation_error.png")
            plt.savefig(error_path)
            plt.close()
            
            return error_path

    def create_cluster_term_importance_plot(self, characteristics, perspective_name):
        """
        Creates a visualization of top terms for each cluster.
        
        Args:
            characteristics: List of cluster characteristic dictionaries
            perspective_name: Name of the clustering perspective
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Creating term importance visualization for {perspective_name}")
        
        try:
            # Check if we have characteristics with top terms
            valid_chars = []
            for cluster in characteristics:
                if 'top_terms' in cluster and cluster['top_terms'] and len(cluster['top_terms']) > 0:
                    valid_chars.append(cluster)
            
            if not valid_chars:
                self.logger.warning("No valid characteristics with top terms provided")
                # Create a simple error image
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "No term importance data available for visualization",
                        horizontalalignment='center', fontsize=14, color='red')
                plt.axis('off')
                
                error_path = os.path.join(self.results_dir, f"{perspective_name}_term_importance_error.png")
                plt.savefig(error_path)
                plt.close()
                
                return error_path
            
            # Determine how many clusters to visualize (max 12 for readability)
            n_clusters = min(12, len(valid_chars))
            
            # Sort clusters by size (descending)
            sorted_clusters = sorted(valid_chars, key=lambda x: x.get('size', 0), reverse=True)
            clusters_to_visualize = sorted_clusters[:n_clusters]
            
            # Determine grid layout based on number of clusters
            if n_clusters <= 4:
                n_rows, n_cols = 2, 2
            elif n_clusters <= 6:
                n_rows, n_cols = 2, 3
            elif n_clusters <= 9:
                n_rows, n_cols = 3, 3
            else:
                n_rows, n_cols = 3, 4
            
            # Create figure with better styling
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            axes = axes.flatten()  # Flatten to easily iterate
            
            # Generate plots for each cluster
            for i, cluster in enumerate(clusters_to_visualize):
                if i >= len(axes):
                    break
                    
                # Get top terms (limit to top 10 for readability)
                top_n = min(10, len(cluster.get('top_terms', [])))
                terms = []
                scores = []
                
                for j in range(top_n):
                    if j < len(cluster.get('top_terms', [])):
                        term, score = cluster['top_terms'][j]
                        terms.append(term)
                        scores.append(score)
                
                if not terms:  # Skip if no terms
                    axes[i].text(0.5, 0.5, "No terms data", ha='center', va='center')
                    axes[i].set_title(f"Cluster {cluster.get('id', i)}")
                    continue
                    
                # Reverse lists for bottom-to-top plotting
                terms.reverse()
                scores.reverse()
                
                # Plot horizontal bar chart with improved styling
                ax = axes[i]
                
                # Use colormap to make bars visually distinct
                cmap = plt.cm.viridis
                colors = cmap(np.linspace(0.2, 0.8, len(terms)))
                
                bars = ax.barh(terms, scores, color=colors, alpha=0.8)
                
                # Add values to bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', ha='left', va='center', fontsize=8)
                
                # Set title and labels
                cluster_size = cluster.get('size', 0)
                cluster_pct = cluster.get('percentage', 0)
                ax.set_title(f"Cluster {cluster.get('id', i)}: {cluster_size} records ({cluster_pct:.1f}%)")
                ax.set_xlabel('Term Importance Score')
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Adjust y-axis to fit long terms
                plt.setp(ax.get_yticklabels(), fontsize=9)
                
                # Set x limit to make bars more visible
                max_score = max(scores) if scores else 0
                ax.set_xlim(0, max_score * 1.15)
            
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
            
            # Add overall title
            fig.suptitle(f'Top Terms by Importance for {perspective_name} Clusters', fontsize=16, y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure with high resolution
            file_path = os.path.join(self.results_dir, f"{perspective_name}_term_importance.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Term importance visualization saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating term importance plot: {str(e)}")
            
            # Create a simple error image
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating term importance plot: {str(e)}",
                    horizontalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective_name}_term_importance_error.png")
            plt.savefig(error_path)
            plt.close()
            
            return error_path
    
    def create_cluster_size_distribution_plot(self, cluster_assignments, cluster_names, perspective_name):
        """
        Creates an enhanced visualization of cluster size distribution.
        
        This method generates both a bar chart showing the absolute size of each cluster 
        and a line chart showing the cumulative percentage, providing insights into
        the distribution of data across clusters.
        
        Args:
            cluster_assignments: Array of cluster assignments for each data point
            cluster_names: Dictionary mapping cluster IDs to names
            perspective_name: Name of the clustering perspective
            
        Returns:
            Path to the saved visualization
        """
        self.logger.info(f"Creating enhanced cluster size distribution for {perspective_name}")
        
        try:
            # Count occurrences of each cluster
            counter = Counter(cluster_assignments)
            
            # Remove noise cluster (-1) if present
            if -1 in counter:
                del counter[-1]
            
            if not counter:
                raise ValueError("No valid clusters to visualize")
            
            # Get cluster IDs and counts
            cluster_ids = list(counter.keys())
            counts = list(counter.values())
            
            # Sort by count (descending)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_ids = [cluster_ids[i] for i in sorted_indices]
            sorted_counts = [counts[i] for i in sorted_indices]
            
            # Calculate percentages and cumulative percentages
            total = sum(sorted_counts)
            percentages = [100 * count / total for count in sorted_counts]
            cumulative = np.cumsum(percentages)
            
            # Create figure with twin y-axes
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ax2 = ax1.twinx()
            
            # Get cluster labels for x-axis
            if cluster_names:
                x_labels = [cluster_names.get(cluster_id, f"Cluster {cluster_id}") for cluster_id in sorted_ids]
                # Truncate long names
                x_labels = [label[:20] + "..." if len(label) > 20 else label for label in x_labels]
            else:
                x_labels = [f"Cluster {cluster_id}" for cluster_id in sorted_ids]
            
            # Create bar chart for counts
            bars = ax1.bar(x_labels, sorted_counts, color='cornflowerblue', alpha=0.7)
            
            # Create line chart for cumulative percentage
            ax2.plot(x_labels, cumulative, color='firebrick', marker='o', linestyle='-', linewidth=2, 
                    markersize=6, label='Cumulative %')
            
            # Add percentage labels above bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{percentages[i]:.1f}%',
                        ha='center', va='bottom', fontsize=9)
            
            # Set labels and title
            ax1.set_xlabel('Clusters', fontsize=12)
            ax1.set_ylabel('Number of Records', fontsize=12)
            ax2.set_ylabel('Cumulative Percentage', fontsize=12)
            plt.title(f'Cluster Size Distribution for {perspective_name}', fontsize=14)
            
            # Set y-axis limits for cumulative percentage
            ax2.set_ylim(0, 110)
            
            # Add gridlines
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add legend
            ax2.legend(loc='upper left')
            
            # Mark the 80% threshold (Pareto principle)
            if len(cumulative) > 0:
                pareto_idx = next((i for i, val in enumerate(cumulative) if val >= 80), len(cumulative) - 1)
                if pareto_idx < len(x_labels):
                    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.6)
                    ax2.text(0, 82, '80% Threshold', ha='left', va='bottom', color='green')
                    ax2.axvline(x=pareto_idx, color='green', linestyle='--', alpha=0.6)
            
            # Add annotations
            plt.figtext(0.02, 0.02, f'Total records: {total}', ha='left', fontsize=9)
            plt.figtext(0.98, 0.02, f'Total clusters: {len(sorted_ids)}', ha='right', fontsize=9)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            file_path = os.path.join(self.results_dir, f"{perspective_name}_size_distribution.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Cluster size distribution saved to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error creating cluster size distribution: {str(e)}")
            
            # Create a simple error image
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating size distribution: {str(e)}",
                    horizontalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            
            error_path = os.path.join(self.results_dir, f"{perspective_name}_size_distribution_error.png")
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
        
    def generate_detailed_cluster_report(self, perspective_name, cluster_characteristics, visualization_paths):
        """
        Generates a detailed HTML report for a clustering perspective.
        
        Args:
            perspective_name: Name of the clustering perspective
            cluster_characteristics: List of cluster characteristic dictionaries
            visualization_paths: Dictionary mapping visualization types to file paths
            
        Returns:
            Path to the generated HTML report
        """
        self.logger.info(f"Generating detailed cluster report for {perspective_name}")
        
        try:
            file_path = os.path.join(self.results_dir, f"{perspective_name}_detailed_report.html")
            
            # Start building HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Detailed Cluster Report - {perspective_name}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f9f9f9;
                    }}
                    h1, h2, h3, h4 {{
                        color: #2c3e50;
                    }}
                    h1 {{
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                        text-align: center;
                    }}
                    .section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .cluster-card {{
                        margin-bottom: 20px;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        overflow: hidden;
                    }}
                    .cluster-header {{
                        background-color: #edf2f7;
                        padding: 12px 15px;
                        border-bottom: 1px solid #ddd;
                    }}
                    .cluster-body {{
                        padding: 15px;
                    }}
                    .cluster-metrics {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 15px;
                        margin-bottom: 15px;
                    }}
                    .metric {{
                        background-color: #f8f9fa;
                        padding: 8px 12px;
                        border-radius: 4px;
                        font-size: 0.9em;
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
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .terms-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 8px;
                        margin-bottom: 15px;
                    }}
                    .term {{
                        background-color: #e9f7fe;
                        border: 1px solid #bde5f8;
                        border-radius: 15px;
                        padding: 4px 10px;
                        font-size: 0.9em;
                    }}
                    .examples {{
                        background-color: #f8f9fa;
                        border-left: 3px solid #3498db;
                        padding: 10px 15px;
                        margin-bottom: 15px;
                        font-style: italic;
                        color: #555;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 0.9em;
                        color: #777;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 10px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }}
                    th {{
                        background-color: #edf2f7;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                </style>
            </head>
            <body>
                <h1>Detailed Cluster Analysis Report: {perspective_name}</h1>
                
                <div class="section">
                    <h2>Overview</h2>
                    <p>This report presents a detailed analysis of the clustering results for the <strong>{perspective_name}</strong> perspective.</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h3>Cluster Distribution Summary</h3>
                    <table>
                        <tr>
                            <th>Cluster ID</th>
                            <th>Size</th>
                            <th>Percentage</th>
                            <th>Top Terms</th>
                        </tr>
            """
            
            # Add cluster distribution table
            sorted_chars = sorted(cluster_characteristics, key=lambda x: x.get('size', 0), reverse=True)
            for cluster in sorted_chars:
                top_terms_str = ", ".join([term for term, _ in cluster.get('top_terms', [])[:5]])
                html_content += f"""
                    <tr>
                        <td>Cluster {cluster.get('id', 'N/A')}</td>
                        <td>{cluster.get('size', 0)}</td>
                        <td>{cluster.get('percentage', 0):.2f}%</td>
                        <td>{top_terms_str}</td>
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
                
                for viz_type, path in visualization_paths.items():
                    if os.path.exists(path):
                        # Create a relative path
                        rel_path = os.path.relpath(path, self.results_dir)
                        title = viz_type.replace('_', ' ').title()
                        
                        html_content += f"""
                        <div class="visualization">
                            <h3>{title}</h3>
                            <img src="{rel_path}" alt="{title}" />
                        </div>
                        """
                
                html_content += """
                </div>
                """
            
            # Add detailed cluster analysis section
            html_content += """
                <div class="section">
                    <h2>Detailed Cluster Analysis</h2>
            """
            
            # Add cards for each cluster
            for cluster in sorted_chars:
                cluster_id = cluster.get('id', 'N/A')
                size = cluster.get('size', 0)
                percentage = cluster.get('percentage', 0)
                
                html_content += f"""
                    <div class="cluster-card">
                        <div class="cluster-header">
                            <h3>Cluster {cluster_id} ({percentage:.2f}% of data)</h3>
                        </div>
                        <div class="cluster-body">
                            <div class="cluster-metrics">
                                <div class="metric"><strong>Size:</strong> {size} records</div>
                """
                
                # Add dispersion and distinctiveness if available
                if 'dispersion' in cluster:
                    html_content += f"""
                                <div class="metric"><strong>Dispersion:</strong> {cluster['dispersion']:.4f}</div>
                    """
                
                if 'distinctiveness' in cluster:
                    html_content += f"""
                                <div class="metric"><strong>Distinctiveness:</strong> {cluster['distinctiveness']:.4f}</div>
                    """
                
                html_content += """
                            </div>
                """
                
                # Add top terms with weights
                if 'top_terms' in cluster and cluster['top_terms']:
                    html_content += """
                            <h4>Key Terms with Weights</h4>
                            <div class="terms-container">
                    """
                    
                    for term, weight in cluster['top_terms'][:15]:
                        html_content += f"""
                                <div class="term">{term} ({weight:.3f})</div>
                        """
                    
                    html_content += """
                            </div>
                    """
                
                # Add representative examples
                if 'examples' in cluster and cluster['examples']:
                    html_content += """
                            <h4>Representative Examples</h4>
                    """
                    
                    for i, example in enumerate(cluster['examples'][:3]):
                        # Truncate long examples
                        display_example = example
                        if len(example) > 300:
                            display_example = example[:300] + "..."
                        
                        html_content += f"""
                            <div class="examples">
                                <strong>Example {i+1}:</strong> {display_example}
                            </div>
                        """
                
                html_content += """
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div class="footer">
                    <p>Generated by the Text Classification System</p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Detailed cluster report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating detailed cluster report: {str(e)}")
            
            # Create a simple error report
            error_path = os.path.join(self.results_dir, f"{perspective_name}_detailed_report_error.html")
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report</title>
                </head>
                <body>
                    <h1>Error Generating Detailed Report</h1>
                    <p>An error occurred while generating the report: {str(e)}</p>
                </body>
                </html>
                """)
            
            return error_path

    def create_cross_perspective_analysis(self, dataframe, perspective1_col, perspective2_col, perspective1_name, perspective2_name):
        """
        Analyzes the relationship between different clustering perspectives.
        
        Args:
            dataframe: DataFrame containing the clustering results
            perspective1_col: Column name for the first clustering perspective
            perspective2_col: Column name for the second clustering perspective
            perspective1_name: Display name for the first perspective
            perspective2_name: Display name for the second perspective
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Creating cross-perspective analysis between {perspective1_name} and {perspective2_name}")
        
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            # Check if columns exist
            if perspective1_col not in dataframe.columns or perspective2_col not in dataframe.columns:
                missing_cols = []
                if perspective1_col not in dataframe.columns:
                    missing_cols.append(perspective1_col)
                if perspective2_col not in dataframe.columns:
                    missing_cols.append(perspective2_col)
                raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_cols)}")
            
            # Get cluster assignments
            df_filtered = dataframe.dropna(subset=[perspective1_col, perspective2_col])
            
            if len(df_filtered) == 0:
                raise ValueError(f"No overlapping data points between {perspective1_name} and {perspective2_name}")
            
            # Create cross-tabulation (raw counts)
            crosstab = pd.crosstab(df_filtered[perspective1_col], df_filtered[perspective2_col])
            
            # Create normalized cross-tabulation (row percentages)
            norm_crosstab = pd.crosstab(
                df_filtered[perspective1_col], 
                df_filtered[perspective2_col],
                normalize='index'
            ) * 100
            
            # Calculate mutual information score
            try:
                mi_score = normalized_mutual_info_score(
                    df_filtered[perspective1_col],
                    df_filtered[perspective2_col]
                )
            except:
                mi_score = None
            
            # Calculate adjusted Rand index
            try:
                ari_score = adjusted_rand_score(
                    df_filtered[perspective1_col],
                    df_filtered[perspective2_col]
                )
            except:
                ari_score = None
            
            # Find strongest relationships between clusters
            key_relationships = []
            
            for idx in norm_crosstab.index:
                row = norm_crosstab.loc[idx]
                max_col = row.idxmax()
                max_value = row[max_col]
                
                if max_value >= 25:  # Only include strong relationships (25% or more)
                    # Get cluster names if available
                    p1_name_col = f"{perspective1_col}_label"
                    p2_name_col = f"{perspective2_col}_label"
                    
                    p1_name = f"Cluster {idx}"
                    p2_name = f"Cluster {max_col}"
                    
                    if p1_name_col in dataframe.columns:
                        names = dataframe.loc[dataframe[perspective1_col] == idx, p1_name_col].dropna()
                        if len(names) > 0:
                            p1_name = names.iloc[0]
                    
                    if p2_name_col in dataframe.columns:
                        names = dataframe.loc[dataframe[perspective2_col] == max_col, p2_name_col].dropna()
                        if len(names) > 0:
                            p2_name = names.iloc[0]
                    
                    key_relationships.append({
                        'p1_cluster': idx,
                        'p2_cluster': max_col,
                        'p1_name': p1_name,
                        'p2_name': p2_name,
                        'percentage': max_value,
                        'count': crosstab.loc[idx, max_col]
                    })
            
            # Sort by percentage (descending)
            key_relationships.sort(key=lambda x: x['percentage'], reverse=True)
            
            # Generate markdown summary
            summary = [
                f"# Cross-Perspective Analysis: {perspective1_name} vs {perspective2_name}",
                "",
                "## Overall Relationship Metrics",
                f"- **Records analyzed**: {len(df_filtered)}",
                f"- **Normalized Mutual Information**: {mi_score:.4f}" if mi_score is not None else "",
                f"- **Adjusted Rand Index**: {ari_score:.4f}" if ari_score is not None else "",
                "",
                "## Key Relationships Between Clusters",
                ""
            ]
            
            for rel in key_relationships[:10]:  # Limit to top 10
                summary.append(f"- **{rel['p1_name']}** → **{rel['p2_name']}**: {rel['percentage']:.1f}% ({rel['count']} records)")
            
            summary.append("")
            summary.append("## Interpretation")
            
            # Add interpretation based on scores
            if mi_score is not None:
                if mi_score > 0.7:
                    summary.append("- The clustering perspectives show **strong alignment** (high mutual information)")
                elif mi_score > 0.4:
                    summary.append("- The clustering perspectives show **moderate alignment** (medium mutual information)")
                else:
                    summary.append("- The clustering perspectives show **weak alignment** (low mutual information)")
            
            if ari_score is not None:
                if ari_score > 0.5:
                    summary.append("- The cluster assignments are **highly consistent** between perspectives (high adjusted Rand index)")
                elif ari_score > 0.2:
                    summary.append("- The cluster assignments show **some consistency** between perspectives (medium adjusted Rand index)")
                else:
                    summary.append("- The cluster assignments show **little consistency** between perspectives (low adjusted Rand index)")
            
            markdown_summary = "\n".join(summary)
            
            # Save the analysis to a markdown file
            save_path = os.path.join(self.results_dir, f"{perspective1_name}_{perspective2_name}_analysis.md")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(markdown_summary)
            
            # Create result dictionary
            result = {
                'crosstab': crosstab,
                'normalized_crosstab': norm_crosstab,
                'mutual_information': mi_score,
                'adjusted_rand_index': ari_score,
                'key_relationships': key_relationships,
                'markdown_summary': markdown_summary,
                'file_path': save_path
            }
            
            self.logger.info(f"Cross-perspective analysis saved to: {save_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating cross-perspective analysis: {str(e)}")
            
            # Create error markdown
            error_text = f"# Error in Cross-Perspective Analysis\n\nAn error occurred: {str(e)}"
            error_path = os.path.join(self.results_dir, f"{perspective1_name}_{perspective2_name}_analysis_error.md")
            
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(error_text)
            
            return {
                'error': str(e),
                'file_path': error_path,
                'markdown_summary': error_text
            }        
        
    def generate_combined_perspectives_report(self, dataframe, perspective_columns, perspective_names):
        """
        Creates a comprehensive report comparing all clustering perspectives.
        
        Args:
            dataframe: DataFrame containing the clustering results
            perspective_columns: List of column names for each clustering perspective
            perspective_names: List of display names for each perspective
            
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating combined perspectives report")
        
        try:
            from sklearn.metrics import normalized_mutual_info_score
            from itertools import combinations
            
            if len(perspective_columns) != len(perspective_names):
                raise ValueError("Number of perspective columns and names must match")
            
            if len(perspective_columns) < 1:
                raise ValueError("At least one perspective is required")
            
            file_path = os.path.join(self.results_dir, "combined_perspectives_report.html")
            
            # Generate HTML content (similar structure as detailed_cluster_report)
            # [HTML content generation code]
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Combined Clustering Perspectives Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f9f9f9;
                    }}
                    h1, h2, h3, h4 {{
                        color: #2c3e50;
                    }}
                    h1 {{
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                        text-align: center;
                    }}
                    .section {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .grid-container {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        margin-top: 20px;
                    }}
                    .grid-item {{
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                    }}
                    .heatmap {{
                        margin: 20px 0;
                        text-align: center;
                    }}
                    .heatmap img {{
                        max-width: 100%;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 10px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }}
                    th {{
                        background-color: #edf2f7;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f8f9fa;
                    }}
                    .relationship-score {{
                        font-weight: bold;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }}
                    .high {{
                        background-color: #d4edda;
                        color: #155724;
                    }}
                    .medium {{
                        background-color: #fff3cd;
                        color: #856404;
                    }}
                    .low {{
                        background-color: #f8d7da;
                        color: #721c24;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 0.9em;
                        color: #777;
                    }}
                </style>
            </head>
            <body>
                <h1>Combined Clustering Perspectives Report</h1>
                
                <div class="section">
                    <h2>Overview</h2>
                    <p>This report provides a comprehensive view of all clustering perspectives and their relationships.</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h3>Clustering Perspectives Summary</h3>
                    <table>
                        <tr>
                            <th>Perspective</th>
                            <th>Column</th>
                            <th>Clusters</th>
                            <th>Records</th>
                        </tr>
            """
            
            # Add summary table for each perspective
            for i in range(len(perspective_columns)):
                col = perspective_columns[i]
                name = perspective_names[i]
                
                # Count non-null records
                valid_records = dataframe[col].notna().sum()
                
                # Count unique clusters
                unique_clusters = dataframe[col].dropna().nunique()
                
                html_content += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{col}</td>
                        <td>{unique_clusters}</td>
                        <td>{valid_records}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
            
            # Add cross-perspective relationships section
            if len(perspective_columns) > 1:
                html_content += """
                <div class="section">
                    <h2>Cross-Perspective Relationships</h2>
                    <p>This section analyzes how different clustering perspectives relate to each other.</p>
                    
                    <h3>Mutual Information Scores</h3>
                    <table>
                        <tr>
                            <th>Perspective A</th>
                            <th>Perspective B</th>
                            <th>Mutual Information</th>
                            <th>Interpretation</th>
                        </tr>
                """
                
                # Calculate mutual information for each pair of perspectives
                for (i, j) in combinations(range(len(perspective_columns)), 2):
                    col1 = perspective_columns[i]
                    col2 = perspective_columns[j]
                    name1 = perspective_names[i]
                    name2 = perspective_names[j]
                    
                    # Get filtered data where both columns have values
                    filtered_data = dataframe.dropna(subset=[col1, col2])
                    
                    if len(filtered_data) > 0:
                        try:
                            mi_score = normalized_mutual_info_score(
                                filtered_data[col1],
                                filtered_data[col2]
                            )
                            
                            # Determine interpretation and class
                            if mi_score > 0.7:
                                interpretation = "Strong alignment"
                                score_class = "high"
                            elif mi_score > 0.4:
                                interpretation = "Moderate alignment"
                                score_class = "medium"
                            else:
                                interpretation = "Weak alignment"
                                score_class = "low"
                            
                            html_content += f"""
                            <tr>
                                <td>{name1}</td>
                                <td>{name2}</td>
                                <td><span class="relationship-score {score_class}">{mi_score:.4f}</span></td>
                                <td>{interpretation}</td>
                            </tr>
                            """
                        except Exception as e:
                            html_content += f"""
                            <tr>
                                <td>{name1}</td>
                                <td>{name2}</td>
                                <td colspan="2">Error calculating score: {str(e)}</td>
                            </tr>
                            """
                
                html_content += """
                    </table>
                """
                
                # Add visualizations grid if available
                html_content += """
                    <h3>Relationship Visualizations</h3>
                    <div class="grid-container">
                """
                
                # Check for correlation heatmaps between perspectives
                for (i, j) in combinations(range(len(perspective_columns)), 2):
                    name1 = perspective_names[i]
                    name2 = perspective_names[j]
                    
                    heatmap_path = os.path.join(self.results_dir, f"{name1}_{name2}_correlation_heatmap.png")
                    
                    if os.path.exists(heatmap_path):
                        # Create a relative path
                        rel_path = os.path.relpath(heatmap_path, self.results_dir)
                        
                        html_content += f"""
                        <div class="grid-item">
                            <h4>{name1} vs {name2}</h4>
                            <div class="heatmap">
                                <img src="{rel_path}" alt="Correlation Heatmap" />
                            </div>
                        </div>
                        """
                
                html_content += """
                    </div>
                </div>
                """
            
            # Add individual perspectives section
            html_content += """
                <div class="section">
                    <h2>Individual Perspectives</h2>
                    <p>This section provides links to detailed reports for each clustering perspective.</p>
                    
                    <ul>
            """
            
            # Add links to individual reports
            for i in range(len(perspective_columns)):
                name = perspective_names[i]
                detailed_report_path = os.path.join(self.results_dir, f"{name}_detailed_report.html")
                
                if os.path.exists(detailed_report_path):
                    # Create a relative path
                    rel_path = os.path.relpath(detailed_report_path, self.results_dir)
                    
                    html_content += f"""
                    <li><a href="{rel_path}">{name} Detailed Report</a></li>
                    """
            
            html_content += """
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated by the Text Classification System</p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Combined perspectives report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error generating combined perspectives report: {str(e)}")
            
            # Create a simple error report
            error_path = os.path.join(self.results_dir, "combined_perspectives_report_error.html")
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error Report</title>
                </head>
                <body>
                    <h1>Error Generating Combined Perspectives Report</h1>
                    <p>An error occurred while generating the report: {str(e)}</p>
                </body>
                </html>
                """)
            
            return error_path


class ClusterAnalyzer:
    """
    Analyzer for extracting detailed insights from clustering results.
    
    This class provides methods for analyzing cluster content, extracting key terms,
    analyzing distributions, and generating human-readable summaries. It works with
    the output of clustering algorithms to provide deeper understanding of what
    each cluster represents.
    """

    def __init__(self, config, logger):
        """
        Initializes the cluster analyzer.

        Args:
            config: Configuration manager
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.analysis_config = config.get_config_value('cluster_analysis', {})
        
        # Default parameters
        self.default_terms_count = self.analysis_config.get('top_terms_count', 15)
        self.default_examples_count = self.analysis_config.get('examples_count', 5)
        
        self.logger.info("ClusterAnalyzer initialized")
    
    def analyze_cluster_content(self, features, vectorizer, cluster_assignments, k):
        """
        Performs a comprehensive analysis of what makes each cluster unique.
        
        This method analyzes the content of each cluster to determine its
        distinctive characteristics, including key terms, central examples,
        and statistical properties.
        
        Args:
            features: Feature matrix used for clustering
            vectorizer: TF-IDF vectorizer used to create the features
            cluster_assignments: Array of cluster assignments for each data point
            k: Number of clusters
            
        Returns:
            List of dictionaries, each containing characteristics of one cluster:
                - id: Cluster identifier
                - size: Number of records in the cluster
                - percentage: Percentage of total records
                - top_terms: List of (term, score) tuples for representative terms
                - center_idx: Index of the most central example
                - dispersion: Measure of cluster dispersion
                - distinctiveness: Measure of cluster separation
        """
        self.logger.info(f"Analyzing content for {k} clusters")
        
        try:
            # Initialize list to store cluster characteristics
            cluster_characteristics = []
            
            # Get unique clusters
            unique_clusters = sorted(np.unique(cluster_assignments))
            total_records = len(cluster_assignments)
            
            # Calculate cluster centers
            cluster_centers = {}
            for cluster_id in unique_clusters:
                cluster_mask = cluster_assignments == cluster_id
                if np.sum(cluster_mask) > 0:  # Ensure cluster has at least one point
                    cluster_features = features[cluster_mask]
                    cluster_centers[cluster_id] = np.mean(cluster_features, axis=0)
            
            # Analyze each cluster
            for cluster_id in unique_clusters:
                # Get indices of points in this cluster
                cluster_mask = cluster_assignments == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size == 0:
                    continue
                
                # Calculate percentage
                percentage = (cluster_size / total_records) * 100
                
                # Get cluster center
                cluster_center = cluster_centers[cluster_id]
                
                # Extract key terms if vectorizer is provided
                top_terms = []
                if vectorizer:
                    if hasattr(cluster_center, 'toarray'):
                        # For sparse matrices
                        cluster_center_array = cluster_center.toarray().flatten()
                    else:
                        # For dense arrays
                        cluster_center_array = cluster_center
                    
                    top_terms = self.extract_key_terms(
                        vectorizer, 
                        cluster_center_array, 
                        n_terms=self.default_terms_count
                    )
                
                # Find the most central example (closest to centroid)
                center_idx = -1
                if cluster_size > 0:
                    cluster_indices = np.where(cluster_mask)[0]
                    distances = []
                    for idx in cluster_indices:
                        point = features[idx]
                        if hasattr(point, 'toarray'):
                            point_array = point.toarray().flatten()
                        else:
                            point_array = point
                        
                        if hasattr(cluster_center, 'toarray'):
                            center_array = cluster_center.toarray().flatten()
                        else:
                            center_array = cluster_center
                        
                        # Calculate Euclidean distance
                        dist = np.linalg.norm(point_array - center_array)
                        distances.append(dist)
                    
                    # Get index of point closest to center
                    if distances:
                        min_dist_idx = np.argmin(distances)
                        center_idx = cluster_indices[min_dist_idx]
                
                # Calculate cluster dispersion (average distance to centroid)
                dispersion = 0.0
                if cluster_size > 1:
                    cluster_indices = np.where(cluster_mask)[0]
                    total_distance = 0.0
                    for idx in cluster_indices:
                        point = features[idx]
                        if hasattr(point, 'toarray'):
                            point_array = point.toarray().flatten()
                        else:
                            point_array = point
                        
                        if hasattr(cluster_center, 'toarray'):
                            center_array = cluster_center.toarray().flatten()
                        else:
                            center_array = cluster_center
                        
                        # Add distance to total
                        total_distance += np.linalg.norm(point_array - center_array)
                    
                    dispersion = total_distance / cluster_size
                
                # Calculate distinctiveness (min distance to other centroids)
                distinctiveness = float('inf')
                for other_id, other_center in cluster_centers.items():
                    if other_id != cluster_id:
                        if hasattr(cluster_center, 'toarray'):
                            center_array = cluster_center.toarray().flatten()
                        else:
                            center_array = cluster_center
                            
                        if hasattr(other_center, 'toarray'):
                            other_array = other_center.toarray().flatten()
                        else:
                            other_array = other_center
                        
                        dist = np.linalg.norm(center_array - other_array)
                        distinctiveness = min(distinctiveness, dist)
                
                # If no other clusters, set distinctiveness to 0
                if distinctiveness == float('inf'):
                    distinctiveness = 0.0
                
                # Store cluster characteristics
                cluster_characteristics.append({
                    'id': int(cluster_id),
                    'size': int(cluster_size),
                    'percentage': round(percentage, 2),
                    'top_terms': top_terms,
                    'center_idx': int(center_idx),
                    'dispersion': float(dispersion),
                    'distinctiveness': float(distinctiveness)
                })
            
            self.logger.info(f"Analyzed {len(cluster_characteristics)} clusters")
            return cluster_characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing cluster content: {str(e)}")
            return []
    
    def extract_key_terms(self, vectorizer, cluster_center, n_terms=15):
        """
        Extracts the most representative terms for a cluster.
        
        This method identifies the terms that best characterize a cluster
        based on their importance in the cluster's centroid vector.
        
        Args:
            vectorizer: TF-IDF vectorizer used to create the features
            cluster_center: Centroid vector for the cluster
            n_terms: Number of terms to extract (default: 15)
            
        Returns:
            List of (term, score) tuples sorted by importance
        """
        try:
            # Get feature names from vectorizer
            feature_names = vectorizer.get_feature_names_out()
            
            # Ensure cluster_center is a 1D array
            if len(cluster_center.shape) > 1:
                cluster_center = cluster_center.flatten()
            
            # Get indices of top terms
            top_indices = cluster_center.argsort()[-n_terms:][::-1]
            
            # Get terms and scores
            top_terms = [(feature_names[idx], float(cluster_center[idx])) for idx in top_indices]
            
            return top_terms
            
        except Exception as e:
            self.logger.error(f"Error extracting key terms: {str(e)}")
            return []
    
    def analyze_cluster_distribution(self, cluster_assignments):
        """
        Analyzes the size distribution of clusters.
        
        This method calculates various statistics about cluster sizes
        to help understand how data is distributed across clusters.
        
        Args:
            cluster_assignments: Array of cluster assignments for each data point
            
        Returns:
            Dictionary containing distribution statistics:
                - counts: Dictionary mapping cluster IDs to sizes
                - largest: ID of the largest cluster
                - smallest: ID of the smallest cluster
                - average_size: Average cluster size
                - median_size: Median cluster size
                - std_dev: Standard deviation of cluster sizes
                - imbalance_ratio: Ratio of largest to smallest cluster
                - gini_coefficient: Measure of distribution inequality (0-1)
        """
        self.logger.info("Analyzing cluster size distribution")
        
        try:
            # Count records in each cluster
            counter = Counter(cluster_assignments)
            
            # Remove -1 (noise) from statistics if present
            if -1 in counter:
                noise_count = counter[-1]
                del counter[-1]
                self.logger.info(f"Removed {noise_count} noise points from distribution analysis")
            
            if not counter:
                self.logger.warning("No valid clusters found for distribution analysis")
                return {
                    'counts': {},
                    'largest': None,
                    'smallest': None,
                    'average_size': 0,
                    'median_size': 0,
                    'std_dev': 0,
                    'imbalance_ratio': 0,
                    'gini_coefficient': 0
                }
            
            # Extract counts
            cluster_ids = list(counter.keys())
            counts = list(counter.values())
            
            # Calculate statistics
            total_samples = sum(counts)
            largest_cluster = max(counter.items(), key=lambda x: x[1])
            smallest_cluster = min(counter.items(), key=lambda x: x[1])
            average_size = total_samples / len(counter)
            median_size = np.median(counts)
            std_dev = np.std(counts)
            imbalance_ratio = largest_cluster[1] / max(smallest_cluster[1], 1)
            
            # Calculate Gini coefficient
            sorted_counts = sorted(counts)
            cumsum = np.cumsum(sorted_counts)
            cumsum = np.insert(cumsum, 0, 0)
            n = len(sorted_counts)
            area_under_curve = np.sum(cumsum) / (n * cumsum[-1])
            gini_coefficient = 1 - 2 * area_under_curve
            
            # Create result dictionary
            distribution = {
                'counts': dict(counter),
                'largest': largest_cluster[0],
                'smallest': smallest_cluster[0],
                'average_size': float(average_size),
                'median_size': float(median_size),
                'std_dev': float(std_dev),
                'imbalance_ratio': float(imbalance_ratio),
                'gini_coefficient': float(gini_coefficient)
            }
            
            self.logger.info(f"Analyzed distribution of {len(counter)} clusters")
            self.logger.debug(f"Distribution stats: largest={distribution['largest']} ({largest_cluster[1]} samples), "
                              f"imbalance={distribution['imbalance_ratio']:.2f}, gini={distribution['gini_coefficient']:.2f}")
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error analyzing cluster distribution: {str(e)}")
            return {
                'counts': {},
                'largest': None,
                'smallest': None,
                'average_size': 0,
                'median_size': 0,
                'std_dev': 0,
                'imbalance_ratio': 0,
                'gini_coefficient': 0
            }
    
    def generate_cluster_summary(self, characteristics):
        """
        Generates a readable summary of cluster characteristics.
        
        This method creates a markdown-formatted summary of cluster analysis results,
        highlighting key insights for easier human interpretation.
        
        Args:
            characteristics: List of cluster characteristic dictionaries
                           (from analyze_cluster_content)
            
        Returns:
            String containing markdown-formatted cluster summary
        """
        self.logger.info(f"Generating summary for {len(characteristics)} clusters")
        
        try:
            # Start building the summary
            summary = ["# Cluster Analysis Summary", ""]
            
            # Overall statistics
            total_size = sum([c['size'] for c in characteristics])
            total_clusters = len(characteristics)
            
            summary.extend([
                f"## Overview",
                f"- **Total clusters:** {total_clusters}",
                f"- **Total records:** {total_size}",
                ""
            ])
            
            # Sort clusters by size (descending)
            sorted_clusters = sorted(characteristics, key=lambda x: x['size'], reverse=True)
            
            # Add information about largest clusters
            top_n = min(5, len(sorted_clusters))
            summary.extend([
                f"## Largest Clusters",
                "| Cluster ID | Size | % of Total | Top Terms |",
                "|------------|------|------------|-----------|"
            ])
            
            for i in range(top_n):
                cluster = sorted_clusters[i]
                top_terms_str = ", ".join([term for term, _ in cluster['top_terms'][:5]])
                summary.append(
                    f"| {cluster['id']} | {cluster['size']} | {cluster['percentage']:.2f}% | {top_terms_str} |"
                )
            
            summary.append("")
            
            # Analysis of cluster distinctiveness
            if len(sorted_clusters) > 1:
                # Sort by distinctiveness (descending)
                distinct_clusters = sorted(characteristics, key=lambda x: x['distinctiveness'], reverse=True)
                least_distinct = distinct_clusters[-1]
                most_distinct = distinct_clusters[0]
                
                summary.extend([
                    f"## Cluster Distinctiveness",
                    f"- **Most distinct cluster:** Cluster {most_distinct['id']} (score: {most_distinct['distinctiveness']:.4f})",
                    f"  - Top terms: {', '.join([term for term, _ in most_distinct['top_terms'][:5]])}",
                    f"- **Least distinct cluster:** Cluster {least_distinct['id']} (score: {least_distinct['distinctiveness']:.4f})",
                    f"  - Top terms: {', '.join([term for term, _ in least_distinct['top_terms'][:5]])}",
                    ""
                ])
            
            # Detailed analysis of each cluster
            summary.extend(["## Detailed Cluster Analysis", ""])
            
            for cluster in sorted_clusters:
                summary.extend([
                    f"### Cluster {cluster['id']} ({cluster['percentage']:.2f}% of total)",
                    f"- **Size:** {cluster['size']} records",
                    f"- **Dispersion:** {cluster['dispersion']:.4f}",
                    f"- **Distinctiveness:** {cluster['distinctiveness']:.4f}",
                    "",
                    "#### Top Terms:",
                    ", ".join([f"{term} ({score:.3f})" for term, score in cluster['top_terms'][:10]]),
                    "",
                ])
            
            # Join the summary into a single string
            full_summary = "\n".join(summary)
            
            self.logger.info(f"Generated cluster summary of {len(full_summary)} characters")
            
            # Optionally save summary to file
            try:
                summary_dir = os.path.join(os.getcwd(), "summaries")
                os.makedirs(summary_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_path = os.path.join(summary_dir, f"cluster_summary_{timestamp}.md")
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(full_summary)
                self.logger.info(f"Saved cluster summary to: {summary_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save cluster summary: {str(e)}")
            
            return full_summary
            
        except Exception as e:
            self.logger.error(f"Error generating cluster summary: {str(e)}")
            return f"# Error Generating Cluster Summary\n\nAn error occurred: {str(e)}"
