import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any
import umap
import warnings
warnings.filterwarnings('ignore')

# Fix font issues on server
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable warnings about too many figures

# Set memory-efficient defaults
matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['font.size'] = 8

# Ensure we can create figures in headless environment
plt.ioff()  # Turn off interactive mode

class ClusteringEvaluator:
    """Simple clustering evaluation with essential metrics."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classification_evaluator = ClassificationEvaluator(config, logger)
    
    def evaluate_clustering(self, features, cluster_assignments):
        """Evaluate clustering with core metrics only."""
        self.logger.info("Evaluating clustering results")
        
        try:
            unique_clusters = np.unique(cluster_assignments)
            if len(unique_clusters) < 2:
                return {"error": "Need at least 2 clusters for evaluation"}
            
            # Core metrics
            metrics = {
                'num_clusters': len(unique_clusters),
                'samples_total': len(cluster_assignments),
                'cluster_sizes': {int(c): int(np.sum(cluster_assignments == c)) for c in unique_clusters}
            }
            
            # Silhouette score
            try:
                if len(cluster_assignments) > 1000:  # Sample for large datasets
                    indices = np.random.choice(len(cluster_assignments), 1000, replace=False)
                    sample_features = features[indices]
                    sample_assignments = cluster_assignments[indices]
                else:
                    sample_features = features
                    sample_assignments = cluster_assignments
                
                metrics['silhouette_score'] = silhouette_score(sample_features, sample_assignments)
            except Exception as e:
                self.logger.warning(f"Silhouette score calculation failed: {e}")
                metrics['silhouette_score'] = None
            
            # Davies-Bouldin score
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, cluster_assignments)
            except Exception as e:
                self.logger.warning(f"Davies-Bouldin score calculation failed: {e}")
                metrics['davies_bouldin_score'] = None
            
            # Calinski-Harabasz score
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, cluster_assignments)
            except Exception as e:
                self.logger.warning(f"Calinski-Harabasz score calculation failed: {e}")
                metrics['calinski_harabasz_score'] = None
            
            metrics['timestamp'] = datetime.now().isoformat()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Clustering evaluation failed: {e}")
            return {"error": str(e)}


class ClassificationEvaluator:
    """Simple classification evaluation."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def evaluate_classification(self, classifications: List[str], categories: List[str], metadata: Dict = None):
        """Evaluate AI classification results with basic metrics."""
        try:
            distribution = Counter(classifications)
            total = len(classifications)
            
            # Basic metrics
            unique_categories = set(classifications)
            coverage_ratio = len(unique_categories) / len(categories) if categories else 0
            
            # Balance metrics
            counts = list(distribution.values())
            balance_ratio = min(counts) / max(counts) if counts and max(counts) > 0 else 0
            
            # Entropy
            probabilities = [count / total for count in counts]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(distribution)) if len(distribution) > 1 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Cost metrics if available
            cost_metrics = {}
            if metadata:
                cost_metrics = {
                    'total_cost': metadata.get('total_cost', 0),
                    'api_calls': metadata.get('api_calls', 0),
                    'cache_hit_rate': metadata.get('cache_hit_rate', 0)
                }
            
            return {
                'total_classified': total,
                'categories_used': len(unique_categories),
                'coverage_ratio': coverage_ratio,
                'balance_ratio': balance_ratio,
                'normalized_entropy': normalized_entropy,
                'distribution': dict(distribution),
                'cost_metrics': cost_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Classification evaluation failed: {e}")
            return {"error": str(e)}


class SimpleVisualizer:
    """FIXED: Simplified visualizer with better error handling and disk space management."""
    
    def __init__(self, config, logger, results_dir):
        self.config = config
        self.logger = logger
        self.results_dir = results_dir
        
        # Create results directory with error handling
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Could not create results directory {results_dir}: {e}")
            # Fallback to current directory
            self.results_dir = "."
        
        # Set matplotlib to use minimal memory and disk space
        plt.ioff()  # Turn off interactive mode
        matplotlib.rcParams['figure.max_open_warning'] = 0
        matplotlib.rcParams['font.size'] = 8  # Smaller fonts
        
        # Test write permissions
        self._test_write_permissions()
        
    def _test_write_permissions(self):
        """Test if we can write to the results directory."""
        try:
            test_file = os.path.join(self.results_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            self.can_write = True
            self.logger.info(f"Write permissions confirmed for {self.results_dir}")
        except Exception as e:
            self.can_write = False
            self.logger.warning(f"Cannot write to {self.results_dir}: {e}")
    
    def _safe_save_plot(self, file_path, dpi=150, bbox_inches='tight'):
        """Safely save plot with error handling and cleanup."""
        if not self.can_write:
            self.logger.warning("Cannot save plot - no write permissions")
            plt.close()
            return None
            
        try:
            # Use lower DPI and tight bbox to save space
            plt.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches, 
                       facecolor='white', edgecolor='none', format='png')
            plt.close()  # Always close to free memory
            
            # Verify file was created and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                self.logger.info(f"Successfully saved plot: {file_path}")
                return file_path
            else:
                self.logger.warning(f"Plot file is empty or missing: {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to save plot {file_path}: {e}")
            plt.close()  # Ensure plot is closed even on error
            
            # Try saving with minimal settings as fallback
            try:
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, f'Visualization failed\nDue to: {str(e)[:50]}...', 
                        ha='center', va='center', fontsize=10)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.title(f"Error in {os.path.basename(file_path)}")
                plt.axis('off')
                
                fallback_path = file_path.replace('.png', '_error.png')
                plt.savefig(fallback_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 0:
                    return fallback_path
                    
            except Exception as e2:
                self.logger.error(f"Even fallback save failed: {e2}")
                plt.close()
                
            return None
    
    def create_embeddings_plot(self, features, cluster_assignments, perspective_name):
        """Create FIXED 2D embeddings plot with better error handling."""
        try:
            self.logger.info(f"Creating embeddings plot for {perspective_name}")
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                features = features.toarray()
            
            # Sample for large datasets to save memory and disk space
            max_points = 1000
            if features.shape[0] > max_points:
                indices = np.random.choice(features.shape[0], max_points, replace=False)
                features_sample = features[indices]
                cluster_sample = cluster_assignments[indices]
                self.logger.info(f"Sampled {max_points} points from {features.shape[0]} for visualization")
            else:
                features_sample = features
                cluster_sample = cluster_assignments
            
            # Reduce dimensionality if needed
            if features_sample.shape[1] > 2:
                try:
                    # Use simpler UMAP settings to save memory
                    reducer = umap.UMAP(
                        n_components=2, 
                        random_state=42, 
                        n_neighbors=min(15, len(features_sample)//4),
                        min_dist=0.1,
                        n_epochs=100,  # Fewer epochs for speed
                        verbose=False
                    )
                    embedding = reducer.fit_transform(features_sample)
                except Exception as e:
                    self.logger.warning(f"UMAP failed: {e}, using PCA fallback")
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    embedding = pca.fit_transform(features_sample)
            else:
                embedding = features_sample
            
            # Create plot with minimal memory usage
            plt.figure(figsize=(8, 6))
            
            # Use fewer colors for clusters
            unique_clusters = np.unique(cluster_sample)
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_clusters), 10)))
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_sample == cluster_id
                if i < len(colors):
                    color = colors[i]
                else:
                    color = 'gray'
                
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[color], label=f'Cluster {cluster_id}', 
                           alpha=0.7, s=20, edgecolors='none')
            
            plt.title(f'Cluster Visualization - {perspective_name}', fontsize=12)
            plt.xlabel('Dimension 1', fontsize=10)
            plt.ylabel('Dimension 2', fontsize=10)
            
            # Limit legend entries
            if len(unique_clusters) <= 10:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            file_path = os.path.join(self.results_dir, f"{perspective_name}_embeddings.png")
            return self._safe_save_plot(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create embeddings plot for {perspective_name}: {e}")
            plt.close()
            return None
    
    def create_distribution_plot(self, assignments_or_classifications, perspective_name, is_classification=False):
        """Create FIXED distribution plot with better error handling."""
        try:
            self.logger.info(f"Creating distribution plot for {perspective_name}")
            
            # Count occurrences
            counter = Counter(assignments_or_classifications)
            
            # Remove noise cluster for clustering
            if not is_classification and -1 in counter:
                del counter[-1]
            
            if not counter:
                self.logger.warning("No data to plot")
                return None
            
            # Limit to top categories to save space
            max_categories = 20
            if len(counter) > max_categories:
                # Keep top categories and group others
                sorted_items = counter.most_common(max_categories - 1)
                other_count = sum(count for item, count in counter.items() 
                                if item not in dict(sorted_items))
                sorted_items.append(('Others', other_count))
                counter = dict(sorted_items)
            
            # Prepare data
            labels = list(counter.keys())
            counts = list(counter.values())
            
            # Sort by count
            sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
            labels, counts = zip(*sorted_data)
            
            # Create plot with appropriate size
            fig_width = max(8, min(len(labels) * 0.8, 16))
            plt.figure(figsize=(fig_width, 6))
            
            bars = plt.bar(range(len(labels)), counts, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
            
            # Add percentage labels on bars
            total = sum(counts)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                pct = count / total * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Format plot
            plt.xlabel('Categories' if is_classification else 'Clusters', fontsize=10)
            plt.ylabel('Count', fontsize=10)
            plt.title(f'Distribution - {perspective_name}', fontsize=12)
            
            # Handle x-axis labels
            if len(labels) > 10:
                # Rotate labels for many categories
                plt.xticks(range(len(labels)), [str(l)[:20] + '...' if len(str(l)) > 20 else str(l) for l in labels], 
                          rotation=45, ha='right', fontsize=8)
            else:
                plt.xticks(range(len(labels)), [str(l) for l in labels], fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            suffix = "classification" if is_classification else "clustering"
            file_path = os.path.join(self.results_dir, f"{perspective_name}_{suffix}_distribution.png")
            return self._safe_save_plot(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create distribution plot for {perspective_name}: {e}")
            plt.close()
            return None
    
    def create_silhouette_plot(self, features, cluster_assignments, perspective_name):
        """Create FIXED silhouette analysis plot."""
        try:
            self.logger.info(f"Creating silhouette plot for {perspective_name}")
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                features = features.toarray()
            
            # Sample for large datasets
            max_samples = 500  # Smaller sample for silhouette analysis
            if features.shape[0] > max_samples:
                indices = np.random.choice(features.shape[0], max_samples, replace=False)
                features_sample = features[indices]
                cluster_sample = cluster_assignments[indices]
            else:
                features_sample = features
                cluster_sample = cluster_assignments
            
            # Calculate silhouette scores
            from sklearn.metrics import silhouette_samples
            silhouette_values = silhouette_samples(features_sample, cluster_sample)
            avg_score = np.mean(silhouette_values)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            y_lower = 10
            unique_clusters = sorted(np.unique(cluster_sample))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster in enumerate(unique_clusters):
                cluster_silhouette_values = silhouette_values[cluster_sample == cluster]
                cluster_silhouette_values.sort()
                
                size_cluster_i = len(cluster_silhouette_values)
                y_upper = y_lower + size_cluster_i
                
                color = colors[i % len(colors)]
                plt.fill_betweenx(np.arange(y_lower, y_upper),
                                 0, cluster_silhouette_values,
                                 facecolor=color, edgecolor=color, alpha=0.7)
                
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster), fontsize=8)
                y_lower = y_upper + 10
            
            plt.axvline(x=avg_score, color="red", linestyle="--", linewidth=2,
                       label=f'Average Score: {avg_score:.3f}')
            plt.xlabel('Silhouette Coefficient', fontsize=10)
            plt.ylabel('Cluster', fontsize=10)
            plt.title(f'Silhouette Analysis - {perspective_name}', fontsize=12)
            plt.legend(fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            file_path = os.path.join(self.results_dir, f"{perspective_name}_silhouette.png")
            return self._safe_save_plot(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create silhouette plot for {perspective_name}: {e}")
            plt.close()
            return None

class SimpleReporter:
    """Simple markdown report generator."""
    
    def __init__(self, config, logger, results_dir):
        self.config = config
        self.logger = logger
        self.results_dir = results_dir
    
    def generate_perspective_report(self, perspective_name, perspective_type, metrics, visualization_paths):
        """Generate simple markdown report for a perspective."""
        try:
            lines = [
                f"# {perspective_name} Results Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"## Perspective Type: {perspective_type.replace('_', ' ').title()}",
                ""
            ]
            
            if perspective_type == 'clustering':
                lines.extend(self._format_clustering_metrics(metrics))
            elif perspective_type == 'openai_classification':
                lines.extend(self._format_classification_metrics(metrics))
            
            # Add visualizations
            if visualization_paths:
                lines.extend([
                    "",
                    "## Visualizations",
                    ""
                ])
                for viz_name, viz_path in visualization_paths.items():
                    if viz_path and os.path.exists(viz_path):
                        rel_path = os.path.relpath(viz_path, self.results_dir)
                        lines.append(f"![{viz_name}]({rel_path})")
                        lines.append("")
            
            # Save report
            report_path = os.path.join(self.results_dir, f"{perspective_name}_report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"Report saved: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return None
    
    def _format_clustering_metrics(self, metrics):
        """Format clustering metrics for markdown."""
        lines = [
            "## Clustering Metrics",
            "",
            f"- **Clusters**: {metrics.get('num_clusters', 'N/A')}",
            f"- **Total Samples**: {metrics.get('samples_total', 'N/A')}",
        ]
        
        # Quality scores
        silhouette = metrics.get('silhouette_score')
        if silhouette is not None:
            quality = "excellent" if silhouette > 0.7 else "good" if silhouette > 0.5 else "fair" if silhouette > 0.25 else "poor"
            lines.append(f"- **Silhouette Score**: {silhouette:.4f} ({quality})")
        
        davies_bouldin = metrics.get('davies_bouldin_score')
        if davies_bouldin is not None:
            lines.append(f"- **Davies-Bouldin Index**: {davies_bouldin:.4f} (lower is better)")
        
        calinski = metrics.get('calinski_harabasz_score')
        if calinski is not None:
            lines.append(f"- **Calinski-Harabasz Index**: {calinski:.4f} (higher is better)")
        
        # Cluster sizes
        cluster_sizes = metrics.get('cluster_sizes', {})
        if cluster_sizes:
            lines.extend([
                "",
                "### Cluster Sizes",
                ""
            ])
            for cluster_id, size in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True):
                total = metrics.get('samples_total', 1)
                pct = size / total * 100
                lines.append(f"- **Cluster {cluster_id}**: {size} samples ({pct:.1f}%)")
        
        return lines
    
    def _format_classification_metrics(self, metrics):
        """Format classification metrics for markdown."""
        lines = [
            "## Classification Metrics",
            "",
            f"- **Total Classified**: {metrics.get('total_classified', 'N/A')}",
            f"- **Categories Used**: {metrics.get('categories_used', 'N/A')}",
            f"- **Coverage Ratio**: {metrics.get('coverage_ratio', 0):.1%}",
            f"- **Balance Ratio**: {metrics.get('balance_ratio', 0):.3f}",
            f"- **Distribution Entropy**: {metrics.get('normalized_entropy', 0):.3f}",
        ]
        
        # Cost information
        cost_metrics = metrics.get('cost_metrics', {})
        if cost_metrics:
            lines.extend([
                "",
                "### Cost Analysis",
                "",
                f"- **Total Cost**: ${cost_metrics.get('total_cost', 0):.4f}",
                f"- **API Calls**: {cost_metrics.get('api_calls', 0)}",
                f"- **Cache Hit Rate**: {cost_metrics.get('cache_hit_rate', 0):.1%}",
            ])
        
        # Distribution
        distribution = metrics.get('distribution', {})
        if distribution:
            lines.extend([
                "",
                "### Category Distribution",
                ""
            ])
            total = sum(distribution.values())
            for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"- **{category}**: {count} ({pct:.1f}%)")
        
        return lines
    
    def generate_summary_report(self, all_results):
        """Generate overall summary report."""
        try:
            lines = [
                "# Classification System Summary Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"## Overview",
                f"Processed {len(all_results)} perspectives",
                ""
            ]
            
            # Perspective summary
            clustering_count = sum(1 for r in all_results.values() if r.get('type') == 'clustering')
            classification_count = sum(1 for r in all_results.values() if r.get('type') == 'openai_classification')
            
            lines.extend([
                f"- **Clustering Perspectives**: {clustering_count}",
                f"- **AI Classification Perspectives**: {classification_count}",
                "",
                "## Perspective Results",
                ""
            ])
            
            # Add link to each perspective report
            for perspective_name, result in all_results.items():
                report_file = f"{perspective_name}_report.md"
                if os.path.exists(os.path.join(self.results_dir, report_file)):
                    lines.append(f"- [{perspective_name}]({report_file})")
            
            # Save summary
            summary_path = os.path.join(self.results_dir, "summary_report.md")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"Summary report saved: {summary_path}")
            return summary_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return None


# FIXED: Update the ClassificationVisualizer class
class ClassificationVisualizer(SimpleVisualizer):
    """FIXED: Classification visualizer with proper error handling."""
    
    def create_classification_distribution_plot(self, classifications, perspective_name, target_categories=None):
        """Create distribution plot for classifications."""
        return self.create_distribution_plot(classifications, perspective_name, is_classification=True)
    
    def create_classification_comparison_plot(self, perspective_results, perspective_names):
        """Create comparison plot for multiple classification perspectives."""
        try:
            self.logger.info("Creating classification comparison plot")
            
            if len(perspective_results) < 2:
                self.logger.warning("Need at least 2 perspectives for comparison")
                return None
            
            # Create subplots
            n_perspectives = len(perspective_results)
            fig, axes = plt.subplots(1, n_perspectives, figsize=(5*n_perspectives, 6))
            
            if n_perspectives == 2:
                axes = [axes[0], axes[1]]
            
            for i, (name, results) in enumerate(perspective_results.items()):
                ax = axes[i] if n_perspectives > 1 else axes
                
                counter = Counter(results)
                labels = list(counter.keys())
                counts = list(counter.values())
                
                # Sort by count
                sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
                labels, counts = zip(*sorted_data)
                
                bars = ax.bar(range(len(labels)), counts, alpha=0.7)
                ax.set_title(f'{name}', fontsize=10)
                ax.set_ylabel('Count', fontsize=9)
                
                # Rotate labels if many categories
                if len(labels) > 5:
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels([str(l)[:10] + '...' if len(str(l)) > 10 else str(l) for l in labels], 
                                      rotation=45, ha='right', fontsize=8)
                else:
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, fontsize=8)
            
            plt.suptitle('Classification Perspectives Comparison', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            file_path = os.path.join(self.results_dir, "classification_comparison.png")
            return self._safe_save_plot(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison plot: {e}")
            plt.close()
            return None


# FIXED: Update the ClusteringVisualizer class  
class ClusteringVisualizer(SimpleVisualizer):
    """FIXED: Clustering visualizer with proper error handling."""
    
    def create_cluster_size_distribution_plot(self, cluster_assignments, cluster_names, perspective_name):
        """Create distribution plot for clusters."""
        return self.create_distribution_plot(cluster_assignments, perspective_name, is_classification=False)

class EvaluationReporter(SimpleReporter):
    """Alias for backward compatibility."""
    
    def generate_report(self, perspective_name, metrics, visualization_paths):
        # Determine perspective type from metrics
        perspective_type = 'clustering' if 'silhouette_score' in metrics else 'openai_classification'
        report_path = self.generate_perspective_report(perspective_name, perspective_type, metrics, visualization_paths)
        return {'markdown': report_path} if report_path else {}


# Simplified ClusterAnalyzer for basic cluster analysis
class ClusterAnalyzer:
    """Minimal cluster analyzer for basic functionality."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def analyze_cluster_content(self, features, vectorizer, cluster_assignments, k):
        """Basic cluster analysis returning minimal characteristics."""
        try:
            unique_clusters = np.unique(cluster_assignments)
            total_records = len(cluster_assignments)
            
            characteristics = []
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise
                    continue
                    
                cluster_size = np.sum(cluster_assignments == cluster_id)
                percentage = (cluster_size / total_records) * 100
                
                characteristics.append({
                    'id': int(cluster_id),
                    'size': int(cluster_size),
                    'percentage': round(percentage, 2),
                    'top_terms': [],  # Simplified - no term extraction
                    'examples': []    # Simplified - no example extraction
                })
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Cluster analysis failed: {e}")
            return []
    
    def generate_cluster_summary(self, characteristics):
        """Generate basic cluster summary."""
        try:
            lines = [
                "# Basic Cluster Summary",
                f"Total clusters: {len(characteristics)}",
                ""
            ]
            
            for cluster in sorted(characteristics, key=lambda x: x['size'], reverse=True):
                lines.append(f"- Cluster {cluster['id']}: {cluster['size']} samples ({cluster['percentage']:.1f}%)")
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed"
