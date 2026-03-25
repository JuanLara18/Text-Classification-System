"""
Evaluation and reporting module for the classification pipeline.
"""

import os
import json
import traceback
from datetime import datetime

import numpy as np

from modules.evaluation import ClusterAnalyzer, ClusteringEvaluator, ClusteringVisualizer, EvaluationReporter
from modules.evaluation import ClassificationEvaluator, ClassificationVisualizer


class PipelineEvaluator:
    """Handles evaluation and reporting for the classification pipeline."""

    def __init__(self, config, logger, perspectives):
        self.config = config
        self.logger = logger
        self.perspectives = perspectives

        results_dir = config.get_results_dir()
        self.evaluator = ClusteringEvaluator(config, logger)
        self.visualizer = ClusteringVisualizer(config, logger, results_dir)
        self.reporter = EvaluationReporter(config, logger, results_dir)

    def evaluate(self, dataframe, features_dict, cluster_assignments_dict):
        """
        Evaluates clustering AND classification results and generates reports.
        UPDATED to handle both clustering and AI classification perspectives.
        """
        return self.evaluate_and_report(dataframe, features_dict, cluster_assignments_dict)

    def evaluate_and_report(self, dataframe, features_dict, cluster_assignments_dict):
        """
        Evaluates clustering AND classification results and generates reports.
        UPDATED to handle both clustering and AI classification perspectives.
        """
        try:
            self.logger.info("Evaluating clustering and classification results")

            # Get visualization types from configuration
            visualization_types = self.config.get_config_value(
                'evaluation.visualizations',
                ['embeddings_plot', 'silhouette_plot', 'distribution_plot']
            )

            # Retrieve all perspectives (clustering + classification)
            all_perspectives = self.config.get_clustering_perspectives()

            # Initialize visualizers for both types
            results_dir = self.config.get_results_dir()
            clustering_visualizer = ClusteringVisualizer(self.config, self.logger, results_dir)
            classification_visualizer = ClassificationVisualizer(self.config, self.logger, results_dir)

            # Dictionary to store all evaluation outputs
            evaluation_results = {}

            # Initialize ClusterAnalyzer for clustering perspectives
            cluster_analyzer = ClusterAnalyzer(self.config, self.logger)

            # Evaluate each perspective
            for perspective_name, perspective_config in all_perspectives.items():
                perspective_type = perspective_config.get('type', 'clustering')

                self.logger.info(f"Evaluating {perspective_type} perspective: {perspective_name}")

                # Initialize entry in evaluation results
                evaluation_results[perspective_name] = {
                    'type': perspective_type,
                    'metrics': {},
                    'visualization_paths': {},
                    'report_paths': {}
                }

                try:
                    if perspective_type == 'clustering':
                        # Handle clustering perspective (existing logic)
                        self._evaluate_clustering_perspective(
                            perspective_name, perspective_config, dataframe,
                            features_dict, cluster_assignments_dict,
                            evaluation_results, clustering_visualizer,
                            cluster_analyzer, visualization_types
                        )

                    elif perspective_type == 'openai_classification':
                        # Handle AI classification perspective (new logic)
                        self._evaluate_classification_perspective(
                            perspective_name, perspective_config, dataframe,
                            evaluation_results, classification_visualizer
                        )

                    self.logger.info(f"Evaluation completed for perspective {perspective_name}")

                except Exception as e:
                    self.logger.error(f"Error evaluating perspective {perspective_name}: {str(e)}")
                    self.logger.error(traceback.format_exc())

            # Create cross-perspective comparisons if we have multiple classification perspectives
            classification_perspectives = {
                name: config for name, config in all_perspectives.items()
                if config.get('type') == 'openai_classification'
            }

            if len(classification_perspectives) > 1:
                self._create_classification_comparisons(
                    dataframe, classification_perspectives,
                    evaluation_results, classification_visualizer
                )

            # Generate AI classification cost report if applicable
            if classification_perspectives:
                self._generate_ai_cost_report(evaluation_results, classification_perspectives)

            if not evaluation_results:
                self.logger.error("No perspectives were evaluated successfully")
                return None

            # ── Generate rich interactive HTML report ──────────────────────────
            try:
                from classifai.reporter import generate_report
                text_cols = self.config.get_config_value("text_columns", [])
                results_dir = self.config.get_results_dir()
                # Auto-detect ground-truth column
                gt_col = None
                for candidate in ("true_category", "true_label", "label", "ground_truth"):
                    if candidate in dataframe.columns:
                        gt_col = candidate
                        break
                report_path = generate_report(
                    df=dataframe,
                    perspectives_config=all_perspectives,
                    results_dir=results_dir,
                    text_columns=text_cols,
                    ground_truth_col=gt_col,
                )
                self.logger.info(f"Interactive HTML report → {report_path}")
            except Exception as report_err:
                self.logger.warning(f"Could not generate rich HTML report: {report_err}")

            self.logger.info("Evaluation and reporting completed for all perspectives")
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error during evaluation and reporting: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _evaluate_clustering_perspective(self, perspective_name, perspective_config, dataframe,
                                         features_dict, cluster_assignments_dict, evaluation_results,
                                         clustering_visualizer, cluster_analyzer, visualization_types):
        """Evaluate a clustering perspective (existing logic extracted)."""
        combined_key = f"{perspective_name}_combined"
        if combined_key not in features_dict or perspective_name not in cluster_assignments_dict:
            self.logger.warning(f"Missing data for clustering perspective {perspective_name}, skipping evaluation")
            return

        features = features_dict[combined_key]
        assignments = cluster_assignments_dict[perspective_name]

        # Compute evaluation metrics
        metrics = self.evaluator.evaluate_clustering(features, assignments)
        evaluation_results[perspective_name]['metrics'] = metrics

        visualization_paths = {}

        # Perform enhanced cluster analysis if enabled
        if self.config.get_config_value('cluster_analysis.enabled', True):
            self.logger.info(f"Performing enhanced cluster analysis for {perspective_name}")

            if hasattr(dataframe, 'attrs') and 'cluster_characteristics' in dataframe.attrs:
                characteristics = dataframe.attrs['cluster_characteristics']
                self.logger.info(f"Using {len(characteristics)} stored cluster characteristics")
            else:
                characteristics = cluster_analyzer.analyze_cluster_content(
                    features, vectorizer=None, cluster_assignments=assignments, k=len(np.unique(assignments))
                )

            cluster_summary = cluster_analyzer.generate_cluster_summary(characteristics)

            # Create clustering visualizations
            try:
                output_column = perspective_config.get('output_column', f"{perspective_name}_cluster")
                label_column = f"{output_column}_label"
                cluster_names = {}

                if label_column in dataframe.columns:
                    for cluster_id in np.unique(assignments):
                        mask = dataframe[output_column] == cluster_id
                        if any(mask):
                            names = dataframe.loc[mask, label_column].dropna()
                            if len(names) > 0:
                                cluster_names[cluster_id] = names.iloc[0]

                viz_path = clustering_visualizer.create_cluster_size_distribution_plot(
                    assignments, cluster_names, perspective_name
                )
                visualization_paths['cluster_size_distribution'] = viz_path
            except Exception as e:
                self.logger.error(f"Error creating size distribution plot: {str(e)}")

            # Additional clustering visualizations...
            try:
                if 'embeddings_plot' in visualization_types:
                    viz_path = clustering_visualizer.create_embeddings_plot(features, assignments, perspective_name)
                    visualization_paths['embeddings_plot'] = viz_path
            except Exception as e:
                self.logger.error(f"Error creating embeddings plot: {str(e)}")

            try:
                if 'silhouette_plot' in visualization_types:
                    viz_path = clustering_visualizer.create_silhouette_plot(features, assignments, perspective_name)
                    visualization_paths['silhouette_plot'] = viz_path
            except Exception as e:
                self.logger.error(f"Error creating silhouette plot: {str(e)}")

        evaluation_results[perspective_name]['visualization_paths'] = visualization_paths

        # Generate reports
        try:
            report_paths = self.reporter.generate_report(perspective_name, metrics, visualization_paths)
            evaluation_results[perspective_name]['report_paths'].update(report_paths)
        except Exception as e:
            self.logger.error(f"Error generating reports: {str(e)}")

    def _evaluate_classification_perspective(self, perspective_name, perspective_config, dataframe,
                                             evaluation_results, classification_visualizer):
        """Evaluate an AI classification perspective (new logic)."""
        output_column = perspective_config.get('output_column', f"{perspective_name}_classification")

        if output_column not in dataframe.columns:
            self.logger.warning(f"Classification column {output_column} not found for perspective {perspective_name}")
            return

        # Get classification results
        classifications = dataframe[output_column].tolist()
        target_categories = perspective_config.get('target_categories', [])

        # Get metadata if available (from classifier manager)
        metadata = None

        # Evaluate classification
        metrics = self.evaluator.classification_evaluator.evaluate_classification(
            classifications, target_categories, metadata
        )
        evaluation_results[perspective_name]['metrics'] = metrics

        # Create visualizations
        visualization_paths = {}
        try:
            viz_path = classification_visualizer.create_classification_distribution_plot(
                classifications, perspective_name, target_categories
            )
            visualization_paths['distribution_plot'] = viz_path
        except Exception as e:
            self.logger.error(f"Error creating classification distribution plot: {str(e)}")

        evaluation_results[perspective_name]['visualization_paths'] = visualization_paths

        # Generate classification report
        try:
            report_paths = self._generate_classification_report(perspective_name, metrics, visualization_paths)
            evaluation_results[perspective_name]['report_paths'].update(report_paths)
        except Exception as e:
            self.logger.error(f"Error generating classification reports: {str(e)}")

    def _create_classification_comparisons(self, dataframe, classification_perspectives,
                                           evaluation_results, classification_visualizer):
        """Create comparison visualizations for multiple classification perspectives."""
        self.logger.info("Creating classification perspective comparisons")

        try:
            # Collect results for comparison
            perspective_results = {}
            perspective_names = list(classification_perspectives.keys())

            for name, config in classification_perspectives.items():
                output_column = config.get('output_column', f"{name}_classification")
                if output_column in dataframe.columns:
                    perspective_results[name] = dataframe[output_column].tolist()

            if len(perspective_results) >= 2:
                comparison_plot_path = classification_visualizer.create_classification_comparison_plot(
                    perspective_results, perspective_names
                )

                # Add to evaluation results
                for name in perspective_names:
                    if name in evaluation_results:
                        evaluation_results[name]['visualization_paths']['comparison_plot'] = comparison_plot_path

        except Exception as e:
            self.logger.error(f"Error creating classification comparisons: {str(e)}")

    def _generate_ai_cost_report(self, evaluation_results, classification_perspectives):
        """Generate a cost summary report for AI classification."""
        self.logger.info("Generating AI classification cost report")

        try:
            total_cost = 0
            total_tokens = 0
            total_api_calls = 0
            perspective_costs = {}

            for perspective_name in classification_perspectives.keys():
                if perspective_name in evaluation_results:
                    metrics = evaluation_results[perspective_name].get('metrics', {})
                    cost_metrics = metrics.get('cost_metrics', {})

                    perspective_cost = cost_metrics.get('total_cost', 0)
                    perspective_tokens = cost_metrics.get('total_tokens', 0)
                    perspective_calls = cost_metrics.get('api_calls', 0)

                    total_cost += perspective_cost
                    total_tokens += perspective_tokens
                    total_api_calls += perspective_calls

                    perspective_costs[perspective_name] = {
                        'cost': perspective_cost,
                        'tokens': perspective_tokens,
                        'api_calls': perspective_calls
                    }

            # Create cost report
            cost_report = {
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'total_api_calls': total_api_calls,
                'perspective_breakdown': perspective_costs,
                'generated_at': datetime.now().isoformat()
            }

            # Save cost report
            results_dir = self.config.get_results_dir()
            cost_report_path = os.path.join(results_dir, 'ai_classification_cost_report.json')
            with open(cost_report_path, 'w') as f:
                json.dump(cost_report, f, indent=2)

            self.logger.info(f"AI classification cost report saved to {cost_report_path}")
            self.logger.info(f"Total AI classification cost: ${total_cost:.4f} ({total_api_calls} API calls, {total_tokens} tokens)")

            # Check cost limits
            max_cost = self.config.get_config_value('ai_classification.cost_management.max_cost_per_run', 50.0)
            if total_cost > max_cost:
                self.logger.warning(f"AI classification cost ${total_cost:.4f} exceeds limit ${max_cost:.4f}")

        except Exception as e:
            self.logger.error(f"Error generating AI cost report: {str(e)}")

    def _generate_classification_report(self, perspective_name, metrics, visualization_paths):
        """Generate HTML report for classification perspective."""
        try:
            results_dir = self.config.get_results_dir()
            file_path = os.path.join(results_dir, f"{perspective_name}_classification_report.html")

            # Build HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Classification Report - {perspective_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
                    .visualization {{ margin: 20px 0; text-align: center; }}
                    .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
                    .cost-alert {{ color: red; font-weight: bold; }}
                    .distribution-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .distribution-table th, .distribution-table td {{
                        border: 1px solid #ddd; padding: 8px; text-align: left;
                    }}
                    .distribution-table th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>AI Classification Report: {perspective_name}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>Classification Metrics</h2>
                <div class="metric">
                    <strong>Total Classified:</strong> {metrics.get('total_classified', 0)}
                </div>
                <div class="metric">
                    <strong>Categories Used:</strong> {metrics.get('categories_used', 0)} / {metrics.get('total_categories', 0)}
                    (Coverage: {metrics.get('coverage_ratio', 0):.1%})
                </div>
                <div class="metric">
                    <strong>Distribution Balance:</strong> {metrics.get('balance_ratio', 0):.3f}
                    (1.0 = perfectly balanced)
                </div>
                <div class="metric">
                    <strong>Entropy:</strong> {metrics.get('normalized_entropy', 0):.3f}
                    (1.0 = maximum diversity)
                </div>
            """

            # Add cost information if available
            cost_metrics = metrics.get('cost_metrics', {})
            if cost_metrics:
                total_cost = cost_metrics.get('total_cost', 0)
                cost_class = 'cost-alert' if total_cost > 10 else ''

                html_content += f"""
                <h2>Cost Analysis</h2>
                <div class="metric {cost_class}">
                    <strong>Total Cost:</strong> ${total_cost:.4f}
                </div>
                <div class="metric">
                    <strong>Cost per Classification:</strong> ${cost_metrics.get('cost_per_classification', 0):.4f}
                </div>
                <div class="metric">
                    <strong>Cache Hit Rate:</strong> {cost_metrics.get('cache_hit_rate', 0):.1%}
                </div>
                <div class="metric">
                    <strong>Error Rate:</strong> {cost_metrics.get('error_rate', 0):.1%}
                </div>
                """

            # Add distribution table
            distribution = metrics.get('distribution', {})
            if distribution:
                html_content += """
                <h2>Category Distribution</h2>
                <table class="distribution-table">
                    <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
                """

                total = sum(distribution.values())
                for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / total * 100 if total > 0 else 0
                    html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                    """

                html_content += "</table>"

            # Add visualizations
            if visualization_paths:
                html_content += "<h2>Visualizations</h2>"
                for viz_name, viz_path in visualization_paths.items():
                    if os.path.exists(viz_path):
                        rel_path = os.path.relpath(viz_path, results_dir)
                        html_content += f"""
                        <div class="visualization">
                            <h3>{viz_name.replace('_', ' ').title()}</h3>
                            <img src="{rel_path}" alt="{viz_name}" />
                        </div>
                        """

            # Add issues if any
            issues = metrics.get('issues', [])
            if issues:
                html_content += "<h2>Issues Detected</h2><ul>"
                for issue in issues:
                    html_content += f"<li>{issue}</li>"
                html_content += "</ul>"

            html_content += """
            </body>
            </html>
            """

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Classification report saved to {file_path}")
            return {'html': file_path}

        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            return {}

    def cross_perspective_analysis(self, dataframe, evaluation_results):
        """
        Performs analysis across different clustering perspectives.

        Args:
            dataframe: DataFrame with all clustering results
            evaluation_results: Dictionary of evaluation results by perspective

        Returns:
            Dictionary with cross-perspective analysis results
        """
        return self.perform_cross_perspective_analysis(dataframe, evaluation_results)

    def perform_cross_perspective_analysis(self, dataframe, evaluation_results):
        """
        Performs analysis across different clustering perspectives.

        Args:
            dataframe: DataFrame with all clustering results
            evaluation_results: Dictionary of evaluation results by perspective

        Returns:
            Dictionary with cross-perspective analysis results
        """
        self.logger.info("Performing cross-perspective analysis")

        try:
            # Check if analysis is enabled
            if not self.config.get_config_value('cluster_analysis.cross_perspective_analysis', True):
                self.logger.info("Cross-perspective analysis is disabled in configuration")
                return {}

            # Get perspectives
            perspectives = self.config.get_clustering_perspectives()
            if len(perspectives) < 2:
                self.logger.info("Need at least 2 perspectives for cross-perspective analysis")
                return {}

            # Prepare for analysis
            perspective_names = list(perspectives.keys())
            perspective_columns = [config.get('output_column', f"{name}_cluster") for name, config in perspectives.items()]

            # Create cross-perspective visualizations and analysis
            cross_analysis_results = {}

            results_dir = self.config.get_results_dir()

            # Initialize the EvaluationReporter if not already available
            if not hasattr(self, 'reporter') or self.reporter is None:
                self.reporter = EvaluationReporter(self.config, self.logger, results_dir)

            # Initialize the ClusteringVisualizer if not already available
            if not hasattr(self, 'visualizer') or self.visualizer is None:
                self.visualizer = ClusteringVisualizer(self.config, self.logger, results_dir)

            # Generate correlation heatmaps between each pair of perspectives
            from itertools import combinations
            for i, j in combinations(range(len(perspective_names)), 2):
                name1 = perspective_names[i]
                name2 = perspective_names[j]
                col1 = perspective_columns[i]
                col2 = perspective_columns[j]

                # Create correlation heatmap
                heatmap_path = self.visualizer.create_cluster_correlation_heatmap(
                    dataframe, col1, col2, name1, name2
                )

                # Create cross-perspective analysis
                analysis = self.reporter.create_cross_perspective_analysis(
                    dataframe, col1, col2, name1, name2
                )

                # Store results
                key = f"{name1}_vs_{name2}"
                cross_analysis_results[key] = {
                    'heatmap_path': heatmap_path,
                    'analysis': analysis
                }

            # Generate combined perspectives report
            combined_report_path = self.reporter.generate_combined_perspectives_report(
                dataframe, perspective_columns, perspective_names
            )

            cross_analysis_results['combined_report'] = combined_report_path

            self.logger.info(f"Cross-perspective analysis completed with {len(cross_analysis_results)-1} perspective pairs")
            return cross_analysis_results

        except Exception as e:
            self.logger.error(f"Error during cross-perspective analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
