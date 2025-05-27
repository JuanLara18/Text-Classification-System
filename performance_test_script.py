#!/usr/bin/env python3
"""
Performance Testing and Analysis Script for Optimized HR Classification

This script helps you:
1. Analyze your dataset to estimate performance improvements
2. Run test classifications with different configurations
3. Compare performance between original and optimized versions
4. Estimate costs and processing times
"""

import pandas as pd
import numpy as np
import time
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

class PerformanceAnalyzer:
    """Analyzer to predict and measure performance improvements."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        self.analysis_results = {}
    
    def load_and_analyze_dataset(self) -> Dict[str, Any]:
        """Load dataset and perform initial analysis."""
        print(f"Loading dataset: {self.data_file}")
        
        try:
            # Load the dataset
            self.df = pd.read_stata(self.data_file, convert_categoricals=False)
            print(f"✅ Loaded {len(self.df):,} records with {self.df.shape[1]} columns")
            
            # Analyze the position_name_english column
            position_col = 'position_name_english'
            if position_col not in self.df.columns:
                print(f"❌ Column '{position_col}' not found")
                return {}
            
            # Basic statistics
            total_records = len(self.df)
            non_null_records = self.df[position_col].notna().sum()
            null_records = total_records - non_null_records
            
            # Unique value analysis - THIS IS THE KEY INSIGHT
            unique_positions = self.df[position_col].dropna().nunique()
            duplicate_reduction = (total_records - unique_positions) / total_records
            
            # Position length analysis
            position_lengths = self.df[position_col].dropna().str.len()
            avg_length = position_lengths.mean()
            max_length = position_lengths.max()
            
            # Most common positions
            position_counts = self.df[position_col].value_counts().head(10)
            
            self.analysis_results = {
                'total_records': total_records,
                'non_null_records': non_null_records,
                'null_records': null_records,
                'unique_positions': unique_positions,
                'duplicate_reduction': duplicate_reduction,
                'avg_position_length': avg_length,
                'max_position_length': max_length,
                'top_positions': position_counts.to_dict()
            }
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return {}
    
    def estimate_performance_improvement(self) -> Dict[str, Any]:
        """Estimate performance improvements from unique value processing."""
        if not self.analysis_results:
            print("❌ No analysis results available. Run load_and_analyze_dataset() first.")
            return {}
        
        total = self.analysis_results['total_records']
        unique = self.analysis_results['unique_positions']
        reduction = self.analysis_results['duplicate_reduction']
        
        # Performance estimates
        estimates = {
            'api_calls_reduction': {
                'original_calls': total,
                'optimized_calls': unique,
                'reduction_ratio': reduction,
                'calls_saved': total - unique
            },
            'time_estimates': {
                'original_time_hours': total * 0.5 / 3600,  # 0.5 seconds per call
                'optimized_time_hours': unique * 0.5 / 3600,
                'time_saved_hours': (total - unique) * 0.5 / 3600
            },
            'cost_estimates': {
                'cost_per_classification': 0.0001,  # Estimated cost for gpt-4o-mini
                'original_cost': total * 0.0001,
                'optimized_cost': unique * 0.0001,
                'cost_saved': (total - unique) * 0.0001
            }
        }
        
        return estimates
    
    def create_performance_visualization(self, save_path: str = "performance_analysis.png"):
        """Create visualization of performance improvements."""
        if not self.analysis_results:
            print("❌ No analysis results available.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Records vs Unique Positions
        categories = ['Total Records', 'Unique Positions']
        values = [self.analysis_results['total_records'], self.analysis_results['unique_positions']]
        colors = ['#ff7f7f', '#7fbf7f']
        
        ax1.bar(categories, values, color=colors)
        ax1.set_title('Dataset Size: Total vs Unique Positions')
        ax1.set_ylabel('Number of Records')
        for i, v in enumerate(values):
            ax1.text(i, v + max(values) * 0.01, f'{v:,}', ha='center', fontweight='bold')
        
        # 2. Performance Improvement
        improvement_data = self.estimate_performance_improvement()
        api_calls = improvement_data['api_calls_reduction']
        
        categories = ['Original API Calls', 'Optimized API Calls']
        values = [api_calls['original_calls'], api_calls['optimized_calls']]
        colors = ['#ff9999', '#66b3ff']
        
        ax2.bar(categories, values, color=colors)
        ax2.set_title('API Calls: Original vs Optimized')
        ax2.set_ylabel('Number of API Calls')
        for i, v in enumerate(values):
            ax2.text(i, v + max(values) * 0.01, f'{v:,}', ha='center', fontweight='bold')
        
        # 3. Cost Comparison
        cost_data = improvement_data['cost_estimates']
        categories = ['Original Cost', 'Optimized Cost']
        values = [cost_data['original_cost'], cost_data['optimized_cost']]
        colors = ['#ffcc99', '#99ffcc']
        
        ax3.bar(categories, values, color=colors)
        ax3.set_title('Estimated Costs: Original vs Optimized')
        ax3.set_ylabel('Cost ($)')
        for i, v in enumerate(values):
            ax3.text(i, v + max(values) * 0.01, f'${v:.2f}', ha='center', fontweight='bold')
        
        # 4. Top Position Frequencies
        top_positions = list(self.analysis_results['top_positions'].items())[:8]
        positions = [pos[:20] + '...' if len(pos) > 20 else pos for pos, _ in top_positions]
        counts = [count for _, count in top_positions]
        
        ax4.barh(positions, counts, color='#ffb366')
        ax4.set_title('Most Common Position Names')
        ax4.set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance visualization saved to: {save_path}")
        plt.show()
    
    def print_analysis_report(self):
        """Print a comprehensive analysis report."""
        if not self.analysis_results:
            print("❌ No analysis results available.")
            return
        
        estimates = self.estimate_performance_improvement()
        
        print("\n" + "="*80)
        print("🚀 PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Total Records: {self.analysis_results['total_records']:,}")
        print(f"   • Valid Positions: {self.analysis_results['non_null_records']:,}")
        print(f"   • Unique Positions: {self.analysis_results['unique_positions']:,}")
        print(f"   • Duplicate Reduction: {self.analysis_results['duplicate_reduction']:.1%}")
        print(f"   • Average Position Length: {self.analysis_results['avg_position_length']:.1f} characters")
        
        print(f"\n⚡ PERFORMANCE IMPROVEMENTS:")
        api_calls = estimates['api_calls_reduction']
        print(f"   • API Calls Saved: {api_calls['calls_saved']:,} ({api_calls['reduction_ratio']:.1%} reduction)")
        
        time_est = estimates['time_estimates']
        print(f"   • Time Saved: {time_est['time_saved_hours']:.1f} hours")
        print(f"   • Original Time: {time_est['original_time_hours']:.1f} hours")
        print(f"   • Optimized Time: {time_est['optimized_time_hours']:.1f} hours")
        
        cost_est = estimates['cost_estimates']
        print(f"   • Cost Saved: ${cost_est['cost_saved']:.2f}")
        print(f"   • Original Cost: ${cost_est['original_cost']:.2f}")
        print(f"   • Optimized Cost: ${cost_est['optimized_cost']:.2f}")
        
        print(f"\n📈 TOP DUPLICATE POSITIONS:")
        for pos, count in list(self.analysis_results['top_positions'].items())[:5]:
            savings = (count - 1) * 0.0001  # Cost saved by not re-processing duplicates
            print(f"   • '{pos}': {count:,} occurrences (saves ${savings:.4f})")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if self.analysis_results['duplicate_reduction'] > 0.5:
            print(f"   ✅ EXCELLENT: {self.analysis_results['duplicate_reduction']:.1%} duplicate reduction - huge performance gains expected!")
        elif self.analysis_results['duplicate_reduction'] > 0.3:
            print(f"   ✅ GOOD: {self.analysis_results['duplicate_reduction']:.1%} duplicate reduction - significant performance gains expected!")
        else:
            print(f"   ⚠️  MODERATE: {self.analysis_results['duplicate_reduction']:.1%} duplicate reduction - some performance gains expected.")
        
        print(f"   • Use unique value processing for maximum efficiency")
        print(f"   • Set batch_size to {min(200, max(50, self.analysis_results['unique_positions'] // 100))}")
        print(f"   • Enable aggressive caching for repeated runs")
        print(f"   • Use {min(11, max(4, self.analysis_results['unique_positions'] // 1000))} parallel workers")


def run_performance_test():
    """Run a comprehensive performance test."""
    print("🧪 HR POSITION CLASSIFICATION PERFORMANCE TEST")
    print("="*60)
    
    # Configuration
    data_file = "input/HR_monthly_panel_translated.dta"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure your data file is in the correct location.")
        return
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(data_file)
    
    # Step 1: Load and analyze dataset
    print("\n1️⃣ Loading and analyzing dataset...")
    results = analyzer.load_and_analyze_dataset()
    
    if not results:
        print("❌ Failed to analyze dataset")
        return
    
    # Step 2: Print comprehensive report
    analyzer.print_analysis_report()
    
    # Step 3: Create visualization
    print("\n2️⃣ Creating performance visualization...")
    analyzer.create_performance_visualization()
    
    # Step 4: Generate test configuration
    print("\n3️⃣ Generating test configuration recommendations...")
    
    unique_count = results['unique_positions']
    
    # Small test configuration
    small_test_size = min(100, unique_count // 10)
    medium_test_size = min(1000, unique_count // 5)
    
    print(f"\n📋 TESTING RECOMMENDATIONS:")
    print(f"   • Small test: {small_test_size} unique positions (~{small_test_size * 2} total records)")
    print(f"   • Medium test: {medium_test_size} unique positions (~{medium_test_size * 2} total records)")
    print(f"   • Full run: {unique_count} unique positions ({results['total_records']:,} total records)")
    
    print(f"\n⚙️  OPTIMAL CONFIGURATION FOR YOUR HARDWARE:")
    print(f"   • batch_size: {min(200, max(50, unique_count // 100))}")
    print(f"   • max_workers: {min(11, max(4, unique_count // 1000))}")
    print(f"   • concurrent_requests: {min(20, max(5, unique_count // 500))}")
    print(f"   • memory_cache_size: {min(500000, unique_count * 5)}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Test with small sample first: Add 'sample_size: {small_test_size}' to testing config")
    print(f"   2. Use the optimized configuration provided")
    print(f"   3. Monitor cache hit rates and adjust settings")
    print(f"   4. Scale up to full dataset once testing is successful")


def create_test_config(sample_size: int = 1000):
    """Create a test configuration file for initial testing."""
    config_content = f"""# TEST CONFIGURATION - Small Sample
# This config is for testing the optimized classification system

# Override the main config for testing
testing:
  enabled: true
  sample_size: {sample_size}
  sample_method: "random"
  output_file: "output/HR_test_sample.dta"

# Use smaller resource allocations for testing
ai_classification:
  parallel_processing:
    max_workers: 4
    chunk_size: 100
  
  rate_limiting:
    requests_per_minute: 500
    concurrent_requests: 8
  
  caching:
    memory_cache_size: 10000

# Test-specific logging
logging:
  level: "DEBUG"
  log_file: "logs/test_classification.log"
"""
    
    with open("test_config.yaml", "w") as f:
        f.write(config_content)
    
    print(f"✅ Test configuration saved to: test_config.yaml")
    print(f"   Sample size: {sample_size}")
    print(f"   Run with: python main.py --config test_config.yaml")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance analysis for HR classification")
    parser.add_argument("--test", action="store_true", help="Run performance test")
    parser.add_argument("--create-test-config", type=int, default=1000, 
                       help="Create test config with specified sample size")
    parser.add_argument("--data-file", default="input/HR_monthly_panel_translated.dta",
                       help="Path to data file")
    
    args = parser.parse_args()
    
    if args.test:
        run_performance_test()
    elif args.create_test_config:
        create_test_config(args.create_test_config)
    else:
        print("Use --test to run performance analysis or --create-test-config N to create test config")