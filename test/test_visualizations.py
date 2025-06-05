#!/usr/bin/env python3
"""
Quick test script to verify visualization fixes work correctly.
Run this before running the full classification to test image generation.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

def test_visualization_system():
    """Test that visualization system works and can create images."""
    print("Testing visualization system...")
    
    # Create test directory
    test_dir = "test_viz"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Test 1: Simple plot
        print("Test 1: Creating simple plot...")
        plt.figure(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.title("Test Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        test_file1 = os.path.join(test_dir, "test_simple.png")
        plt.savefig(test_file1, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_file1) and os.path.getsize(test_file1) > 0:
            print(f"✓ Simple plot created successfully: {os.path.getsize(test_file1)} bytes")
        else:
            print("✗ Simple plot failed")
            return False
        
        # Test 2: Distribution plot (like we use in classification)
        print("Test 2: Creating distribution plot...")
        data = ['A', 'B', 'C', 'A', 'B', 'A'] * 100
        counter = Counter(data)
        labels = list(counter.keys())
        counts = list(counter.values())
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(len(labels)), counts, color='skyblue', alpha=0.8)
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.title('Test Distribution')
        plt.xticks(range(len(labels)), labels)
        
        # Add percentage labels
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = count / total * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        test_file2 = os.path.join(test_dir, "test_distribution.png")
        plt.savefig(test_file2, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_file2) and os.path.getsize(test_file2) > 0:
            print(f"✓ Distribution plot created successfully: {os.path.getsize(test_file2)} bytes")
        else:
            print("✗ Distribution plot failed")
            return False
        
        # Test 3: Scatter plot (like embeddings)
        print("Test 3: Creating scatter plot...")
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        colors = np.random.randint(0, 5, 500)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=colors, alpha=0.7, s=20)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Test Scatter Plot')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        test_file3 = os.path.join(test_dir, "test_scatter.png")
        plt.savefig(test_file3, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_file3) and os.path.getsize(test_file3) > 0:
            print(f"✓ Scatter plot created successfully: {os.path.getsize(test_file3)} bytes")
        else:
            print("✗ Scatter plot failed")
            return False
        
        print("\n🎉 All visualization tests PASSED!")
        print(f"Test images saved in: {test_dir}/")
        print("You can now run the full classification with confidence.")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization_system()
    sys.exit(0 if success else 1)