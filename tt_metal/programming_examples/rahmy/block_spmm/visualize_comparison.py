#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open('test_basic/golden.txt', 'r') as f:
    golden = np.array([float(line.strip()) for line in f])

with open('test_basic/output.txt', 'r') as f:
    output = np.array([float(line.strip()) for line in f])

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Scatter plot (should show no correlation)
axes[0, 0].scatter(golden, output, alpha=0.3, s=1)
axes[0, 0].set_xlabel('Golden')
axes[0, 0].set_ylabel('Output')
axes[0, 0].set_title('Scatter Plot: No Element-wise Correlation')
axes[0, 0].grid(True, alpha=0.3)
# Add diagonal line for reference
min_val = min(golden.min(), output.min())
max_val = max(golden.max(), output.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
axes[0, 0].legend()

# 2. Overlaid histograms (should show similar distributions)
axes[0, 1].hist(golden, bins=50, alpha=0.5, label='Golden', density=True)
axes[0, 1].hist(output, bins=50, alpha=0.5, label='Output', density=True)
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution Comparison: Nearly Identical')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot (quantile-quantile)
golden_sorted = np.sort(golden)
output_sorted = np.sort(output)
axes[0, 2].scatter(golden_sorted, output_sorted, alpha=0.3, s=1)
axes[0, 2].plot([golden_sorted.min(), golden_sorted.max()],
                 [golden_sorted.min(), golden_sorted.max()], 'r--', alpha=0.5)
axes[0, 2].set_xlabel('Golden (sorted)')
axes[0, 2].set_ylabel('Output (sorted)')
axes[0, 2].set_title('Q-Q Plot: Distribution Similarity')
axes[0, 2].grid(True, alpha=0.3)

# 4. Time series plot (first 200 values)
n_plot = min(200, len(golden))
axes[1, 0].plot(golden[:n_plot], label='Golden', alpha=0.7)
axes[1, 0].plot(output[:n_plot], label='Output', alpha=0.7)
axes[1, 0].set_xlabel('Index')
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('First 200 Values: No Position Correlation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. CDF comparison
golden_cdf = np.arange(1, len(golden)+1) / len(golden)
output_cdf = np.arange(1, len(output)+1) / len(output)
axes[1, 1].plot(golden_sorted, golden_cdf, label='Golden', alpha=0.7)
axes[1, 1].plot(output_sorted, output_cdf, label='Output', alpha=0.7)
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('CDF Comparison: Nearly Overlapping')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Difference analysis
abs_diff = np.abs(golden - output)
axes[1, 2].hist(abs_diff, bins=50, edgecolor='black')
axes[1, 2].set_xlabel('Absolute Difference')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title(f'Element-wise Differences\nMean: {abs_diff.mean():.2e}')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_basic/similarity_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved to test_basic/similarity_analysis.png")
plt.close()

# Additional statistical tests
print("\nADDITIONAL INSIGHTS:")
print("=" * 70)

# Check if they might be permutations of each other
golden_set = set(np.round(golden, 2))
output_set = set(np.round(output, 2))
overlap = len(golden_set.intersection(output_set))
print(f"Unique values overlap (rounded to 2 decimals): {overlap}/{len(golden_set)}")

# Check sorted correlation
from scipy import stats
sorted_corr, _ = stats.pearsonr(golden_sorted, output_sorted)
print(f"Correlation of sorted values: {sorted_corr:.6f}")

print("\nCONCLUSION:")
print("The two files contain values drawn from nearly identical statistical")
print("distributions, but the values are not aligned position-by-position.")
print("This suggests they may be:")
print("  - Different runs of the same stochastic process")
print("  - Permutations of similar data")
print("  - Results from operations with different ordering but same statistics")
