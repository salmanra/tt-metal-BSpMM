#!/usr/bin/env python3
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
with open('test_basic/golden.txt', 'r') as f:
    golden = np.array([float(line.strip()) for line in f])

with open('test_basic/output.txt', 'r') as f:
    output = np.array([float(line.strip()) for line in f])

print(f"Number of values in each file: {len(golden)}, {len(output)}")
print("=" * 70)

# Basic statistics
print("\n1. BASIC STATISTICS:")
print("-" * 70)
print(f"Golden - Mean: {golden.mean():.4e}, Std: {golden.std():.4e}")
print(f"         Min: {golden.min():.4e}, Max: {golden.max():.4e}")
print(f"Output - Mean: {output.mean():.4e}, Std: {output.std():.4e}")
print(f"         Min: {output.min():.4e}, Max: {output.max():.4e}")

# Pearson correlation
print("\n2. CORRELATION MEASURES:")
print("-" * 70)
pearson_corr, pearson_p = stats.pearsonr(golden, output)
print(f"Pearson correlation: {pearson_corr:.6f} (p-value: {pearson_p:.4e})")

# Spearman rank correlation (checks monotonic relationship)
spearman_corr, spearman_p = stats.spearmanr(golden, output)
print(f"Spearman rank correlation: {spearman_corr:.6f} (p-value: {spearman_p:.4e})")

# Cosine similarity
cosine_sim = np.dot(golden, output) / (np.linalg.norm(golden) * np.linalg.norm(output))
print(f"Cosine similarity: {cosine_sim:.6f}")

# Distribution comparison
print("\n3. DISTRIBUTION SIMILARITY:")
print("-" * 70)
# Kolmogorov-Smirnov test (tests if distributions are the same)
ks_stat, ks_p = stats.ks_2samp(golden, output)
print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.6f}, p-value={ks_p:.4e}")
print(f"  (p < 0.05 suggests different distributions)")

# Compare normalized distributions
golden_norm = (golden - golden.mean()) / golden.std()
output_norm = (output - output.mean()) / output.std()
pearson_norm, _ = stats.pearsonr(golden_norm, output_norm)
print(f"Pearson correlation (normalized): {pearson_norm:.6f}")

# Element-wise comparison
print("\n4. ELEMENT-WISE DIFFERENCES:")
print("-" * 70)
abs_diff = np.abs(golden - output)
rel_diff = abs_diff / (np.abs(golden) + 1e-10)  # Avoid division by zero

print(f"Mean absolute difference: {abs_diff.mean():.4e}")
print(f"Median absolute difference: {np.median(abs_diff):.4e}")
print(f"Max absolute difference: {abs_diff.max():.4e}")
print(f"Mean relative difference: {rel_diff.mean():.6f}")
print(f"Median relative difference: {np.median(rel_diff):.6f}")

# Check for sign agreement
sign_agreement = np.sum(np.sign(golden) == np.sign(output)) / len(golden)
print(f"Sign agreement: {sign_agreement:.2%}")

# Check for order of magnitude agreement
golden_mag = np.floor(np.log10(np.abs(golden) + 1e-10))
output_mag = np.floor(np.log10(np.abs(output) + 1e-10))
mag_agreement = np.sum(golden_mag == output_mag) / len(golden)
print(f"Order of magnitude agreement: {mag_agreement:.2%}")

# Check for potential linear relationship with offset
print("\n5. LINEAR RELATIONSHIP ANALYSIS:")
print("-" * 70)
slope, intercept, r_value, p_value, std_err = stats.linregress(golden, output)
print(f"Linear fit: output = {slope:.6f} * golden + {intercept:.4e}")
print(f"R-squared: {r_value**2:.6f}")
print(f"Standard error: {std_err:.4e}")

# Check if ratio is constant
ratios = output / (golden + 1e-10)
print(f"\n6. RATIO ANALYSIS (output/golden):")
print("-" * 70)
print(f"Mean ratio: {ratios.mean():.6f}")
print(f"Std of ratio: {ratios.std():.6f}")
print(f"Median ratio: {np.median(ratios):.6f}")

# Quantile comparison
print("\n7. QUANTILE COMPARISON:")
print("-" * 70)
quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
for q in quantiles:
    g_q = np.quantile(golden, q)
    o_q = np.quantile(output, q)
    print(f"  {int(q*100):3d}%: Golden={g_q:+.4e}, Output={o_q:+.4e}")

# Histogram similarity (Bhattacharyya coefficient)
print("\n8. HISTOGRAM SIMILARITY:")
print("-" * 70)
bins = 50
hist_golden, bin_edges = np.histogram(golden, bins=bins, density=True)
hist_output, _ = np.histogram(output, bins=bin_edges, density=True)

# Normalize histograms
hist_golden = hist_golden / hist_golden.sum()
hist_output = hist_output / hist_output.sum()

# Bhattacharyya coefficient (1 = identical, 0 = completely different)
bhattacharyya = np.sum(np.sqrt(hist_golden * hist_output))
print(f"Bhattacharyya coefficient: {bhattacharyya:.6f} (1=identical, 0=different)")

# Hellinger distance
hellinger = np.sqrt(1 - bhattacharyya)
print(f"Hellinger distance: {hellinger:.6f} (0=identical, 1=different)")

# Chi-squared distance
chi_squared = np.sum((hist_golden - hist_output)**2 / (hist_golden + hist_output + 1e-10))
print(f"Chi-squared distance: {chi_squared:.6f}")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
if abs(pearson_corr) < 0.1:
    print("✗ No linear correlation (Pearson ≈ 0)")
else:
    print("✓ Linear correlation detected")

if abs(spearman_corr) < 0.1:
    print("✗ No monotonic relationship (Spearman ≈ 0)")
else:
    print("✓ Monotonic relationship detected")

if sign_agreement > 0.7:
    print(f"✓ Signs mostly agree ({sign_agreement:.1%})")
else:
    print(f"✗ Signs mostly disagree ({sign_agreement:.1%})")

if mag_agreement > 0.7:
    print(f"✓ Order of magnitude mostly agrees ({mag_agreement:.1%})")
else:
    print(f"✗ Order of magnitude mostly disagrees ({mag_agreement:.1%})")

if bhattacharyya > 0.7:
    print(f"✓ Distributions are similar (Bhattacharyya={bhattacharyya:.3f})")
else:
    print(f"✗ Distributions are different (Bhattacharyya={bhattacharyya:.3f})")
