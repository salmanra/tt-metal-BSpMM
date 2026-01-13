#!/usr/bin/env python3
import numpy as np

# Load the data
with open('test_1_block_simplified/golden.txt', 'r') as f:
    golden = np.array([float(line.strip()) for line in f])

with open('test_1_block_simplified/output.txt', 'r') as f:
    output = np.array([float(line.strip()) for line in f])

# Get the permutation indices that sort each array
golden_perm = np.argsort(golden)
output_perm = np.argsort(output)

print(f"Length of each permutation vector: {len(golden_perm)}")
print()

# Save the permutation vectors
np.savetxt('test_1_block_simplified/golden_permutation.txt', golden_perm, fmt='%d')
np.savetxt('test_1_block_simplified/output_permutation.txt', output_perm, fmt='%d')

print("Permutation vectors saved to:")
print("  - test_1_block_simplified/golden_permutation.txt")
print("  - test_1_block_simplified/output_permutation.txt")
print()

# Show first 20 elements of each permutation
print("First 20 elements of golden_permutation:")
print(golden_perm[:20])
print()

print("First 20 elements of output_permutation:")
print(output_perm[:20])
print()

# Show last 20 elements
print("Last 20 elements of golden_permutation:")
print(golden_perm[-20:])
print()

print("Last 20 elements of output_permutation:")
print(output_perm[-20:])
print()

# Verify the permutations work
golden_sorted = golden[golden_perm]
output_sorted = output[output_perm]

print("Verification - First 10 sorted values from golden:")
print(golden_sorted[:10])
print()

print("Verification - First 10 sorted values from output:")
print(output_sorted[:10])
print()

# Check correlation between the permutations themselves
from scipy import stats
perm_corr, _ = stats.pearsonr(golden_perm, output_perm)
print(f"Correlation between permutation vectors: {perm_corr:.6f}")

# Check Spearman correlation between permutations
perm_spearman, _ = stats.spearmanr(golden_perm, output_perm)
print(f"Spearman correlation between permutation vectors: {perm_spearman:.6f}")

# Check how many elements are in the same position after sorting
same_position = np.sum(golden_perm == output_perm)
print(f"Elements in same position: {same_position}/{len(golden_perm)} ({100*same_position/len(golden_perm):.2f}%)")
