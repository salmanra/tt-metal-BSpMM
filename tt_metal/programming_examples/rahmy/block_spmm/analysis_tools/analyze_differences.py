#!/usr/bin/env python3
"""
Analyze magnitude of differences between two numerical files.
Find regions where tolerance is reasonable.
"""

import sys
import argparse
import numpy as np
from pathlib import Path


def analyze_numerical_differences(file1_path, file2_path, tolerance_percent=5.0, region_size=1000):
    """
    Analyze numerical differences between two files.

    Args:
        file1_path: Path to golden/reference file
        file2_path: Path to output file
        tolerance_percent: Acceptable relative error percentage
        region_size: Number of consecutive lines to consider a "region"
    """

    print(f"Analyzing numerical differences:")
    print(f"  Golden: {file1_path}")
    print(f"  Output: {file2_path}")
    print(f"  Tolerance: {tolerance_percent}%")
    print(f"  Region size: {region_size} lines")
    print("=" * 80)

    absolute_errors = []
    relative_errors = []
    golden_values = []
    output_values = []
    line_numbers = []

    within_tolerance_count = 0
    total_valid_comparisons = 0

    print("\nReading and analyzing files...")

    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            line_num = 0

            while True:
                line1 = f1.readline()
                line2 = f2.readline()

                if not line1 and not line2:
                    break

                line_num += 1

                # Progress indicator
                if line_num % 100000 == 0:
                    print(f"Processed {line_num:,} lines...", end='\r', flush=True)

                # Parse numerical values
                try:
                    val1 = float(line1.strip())
                    val2 = float(line2.strip())

                    golden_values.append(val1)
                    output_values.append(val2)
                    line_numbers.append(line_num)

                    # Calculate errors
                    abs_error = abs(val2 - val1)
                    absolute_errors.append(abs_error)

                    # Relative error (handle division by zero)
                    if abs(val1) > 1e-10:  # Avoid division by very small numbers
                        rel_error = abs((val2 - val1) / val1) * 100
                    else:
                        rel_error = 0.0 if abs_error < 1e-10 else float('inf')

                    relative_errors.append(rel_error)

                    # Check tolerance
                    total_valid_comparisons += 1
                    if rel_error <= tolerance_percent or (abs(val1) < 1e-10 and abs_error < 1e-10):
                        within_tolerance_count += 1

                except (ValueError, AttributeError):
                    # Skip non-numeric lines
                    continue

        print(" " * 80, end='\r')

        # Convert to numpy arrays for statistical analysis
        absolute_errors = np.array(absolute_errors)
        relative_errors = np.array(relative_errors)
        golden_values = np.array(golden_values)
        output_values = np.array(output_values)
        line_numbers = np.array(line_numbers)

        # Remove infinite relative errors for statistics
        finite_rel_errors = relative_errors[np.isfinite(relative_errors)]

        print("\n" + "=" * 80)
        print("ERROR STATISTICS")
        print("=" * 80)

        print(f"\nTotal comparisons: {total_valid_comparisons:,}")
        print(f"Within tolerance ({tolerance_percent}%): {within_tolerance_count:,} ({within_tolerance_count/total_valid_comparisons*100:.2f}%)")
        print(f"Outside tolerance: {total_valid_comparisons - within_tolerance_count:,} ({(total_valid_comparisons - within_tolerance_count)/total_valid_comparisons*100:.2f}%)")

        print("\n--- ABSOLUTE ERRORS ---")
        print(f"Min:     {np.min(absolute_errors):.6e}")
        print(f"Max:     {np.max(absolute_errors):.6e}")
        print(f"Mean:    {np.mean(absolute_errors):.6e}")
        print(f"Median:  {np.median(absolute_errors):.6e}")
        print(f"Std Dev: {np.std(absolute_errors):.6e}")
        print(f"\nPercentiles:")
        print(f"  25th: {np.percentile(absolute_errors, 25):.6e}")
        print(f"  50th: {np.percentile(absolute_errors, 50):.6e}")
        print(f"  75th: {np.percentile(absolute_errors, 75):.6e}")
        print(f"  90th: {np.percentile(absolute_errors, 90):.6e}")
        print(f"  95th: {np.percentile(absolute_errors, 95):.6e}")
        print(f"  99th: {np.percentile(absolute_errors, 99):.6e}")

        print("\n--- RELATIVE ERRORS (%) ---")
        if len(finite_rel_errors) > 0:
            print(f"Min:     {np.min(finite_rel_errors):.4f}%")
            print(f"Max:     {np.max(finite_rel_errors):.4f}%")
            print(f"Mean:    {np.mean(finite_rel_errors):.4f}%")
            print(f"Median:  {np.median(finite_rel_errors):.4f}%")
            print(f"Std Dev: {np.std(finite_rel_errors):.4f}%")
            print(f"\nPercentiles:")
            print(f"  25th: {np.percentile(finite_rel_errors, 25):.4f}%")
            print(f"  50th: {np.percentile(finite_rel_errors, 50):.4f}%")
            print(f"  75th: {np.percentile(finite_rel_errors, 75):.4f}%")
            print(f"  90th: {np.percentile(finite_rel_errors, 90):.4f}%")
            print(f"  95th: {np.percentile(finite_rel_errors, 95):.4f}%")
            print(f"  99th: {np.percentile(finite_rel_errors, 99):.4f}%")

        # Find regions with reasonable tolerance
        print("\n" + "=" * 80)
        print(f"REGIONS WITH REASONABLE TOLERANCE (<= {tolerance_percent}%)")
        print("=" * 80)

        # Create boolean array for within tolerance
        within_tolerance = (relative_errors <= tolerance_percent) | \
                          ((np.abs(golden_values) < 1e-10) & (absolute_errors < 1e-10))

        # Find consecutive regions
        regions = []
        in_region = False
        region_start = 0

        for i, is_good in enumerate(within_tolerance):
            if is_good and not in_region:
                # Start of a good region
                region_start = i
                in_region = True
            elif not is_good and in_region:
                # End of a good region
                if i - region_start >= region_size:
                    regions.append((region_start, i - 1))
                in_region = False

        # Check if we ended in a good region
        if in_region and len(within_tolerance) - region_start >= region_size:
            regions.append((region_start, len(within_tolerance) - 1))

        if regions:
            print(f"\nFound {len(regions)} regions with >= {region_size} consecutive lines within tolerance:\n")

            for idx, (start_idx, end_idx) in enumerate(regions[:20]):  # Show first 20
                start_line = line_numbers[start_idx]
                end_line = line_numbers[end_idx]
                region_len = end_idx - start_idx + 1

                # Calculate stats for this region
                region_rel_errors = relative_errors[start_idx:end_idx+1]
                region_rel_errors_finite = region_rel_errors[np.isfinite(region_rel_errors)]

                if len(region_rel_errors_finite) > 0:
                    max_err = np.max(region_rel_errors_finite)
                    mean_err = np.mean(region_rel_errors_finite)
                else:
                    max_err = 0.0
                    mean_err = 0.0

                print(f"Region {idx+1}: Lines {start_line:,} - {end_line:,} ({region_len:,} lines)")
                print(f"  Max relative error: {max_err:.4f}%")
                print(f"  Mean relative error: {mean_err:.4f}%")

            if len(regions) > 20:
                print(f"\n... and {len(regions) - 20} more regions")
        else:
            print(f"\nNo regions found with >= {region_size} consecutive lines within {tolerance_percent}% tolerance")

        # Show worst regions
        print("\n" + "=" * 80)
        print("WORST ERROR REGIONS")
        print("=" * 80)

        # Find regions with highest average error
        worst_regions = []
        window_size = 100

        for i in range(0, len(relative_errors) - window_size, window_size):
            window = relative_errors[i:i+window_size]
            window_finite = window[np.isfinite(window)]
            if len(window_finite) > 0:
                avg_error = np.mean(window_finite)
                max_error = np.max(window_finite)
                worst_regions.append((i, avg_error, max_error))

        # Sort by average error
        worst_regions.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 worst regions (window size: {window_size} lines):\n")
        for idx, (start_idx, avg_err, max_err) in enumerate(worst_regions[:10]):
            start_line = line_numbers[start_idx] if start_idx < len(line_numbers) else start_idx
            end_line = line_numbers[min(start_idx + window_size - 1, len(line_numbers) - 1)]

            print(f"Region {idx+1}: Lines {start_line:,} - {end_line:,}")
            print(f"  Avg relative error: {avg_err:.4f}%")
            print(f"  Max relative error: {max_err:.4f}%")

            # Show a sample value from this region
            if start_idx < len(golden_values):
                sample_golden = golden_values[start_idx]
                sample_output = output_values[start_idx]
                print(f"  Sample: {sample_golden:.6e} (golden) vs {sample_output:.6e} (output)")

        # Create error distribution histogram
        print("\n" + "=" * 80)
        print("RELATIVE ERROR DISTRIBUTION")
        print("=" * 80)

        if len(finite_rel_errors) > 0:
            bins = [0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
            hist, _ = np.histogram(finite_rel_errors, bins=bins)

            print("\nError Range          | Count       | Percentage")
            print("-" * 60)
            for i in range(len(bins) - 1):
                lower = bins[i]
                upper = bins[i+1]
                count = hist[i]
                pct = count / len(finite_rel_errors) * 100

                if upper == float('inf'):
                    print(f"{lower:6.1f}% - inf        | {count:11,} | {pct:6.2f}%")
                else:
                    print(f"{lower:6.1f}% - {upper:6.1f}% | {count:11,} | {pct:6.2f}%")

        print("\n" + "=" * 80)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    parser = argparse.ArgumentParser(
        description='Analyze numerical differences between golden and output files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze files in a test directory with 5% tolerance
  %(prog)s test_my_spmm/golden.txt test_my_spmm/output.txt -t 5

  # Analyze with 10% tolerance and 500-line regions
  %(prog)s golden.txt output.txt -t 10 -r 500

  # Use default names in a directory
  %(prog)s -d test_big_dense_large_Rv3 -t 10
        """
    )

    parser.add_argument('file1', nargs='?', default=None,
                        help='Golden/reference file (default: golden.txt)')
    parser.add_argument('file2', nargs='?', default=None,
                        help='Output file to compare (default: output.txt)')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory containing golden.txt and output.txt')
    parser.add_argument('-t', '--tolerance', type=float, default=5.0,
                        help='Acceptable relative error percentage (default: 5.0)')
    parser.add_argument('-r', '--region-size', type=int, default=1000,
                        help='Number of consecutive lines to consider a region (default: 1000)')

    args = parser.parse_args()

    # Determine file paths
    if args.directory:
        file1 = Path(args.directory) / 'golden.txt'
        file2 = Path(args.directory) / 'output.txt'
    elif args.file1 and args.file2:
        file1 = Path(args.file1)
        file2 = Path(args.file2)
    elif args.file1:
        parser.error("If not using -d/--directory, both file1 and file2 must be provided")
    else:
        file1 = Path('golden.txt')
        file2 = Path('output.txt')

    # Check files exist
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)

    analyze_numerical_differences(file1, file2, args.tolerance, args.region_size)


if __name__ == "__main__":
    main()
