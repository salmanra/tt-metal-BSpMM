#!/usr/bin/env python3
"""
Export good and bad regions to files for detailed analysis.
"""

import sys
import argparse
import numpy as np
from pathlib import Path


def export_regions(golden_file, output_file, output_dir=None, tolerance_percent=10.0, min_region_size=50):
    """Export good and bad regions to CSV files for analysis."""

    print(f"Finding regions within {tolerance_percent}% tolerance (min size: {min_region_size})")
    print("=" * 80)

    absolute_errors = []
    relative_errors = []
    golden_values = []
    output_values = []
    line_numbers = []

    # Read and analyze
    with open(golden_file, 'r') as f1, open(output_file, 'r') as f2:
        line_num = 0

        while True:
            line1 = f1.readline()
            line2 = f2.readline()

            if not line1 and not line2:
                break

            line_num += 1

            if line_num % 100000 == 0:
                print(f"Processed {line_num:,} lines...", end='\r', flush=True)

            try:
                val1 = float(line1.strip())
                val2 = float(line2.strip())

                golden_values.append(val1)
                output_values.append(val2)
                line_numbers.append(line_num)

                abs_error = abs(val2 - val1)
                absolute_errors.append(abs_error)

                if abs(val1) > 1e-10:
                    rel_error = abs((val2 - val1) / val1) * 100
                else:
                    rel_error = 0.0 if abs_error < 1e-10 else float('inf')

                relative_errors.append(rel_error)

            except (ValueError, AttributeError):
                continue

    print(" " * 80, end='\r')

    # Convert to numpy
    relative_errors = np.array(relative_errors)
    absolute_errors = np.array(absolute_errors)
    golden_values = np.array(golden_values)
    output_values = np.array(output_values)
    line_numbers = np.array(line_numbers)

    # Find good regions
    within_tolerance = (relative_errors <= tolerance_percent) | \
                      ((np.abs(golden_values) < 1e-10) & (absolute_errors < 1e-10))

    good_regions = []
    bad_regions = []

    in_good_region = False
    good_start = 0

    in_bad_region = False
    bad_start = 0

    for i, is_good in enumerate(within_tolerance):
        # Track good regions
        if is_good and not in_good_region:
            good_start = i
            in_good_region = True
        elif not is_good and in_good_region:
            if i - good_start >= min_region_size:
                good_regions.append((good_start, i - 1))
            in_good_region = False

        # Track bad regions
        if not is_good and not in_bad_region:
            bad_start = i
            in_bad_region = True
        elif is_good and in_bad_region:
            if i - bad_start >= min_region_size:
                bad_regions.append((bad_start, i - 1))
            in_bad_region = False

    # Handle regions that go to end of file
    if in_good_region and len(within_tolerance) - good_start >= min_region_size:
        good_regions.append((good_start, len(within_tolerance) - 1))
    if in_bad_region and len(within_tolerance) - bad_start >= min_region_size:
        bad_regions.append((bad_start, len(within_tolerance) - 1))

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Export good regions
    print(f"\nFound {len(good_regions)} good regions (>= {min_region_size} lines)")
    print(f"Found {len(bad_regions)} bad regions (>= {min_region_size} lines)")

    good_regions_file = output_dir / "good_regions.csv"
    with open(good_regions_file, 'w') as f:
        f.write("Region,StartLine,EndLine,Length,MaxRelError%,MeanRelError%,MaxAbsError,MeanAbsError\n")

        for idx, (start_idx, end_idx) in enumerate(good_regions):
            start_line = line_numbers[start_idx]
            end_line = line_numbers[end_idx]
            length = end_idx - start_idx + 1

            region_rel = relative_errors[start_idx:end_idx+1]
            region_abs = absolute_errors[start_idx:end_idx+1]

            region_rel_finite = region_rel[np.isfinite(region_rel)]

            max_rel = np.max(region_rel_finite) if len(region_rel_finite) > 0 else 0.0
            mean_rel = np.mean(region_rel_finite) if len(region_rel_finite) > 0 else 0.0
            max_abs = np.max(region_abs)
            mean_abs = np.mean(region_abs)

            f.write(f"{idx+1},{start_line},{end_line},{length},{max_rel:.4f},{mean_rel:.4f},{max_abs:.6e},{mean_abs:.6e}\n")

    print(f"✓ Good regions exported to: {good_regions_file}")

    # Export bad regions
    bad_regions_file = output_dir / "bad_regions.csv"
    with open(bad_regions_file, 'w') as f:
        f.write("Region,StartLine,EndLine,Length,MaxRelError%,MeanRelError%,MaxAbsError,MeanAbsError,SampleGolden,SampleOutput\n")

        for idx, (start_idx, end_idx) in enumerate(bad_regions[:100]):  # Limit to top 100
            start_line = line_numbers[start_idx]
            end_line = line_numbers[end_idx]
            length = end_idx - start_idx + 1

            region_rel = relative_errors[start_idx:end_idx+1]
            region_abs = absolute_errors[start_idx:end_idx+1]

            region_rel_finite = region_rel[np.isfinite(region_rel)]

            max_rel = np.max(region_rel_finite) if len(region_rel_finite) > 0 else 0.0
            mean_rel = np.mean(region_rel_finite) if len(region_rel_finite) > 0 else 0.0
            max_abs = np.max(region_abs)
            mean_abs = np.mean(region_abs)

            sample_golden = golden_values[start_idx]
            sample_output = output_values[start_idx]

            f.write(f"{idx+1},{start_line},{end_line},{length},{max_rel:.4f},{mean_rel:.4f},{max_abs:.6e},{mean_abs:.6e},{sample_golden:.6e},{sample_output:.6e}\n")

    print(f"✓ Bad regions (first 100) exported to: {bad_regions_file}")

    # Pattern analysis
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    # Check if good regions follow a pattern (e.g., every N lines)
    if len(good_regions) > 1:
        good_starts = [line_numbers[start] for start, _ in good_regions]
        gaps = np.diff(good_starts)

        print(f"\nGood region spacing:")
        print(f"  Average gap: {np.mean(gaps):.1f} lines")
        print(f"  Median gap: {np.median(gaps):.1f} lines")
        print(f"  Min gap: {np.min(gaps)} lines")
        print(f"  Max gap: {np.max(gaps)} lines")

        # Check for periodicity
        from collections import Counter
        gap_counts = Counter(gaps)
        most_common_gaps = gap_counts.most_common(5)

        if most_common_gaps:
            print(f"\nMost common gaps:")
            for gap, count in most_common_gaps:
                print(f"  {gap:,} lines: {count} occurrences")

    # Calculate coverage
    total_good_lines = sum(end - start + 1 for start, end in good_regions)
    total_bad_lines = sum(end - start + 1 for start, end in bad_regions)
    coverage_good = total_good_lines / len(line_numbers) * 100
    coverage_bad = total_bad_lines / len(line_numbers) * 100

    print(f"\nCoverage:")
    print(f"  Good regions: {total_good_lines:,} lines ({coverage_good:.2f}%)")
    print(f"  Bad regions: {total_bad_lines:,} lines ({coverage_bad:.2f}%)")
    print(f"  Scattered: {len(line_numbers) - total_good_lines - total_bad_lines:,} lines ({100 - coverage_good - coverage_bad:.2f}%)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Export good and bad regions to CSV files for analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export regions from files in a test directory
  %(prog)s test_my_spmm/golden.txt test_my_spmm/output.txt -o test_my_spmm

  # Use 10% tolerance and minimum 50-line regions
  %(prog)s golden.txt output.txt -t 10 -m 50

  # Use default names in a directory
  %(prog)s -d test_big_dense_large_Rv3 -o test_big_dense_large_Rv3
        """
    )

    parser.add_argument('file1', nargs='?', default=None,
                        help='Golden/reference file (default: golden.txt)')
    parser.add_argument('file2', nargs='?', default=None,
                        help='Output file to compare (default: output.txt)')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory containing golden.txt and output.txt')
    parser.add_argument('-o', '--output-dir', type=str,
                        help='Output directory for CSV files (default: current directory or -d directory)')
    parser.add_argument('-t', '--tolerance', type=float, default=10.0,
                        help='Acceptable relative error percentage (default: 10.0)')
    parser.add_argument('-m', '--min-size', type=int, default=50,
                        help='Minimum region size in lines (default: 50)')

    args = parser.parse_args()

    # Determine file paths
    if args.directory:
        file1 = Path(args.directory) / 'golden.txt'
        file2 = Path(args.directory) / 'output.txt'
        # Default output to same directory if not specified
        output_dir = args.output_dir if args.output_dir else args.directory
    elif args.file1 and args.file2:
        file1 = Path(args.file1)
        file2 = Path(args.file2)
        output_dir = args.output_dir
    elif args.file1:
        parser.error("If not using -d/--directory, both file1 and file2 must be provided")
    else:
        file1 = Path('golden.txt')
        file2 = Path('output.txt')
        output_dir = args.output_dir

    # Check files exist
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)

    export_regions(file1, file2, output_dir, args.tolerance, args.min_size)


if __name__ == "__main__":
    main()
