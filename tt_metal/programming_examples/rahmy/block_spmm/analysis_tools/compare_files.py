#!/usr/bin/env python3
"""
Fast line-by-line comparison of two text files.
Optimized for large files.
"""

import sys
import argparse
from pathlib import Path


def compare_files(file1_path, file2_path, show_first_n_diffs=20, context_lines=2):
    """
    Compare two files line by line.

    Args:
        file1_path: Path to first file
        file2_path: Path to second file
        show_first_n_diffs: Number of differences to display in detail
        context_lines: Number of context lines to show around differences
    """

    print(f"Comparing files:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    print("-" * 80)

    diff_count = 0
    line_num = 0
    total_lines_file1 = 0
    total_lines_file2 = 0
    differences = []

    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            # Read both files line by line simultaneously
            while True:
                line1 = f1.readline()
                line2 = f2.readline()

                # Check if we've reached end of either file
                if not line1 and not line2:
                    # Both files ended at the same time
                    break

                line_num += 1

                if line1:
                    total_lines_file1 += 1
                if line2:
                    total_lines_file2 += 1

                # Check if lines differ
                if line1 != line2:
                    diff_count += 1

                    # Store first N differences for detailed display
                    if diff_count <= show_first_n_diffs:
                        differences.append({
                            'line_num': line_num,
                            'line1': line1.rstrip('\n') if line1 else '<EOF>',
                            'line2': line2.rstrip('\n') if line2 else '<EOF>',
                        })

                # Progress indicator for large files (every 100k lines)
                if line_num % 100000 == 0:
                    print(f"Processed {line_num:,} lines, found {diff_count:,} differences...",
                          end='\r', flush=True)

        # Clear progress line
        if line_num >= 100000:
            print(" " * 80, end='\r')

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total lines in file 1: {total_lines_file1:,}")
        print(f"Total lines in file 2: {total_lines_file2:,}")
        print(f"Total differences found: {diff_count:,}")

        if total_lines_file1 == total_lines_file2:
            print(f"Line count: SAME ({total_lines_file1:,} lines)")
        else:
            print(f"Line count: DIFFERENT (delta: {abs(total_lines_file1 - total_lines_file2):,})")

        if diff_count == 0:
            print("\n✓ FILES ARE IDENTICAL")
            return True
        else:
            match_rate = ((max(total_lines_file1, total_lines_file2) - diff_count) /
                         max(total_lines_file1, total_lines_file2) * 100)
            print(f"Match rate: {match_rate:.2f}%")

            # Display first N differences
            print("\n" + "=" * 80)
            print(f"FIRST {min(diff_count, show_first_n_diffs)} DIFFERENCES")
            print("=" * 80)

            for diff in differences:
                print(f"\nLine {diff['line_num']}:")
                print(f"  File 1: {diff['line1'][:120]}")  # Truncate long lines
                print(f"  File 2: {diff['line2'][:120]}")

                # Show character-level difference for short lines
                if len(diff['line1']) < 200 and len(diff['line2']) < 200:
                    if diff['line1'] != '<EOF>' and diff['line2'] != '<EOF>':
                        # Find first difference position
                        for i, (c1, c2) in enumerate(zip(diff['line1'], diff['line2'])):
                            if c1 != c2:
                                print(f"  First diff at position {i}: '{c1}' vs '{c2}'")
                                break

            if diff_count > show_first_n_diffs:
                print(f"\n... and {diff_count - show_first_n_diffs:,} more differences")

            print("\n✗ FILES ARE DIFFERENT")
            return False

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fast line-by-line comparison of two text files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare files in a test directory
  %(prog)s test_my_spmm/golden.txt test_my_spmm/output.txt

  # Compare with custom diff display count
  %(prog)s golden.txt output.txt -n 50

  # Use default names in a directory
  %(prog)s -d test_big_dense_large_Rv3
        """
    )

    parser.add_argument('file1', nargs='?', default=None,
                        help='First file to compare (default: golden.txt)')
    parser.add_argument('file2', nargs='?', default=None,
                        help='Second file to compare (default: output.txt)')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory containing golden.txt and output.txt')
    parser.add_argument('-n', '--num-diffs', type=int, default=20,
                        help='Number of differences to display in detail (default: 20)')
    parser.add_argument('-c', '--context', type=int, default=2,
                        help='Number of context lines to show (default: 2)')

    args = parser.parse_args()

    # Determine file paths
    if args.directory:
        file1 = Path(args.directory) / 'golden.txt'
        file2 = Path(args.directory) / 'output.txt'
    elif args.file1 and args.file2:
        file1 = Path(args.file1)
        file2 = Path(args.file2)
    elif args.file1:
        # Only one file provided - error
        parser.error("If not using -d/--directory, both file1 and file2 must be provided")
    else:
        # No arguments - use defaults in current directory
        file1 = Path('golden.txt')
        file2 = Path('output.txt')

    # Check files exist
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)

    result = compare_files(file1, file2, show_first_n_diffs=args.num_diffs,
                          context_lines=args.context)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
