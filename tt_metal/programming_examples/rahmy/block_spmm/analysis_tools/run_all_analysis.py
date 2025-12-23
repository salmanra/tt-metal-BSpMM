#!/usr/bin/env python3
"""
Master script to run all analysis tools on a test directory.
Runs compare_files, analyze_differences, find_good_regions, and analyze_tiles.
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n⚠ Warning: {description} exited with code {result.returncode}")
        return False
    else:
        print(f"\n✓ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run all analysis tools on a test directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the following analyses:
  1. compare_files.py - Line-by-line comparison
  2. analyze_differences.py - Numerical error analysis
  3. find_good_regions.py - Export good/bad regions to CSV
  4. analyze_tiles.py - Tile-based pattern analysis with visualizations

Examples:
  # Run all analyses on a test directory
  %(prog)s test_big_dense_large_Rv3

  # Run with custom tolerance and tile size
  %(prog)s test_my_spmm -t 5 -s 512

  # Run on specific files
  %(prog)s -g path/to/golden.txt -o path/to/output.txt --output-dir results
        """
    )

    parser.add_argument('directory', nargs='?', default=None,
                        help='Directory containing golden.txt and output.txt')
    parser.add_argument('-g', '--golden', type=str,
                        help='Path to golden file (overrides directory/golden.txt)')
    parser.add_argument('-o', '--output', type=str,
                        help='Path to output file (overrides directory/output.txt)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory for analysis results (default: same as input directory)')
    parser.add_argument('-t', '--tolerance', type=float, default=10.0,
                        help='Tolerance percentage for error analysis (default: 10.0)')
    parser.add_argument('-s', '--tile-size', type=int, default=1024,
                        help='Tile size for tile analysis (default: 1024)')
    parser.add_argument('-r', '--region-size', type=int, default=1000,
                        help='Minimum region size for region analysis (default: 1000)')
    parser.add_argument('--skip-compare', action='store_true',
                        help='Skip file comparison step')
    parser.add_argument('--skip-differences', action='store_true',
                        help='Skip difference analysis step')
    parser.add_argument('--skip-regions', action='store_true',
                        help='Skip region export step')
    parser.add_argument('--skip-tiles', action='store_true',
                        help='Skip tile analysis step')

    args = parser.parse_args()

    # Determine file paths
    if args.directory:
        golden_file = Path(args.directory) / 'golden.txt'
        output_file = Path(args.directory) / 'output.txt'
        output_dir = args.output_dir if args.output_dir else args.directory
    elif args.golden and args.output:
        golden_file = Path(args.golden)
        output_file = Path(args.output)
        output_dir = args.output_dir if args.output_dir else Path.cwd()
    else:
        parser.error("Either provide a directory or both --golden and --output")

    # Check files exist
    if not golden_file.exists():
        print(f"Error: Golden file not found: {golden_file}")
        sys.exit(1)
    if not output_file.exists():
        print(f"Error: Output file not found: {output_file}")
        sys.exit(1)

    # Get script directory
    script_dir = Path(__file__).parent

    print("=" * 80)
    print("SPMM TEST ANALYSIS SUITE")
    print("=" * 80)
    print(f"Golden file: {golden_file}")
    print(f"Output file: {output_file}")
    print(f"Output directory: {output_dir}")
    print(f"Tolerance: {args.tolerance}%")
    print(f"Tile size: {args.tile_size} lines")
    print(f"Region size: {args.region_size} lines")
    print("=" * 80)

    results = {}

    # 1. Compare files
    if not args.skip_compare:
        cmd = [
            sys.executable,
            script_dir / 'compare_files.py',
            str(golden_file),
            str(output_file),
            '-n', '20'
        ]
        results['compare'] = run_command(cmd, "File Comparison")
    else:
        print("\n⊘ Skipping file comparison")

    # 2. Analyze differences
    if not args.skip_differences:
        cmd = [
            sys.executable,
            script_dir / 'analyze_differences.py',
            str(golden_file),
            str(output_file),
            '-t', str(args.tolerance),
            '-r', str(args.region_size)
        ]
        results['differences'] = run_command(cmd, "Numerical Difference Analysis")
    else:
        print("\n⊘ Skipping difference analysis")

    # 3. Find and export regions
    if not args.skip_regions:
        cmd = [
            sys.executable,
            script_dir / 'find_good_regions.py',
            str(golden_file),
            str(output_file),
            '-t', str(args.tolerance),
            '-m', '50',
            '-o', str(output_dir)
        ]
        results['regions'] = run_command(cmd, "Region Export")
    else:
        print("\n⊘ Skipping region export")

    # 4. Tile analysis
    if not args.skip_tiles:
        cmd = [
            sys.executable,
            script_dir / 'analyze_tiles.py',
            str(golden_file),
            str(output_file),
            '-s', str(args.tile_size),
            '-t', str(args.tolerance),
            '-o', str(output_dir)
        ]
        results['tiles'] = run_command(cmd, "Tile Analysis")
    else:
        print("\n⊘ Skipping tile analysis")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for v in results.values() if v)

    print(f"\nTotal steps run: {total}")
    print(f"Successful: {successful}")
    print(f"Failed/Warned: {total - successful}")

    if not args.skip_regions or not args.skip_tiles:
        print(f"\nOutput files saved to: {output_dir}")
        if not args.skip_regions:
            print(f"  - {Path(output_dir) / 'good_regions.csv'}")
            print(f"  - {Path(output_dir) / 'bad_regions.csv'}")
        if not args.skip_tiles:
            print(f"  - {Path(output_dir) / 'tile_analysis.csv'}")
            print(f"  - {Path(output_dir) / 'tile_error_plot.png'}")
            print(f"  - {Path(output_dir) / 'tile_quality_heatmap.png'}")

    print("\n" + "=" * 80)

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
