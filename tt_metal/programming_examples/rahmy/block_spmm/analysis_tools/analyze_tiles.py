#!/usr/bin/env python3
"""
Analyze files in tiles of N consecutive lines to find patterns.
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_tiles(golden_file, output_file, output_dir=None, tile_size=1024, tolerance_percent=10.0):
    """Analyze files in fixed-size tiles."""

    print(f"Analyzing files in tiles of {tile_size} lines")
    print(f"Tolerance: {tolerance_percent}%")
    print("=" * 80)

    tiles = []
    current_tile = {
        'tile_num': 0,
        'start_line': 1,
        'golden_vals': [],
        'output_vals': [],
        'abs_errors': [],
        'rel_errors': [],
    }

    line_num = 0

    print("\nReading files and computing tile statistics...")

    with open(golden_file, 'r') as f1, open(output_file, 'r') as f2:
        while True:
            line1 = f1.readline()
            line2 = f2.readline()

            if not line1 and not line2:
                # Save last partial tile if it exists
                if current_tile['golden_vals']:
                    tiles.append(finalize_tile(current_tile, tolerance_percent))
                break

            line_num += 1

            if line_num % 100000 == 0:
                print(f"Processed {line_num:,} lines ({len(tiles)} tiles completed)...", end='\r', flush=True)

            try:
                val1 = float(line1.strip())
                val2 = float(line2.strip())

                current_tile['golden_vals'].append(val1)
                current_tile['output_vals'].append(val2)

                abs_error = abs(val2 - val1)
                current_tile['abs_errors'].append(abs_error)

                if abs(val1) > 1e-10:
                    rel_error = abs((val2 - val1) / val1) * 100
                else:
                    rel_error = 0.0 if abs_error < 1e-10 else float('inf')

                current_tile['rel_errors'].append(rel_error)

                # Check if tile is complete
                if len(current_tile['golden_vals']) == tile_size:
                    tiles.append(finalize_tile(current_tile, tolerance_percent))
                    current_tile = {
                        'tile_num': len(tiles),
                        'start_line': line_num + 1,
                        'golden_vals': [],
                        'output_vals': [],
                        'abs_errors': [],
                        'rel_errors': [],
                    }

            except (ValueError, AttributeError):
                continue

    print(" " * 80, end='\r')

    # Convert to structured data
    tile_data = {
        'tile_num': [t['tile_num'] for t in tiles],
        'start_line': [t['start_line'] for t in tiles],
        'end_line': [t['end_line'] for t in tiles],
        'size': [t['size'] for t in tiles],
        'mean_rel_error': [t['mean_rel_error'] for t in tiles],
        'median_rel_error': [t['median_rel_error'] for t in tiles],
        'max_rel_error': [t['max_rel_error'] for t in tiles],
        'mean_abs_error': [t['mean_abs_error'] for t in tiles],
        'within_tolerance_pct': [t['within_tolerance_pct'] for t in tiles],
        'is_good': [t['is_good'] for t in tiles],
        'sign_flips': [t['sign_flips'] for t in tiles],
    }

    print(f"\nTotal tiles: {len(tiles)}")
    print(f"Complete tiles ({tile_size} lines): {sum(1 for t in tiles if t['size'] == tile_size)}")
    print(f"Partial tile: {sum(1 for t in tiles if t['size'] < tile_size)}")

    # Analysis
    print("\n" + "=" * 80)
    print("TILE STATISTICS")
    print("=" * 80)

    good_tiles = [t for t in tiles if t['is_good']]
    bad_tiles = [t for t in tiles if not t['is_good']]

    print(f"\nGood tiles (≥50% within tolerance): {len(good_tiles)} ({len(good_tiles)/len(tiles)*100:.2f}%)")
    print(f"Bad tiles (<50% within tolerance): {len(bad_tiles)} ({len(bad_tiles)/len(tiles)*100:.2f}%)")

    # Tile quality distribution
    within_tol_values = tile_data['within_tolerance_pct']
    print(f"\nTile Quality Distribution:")
    print(f"  Excellent (>90% within tol): {sum(1 for x in within_tol_values if x > 90)} tiles")
    print(f"  Good (70-90% within tol):    {sum(1 for x in within_tol_values if 70 <= x <= 90)} tiles")
    print(f"  Fair (50-70% within tol):    {sum(1 for x in within_tol_values if 50 <= x < 70)} tiles")
    print(f"  Poor (30-50% within tol):    {sum(1 for x in within_tol_values if 30 <= x < 50)} tiles")
    print(f"  Bad (10-30% within tol):     {sum(1 for x in within_tol_values if 10 <= x < 30)} tiles")
    print(f"  Terrible (<10% within tol):  {sum(1 for x in within_tol_values if x < 10)} tiles")

    # Sign flip analysis
    tiles_with_sign_flips = [t for t in tiles if t['sign_flips'] > 0]
    print(f"\nTiles with sign flips: {len(tiles_with_sign_flips)} ({len(tiles_with_sign_flips)/len(tiles)*100:.2f}%)")
    if tiles_with_sign_flips:
        avg_flips = np.mean([t['sign_flips'] for t in tiles_with_sign_flips])
        max_flips = max(t['sign_flips'] for t in tiles_with_sign_flips)
        print(f"  Average sign flips per affected tile: {avg_flips:.1f}")
        print(f"  Max sign flips in a tile: {max_flips}")

    # Pattern detection
    print("\n" + "=" * 80)
    print("PATTERN DETECTION")
    print("=" * 80)

    # Check for periodic patterns
    good_tile_nums = [t['tile_num'] for t in good_tiles]
    bad_tile_nums = [t['tile_num'] for t in bad_tiles]

    if len(good_tiles) > 1:
        good_gaps = np.diff(good_tile_nums)
        print(f"\nGood tile spacing:")
        print(f"  Average gap: {np.mean(good_gaps):.2f} tiles")
        print(f"  Median gap: {np.median(good_gaps):.0f} tiles")
        print(f"  Min gap: {np.min(good_gaps)} tiles")
        print(f"  Max gap: {np.max(good_gaps)} tiles")

    # Check if good/bad tiles appear in clusters
    print("\nClustering analysis:")
    good_clusters = find_clusters([t['tile_num'] for t in good_tiles])
    bad_clusters = find_clusters([t['tile_num'] for t in bad_tiles])

    print(f"  Good tile clusters: {len(good_clusters)}")
    if good_clusters:
        avg_cluster_size = np.mean([c[1] - c[0] + 1 for c in good_clusters])
        max_cluster_size = max(c[1] - c[0] + 1 for c in good_clusters)
        print(f"    Average cluster size: {avg_cluster_size:.1f} tiles")
        print(f"    Largest cluster: {max_cluster_size} tiles (tiles {good_clusters[np.argmax([c[1]-c[0] for c in good_clusters])][0]}-{good_clusters[np.argmax([c[1]-c[0] for c in good_clusters])][1]})")

    print(f"  Bad tile clusters: {len(bad_clusters)}")
    if bad_clusters:
        avg_cluster_size = np.mean([c[1] - c[0] + 1 for c in bad_clusters])
        max_cluster_size = max(c[1] - c[0] + 1 for c in bad_clusters)
        print(f"    Average cluster size: {avg_cluster_size:.1f} tiles")
        print(f"    Largest cluster: {max_cluster_size} tiles")

    # Check for modulo patterns
    print("\nModulo pattern analysis (checking if good/bad tiles follow periodic pattern):")
    for modulo in [2, 4, 8, 16, 32, 64]:
        good_mod_dist = {}
        bad_mod_dist = {}

        for t in good_tiles:
            mod_val = t['tile_num'] % modulo
            good_mod_dist[mod_val] = good_mod_dist.get(mod_val, 0) + 1

        for t in bad_tiles:
            mod_val = t['tile_num'] % modulo
            bad_mod_dist[mod_val] = bad_mod_dist.get(mod_val, 0) + 1

        # Check if there's a strong pattern (one mod value dominates)
        if good_mod_dist:
            max_good_count = max(good_mod_dist.values())
            if max_good_count > len(good_tiles) * 0.7:  # 70% threshold
                dominant_mod = [k for k, v in good_mod_dist.items() if v == max_good_count][0]
                print(f"  Mod {modulo}: {max_good_count}/{len(good_tiles)} ({max_good_count/len(good_tiles)*100:.1f}%) good tiles at position {dominant_mod}")

    # Top 10 worst tiles
    print("\n" + "=" * 80)
    print("TOP 10 WORST TILES")
    print("=" * 80)

    sorted_tiles = sorted(tiles, key=lambda t: t['mean_rel_error'], reverse=True)
    print("\nTile # | Lines           | Within Tol | Mean Err  | Max Err   | Sign Flips")
    print("-" * 80)
    for t in sorted_tiles[:10]:
        print(f"{t['tile_num']:6d} | {t['start_line']:7,}-{t['end_line']:7,} | {t['within_tolerance_pct']:6.2f}%   | {t['mean_rel_error']:8.2f}% | {t['max_rel_error']:8.2f}% | {t['sign_flips']:4d}")

    # Top 10 best tiles
    print("\n" + "=" * 80)
    print("TOP 10 BEST TILES")
    print("=" * 80)

    sorted_tiles_best = sorted([t for t in tiles if np.isfinite(t['mean_rel_error'])],
                               key=lambda t: t['mean_rel_error'])
    print("\nTile # | Lines           | Within Tol | Mean Err  | Max Err   | Sign Flips")
    print("-" * 80)
    for t in sorted_tiles_best[:10]:
        print(f"{t['tile_num']:6d} | {t['start_line']:7,}-{t['end_line']:7,} | {t['within_tolerance_pct']:6.2f}%   | {t['mean_rel_error']:8.2f}% | {t['max_rel_error']:8.2f}% | {t['sign_flips']:4d}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Export CSV
    print("\n" + "=" * 80)
    print("EXPORTING DATA")
    print("=" * 80)

    csv_file = output_dir / 'tile_analysis.csv'
    with open(csv_file, 'w') as f:
        f.write("TileNum,StartLine,EndLine,Size,WithinTolPct,MeanRelErr%,MedianRelErr%,MaxRelErr%,MeanAbsErr,SignFlips,IsGood\n")
        for t in tiles:
            f.write(f"{t['tile_num']},{t['start_line']},{t['end_line']},{t['size']},{t['within_tolerance_pct']:.2f},")
            f.write(f"{t['mean_rel_error']:.4f},{t['median_rel_error']:.4f},{t['max_rel_error']:.4f},")
            f.write(f"{t['mean_abs_error']:.6e},{t['sign_flips']},{1 if t['is_good'] else 0}\n")

    print(f"✓ Tile analysis exported to: {csv_file}")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(tiles, tile_size, output_dir)
    print(f"✓ Visualizations saved to: {output_dir / 'tile_quality_heatmap.png'}, {output_dir / 'tile_error_plot.png'}")

    print("\n" + "=" * 80)


def finalize_tile(tile_dict, tolerance_percent):
    """Compute statistics for a tile."""
    golden = np.array(tile_dict['golden_vals'])
    output = np.array(tile_dict['output_vals'])
    abs_errors = np.array(tile_dict['abs_errors'])
    rel_errors = np.array(tile_dict['rel_errors'])

    # Filter finite relative errors for stats
    rel_errors_finite = rel_errors[np.isfinite(rel_errors)]

    within_tolerance = (rel_errors <= tolerance_percent) | \
                      ((np.abs(golden) < 1e-10) & (abs_errors < 1e-10))

    # Detect sign flips
    sign_flips = np.sum((golden > 0) & (output < 0)) + np.sum((golden < 0) & (output > 0))

    tile = {
        'tile_num': tile_dict['tile_num'],
        'start_line': tile_dict['start_line'],
        'end_line': tile_dict['start_line'] + len(golden) - 1,
        'size': len(golden),
        'mean_rel_error': np.mean(rel_errors_finite) if len(rel_errors_finite) > 0 else float('inf'),
        'median_rel_error': np.median(rel_errors_finite) if len(rel_errors_finite) > 0 else float('inf'),
        'max_rel_error': np.max(rel_errors_finite) if len(rel_errors_finite) > 0 else float('inf'),
        'mean_abs_error': np.mean(abs_errors),
        'within_tolerance_pct': np.sum(within_tolerance) / len(within_tolerance) * 100,
        'is_good': np.sum(within_tolerance) / len(within_tolerance) >= 0.5,  # 50% threshold
        'sign_flips': int(sign_flips),
    }

    return tile


def find_clusters(tile_nums):
    """Find consecutive clusters of tile numbers."""
    if not tile_nums:
        return []

    tile_nums = sorted(tile_nums)
    clusters = []
    cluster_start = tile_nums[0]
    prev = tile_nums[0]

    for num in tile_nums[1:]:
        if num != prev + 1:
            # End of cluster
            clusters.append((cluster_start, prev))
            cluster_start = num
        prev = num

    # Add last cluster
    clusters.append((cluster_start, prev))

    return clusters


def create_visualizations(tiles, tile_size, output_dir):
    """Create visualization plots."""

    # Extract data
    tile_nums = [t['tile_num'] for t in tiles]
    within_tol_pct = [t['within_tolerance_pct'] for t in tiles]
    mean_errors = [t['mean_rel_error'] if np.isfinite(t['mean_rel_error']) else 100 for t in tiles]

    # Figure 1: Tile quality over tile number
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Within tolerance percentage
    ax1.plot(tile_nums, within_tol_pct, linewidth=0.5, alpha=0.7)
    ax1.axhline(y=50, color='r', linestyle='--', label='50% threshold', linewidth=1)
    ax1.set_xlabel('Tile Number')
    ax1.set_ylabel('% Within Tolerance')
    ax1.set_title(f'Tile Quality: % of Values Within Tolerance (Tile Size: {tile_size} lines)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Plot 2: Mean relative error (capped for visualization)
    capped_errors = [min(e, 1000) for e in mean_errors]  # Cap at 1000% for viz
    ax2.plot(tile_nums, capped_errors, linewidth=0.5, alpha=0.7, color='red')
    ax2.axhline(y=10, color='g', linestyle='--', label='10% error', linewidth=1)
    ax2.set_xlabel('Tile Number')
    ax2.set_ylabel('Mean Relative Error % (capped at 1000%)')
    ax2.set_title('Mean Relative Error per Tile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'tile_error_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Heatmap (if we can arrange tiles in a grid)
    # Try to arrange as close to square as possible
    n_tiles = len(tiles)
    grid_size = int(np.ceil(np.sqrt(n_tiles)))
    grid = np.full((grid_size, grid_size), np.nan)

    for i, pct in enumerate(within_tol_pct):
        row = i // grid_size
        col = i % grid_size
        if row < grid_size:
            grid[row, col] = pct

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    ax.set_xlabel('Tile Column')
    ax.set_ylabel('Tile Row')
    ax.set_title(f'Tile Quality Heatmap ({grid_size}x{grid_size} grid, {tile_size} lines/tile)\nGreen=Good, Red=Bad')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% Within Tolerance', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'tile_quality_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze files in fixed-size tiles to identify patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with 1024-line tiles and 10% tolerance
  %(prog)s test_my_spmm/golden.txt test_my_spmm/output.txt -o test_my_spmm

  # Use custom tile size and tolerance
  %(prog)s golden.txt output.txt -s 512 -t 5 -o results

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
                        help='Output directory for results (default: current directory or -d directory)')
    parser.add_argument('-s', '--tile-size', type=int, default=1024,
                        help='Number of lines per tile (default: 1024)')
    parser.add_argument('-t', '--tolerance', type=float, default=10.0,
                        help='Acceptable relative error percentage (default: 10.0)')

    args = parser.parse_args()

    # Determine file paths
    if args.directory:
        file1 = Path(args.directory) / 'golden.txt'
        file2 = Path(args.directory) / 'output.txt'
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

    analyze_tiles(file1, file2, output_dir, args.tile_size, args.tolerance)


if __name__ == "__main__":
    main()
