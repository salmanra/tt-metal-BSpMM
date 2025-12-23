# Analysis Tools for SPMM Tests

A suite of Python scripts for analyzing differences between golden (expected) and output (actual) results from SPMM tests. These tools help identify patterns in errors, find good/bad regions, and visualize tile-based accuracy.

## Overview

This directory contains reusable analysis scripts that can be run on any test output directory containing `golden.txt` and `output.txt` files.

### Available Tools

1. **compare_files.py** - Fast line-by-line comparison
2. **analyze_differences.py** - Statistical analysis of numerical errors
3. **find_good_regions.py** - Export good/bad regions to CSV files
4. **analyze_tiles.py** - Tile-based pattern analysis with visualizations
5. **run_all_analysis.py** - Master script to run all analyses

## Quick Start

### Run all analyses on a test directory:

```bash
./analysis_tools/run_all_analysis.py test_big_dense_large_Rv3
```

This will:
- Compare the files line-by-line
- Analyze numerical errors and statistics
- Export good/bad regions to CSV
- Perform tile-based analysis with visualizations
- Save all results to the test directory

### Run all analyses with custom parameters:

```bash
./analysis_tools/run_all_analysis.py test_my_spmm -t 5 -s 512
```

- `-t 5`: Use 5% tolerance (default: 10%)
- `-s 512`: Use 512-line tiles (default: 1024)

## Individual Tools

### 1. compare_files.py

Fast line-by-line comparison of two text files.

**Usage:**
```bash
# Compare files in a test directory
./analysis_tools/compare_files.py test_my_spmm/golden.txt test_my_spmm/output.txt

# Show more differences
./analysis_tools/compare_files.py golden.txt output.txt -n 50

# Use directory shorthand
./analysis_tools/compare_files.py -d test_big_dense_large_Rv3
```

**Output:**
- Total line counts
- Number of differences
- Match rate percentage
- First N differences with line numbers

### 2. analyze_differences.py

Statistical analysis of numerical differences between files.

**Usage:**
```bash
# Analyze with 5% tolerance
./analysis_tools/analyze_differences.py test_my_spmm/golden.txt test_my_spmm/output.txt -t 5

# Analyze with 10% tolerance and 500-line regions
./analysis_tools/analyze_differences.py golden.txt output.txt -t 10 -r 500

# Use directory shorthand
./analysis_tools/analyze_differences.py -d test_big_dense_large_Rv3 -t 10
```

**Output:**
- Absolute error statistics (min, max, mean, median, percentiles)
- Relative error statistics
- Regions within tolerance threshold
- Worst error regions
- Error distribution histogram

### 3. find_good_regions.py

Export good and bad regions to CSV files for detailed analysis.

**Usage:**
```bash
# Export regions with 10% tolerance
./analysis_tools/find_good_regions.py test_my_spmm/golden.txt test_my_spmm/output.txt -o test_my_spmm

# Custom tolerance and minimum region size
./analysis_tools/find_good_regions.py golden.txt output.txt -t 10 -m 50 -o results

# Use directory shorthand
./analysis_tools/find_good_regions.py -d test_big_dense_large_Rv3 -o test_big_dense_large_Rv3
```

**Output Files:**
- `good_regions.csv` - All regions meeting tolerance threshold
- `bad_regions.csv` - First 100 regions exceeding tolerance

**CSV Columns:**
- Region number
- Start/End line numbers
- Region length
- Max/Mean relative error
- Max/Mean absolute error
- Sample values (bad regions only)

**Console Output:**
- Region spacing patterns
- Most common gaps between good regions
- Coverage percentages

### 4. analyze_tiles.py

Analyze files in fixed-size tiles (default 1024 lines) to identify patterns.

**Usage:**
```bash
# Analyze with default settings (1024-line tiles, 10% tolerance)
./analysis_tools/analyze_tiles.py test_my_spmm/golden.txt test_my_spmm/output.txt -o test_my_spmm

# Custom tile size and tolerance
./analysis_tools/analyze_tiles.py golden.txt output.txt -s 512 -t 5 -o results

# Use directory shorthand
./analysis_tools/analyze_tiles.py -d test_big_dense_large_Rv3 -o test_big_dense_large_Rv3
```

**Output Files:**
- `tile_analysis.csv` - Per-tile statistics
- `tile_error_plot.png` - Tile quality and error plots over tile number
- `tile_quality_heatmap.png` - 2D heatmap visualization of tile quality

**Console Output:**
- Good vs bad tile counts
- Tile quality distribution (excellent/good/fair/poor/bad/terrible)
- Sign flip detection
- Clustering analysis
- Modulo pattern detection (periodic patterns)
- Top 10 best/worst tiles

### 5. run_all_analysis.py

Master script to run all analyses in sequence.

**Usage:**
```bash
# Run all analyses
./analysis_tools/run_all_analysis.py test_big_dense_large_Rv3

# Run with custom parameters
./analysis_tools/run_all_analysis.py test_my_spmm -t 5 -s 512

# Skip specific analyses
./analysis_tools/run_all_analysis.py test_my_spmm --skip-compare --skip-tiles

# Use specific files
./analysis_tools/run_all_analysis.py -g path/to/golden.txt -o path/to/output.txt --output-dir results
```

**Options:**
- `-t, --tolerance`: Tolerance percentage (default: 10.0)
- `-s, --tile-size`: Tile size in lines (default: 1024)
- `-r, --region-size`: Minimum region size (default: 1000)
- `--skip-compare`: Skip file comparison
- `--skip-differences`: Skip difference analysis
- `--skip-regions`: Skip region export
- `--skip-tiles`: Skip tile analysis

## Understanding the Output

### Error Metrics

**Absolute Error:**
```
abs_error = |output_value - golden_value|
```

**Relative Error (%):**
```
rel_error = |output_value - golden_value| / |golden_value| × 100
```

For values near zero (|golden_value| < 1e-10), relative error is considered 0 if absolute error is also near zero, otherwise infinity.

### Tolerance Thresholds

The default tolerance is **10%** relative error. A value is considered "within tolerance" if:
- Relative error ≤ tolerance%, OR
- Both golden and output values are near zero (< 1e-10)

### Tile Classification

Tiles are classified based on the percentage of values within tolerance:
- **Excellent**: >90% within tolerance
- **Good**: 70-90% within tolerance
- **Fair**: 50-70% within tolerance
- **Poor**: 30-50% within tolerance
- **Bad**: 10-30% within tolerance
- **Terrible**: <10% within tolerance

A tile is marked as "good" if ≥50% of values are within tolerance.

## Example Workflow

```bash
# 1. Run all analyses on a new test
./analysis_tools/run_all_analysis.py test_new_feature

# 2. If you need more detail on a specific aspect, run individual tools:

# Check specific error statistics
./analysis_tools/analyze_differences.py -d test_new_feature -t 5

# Export regions for manual inspection
./analysis_tools/find_good_regions.py -d test_new_feature -t 5 -m 100 -o test_new_feature

# Try different tile sizes to find patterns
./analysis_tools/analyze_tiles.py -d test_new_feature -s 512 -o test_new_feature
./analysis_tools/analyze_tiles.py -d test_new_feature -s 2048 -o test_new_feature
```

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (for visualizations)

Install dependencies:
```bash
pip install numpy matplotlib
```

## Tips

1. **Start with run_all_analysis.py** to get a comprehensive overview
2. **Adjust tolerance** based on your expected precision (use `-t` flag)
3. **Try different tile sizes** to reveal patterns at different scales
4. **Check the visualizations** in the PNG files for quick insights
5. **Examine good_regions.csv** to understand what's working correctly
6. **Review bad_regions.csv** to focus debugging efforts

## File Format

Input files (`golden.txt` and `output.txt`) should contain one numerical value per line:
```
1.234567
-0.891234
42.000000
...
```

Non-numeric lines are automatically skipped during analysis.

## Integration with Tests

After running your SPMM test, simply point the analysis tools at the output directory:

```bash
# Run your test
./build/programming_examples/rahmy/block_spmm/test_my_kernel

# Analyze results
./analysis_tools/run_all_analysis.py test_my_kernel

# Review the generated CSVs and visualizations
ls test_my_kernel/*.csv test_my_kernel/*.png
```

## Common Issues

**Issue:** Scripts fail with "File not found"
**Solution:** Ensure `golden.txt` and `output.txt` exist in the specified directory

**Issue:** "ModuleNotFoundError: No module named 'numpy'"
**Solution:** Install required packages: `pip install numpy matplotlib`

**Issue:** Visualizations not generated
**Solution:** Check that matplotlib is installed and the output directory is writable

## Contributing

When adding new test directories, run the analysis tools to document the behavior. Include the generated CSVs and visualizations in your analysis.
