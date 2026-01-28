import os
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import numpy as np
import json


profiles_dir = "/home/user/tt-metal/profiles/"
csv_dir = profiles_dir + "csvs/"
json_output_dir = profiles_dir + "jsons/"
png_output_dir = profiles_dir + "pngs/"

csv_dir_v5 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_load_balanced"
csv_dir_v4 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse_iteration"
csv_dir_v3 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse_many_blocks_per_core"
csv_dir_v2 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse"
csv_dir_v1 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse_naive"

csv_data_dirs = [csv_dir_v5, csv_dir_v4, csv_dir_v3, csv_dir_v2, csv_dir_v1]

test_cases = [        
        # "profile_case_sparse_single_block_R32_C32.csv",
        # "profile_case_sparse_single_block_R64_C64.csv",
        # "profile_case_sparse_single_block_R128_C128.csv",
        f"profile_case_sparse_diagonal_R32_C32.csv\n(NNZ Blocks = 32)", 
        f"profile_case_sparse_diagonal_R64_C64.csv\n(NNZ Blocks = 16)", 
        f"profile_case_sparse_diagonal_R128_C128.csv\n(NNZ Blocks = 8)",
        f"profile_case_sparse_fill_column_R32_C32.csv\n(NNZ Blocks = 32)", 
        f"profile_case_sparse_fill_column_R64_C64.csv\n(NNZ Blocks = 16)", 
        f"profile_case_sparse_fill_column_R128_C128.csv\n(NNZ Blocks = 8)",
        f"profile_case_sparse_fill_row_R32_C32.csv\n(NNZ Blocks = 32)", 
        f"profile_case_sparse_fill_row_R64_C64.csv\n(NNZ Blocks = 16)", 
        f"profile_case_sparse_fill_row_R128_C128.csv\n(NNZ Blocks = 8)",
        f"profile_case_sparse_fill_random_R32_C32.csv\n(NNZ Blocks = 32)",
        f"profile_case_sparse_fill_random_R64_C64.csv\n(NNZ Blocks = 16)", 
    ]

test_case_files = [name.split(".csv")[0] + ".csv" for name in test_cases]

test_cases_short = [name.replace("profile_case_sparse_", "") for name in test_cases]
test_cases_short = [name.replace(".csv", "") for name in test_cases_short]
test_cases_short = [name.replace("fill_", "") for name in test_cases_short]

# for each host program
#   make dict of {csv file name, dict} pairs (empty)
#   for each csv file
#       read the csv file
#       make dict of {Zone-name,total time} pairs (empty)
#       add {Program Loop, time} to dict
#       add {,} some other pairs if you feel it now
#       add Zone dict to csv file name dict

v5_data = {}
v4_data = {}
v3_data = {}
v2_data = {}
v1_data = {}

data_dicts = [v5_data, v4_data, v3_data, v2_data, v1_data]

for i, csv_data_dir in enumerate(csv_data_dirs):
    # csv_file_names = sorted(os.listdir(csv_data_dir))
    # csv_file_names = os.listdir(csv_data_dir)
    csv_files = [os.path.join(csv_data_dir, f) for f in test_case_files]
    for j, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # print(f'We are in the {j}th csv file of the {i}th host')
        # print(df[df["name"] == "Program Loop"].size) # what do you mean not all of these dfs have a Program Loop?
        # print(df.shape)
        
        zones_data = {}
        if df[df["name"] == "Program Loop"].size == 0:
            zones_data["Program Loop total ns"] = np.nan
        else:
            zones_data["Program Loop total ns"] = int(df.loc[df["name"] == "Program Loop", "total_ns"].array[0])

        # print(type(df[df["name"] == "Program Loop"]))
        # print(type(df[df["name"] == "Program Loop"]["total_ns"]))
        # total_ns = df.get("total_ns")["Program Loop"]
        # zones_data["Program Loop total ns"] = total_ns

        data_dicts[i][test_cases_short[j]] = zones_data

# pprint.pp(data_dicts)
with open(json_output_dir + "data_dicts.json", "w") as f:
    json.dump(data_dicts, f, indent=4)
# Now we can make a simple bar chart? And we could write this to a more easily readable JSON file.
#

# Extract keys (csv file names) and values ("Program Loop total ns") for each dict
group_labels = list(data_dicts[0].keys())
n_groups = len(group_labels)
n_dicts = len(data_dicts)

# Prepare data for plotting
bar_values = []
for d in data_dicts:
    bar_values.append([d[k]["Program Loop total ns"] for k in group_labels])

bar_values = np.array(bar_values)  # shape: (n_dicts, n_groups)
max_val = np.nanmax(bar_values) * 1.05

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
bar_width = 0.18
x = np.arange(n_groups)

group_colors = ["red", "orange", "steelblue", "mediumblue", "midnightblue"]

for i in range(n_dicts):
    mask_ok = ~np.isnan(bar_values[i])
    ax.bar(x[mask_ok] + i * bar_width,
           bar_values[i, mask_ok], 
           width=bar_width, 
           color=group_colors[i],
           label=f'SpMM V{5-i}')

    mask_nan = np.isnan(bar_values[i])
    if mask_nan.sum() > 0:
        # Plot a red X at the center of where the bar would be, but center vertically (y axis)
        for idx in np.where(mask_nan)[0]:
            xpos = x[idx] + i * bar_width + bar_width / 2
            ypos = max_val / 5
            ax.plot(xpos, ypos, marker='x', color='red', markersize=12, markeredgewidth=3, label=None if i != 0 or idx != np.where(mask_nan)[0][0] else 'Missing')

ax.set_xlabel('Test Case')
ax.set_ylabel('Execution Time in Nanoseconds (10 iterations)')
ax.set_title('Sparse Algorithms Runtime Comparison')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(group_labels, rotation=45, ha='right')
ax.legend()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
# ax.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5))
ax.legend()

plt.tight_layout()
plt.show()

plt.savefig(png_output_dir + "fig2_updated.png")
