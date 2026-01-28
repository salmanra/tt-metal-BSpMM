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

csv_dir_spmm = csv_dir + "ProfileSuite/bsr_spmm_multicore_reuse_many_blocks_per_core"
csv_dir_basic = csv_dir + "ProfileSuite/matmul_multicore_reuse"
csv_dir_mcast = csv_dir + "ProfileSuite/matmul_multicore_reuse_mcast"

csv_data_dirs = [csv_dir_spmm, csv_dir_basic, csv_dir_mcast]
test_cases = [        
        # "profile_case_sparse_single_block_R32_C32.csv",
        # "profile_case_sparse_single_block_R64_C64.csv",
        # "profile_case_sparse_single_block_R128_C128.csv",
        "profile_case_sparse_diagonal_R32_C32.csv", 
        "profile_case_sparse_diagonal_R64_C64.csv", 
        "profile_case_sparse_diagonal_R128_C128.csv",
        "profile_case_sparse_fill_column_R32_C32.csv", 
        "profile_case_sparse_fill_column_R64_C64.csv", 
        "profile_case_sparse_fill_column_R128_C128.csv",
        "profile_case_sparse_fill_row_R32_C32.csv", 
        "profile_case_sparse_fill_row_R64_C64.csv", 
        "profile_case_sparse_fill_row_R128_C128.csv",
        "profile_case_sparse_fill_random_R32_C32.csv",
        "profile_case_sparse_fill_random_R64_C64.csv", 
    ]

test_cases_short = [name.replace("profile_case_sparse_", "") for name in test_cases]
test_cases_short = [name.replace(".csv", "") for name in test_cases_short]

# for each host program
#   make dict of {csv file name, dict} pairs (empty)
#   for each csv file
#       read the csv file
#       make dict of {Zone-name,total time} pairs (empty)
#       add {Program Loop, time} to dict
#       add {,} some other pairs if you feel it now
#       add Zone dict to csv file name dict

spmm_data = {}
gemm_basic_data = {}
gemm_mcast_data = {}

data_dicts = [spmm_data, gemm_basic_data, gemm_mcast_data]

for i, csv_data_dir in enumerate(csv_data_dirs):
    # csv_file_names = sorted(os.listdir(csv_data_dir))
    # csv_file_names = os.listdir(csv_data_dir)
    csv_files = [os.path.join(csv_data_dir, f) for f in test_cases]
    for j, csv_file in enumerate(csv_files):
        print(csv_file)
        df = pd.read_csv(csv_file)
        # print(f'We are in the {j}th csv file of the {i}th host')
        # print(df[df["name"] == "Program Loop"].size) # what do you mean not all of these dfs have a Program Loop?
        # print(df.shape)
        
        zones_data = {}
        if df[df["name"] == "Program Loop"].size == 0:
            zones_data["Program Loop total ns"] = np.nan
        else:
            zones_data["Program Loop total ns"] = int(df.loc[df["name"] == "Program Loop", "total_ns"].array[0])

        data_dicts[i][test_cases_short[j]] = zones_data

pprint.pp(data_dicts)
with open(json_output_dir + "data_dicts.json", "w") as f:
    json.dump(data_dicts, f, indent=4)
# Now we can make a simple bar chart? And we could write this to a more easily readable JSON file.
#


# Extract keys (csv file names) and values ("Program Loop total ns") for each dict
# Calculate speedup relative to gemm_basic_data (baseline)
group_labels = list(data_dicts[0].keys())
n_groups = len(group_labels)
n_dicts = len(data_dicts)

# Prepare speedup data: speedup = baseline_time / method_time
baseline = np.array([gemm_basic_data[k]["Program Loop total ns"] for k in group_labels], dtype=np.float64)
speedup_values = []
for d in data_dicts:
    vals = np.array([d[k]["Program Loop total ns"] for k in group_labels], dtype=np.float64)
    speedup = baseline / vals
    # If baseline or vals is nan, speedup is nan
    speedup[np.isnan(baseline) | np.isnan(vals)] = np.nan
    speedup_values.append(speedup)

speedup_values = np.array(speedup_values)  # shape: (n_dicts, n_groups)

group_names = ['Block SpMM', 'Dense Matmul Basic', 'Dense Matmul Multicast']
group_colors = ["steelblue", "darkorange", "green"]

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25
x = np.arange(n_groups)

for i in range(n_dicts):
    mask_ok = ~np.isnan(speedup_values[i])
    bars = ax.bar(x[mask_ok] + i * bar_width,
                  speedup_values[i, mask_ok],
                  width=bar_width,
                  color=group_colors[i],
                  label=group_names[i])
    # Plot red X for missing bars
    mask_nan = np.isnan(speedup_values[i])
    for idx in np.where(mask_nan)[0]:
        xpos = x[idx] + i * bar_width + bar_width / 2
        ypos = 1.0  # Place X at y=1 for visibility
        ax.plot(xpos, ypos, marker='x', color='red', markersize=14, markeredgewidth=3, label=None if i > 0 else 'Missing')

ax.set_xlabel('Test Case')
ax.set_ylabel('Speedup vs Dense Matmul Basic')
ax.set_title('Speedup of Sparse/Dense Methods (Higher is Better)')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(group_labels, rotation=45, ha='right')
ax.set_yticks(np.arange(0, int(np.nanmax(speedup_values)) + 2, 1))
ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Baseline (x1)')


# Only show one legend entry for 'Missing'
handles, labels = ax.get_legend_handles_labels()
if 'Missing' in labels:
    first_missing = labels.index('Missing')
    handles = handles[:first_missing+1]
    labels = labels[:first_missing+1]
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(png_output_dir + "fig1_speedup.png")
plt.show()
# ...existing