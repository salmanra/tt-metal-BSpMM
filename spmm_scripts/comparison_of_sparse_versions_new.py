import os
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import numpy as np
import json

snf_profiles_dir = "/home/user/tt-metal/profiles_noc_flipped/"
naive_profiles_dir = "/home/user/tt-metal/profiles_new/"
snf_csv_dir = snf_profiles_dir + "csvs/"
naive_csv_dir = naive_profiles_dir + "csvs/"

json_output_dir = snf_profiles_dir + "jsons/"
png_output_dir = snf_profiles_dir + "pngs/"
os.makedirs(snf_csv_dir, exist_ok=True)
os.makedirs(naive_csv_dir, exist_ok=True)
os.makedirs(png_output_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)

csv_dir_v6 = snf_csv_dir + "ProfileSuiteLargeSparseVersioning/bsr_spmm_multicore_snf"
csv_dir_v5 = naive_csv_dir + "ProfileSuiteLargeSparseVersioning/bsr_spmm_multicore_load_balanced"
csv_dir_v4 = naive_csv_dir + "ProfileSuiteLargeSparseVersioning/bsr_spmm_multicore_reuse_iteration"
# csv_dir_v3 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse_many_blocks_per_core"
# csv_dir_v2 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse"
# csv_dir_v1 = csv_dir + "ProfileSuiteSparseVersioning/bsr_spmm_multicore_reuse_naive"

csv_data_dirs = [csv_dir_v6, csv_dir_v5, csv_dir_v4]
csv_data_labels = [os.path.basename(d) for d in csv_data_dirs]

test_cases = sorted(
    set.union(*(
        {f for f in os.listdir(d) if f.endswith(".csv")}
        for d in csv_data_dirs
    ))
)

sparse_logs = sorted(
    set.union(*(
        {f for f in os.listdir(d) if f.endswith("sparse.log")}
        for d in csv_data_dirs
    ))
)

dense_logs = sorted(
    set.union(*(
        {f for f in os.listdir(d) if f.endswith("dense.log")}
        for d in csv_data_dirs
    ))
)

test_case_files = test_cases

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

def parse_log_metadata(filepath):
    """Parse matrix metadata from a pretty_print log file."""
    result = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if '(H x W)' in line:
                    parts = line.split(':')[1].strip().split(' x ')
                    result['H'], result['W'] = int(parts[0]), int(parts[1])
                elif '(R x C)' in line:
                    parts = line.split(':')[1].strip().split(' x ')
                    result['R'], result['C'] = int(parts[0]), int(parts[1])
                elif 'Number of blocks' in line:
                    result['nblocks'] = int(line.split(':')[1].strip())
    except FileNotFoundError:
        pass
    return result

v6_data = {}
v5_data = {}
v4_data = {}

data_dicts = [v6_data, v5_data, v4_data]

for i, csv_data_dir in enumerate(csv_data_dirs):
    # csv_file_names = sorted(os.listdir(csv_data_dir))
    # csv_file_names = os.listdir(csv_data_dir)
    csv_files = [os.path.join(csv_data_dir, f) for f in test_case_files]
    sparse_log_files = [os.path.join(csv_data_dir, f) for f in sparse_logs]
    dense_log_files = [os.path.join(csv_data_dir, f) for f in dense_logs]
    for j, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        sparse_log = sparse_logs[j]
        dense_log = dense_logs[j]

        sparse_meta = parse_log_metadata(sparse_log_files[j])
        dense_meta = parse_log_metadata(dense_log_files[j])

        M = sparse_meta['H']
        K = sparse_meta['W']
        N = dense_meta['W']
        R = sparse_meta['R']
        C = sparse_meta['C']
        nblocks = sparse_meta['nblocks']

        nnz_elts = nblocks * R * C
        total_ops = nnz_elts * N * 2 # 1 add and 1 mul for each nz elt for each column of the dense matrix
        tflop_count = total_ops / 1e12

        # print(f'We are in the {j}th csv file of the {i}th host')
        # print(df[df["name"] == "Program Loop"].size) # what do you mean not all of these dfs have a Program Loop?
        # print(df.shape)
        
        zones_data = {}
        if df[df["name"] == "Device program Loop"].size == 0:
            zones_data["Program Loop total ns"] = np.nan
        else:
            nanosec = int(df.loc[df["name"] == "Device program Loop", "total_ns"].array[0])
            zones_data["Program Loop total seconds"] = nanosec / 1e9 

        # print(type(df[df["name"] == "Program Loop"]))
        # print(type(df[df["name"] == "Program Loop"]["total_ns"]))
        # total_ns = df.get("total_ns")["Program Loop"]
        # zones_data["Program Loop total ns"] = total_ns
        zones_data["FLOP count"] = tflop_count
        data_dicts[i][test_cases_short[j]] = zones_data

# pprint.pp(data_dicts)
with open(json_output_dir + "data_dicts.json", "w") as f:
    json.dump(data_dicts, f, indent=4)

# Extract keys (csv file names) and values ("Program Loop total ns") for each dict
group_labels = list(data_dicts[0].keys())
n_groups = len(group_labels)
n_dicts = len(data_dicts)

# Prepare data for plotting
num_iters = 10 # TODO : coordinate num iters
bar_values = []
for d in data_dicts:
    bar_values.append([d[k]["FLOP count"] / (d[k]["Program Loop total seconds"] / num_iters) for k in group_labels])

bar_values = np.array(bar_values)  # shape: (n_dicts, n_groups)
# print(bar_values)
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
           label=csv_data_labels[i])

    mask_nan = np.isnan(bar_values[i])
    if mask_nan.sum() > 0:
        # Plot a red X at the center of where the bar would be, but center vertically (y axis)
        for idx in np.where(mask_nan)[0]:
            xpos = x[idx] + i * bar_width + bar_width / 2
            ypos = max_val / 5
            ax.plot(xpos, ypos, marker='x', color='red', markersize=12, markeredgewidth=3, label=None if i != 0 or idx != np.where(mask_nan)[0][0] else 'Missing')

ax.set_xlabel('Test Case')
ax.set_ylabel('TFLOPs (Peak is 74 TFLOPs)')
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

plt.savefig(png_output_dir + "fig2_tflops_opt_nocs.png")
