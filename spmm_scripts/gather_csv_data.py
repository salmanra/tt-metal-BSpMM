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

csv_dir_spmm = csv_dir + "bsr/bsr_spmm_multicore_reuse_many_blocks_per_core"
csv_dir_basic = csv_dir + "dense/matmul_multicore_reuse"
csv_dir_mcast = csv_dir + "dense/matmul_multicore_reuse_mcast"

csv_data_dirs = [csv_dir_spmm, csv_dir_basic, csv_dir_mcast]

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
    csv_file_names = sorted(os.listdir(csv_data_dir))
    csv_files = [os.path.join(csv_data_dir, f) for f in csv_file_names]
    for j, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # print(f'We are in the {j}th csv file of the {i}th host')
        # print(df[df["name"] == "Program Loop"].size) # what do you mean not all of these dfs have a Program Loop?
        # print(df.shape)
        
        zones_data = {}
        if df[df["name"] == "Program Loop"].size == 0:
            zones_data["Program Loop total ns"] = 0
        else:
            zones_data["Program Loop total ns"] = int(df.loc[df["name"] == "Program Loop", "total_ns"].array[0])

        # print(type(df[df["name"] == "Program Loop"]))
        # print(type(df[df["name"] == "Program Loop"]["total_ns"]))
        # total_ns = df.get("total_ns")["Program Loop"]
        # zones_data["Program Loop total ns"] = total_ns

        data_dicts[i][csv_file_names[j]] = zones_data

pprint.pp(data_dicts)
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

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = np.arange(n_groups)

for i in range(n_dicts):
    ax.bar(x + i * bar_width, bar_values[i], width=bar_width, label=f'Dict {i+1}')

ax.set_xlabel('CSV File')
ax.set_ylabel('Program Loop total ns')
ax.set_title('Grouped Bar Chart of Program Loop total ns')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(group_labels, rotation=45, ha='right')
ax.legend(['spmm', 'gemm_basic', 'gemm_mcast'])

plt.tight_layout()
plt.show()

plt.savefig(png_output_dir + "program_loop_total_ns.png")
