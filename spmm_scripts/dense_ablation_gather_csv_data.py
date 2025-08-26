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

csv_dir_basic = csv_dir + "DenseAblationKProfileSuite/matmul_multicore_reuse"
csv_dir_mcast = csv_dir + "DenseAblationKProfileSuite/matmul_multicore_reuse_mcast"

csv_data_dirs = [csv_dir_basic, csv_dir_mcast]

# for each host program
#   make dict of {csv file name, dict} pairs (empty)
#   for each csv file
#       read the csv file
#       make dict of {Zone-name,total time} pairs (empty)
#       add {Program Loop, time} to dict
#       add {,} some other pairs if you feel it now
#       add Zone dict to csv file name dict

gemm_basic_data = {}
gemm_mcast_data = {}

data_dicts = [gemm_basic_data, gemm_mcast_data]

for i, csv_data_dir in enumerate(csv_data_dirs):
    csv_file_names = sorted(os.listdir(csv_data_dir))
    csv_files = [os.path.join(csv_data_dir, f) for f in csv_file_names]
    for j, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
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
with open(json_output_dir + "data_dicts_dense_ablation.json", "w") as f:
    json.dump(data_dicts, f, indent=4)


k_vals = [512, 1024, 2048, 4096] 
group_labels = list(data_dicts[0].keys())

# Map each group label to its corresponding k value
group_labels_sorted = []
for k in k_vals:
    for label in group_labels:
        if str(k) in label:
            group_labels_sorted.append(label)
            break

n_groups = len(group_labels_sorted)
n_dicts = len(data_dicts)

# Prepare data for plotting
bar_values = []
for d in data_dicts:

    bar_values.append([d[k]["Program Loop total ns"] for k in group_labels_sorted])

bar_values = np.array(bar_values)  # shape: (n_dicts, n_groups)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = np.arange(n_groups)
colors = ['darkorange', 'green']
for i in range(n_dicts):
    ax.bar(x + i * bar_width, bar_values[i], width=bar_width, color=colors[i], label=f'Dict {i+1}')

ax.set_xlabel('Reduction Dimension Size')
ax.set_ylabel('Nanoseconds elapsed (10 iterations)')
ax.set_title('Basic vs Multicasting Dense Matmul Runtime')
ax.set_xticks(x + bar_width)
ax.set_xticklabels([f'K={k}' for k in k_vals], rotation=45, ha='right')
ax.legend(['Dense Matmul Basic', 'Dense Matmul Multicast'])

plt.tight_layout()
plt.show()

plt.savefig(png_output_dir + "program_loop_total_ns_dense_ablation.png")
