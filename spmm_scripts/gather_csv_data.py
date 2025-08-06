import os
import pandas as pd
import pprint

csv_dir = "/home/user/tt-metal/profiles/csvs/"

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
    csv_file_names = os.listdir(csv_data_dir)
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
            zones_data["Program Loop total ns"] = df.loc[df["name"] == "Program Loop", "total_ns"].array[0]

        # print(type(df[df["name"] == "Program Loop"]))
        # print(type(df[df["name"] == "Program Loop"]["total_ns"]))
        # total_ns = df.get("total_ns")["Program Loop"]
        # zones_data["Program Loop total ns"] = total_ns

        data_dicts[i][csv_file_names[j]] = zones_data

pprint.pp(data_dicts)
# Now we can make a simple bar chart? And we could write this to a more easily readable JSON file.
#

