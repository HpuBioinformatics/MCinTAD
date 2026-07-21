import numpy as np
import json
import os

#keep the hypergraph structure as a json file
chrom_list = [
    "chr1", "chr2","chr3", "chr4", "chr5", 
    "chr6", "chr7", "chr8", "chr9", "chr10", 
    "chr11", "chr12", "chr13", "chr14", "chr15", 
    "chr16", "chr17", "chr18", "chr19", "chr20", 
    "chr21", "chr22"
]
resolution_list = [10000, 25000, 50000]

if not os.path.exists("hypergraph"):
    os.makedirs("hypergraph")

for res in resolution_list:
    for chrom in chrom_list:
        resolution_str = f"{int(res/1000)}KB"
        hyperedges = np.load(f'temp3/{chrom}_{resolution_str}_hyperedges.npy', allow_pickle=True).item()
        chrom_range = np.load(f'temp1/{resolution_str}_chrom_ranges.npy',allow_pickle=True)
        i = int(chrom[3:])
        bin_num = int(chrom_range[i-1][1])
        he_list = []
        he_weight = []
        for key,value in hyperedges.items():
            if len(key) < 2:
                continue
            node_list = [node - 1 for node in key]
            he_list.append(node_list)
            he_weight.append(value)
        
        hypergraph_dict = {
            "num_v": bin_num,
            "e_list": he_list, 
            "e_weight": he_weight
        }

        with open(f"hypergraph/{chrom}_{resolution_str}_hypergraph.json", "w") as f:
            json.dump(hypergraph_dict, f)

        