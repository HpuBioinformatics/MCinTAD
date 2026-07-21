import numpy as np
import pandas as pd
import math
import os
from collections import Counter

#chromosome size file
#chromosome list
#resolution list
chrom_size = "chrom_hg38_sizes.txt"
chrom_list = [
    "chr1", "chr2", "chr3", "chr4", "chr5", 
    "chr6", "chr7", "chr8", "chr9", "chr10", 
    "chr11", "chr12", "chr13", "chr14", "chr15", 
    "chr16", "chr17", "chr18", "chr19", "chr20", 
    "chr21", "chr22"
]
resolution_list = [10000, 25000, 50000]


def build_node_dict():
    chrom_sizes_tab = pd.read_table(chrom_size, header=None, sep="\t")
    chrom_sizes_tab.columns = ['chr', 'size']
    print(chrom_sizes_tab)

    for res in resolution_list:
        bin_to_node = {}
        chrom_ranges = []
        
        for j, chrom in enumerate(chrom_list):
            
            size = np.max(chrom_sizes_tab['size'][chrom_sizes_tab['chr'] == chrom])
            
            max_bin_chrom = math.ceil(size / res)
            
          
            node_id = 1
            temp_range = [node_id]
            
            for i in range(max_bin_chrom):
                bin_start = i * res
                bin_key = f"{chrom}:{bin_start}"
                bin_to_node[bin_key] = node_id
                node_id += 1
            
            temp_range.append(node_id - 1)
            chrom_ranges.append(temp_range)
        
       
        resolution_str = f"{int(res/1000)}KB"
        np.save(f"temp1/{resolution_str}_chrom_ranges.npy", chrom_ranges)
        np.save(f"temp1/{resolution_str}_bin_to_node.npy", bin_to_node)

if not os.path.exists("temp1"):
    os.makedirs("temp1")

build_node_dict()

#clusters file
#the max number of fragments for each reads
clusters_file_path = "GM12878_FC6_filter.clusters"
max_cluster_length = 32
def parse_file():
    for chrom in chrom_list:
        for res in resolution_list:
            print(f"Processing chromosome: {chrom}, resolution: {res}")
            
         
            resolution_str = f"{int(res/1000)}KB"
            bin_to_node = np.load(f"temp1/{resolution_str}_bin_to_node.npy", allow_pickle=True).item()
      
            with open(clusters_file_path, "r") as file1:
                line = file1.readline()
                edge_lists = []
                
                while line:
                    info_list = line.strip().split("\t")[1:]
                    node_list = []
                    
                    if (len(info_list) < 2) or (len(info_list) > max_cluster_length):
                        line = file1.readline()
                        continue
                    for info in info_list:
                        try:
                            if ":" in info:
                                parts = info.split(":", 1)
                                if len(parts) == 2:
                                    chr, bin_pos = parts
                                else:
                                    print(f"Skipping invalid data: {info}")
                                    continue
                            else:
                                print(f"Skipping invalid data: {info}")
                                continue
                        except:
                            print(f"Error processing: {info}")
                            raise EOFError
                        if chr not in chrom_list:
                            continue
                        
                        if chr == chrom:

                            bin_floor = int(math.floor(int(bin_pos) / res))
                            bin_start = bin_floor * res
                            bin_key = f"{chr}:{bin_start}"
                            node_id = bin_to_node.get(bin_key)
                            
                            if node_id is not None:
                                node_list.append(node_id)
                    

                    unique_nodes = list(set(node_list))
                    
                    if len(unique_nodes) > max_cluster_length:
                        line = file1.readline()
                        continue
                    
                    unique_nodes.sort()
                    
                    if len(unique_nodes) > 1:
                        edge_lists.append(unique_nodes)
                                              
                    line = file1.readline()
                
                sort_edge_lists = sorted([tuple(edge) for edge in edge_lists], key=lambda x: (x[0], x[1]))
                np.save(f"temp2/{chrom}_{resolution_str}_edge_lists.npy", np.array(sort_edge_lists, dtype=object))
                counts = Counter(sort_edge_lists)
                hyperedges = {}
                for edge, num in counts.items():
                    hyperedges[edge] = num
                np.save(f"temp3/{chrom}_{resolution_str}_hyperedges.npy", hyperedges) 





if not os.path.exists("temp2"):
    os.makedirs("temp2")

if not os.path.exists("temp3"):
    os.makedirs("temp3")

parse_file()