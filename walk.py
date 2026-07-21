import numpy as np
import dhg
import json
import os
from scipy.sparse import lil_matrix

# chrom_list = [
#     "chr1", "chr2","chr3", "chr4", "chr5", 
#     "chr6", "chr7", "chr8", "chr9", "chr10", 
#     "chr11", "chr12", "chr13", "chr14", "chr15", 
#     "chr16", "chr17", "chr18", "chr19", "chr20", 
#     "chr21", "chr22"
# ]
chrom_list = [
    "chr20", "chr21", "chr22"
]

resolution_list = [10000, 25000, 50000]

def cal_Pr(G):
    nodes = G.v
    n = len(nodes)
    P = lil_matrix((n, n))
    node_deg = G.deg_v
    elist = G.e[0]
    wlist = G.e[1]

    for v1 in nodes:
        if node_deg[v1] == 0:
            continue
        for eidx in G.nbr_e(v1):
            bins = elist[eidx]
            for v2 in bins:
                if v1 != v2:
                    De = len(bins)
                    P[v1, v2] += wlist[eidx] / ( node_deg[v1] * (De - 1))
    
    Pr_matrix = P.toarray()
    return Pr_matrix


def get_probs_mat():
    for res in resolution_list:
        for chrom in chrom_list:
            print(f"Calculate transition probability of walk for chromosome: {chrom}, resolution: {res}")
            resolution_str = f"{int(res/1000)}KB"
            with open(f"prep_data/hypergraph/{chrom}_{resolution_str}_hypergraph.json", "r") as f:
                hg = json.load(f)

            G = dhg.Hypergraph(num_v=hg["num_v"], e_list=hg["e_list"], e_weight=hg["e_weight"])

            probs_mat = cal_Pr(G)

            np.savetxt(f'probs_mat/{chrom}_{resolution_str}_probs_mat.txt', probs_mat, fmt='%g', delimiter='\t')           

if not os.path.exists("probs_mat"):
    os.makedirs("probs_mat")

get_probs_mat()


