import numpy as np
import dhg
import json
import heapq
import os

class TADNode:
    def __init__(self, left, right, left_child=None, right_child=None, volume=None):
        self.left = left          
        self.right = right        
        self.left_child = left_child
        self.right_child = right_child
        self.volume = volume      
        self.merge_ncut = None    

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def __repr__(self):
        return f"[{self.left},{self.right}]"
    
def bin_to_edge_idx(G):
    nodes = G.v
    elist = G.e[0]
    bin_to_eidx = [[] for _ in range(len(nodes))]
    for eidx, bins in enumerate(elist):
        for node in bins:
            bin_to_eidx[node].append(eidx)
    
    return bin_to_eidx



def cal_volume(G, tad):
    node_deg = G.deg_v
    volume = sum(node_deg[v] for v in range(tad['left'], tad['right'] + 1))
    return volume


def merge_tads(G, tads, threshold):
    if not tads or len(tads) < 2:
        return tads
    
    converted_tads = []
    for tad in tads:
        converted_tads.append({
            'left': tad['left'] - 1,
            'right': tad['right'] - 1
        })
    all_tads = [{'left': tad['left'], 'right': tad['right']} for tad in converted_tads]
    sorted_tads = sorted(converted_tads, key=lambda x: x['left'])
    bin_to_eidx = bin_to_edge_idx(G)
    wlist = G.e[1]

    edge_cache = {}
    def get_edge_set(tad):
        key = (tad['left'], tad['right'])
        if key not in edge_cache:
            es = set()
            for v in range(tad['left'], tad['right'] + 1):
                es.update(bin_to_eidx[v])
            edge_cache[key] = es
        return edge_cache[key]
  
    def cal_cross_weight(tad1, tad2):
        es1 = get_edge_set(tad1)
        es2 = get_edge_set(tad2)
        if len(es1) > len(es2):
            es1, es2 = es2, es1
        
        cross = 0
        for eidx in es1:
            if eidx in es2:
                cross += wlist[eidx]
        return cross
    
    def cal_cut_ratio(tad1, tad2):
        vol1 = cal_volume(G, tad1)
        vol2 = cal_volume(G, tad2)
        cross_weight = cal_cross_weight(tad1, tad2)
        return cross_weight / min(vol1, vol2)


    nodes = []
    for tad in sorted_tads:
        vol = cal_volume(G, tad)
        node = TADNode(tad['left'], tad['right'], volume=vol)
        nodes.append(node)
    
    heap = []
    for i in range(len(nodes) - 1):
        if nodes[i].right + 1 == nodes[i+1].left:
            tad1 = {'left': nodes[i].left, 'right': nodes[i].right}
            tad2 = {'left': nodes[i+1].left, 'right': nodes[i+1].right}
            ncut = cal_cut_ratio(tad1, tad2)
            heapq.heappush(heap, (-ncut, i, i+1, nodes[i], nodes[i+1]))

    active = [True] * len(nodes)
    while len([n for n in nodes if n is not None]) > 1:
        while heap:
            neg_ncut, i, j, node_i, node_j = heapq.heappop(heap)
            ncut= -neg_ncut
            if i >= len(nodes) or j >= len(nodes):
                continue
            if not (active[i] and active[j]):
                continue
            if nodes[i] is node_i and nodes[j] is node_j:
                break
        
        else:
            break

        if ncut < threshold:
            break

        if node_i.right + 1 != node_j.left:
            continue

        parent = TADNode(
            left=node_i.left,
            right=node_j.right,
            left_child=node_i,
            right_child=node_j,
            volume=node_i.volume + node_j.volume
        )
        parent.merge_ncut = ncut
        all_tads.append({'left': parent.left, 'right': parent.right})
        nodes[i] = parent
        nodes[j] = None
        active[i] = True
        active[j] = False

        new_nodes = [node for node in nodes if node is not None]
        heap = []
        for k in range(len(new_nodes) - 1):
            if new_nodes[k].right + 1 == new_nodes[k+1].left:
                tad1 = {'left': new_nodes[k].left, 'right': new_nodes[k].right}
                tad2 = {'left': new_nodes[k+1].left, 'right': new_nodes[k+1].right}
                ncut = cal_cut_ratio(tad1, tad2)
                heapq.heappush(heap, (-ncut, k, k+1, new_nodes[k], new_nodes[k+1]))
        nodes = new_nodes
        active = [True] * len(nodes)
    final_tads = [{'left': tad['left'] + 1, 'right': tad['right'] + 1} for tad in all_tads]
    sorted_results = sorted(final_tads, key=lambda x: (x['left'], x['right']))
    return sorted_results


if __name__ == '__main__':

    resolution_list = [10000, 25000, 50000]
    chrom_list = [
        "chr20", "chr21", "chr22"
    ]

    merge_threshold = 0.2
    max_tad_size = 3000000

    for res in resolution_list:
        res_str = f"{int(res/1000)}KB"
        if not os.path.exists(f"TADs/{res_str}"):
            os.makedirs(f"TADs/{res_str}")
        for chrom in chrom_list:
            print(f"Generate TADs from the boundary for chromosome: {chrom}, resolution: {res_str}")
            boundaries = np.loadtxt(f'results/{res_str}/{chrom}_{res_str}_boundaries.txt', dtype=int)
            with open(f"prep_data/hypergraph/{chrom}_{res_str}_hypergraph.json", "r") as f:
                hg = json.load(f)

            G = dhg.Hypergraph(num_v=hg["num_v"], e_list=hg["e_list"], e_weight=hg["e_weight"])
            node_deg = G.deg_v

            tads = []
            for i in range(len(boundaries) - 1):
                if sum(node_deg[v] for v in range(boundaries[i] - 1, boundaries[i+1] - 1)) == 0:
                    continue
                tads.append({'left': boundaries[i], 'right': boundaries[i+1] - 1})
            
            with open(f'results/{res_str}/{chrom}_{res_str}_tads.txt', "w") as out:
                for original_tad in tads:
                    out.write(f"{original_tad['left']}\t{original_tad['right']}\n")

            nested_tads = merge_tads(G, tads, merge_threshold)

            with open(f'TADs/{res_str}/{chrom}.txt', "w") as out:
                for tad in nested_tads:
                    if tad['right'] - tad['left'] > max_tad_size // res:
                        continue
                    out.write(f"{tad['left']}\t{tad['right']}\n")
            


