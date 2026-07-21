import numpy as np

resolution_list = [10000, 25000, 50000]
chrom_list = [
    "chr20", "chr21", "chr22"
]

def get_up_down_mean(mat):
    all_values = mat[~np.eye(len(mat), dtype=bool)]
    if mat.size == 0:
        return 0.0
    
    return all_values.mean()

def get_cross_mean(mat):
    if mat.size == 0:
        return 0.0
    
    return np.mean(mat)

for resolution in resolution_list:
    res_str = f"{int(resolution/1000)}KB"
    for chrom in chrom_list:
        boundaries = np.loadtxt(f'results/{res_str}/{chrom}_{res_str}_predicted_boundaries.txt', dtype=int)
        probs_matrix = np.loadtxt(f'probs_mat/{chrom}_{res_str}_probs_mat.txt')
        filtered_boundaries = []
        n = len(probs_matrix)
        window = 5

        for boundary in boundaries:
            if boundary <= window or boundary >= (n - window):
                filtered_boundaries.append(boundary)
                continue
            bidx = boundary - 1
            sub_matrix = probs_matrix[max(0, bidx - window) : min(n, bidx + window), max(0, bidx - window) : min(n, bidx + window)]
            row_sums = sub_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            norm_probs = sub_matrix / row_sums
            up_probs = norm_probs[0 : window, 0 : window]
            down_probs = norm_probs[window : 2 * window, window : 2 * window]
            up_value = get_up_down_mean(up_probs)
            down_value = get_up_down_mean(down_probs)
            cross_probs1 = norm_probs[0 : window, window : 2 * window]
            cross_probs2 = norm_probs[window : 2 * window, 0 : window]
            cross_value1 = get_cross_mean(cross_probs1)
            cross_value2 = get_cross_mean(cross_probs2)
            if (up_value == 0 or down_value == 0) and (cross_value1 == 0 or cross_value2 == 0):
                filtered_boundaries.append(boundary)
                continue
            if up_value > 2 * cross_value1 and down_value > 2 * cross_value2:
                filtered_boundaries.append(boundary)
       
        np.savetxt(f'results/{res_str}/{chrom}_{res_str}_boundaries.txt', filtered_boundaries, fmt='%d', delimiter='\t')


