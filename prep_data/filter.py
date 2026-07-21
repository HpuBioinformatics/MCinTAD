import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import os

#input reads file
#converted parquet file
#the output clusters file
input_csv = "/mnt/sdb/zme/datasets/HiPoreC/GSM6284587_GM12878_FC6_reads_alignment.csv"
input_parquet = "GM12878_FC6.parquet"
output_file = "GM12878_FC6_filter.clusters"

if os.path.exists(input_parquet):
    print(f"The parquet file already exists.")
else:
    print("Convert to parquet file.")
    csv_df = pd.read_csv(input_csv, engine="pyarrow")
    csv_df.to_parquet(input_parquet, engine="pyarrow")
    print(f"Saved as a parquet file:{input_parquet}")


dtype_dict = {
    "read_name": str,
    "chrom": str,
    "start": int,
    "end": int,
    "Matches": int,
    "MapQual": int,
}

columns = list(dtype_dict.keys())

def process_batch(batch, dtype_dict):
    df = batch.to_pandas().astype(dtype_dict)
    df = df[
        (df["MapQual"] >= 30) &
        (df["Matches"] >= 100)
    ].copy()

    if df.empty:
        return {}
    
    df["mid_pos"] = (df["start"] + df["end"]) // 2 + 1
    df["position"] = df["chrom"].astype(str) + ":" + df["mid_pos"].astype(str)

    grouped = df.groupby("read_name")["position"].apply(list).to_dict()

    return grouped

pf = pq.ParquetFile(input_parquet)
all_fragments = defaultdict(list)

with ProcessPoolExecutor(max_workers=24) as executor:
    futures = []
    for batch in pf.iter_batches(batch_size=1000000, columns=columns):
        futures.append(executor.submit(process_batch, batch, dtype_dict))
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
        try:
            batch_result = future.result()
            for rid, pos_list in batch_result.items():
                all_fragments[rid].extend(pos_list)
        except Exception as e:
            print(f"Error processing batch: {e}")

#at least for pair-wise contact
min_frag_per_read = 2   
count = 0
with open(output_file, "w") as f:
    for read_name, positions in all_fragments.items():
        if len(positions) >= min_frag_per_read:
            bins_str = "\t".join(positions)
            f.write(f"{read_name}\t{bins_str}\n")
            count += 1

print(f"The results of {count} filtered reads have been saved to {output_file}")