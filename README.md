
# About

MCinTAD aims to utilize high-order interaction data to identify nested topologically associating domains.


# Requirements
<pre>python==3.8.20
dhg==0.9.5
pytorch==1.12.1
pyarrow==17.0.0
scipy==1.10.1
pandas==2.0.3 
numpy==1.24.4
tqdm==4.67.1</pre>  


# Usage

## Step 1. Data processing
(i) Enter the data processing directory.
<pre>
cd prep_data</pre>

(ii) Filter and align reads to obtain the multi-way interaction clusters file, if the data is already clusters file, this step can be skipped.
```python
python filter.py  
```

(iii) Construct the hypergraph structure and save as the corresponding json file.
```python
python chr.py
python save.py 
```

## Step 2. Feature generation
Generate the transition probability matrix via hypergraph random walk.
```python
cd ..
python walk.py
```

## Step 3. Model prediction
Load model weights and predict TAD boundaries.
```python
python predict.py
```

## Step 4. Filter boundaries and generate nested TADs
(i) Filter TAD boundaries by the original transition probability matrix.
```python
python filter_boundaries.py
```
(ii) Generate nested TADs based on hyperedge cut ratio.
```python
python merge.py
```

# Parameter setting

| parameters | description | used in |
|------------|-------------|---------|
| input_csv | path to the raw reads file | filter.py |
| input_parquet | path to the converted parquet file   | filter.py |
| output_file | path to the output clusters file | filter.py |
| chrom_size | path to the chromosome size file  | chr.py |
| chrom_list | list of the chromosomes | chr.py, save.py, walk.py, predict.py, filter_boundaries.py, merge.py |
| resolution_list | list of the resolutions | chr.py, save.py, walk.py, predict.py, filter_boundaries.py, merge.py |
| clusters_file_path | path to the clusters file | chr.py |
| max_cluster_length | maximum number of fragments per read [32] | chr.py |
| bg_percent | quantile cutoff for predicting scores to set adaptive threshold [None] | predict.py |
| merge_threshold | threshold of hyperedge cut ratio for merging TADs [0.2] | merge.py |
| max_tad_size | maximum TAD size in bp [3000000] | merge.py |

