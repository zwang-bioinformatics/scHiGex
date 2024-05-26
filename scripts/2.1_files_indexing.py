# -------------------------------------------------- #
# Author: Bishal Shrestha
# Last Updated: 5/25/2024
# -------------------------------------------------- #

import os
import sys
import json
import pandas as pd


pairs_files_location = '../assets/pairs'
    
rna_celltype_file_path = f'../assets/metadata.xlsx'
rna_umicount_location = f'../assets/rna_umicount.tsv'

rna_celltype_df = pd.read_excel(rna_celltype_file_path, header=0)
rna_celltype_df = rna_celltype_df[['Cellname', 'Cell type']]
rna_celltype_df = rna_celltype_df.set_index('Cellname')
print(rna_celltype_df.head())

rna_umicount_df = pd.read_csv(rna_umicount_location, delimiter='\t', header=0)
sample_names = rna_umicount_df.columns[1:]
rna_umicount_df = rna_umicount_df.sort_index() # arrange the rows in the ascending order of the index
rna_umicount_df = rna_umicount_df.transpose()
sample_name_after_transpose = rna_umicount_df.index[1:]
print(rna_umicount_df.head())

# Checking if the order of the sample names in UMI count and after transpose are the same
assert all(sample_names == sample_name_after_transpose), f"Sample names in UMI count and after transpose do not match"

rna_celltype_list = []
# The order of the samples in rna_umicount_df and rna_celltype_df may be different. I need to reorder the rows of rna_celltype_df to match the order of rna_umicount_df so the colors match. Only select the ones that is in rna_umicount_df
for index, row in rna_umicount_df.iterrows():
    if index in rna_celltype_df.index:
        rna_celltype_list.append(rna_celltype_df.loc[index, 'Cell type'])
    else:
        print(f'{index} is not in rna_celltype_df')
        
# Create a dictionary with unique cell type as key and its count as value
celltype_count = {}
for celltype in rna_celltype_list:
    if celltype in celltype_count:
        celltype_count[celltype] += 1
    else:
        celltype_count[celltype] = 1
        
# Select first top 3 cell types
celltype_count = dict(sorted(celltype_count.items(), key=lambda item: item[1], reverse=True))
top_3_cell_types = list(celltype_count.keys())[:3]
print(celltype_count)

# Create a dictionary where the key is the cell type and the value is a list of tuple with its new index for the celltype and sample names
celltype_sample_dict = {}

for index, row in rna_umicount_df.iterrows():
    if index in rna_celltype_df.index:
        celltype = rna_celltype_df.loc[index, 'Cell type']
        if celltype in celltype_sample_dict:
            celltype_sample_dict[celltype].append(index)
        else:
            celltype_sample_dict[celltype] = [index]

# The order of keys of celltype_sample_dict must be same as celltype_count
celltype_sample_dict_final = {}
for celltype in celltype_count:
    celltype_sample_dict_final[celltype] = celltype_sample_dict[celltype]
    
# Check if the order is same
assert list(celltype_sample_dict_final.keys()) == list(celltype_count.keys()), "Order of keys of celltype_sample_dict_final is not same as celltype_count"
    
# Check if the number of samples for each cell type is the same
for celltype in celltype_sample_dict_final:
    assert len(celltype_sample_dict_final[celltype]) == celltype_count[celltype], f"Number of samples for {celltype} does not match the count for {celltype}"
    
# Read pairs file names
pairs_files = [filename for filename in os.listdir(pairs_files_location)]
    
# Create a dictionary with cell type as key and its value is another dictionary with sample name as key and its value is [index, pairs_file]
celltype_sample_index_dict = {}
for celltype in celltype_sample_dict_final:
    celltype_sample_index_dict[celltype] = {}
    for index, sample in enumerate(celltype_sample_dict_final[celltype]):
        for pairs_file in pairs_files:
            if sample in pairs_file:
                celltype_sample_index_dict[celltype][index] = [sample, pairs_file]
                break
        else:
            print(f"Pairs file not found for {sample}")
            sys.exit()

# Confirm the keys/index of celltype_sample_index_dict[celltype] is same as celltype_sample_dict_final[celltype]
for celltype in celltype_sample_index_dict:
    assert list(celltype_sample_index_dict[celltype].keys()) == list(range(len(celltype_sample_dict_final[celltype]))), f"Keys/index of celltype_sample_index_dict[{celltype}] is not same as celltype_sample_dict_final[{celltype}]"
    for index, sample in enumerate(celltype_sample_dict_final[celltype]):
        assert celltype_sample_index_dict[celltype][index][0] == sample, f"Sample name for index {index} does not match in celltype_sample_index_dict[{celltype}]"
        
# Save the celltype_sample_dict_final to a file
with open(f'../assets/cell_types.json', 'w') as f:
    json.dump(celltype_sample_dict_final, f)
    
# Save the celltype_count to a file
with open(f'../assets/cell_types_count.json', 'w') as f:
    json.dump(celltype_count, f)
    
# Save the celltype_sample_index_dict to a file
with open(f'../assets/cell_types_sample_index.json', 'w') as f:
    json.dump(celltype_sample_index_dict, f)
        
    