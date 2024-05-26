# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import time
import math
import pandas as pd
import json
import numpy as np
import torch
import multiprocessing
from safetensors import safe_open
from safetensors.torch import save_file

os.environ["NUMEXPR_MAX_THREADS"] = "128"
pd.set_option("display.max_colwidth", None)
print("Process ID: ", os.getpid())


Hi_C_FOLDER = f'../assets/contact_matrices/'
OUTPUT_FOLDER = f'../assets/edge_features/'  

chromosomes = []

cell_types_sample_index = json.load(open(f'../assets/cell_types_sample_index.json', 'r'))

gene_indexes = json.load(open(f'../assets/exclusive_gene_index.json', 'r'))
gene_definitions = json.load(open(f'../assets/exclusive_gene_definitions.json', 'r'))
chrom_sizes = json.load(open(f'../assets/chrom_size.json', 'r'))
chromosomes = list(gene_indexes.keys())


def process_meta_cell(cell_type, meta_cell):
    meta_cell = meta_cell.strip().split()
            
    for chromosome in chromosomes:
        
        if os.path.exists(f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt'):
            # open to check if the file is corrupted
            try:
                safe_open(f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt', framework="pt", device="cpu")
                print(f"\nFile already exists: {cell_type} | {meta_cell[0]} | {chromosome}")
                continue
            except:
                os.remove(f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt')
                
        
        start = time.time()
        
        hics = {}
        shape = (len(gene_indexes[chromosome]), len(gene_indexes[chromosome])) 
        
        for index in meta_cell:
            name = cell_types_sample_index[cell_type][index][0]+'_'+chromosome+'.mmap'
            hics[index] = np.memmap(f'{Hi_C_FOLDER}/{name}', dtype=np.int32, mode='r', shape=shape)
        
        edge_feats = []
        edge_index = []
        
        for gn, gene_index in gene_indexes[chromosome].items():
            for gn2, gene_index2 in gene_indexes[chromosome].items():
                if gene_index == gene_index2:
                    continue
                
                num_of_cells_with_contacts = 0
                temp_edge_feats = []
                mid_point_of_gn = (gene_definitions[chromosome][gn]["end"] + gene_definitions[chromosome][gn]["start"]) / 2
                mid_point_of_gn2 = (gene_definitions[chromosome][gn2]["end"] + gene_definitions[chromosome][gn2]["start"]) / 2
                genomic_distance = math.log10(abs(mid_point_of_gn - mid_point_of_gn2) + 1e-10)
                for index in meta_cell:
                    contact = hics[index][gene_index][gene_index2]
                    temp_edge_feats.append(contact)
                    if contact > 0: num_of_cells_with_contacts += 1
                
                try:
                    assert len(temp_edge_feats) == 21, f"Length of temp_edge_feats is not 21: {len(temp_edge_feats)} | {cell_type} | {meta_cell[0]} | {chromosome} | {gn} | {gn2}"
                except AssertionError as e:
                    if len(temp_edge_feats) == 20:
                        if cell_type == "Ast" or cell_type == "Oli":
                            temp_edge_feats.append(0)

                assert len(temp_edge_feats) == 21, f"Length of temp_edge_feats is not 21: {len(temp_edge_feats)} | {cell_type} | {meta_cell[0]} | {chromosome} | {gn} | {gn2}"
                
                temp_edge_feats.append(num_of_cells_with_contacts/21)
                temp_edge_feats.append(genomic_distance)
                assert len(temp_edge_feats) == 23, f"Length of temp_edge_feats is not 23: {len(temp_edge_feats)} | {cell_type} | {meta_cell[0]} | {chromosome} | {gn} | {gn2}"
                
                if num_of_cells_with_contacts > 0:
                    edge_feats.append(temp_edge_feats)
                    edge_index.append((gene_index, gene_index2))

        # check how many 
        if len(edge_feats) == 0: continue
        
        assert len(edge_feats) == len(edge_index), f"Length of edge_feats and edge_index do not match: {len(edge_feats)} != {len(edge_index)} | {cell_type} | {meta_cell[0]} | {chromosome}"
        
        tensors = {
            "edge_features": torch.tensor(edge_feats, dtype=torch.float),
            "edge_index": torch.tensor(edge_index, dtype=torch.long)
        }
        
        if not os.path.exists(f'{OUTPUT_FOLDER}/{cell_type}'): os.makedirs(f'{OUTPUT_FOLDER}/{cell_type}')
        save_file(tensors, f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt')
        
        with open(f'{OUTPUT_FOLDER}/{cell_type}/list', 'a') as f: f.write(f'{meta_cell[0]}_{chromosome}.pt\n')
        
        print(f"\nTime taken for {cell_type} | {meta_cell[0]} | {chromosome}: {(time.time()-start)/60:.2f} min")


def main():

    for cell_type in cell_types_sample_index:
        print (f"\n\ncell_type: {cell_type} | len: {len(cell_types_sample_index[cell_type])}")
        
        with open(f'../assets/metacell/{cell_type}', 'r') as f:
            meta_cells = f.read().splitlines()
            
        pool = multiprocessing.Pool(processes=25)
        pool.starmap(process_meta_cell, [(cell_type, meta_cell) for meta_cell in meta_cells])
            



if __name__ == "__main__":
    main()
    print("\n\nEnd of the program")
    

# nohup python -u efeats_gen.py > output/efeats_gen.log 2>&1 &
# pkill -f "python -u efeats_gen.py"
# pgrep -f "efeats_gen.py"
