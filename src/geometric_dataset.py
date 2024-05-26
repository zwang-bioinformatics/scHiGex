# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
from safetensors import safe_open


def get_node_features():

    gene_index = json.load(open(f'../assets/exclusive_gene_index.json', 'r'))
    chromosomes = gene_index.keys()

    all_chromosome_node_features = {}

    for chromosome in chromosomes:
        pca_node_features_filename = f'../assets/node_embeddings/pca_node_features_{chromosome}.npy'
        pca_node_features = np.load(pca_node_features_filename, allow_pickle=True)
        all_chromosome_node_features[chromosome] = torch.tensor(pca_node_features, dtype=torch.float)

    return all_chromosome_node_features


def get_targets(gexp_dict, chr):

    gene_index = json.load(open(f'../assets/exclusive_gene_index.json', 'r'))
    total_indexed_genes_not_found_in_result_dict = 0

    target = []
    genes = gene_index[chr].keys()
    for idx, gene in enumerate(genes):
        assert idx == gene_index[chr][gene], "Gene index mismatch!"
        if gene in gexp_dict: target.append(gexp_dict[gene])
        else:
            total_indexed_genes_not_found_in_result_dict += 1
            target.append(0.0)

    assert len(target) == len(genes), "Length of target and number of genes mismatch!"
    if total_indexed_genes_not_found_in_result_dict > 0: print(f"\t\t> Total indexed genes not found in result_dict sample name: {total_indexed_genes_not_found_in_result_dict}")
    
    target = torch.tensor(target, dtype=torch.float)
    target = torch.clamp(target, min=0, max=3)
    target = target / 3 # Normalizing the target values between 0 and 1
    
    return target




class GeometricDataset(Dataset):
    
    def __init__ (self, partition = None, params = None):
        
        #### Checks ####
        
        assert partition is not None, "partition is required!"
        assert params is not None, "params is required!"
  
        assert partition in ["training", "validation", "blind_test"], "invalid partition!"
        
        #### Inits ####
  
        self.partition = partition
        self.params = params
        self.graph_map = {}
        
        data_map = json.load(open('../assets/data_map.json', 'r'))
        data_map_array= []
        
        for cell_type in params["cell_types"]:
            data_list = data_map[cell_type]
            # Split the data into training, validation and blind test on the basis of ratios
            if params["ratios"][0] == 1.0:      train, val, test = data_list, [], []
            elif params["ratios"][1] == 1.0:    train, val, test = [], data_list, []
            elif params["ratios"][2] == 1.0:    train, val, test = [], [], data_list
            else:
                train, test = train_test_split(data_list, train_size = params["ratios"][0], random_state = 42)
                val, test = train_test_split(test, train_size = params["ratios"][1]/(params["ratios"][1] + params["ratios"][2]), random_state = 42)
            if partition == "training": data_map_array += train
            if partition == "validation": data_map_array += val
            if partition == "blind_test": data_map_array += test
            
        # np.random.shuffle(data_map_array)
        self.data_map2 = {idx: data_map_array[idx] for idx in range(len(data_map_array))}
        
        print(f"- Number of graphs in the {partition}: {len(self.data_map2)}")
        
        self.node_features = get_node_features()
        
        self.rna_umi_count_df = pd.read_csv(f'../assets/rna_umicount.tsv', delimiter = "\t", header=0)
        
        super().__init__()
        
    def len(self): return len(self.data_map2)
        
    def get(self, idx):
        
        graph = Data()
        
        with safe_open(self.data_map2[idx]["path"], framework="pt", device="cpu") as raw_graph:
            
            graph.node_features = self.node_features[self.data_map2[idx]["chr"]]
            gexp_dict = dict(zip(self.rna_umi_count_df["gene"], self.rna_umi_count_df[self.data_map2[idx]["sample_name"][0]]))
            graph.gene_exp = get_targets(gexp_dict, self.data_map2[idx]["chr"])
            graph.edge_index = raw_graph.get_tensor("edge_index")
            graph.edge_features = raw_graph.get_tensor("edge_features")
            graph.sample_name = self.data_map2[idx]["sample_name"][0]
            graph.chr = self.data_map2[idx]["chr"]
            graph.cell_type = self.data_map2[idx]["cell_type"]
            
        return graph
            
        