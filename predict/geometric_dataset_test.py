# -------------------------------------------------- #
# Author: Bishal Shrestha
# Last Updated: 5/25/2024
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
        pca_node_features_filename = f'./node_embeddings/pca_node_features_{chromosome}.npy'
        pca_node_features = np.load(pca_node_features_filename, allow_pickle=True)
        all_chromosome_node_features[chromosome] = torch.tensor(pca_node_features, dtype=torch.float)

    return all_chromosome_node_features


class GeometricDataset(Dataset):
    
    def __init__ (self, partition = None, params = None):
        
        #### Checks ####
        
        assert partition is not None, "partition is required!"
        assert params is not None, "params is required!"
        
        #### Inits ####
  
        self.partition = partition
        self.params = params
        self.graph_map = {}
        
        data_map = json.load(open('./data_map.json', 'r'))
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
        
        super().__init__()
        
    def len(self): return len(self.data_map2)
        
    def get(self, idx):
        
        graph = Data()
        
        with safe_open(self.data_map2[idx]["path"], framework="pt", device="cpu") as raw_graph:
            
            graph.node_features = self.node_features[self.data_map2[idx]["chr"]]
            graph.edge_index = raw_graph.get_tensor("edge_index")
            graph.edge_features = raw_graph.get_tensor("edge_features")
            graph.sample_name = self.data_map2[idx]["sample_name"][0]
            graph.chr = self.data_map2[idx]["chr"]
            graph.cell_type = self.data_map2[idx]["cell_type"]
            
        return graph
            
        