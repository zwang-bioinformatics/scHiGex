# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import torch
from torch_geometric.data import Data

def collate_fn(batch): 

    tensor_keys = set(["node_features", "edge_features", "gene_exp"])

    batched_graph = Data()

    for batch_idx, graph in enumerate(batch):
        
        for key in graph.keys():
            if key == "edge_index":
                node_len = 0 if "node_features" not in batched_graph else batched_graph["node_features"].shape[0]
                if "edge_index" not in batched_graph: batched_graph["edge_index"] = graph["edge_index"].long()
                else: batched_graph["edge_index"] = torch.cat((batched_graph["edge_index"], torch.add(graph["edge_index"], node_len).long()), dim=0)   
            elif key in tensor_keys:
                if key not in batched_graph: batched_graph[key] = graph[key]
                else: batched_graph[key] = torch.cat((batched_graph[key], graph[key]), dim=0)
            else:
                if key not in graph: continue 
                if key not in batched_graph: batched_graph[key] = [graph[key]]
                else: batched_graph[key] += [graph[key]]

    return batched_graph
