# -------------------------------------------------- #
# Author: Bishal Shrestha
# Last Updated: 5/25/2024
# -------------------------------------------------- #

import sys
sys.path.append('../src/')

import json
import torch
import pandas as pd
from tqdm import tqdm

from rec import *
from utils import *
from models import *

from collate_fn import *
from geometric_dataset_test import *

model_name = "GTransCustomModular"

# load gene index
gene_index = json.load(open(f'../assets/exclusive_gene_index.json', 'r'))

def test(model, test_loader, device):
    checkpoint = torch.load(f"./model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    gene_list = []
    for chromosome in gene_index.keys():
        gene_list += list(gene_index[chromosome].keys())
    
    predictions_df = pd.DataFrame(index=gene_list)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting"):
            data = batch.to(device)
            data.edge_index = data.edge_index.transpose(0, 1)
            out = model(data.node_features, data.edge_index, data.edge_features).squeeze()
            
            # check if the column name is already in the dataframe
            column_name = data.sample_name[0]
            if column_name not in predictions_df.columns:
                predictions_df[column_name] = None
            
            gene_list = list(gene_index[data.chr[0]].keys())
            for i, gene_name in enumerate(gene_list):
                predictions_df.loc[gene_name, column_name] = out[i].item()
            
    predictions_df.to_csv("predictions.csv")
                
                
            
if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'); print(device)
    
    # get cell types
    cell_types = list(json.load(open(f'./cell_types_count.json', 'r')).keys())
    
    torch.manual_seed(101)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(101)
    
    test_data = GeometricDataset(partition="blind_test",
        params={
            "cell_types": cell_types,
            "ratios": [0.0, 0.0, 1.0]
        }
    )
    
    test_loader = get_dataloader(test_data, batch_size=1)
    
    temp_data = next(iter(test_loader))
    node_dim, edge_dim = temp_data.node_features.shape[1], temp_data.edge_features.shape[1]

    model = create_model(model_name=model_name, config={"node_dim": node_dim, "edge_dim":edge_dim, "device": device}).to(device)
    print(model)
    
    test(model, test_loader, device)