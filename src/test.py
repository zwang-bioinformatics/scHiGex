# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import sys
import torch
from tqdm import tqdm

from rec import *
from models import *
from utils import *

from geometric_dataset import *
from collate_fn import *


model_name = open("model").read().strip(); print(f"Model: {model_name}")

def test(model, epoch, rec, test_loader, device):
    print(f"\nPerforming blind test on epoch {epoch}")
    checkpoint = torch.load(f"../checkpoints/{epoch}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        rec.set_epoch(epoch)
        # rec.reset()
        for batch in tqdm(test_loader, desc=f"Test on Epoch {epoch}"):
            data = batch.to(device)
            data.edge_index = data.edge_index.transpose(0, 1)
            out = model(data.node_features, data.edge_index, data.edge_features).squeeze() #, data.batch_indices
            rec.commit("blind_test", data.gene_exp, out, [data.sample_name, data.chr, data.cell_type])
    
    # rec.save()
    print_results("blind_test", rec)
            
def print_results(tag, rec):
    
    rec_data = {}
    for metric in ["loss", "acc", "bin_acc", "bin_bal_acc", "zero_rate", "precision", "precision_weighted", "recall", "recall_weighted", "roc_auc", "avg_precision", "r2", "pearson", "spearman", "matthews_corr", "f1_score", "f1_score_weighted",  "max"]:
            if metric in ["zero_rate", "max"]: 
                for arg in ["y_true", "y_pred"]: rec_data[f"{tag}_{metric}_{arg}"] = rec.calc(tag, metric, arg)
            else: rec_data[f"{tag}_{metric}"] = rec.calc(tag, metric)
    for results in rec_data: print(f"Result {results}: {rec_data[results]}")
        
    test_avg_diff_over_group = rec.plot(tag, "boxplot")
    rec.plot(tag, "box_plot_chr")
    print(f"Result Avg Diff over group: {test_avg_diff_over_group}")
    
    rec.save_results()
    rec.plot(tag, "densityplot")
    rec.plot(tag, "histogramplot")
    rec.plot(tag, "roc_auc_plot")
    rec.plot(tag, "pr_auc_plot")
    rec.cluster_gene_exp(tag, "y_true")
    rec.cluster_gene_exp(tag, "y_pred")
        
if __name__ == "__main__":
    epoch = sys.argv[1]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print(device)
    
    rec = REC()
    rec.load(epoch)
    
    if len(sys.argv) > 2:
        print_results(sys.argv[2], rec)
    else:
        torch.manual_seed(101)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(101)
        
        test_data = GeometricDataset(partition="blind_test",
            params={
                "cell_types": ["neural ectoderm", "early mesenchyme", "ExE ectoderm"],
                "ratios": [0.0, 0.0, 1.0]
            }
        )
        
        test_loader = get_dataloader(test_data, batch_size=1)
        
        temp_data = next(iter(test_loader))
        node_dim, edge_dim = temp_data.node_features.shape[1], temp_data.edge_features.shape[1]

        model = create_model(model_name=model_name, config={"node_dim": node_dim, "edge_dim":edge_dim, "device": device}).to(device)
        print(model)
        
        test(model, epoch, rec, test_loader, device)