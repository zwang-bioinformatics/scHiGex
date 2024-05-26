# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import sys
import argparse
from tqdm import tqdm

from rec import *
from plot import *
from utils import *
from models import *
from test import test
from geometric_dataset import *

print("Process ID: ", os.getpid())

model_name = open("model").read().strip(); print(f"Model: {model_name}")

### Argument parsing ###
parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('-model', type=str, default=model_name, help='Model name')
parser.add_argument('-batch_size', type=int, default=8, help='Batch size')
parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('-resume', type=str, default=None, help='Resume from the last checkpoint')
parser.add_argument('-gpu', type=str, default=3, help='GPU number')
args = parser.parse_args()
print(args)


device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'); print(device)
torch.manual_seed(101)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(101)
# torch.multiprocessing.set_sharing_strategy('file_system')

train_data = GeometricDataset(partition="training",
    params={
        "cell_types": ["mix late mesenchyme", "blood", "mitosis", "ExE endoderm", "early neurons"],
        "ratios": [1.0, 0.0, 0.0]
    }
)

val_data = GeometricDataset(partition="validation",
    params={
        "cell_types": ["radial glias", "early mesoderm"],
        "ratios": [0.0, 1.0, 0.0]
    }
)

test_data = GeometricDataset(partition="blind_test",
    params={
        "cell_types": ["neural ectoderm", "early mesenchyme", "ExE ectoderm"],
        "ratios": [0.0, 0.0, 1.0]
    }
)
    
train_loader = get_dataloader(train_data, batch_size=args.batch_size)
val_loader = get_dataloader(val_data, batch_size=args.batch_size)
test_loader = get_dataloader(test_data, batch_size=1)

# Get the edge feature and node feature dimensions
temp_data = next(iter(train_data))
node_dim, edge_dim = temp_data.node_features.shape[1], temp_data.edge_features.shape[1]
print(f"Node dim: {node_dim}, Edge dim: {edge_dim}")

while True:
    try: model = create_model(model_name=args.model, config={"node_dim": node_dim, "edge_dim":edge_dim, "device": device}).to(device); break
    except: print(".", end=""); continue
print(model)

rec = REC();
rec_s = REC_SUMMARY()

criterion = CustomLoss(n_segments=False, use_sampling=True, use_weights=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

start_epoch = 1
best_val_loss, patience, patience_limit = float('inf'), 0, 10

if args.resume == 'True': 
    model, optimizer, schedular, start_epoch = load_checkpoint(model, optimizer, schedular)
    rec_s.load()

print("Training started")
for epoch in range(start_epoch, args.num_epochs + 1):
    model.train()
    rec.reset()
    rec.set_epoch(epoch)

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.num_epochs}"):
        data = batch.to(device)
        data.edge_index = data.edge_index.transpose(0, 1)
        optimizer.zero_grad()
        out = model(data.node_features, data.edge_index, data.edge_features).squeeze()
        loss = criterion(out, data.gene_exp)
        rec.commit("train", data.gene_exp, out, [data.sample_name, data.chr, data.cell_type])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}/{args.num_epochs}"):
            data = batch.to(device)
            data.edge_index = data.edge_index.transpose(0, 1)
            out = model(data.node_features, data.edge_index, data.edge_features).squeeze()
            rec.commit("val", data.gene_exp, out, [data.sample_name, data.chr, data.cell_type])

        val_loss, val_acc = rec.calc("val", "loss"), rec.calc("val", "acc")
        schedular.step(val_loss)
        
        rec_data = {"epoch": epoch, "lr": optimizer.param_groups[0]['lr']}
        for phase in ["train", "val"]:
            for metric in ["loss", "acc", "bin_acc", "bin_bal_acc", "zero_rate", "precision", "precision_weighted", "recall", "recall_weighted", "roc_auc", "avg_precision", "r2", "pearson", "spearman", "matthews_corr", "f1_score", "f1_score_weighted",  "max"]:
                if metric in ["zero_rate", "max"]: 
                    for arg in ["y_true", "y_pred"]: rec_data[f"{phase}_{metric}_{arg}"] = rec.calc(phase, metric, arg)
                else: rec_data[f"{phase}_{metric}"] = rec.calc(phase, metric)
        rec_s.commit(rec_data)
        for results in rec_data: print(f"{results}: {rec_data[results]}", end="    ")
    
    print()
    rec_s.save()
    # rec_s.plot()
    rec.save()
    # rec.plot("val", "densityplot")
    # rec.plot("val", "histogramplot")
    # rec.plot("val", "boxplot")
    # rec.plot("val", "roc_auc_plot")
    # rec.plot("val", "pr_auc_plot")
    # if (args.batch_size == 1):
    #     if epoch == 1: rec.cluster_gene_exp("train", sample, "y_true"); rec.cluster_gene_exp("val", sample, "y_true")
    #     rec.cluster_gene_exp("train", sample, "y_pred")
    #     rec.cluster_gene_exp("val", sample, "y_pred")
        
    # if epoch == 1: rec.cluster_gene_exp("val", sample, "y_true") 
    # rec.cluster_gene_exp("val", sample, "y_pred")

    save_checkpoint(model, optimizer, schedular, epoch)
    
    if val_loss < best_val_loss: best_val_loss, patience = val_loss, 0
    else: patience += 1
    if patience >= patience_limit:
        print("Early stopping triggered")
        break
        
    
### Blind Test ###
best_epoch = rec_s.get_best_epoch()
test(model, best_epoch, rec, test_loader, device)

best_epoch_acc = rec_s.get_best_epoch_acc()
if (best_epoch_acc != best_epoch): test(model, best_epoch_acc, rec, test_loader, device)

print("COMPLETE")