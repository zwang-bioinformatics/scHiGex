# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from geometric_dataset import *
from collate_fn import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:1024"

def get_dataloader(data, batch_size=1):
    return(DataLoader(
        data,
        pin_memory=False,
        num_workers=15,
        persistent_workers=True,
        prefetch_factor=5,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
    ))
    
    
def calc_mean(x, n_segments = 5):
    if isinstance(x, list): x = torch.tensor(x)
    if x.dtype != torch.float32: x = x.float()
    if n_segments is False: return torch.nanmean(x)
    elif n_segments: return avg_segments_avg(x, n_segments)
    
def avg_segments_avg(x, n_segments=5):
    if isinstance(x, list): x = torch.tensor(x)
    if x.dtype != torch.float32: x = x.float()
    segment_size = x.numel() // n_segments
    avgs = []
    for i in range(n_segments):
        start = i * segment_size
        end = None if i == n_segments - 1 else (i + 1) * segment_size
        segment = x[start:end]
        avgs.append(torch.nanmean(segment))
    return torch.nanmean(torch.stack(avgs))


class CustomLoss(nn.Module):
    def __init__(self, n_segments=False, use_sampling=False, use_weights=False):
        super(CustomLoss, self).__init__()
        self.n_segments = n_segments
        self.use_sampling = use_sampling
        self.use_weights = use_weights

    def forward(self, output, target):
        default = True if np.random.uniform(0, 1) < 0.1 else False
        if self.use_sampling and not default:
            output, target = custom_y_sampling(output, target)
        diff = torch.abs(output - target)
        weights = torch.where(target == 0, 0.4, 0.6)
        if self.use_weights and not default:
            return calc_mean((diff * weights), self.n_segments)
        else:
            return calc_mean(diff, self.n_segments)

    
def custom_y_sampling(y_pred, y_true):
    random.seed(111)
    
    non_zero_indices = torch.where(y_true > 0)[0]
    num_samples_to_keep = int(1 * len(non_zero_indices))
    sampled_non_zero_indices = random.sample(non_zero_indices.tolist(), num_samples_to_keep)
    y_true_non_zero = y_true[sampled_non_zero_indices]
    y_pred_non_zero = y_pred[sampled_non_zero_indices]
    
    zero_indices = torch.where(y_true == 0)[0]
    if len(zero_indices) < num_samples_to_keep: num_samples_to_keep = len(zero_indices)
    random_zero_indices = random.sample(zero_indices.tolist(), num_samples_to_keep)
    y_true_zero = y_true[random_zero_indices]
    y_pred_zero = y_pred[random_zero_indices]
    
    y_true_sampled = torch.cat((y_true_non_zero, y_true_zero))
    y_pred_sampled = torch.cat((y_pred_non_zero, y_pred_zero))
    
    return y_pred_sampled, y_true_sampled

def cont_to_cat(y, tag=None):
    assert tag in ["pred", "true"], "Invalid tag"
    if tag == "pred": return torch.where(y < 0.05, 0.0, 1.0)
    elif tag == "true": return torch.where(y == 0.0, 0.0, 1.0) 


def getgeneindex():
    gene_index = json.load(open(f'../assets/exclusive_gene_index.json', 'rb'))
    return gene_index

def getgenelist():
    gene_index = getgeneindex()
    gene_names = []
    total_genes_count = 0
    for chromosome in gene_index:
        genes = list(gene_index[chromosome].keys())
        gene_names.extend(genes)
        total_genes_count += len(genes)
    assert len(gene_names) == total_genes_count
    return gene_names

def getmetadata():
    df = pd.read_excel(f'../assets/metadata.xlsx', header=0)
    return df

def get_original_gene_exp():
    rna_umicount_df = pd.read_csv(f'../assets/rna.umicount.tsv', delimiter='\t', header=0, index_col=0)
    return rna_umicount_df

def validate_true_gene_exp_with_original(true):
    original = get_original_gene_exp()
    # Check if data in the true gene expression is same as the original gene expression. Here, the 1st row is sample names and 1st column is gene names. Is is not necessary they need to same. It is just that content in true gene expression should be same as the original gene expression.
    
    columns = true.columns
    rows = true.index
    for column in columns:
        assert column in original.columns
    for row in rows:
        assert row in original.index
    
    total = 0
    unmatched = 0
    for column in columns:
        for row in rows:
            total += 1
            if (true.loc[row, column] != original.loc[row, column]):
                unmatched += 1

            
def load_checkpoint(model, optimizer, scheduler):
    checkpoint_path = f"../checkpoints/"
    files = os.listdir(checkpoint_path)
    sorted_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    latest_file = checkpoint_path + sorted_files[-1]
    checkpoint = torch.load(latest_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Loaded checkpoint from epoch {start_epoch}")
    return model, optimizer, scheduler, start_epoch
    
def save_checkpoint(model, optimizer, scheduler, epoch):
    if not os.path.exists("../checkpoints"): os.mkdir("../checkpoints")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }, f"../checkpoints/{epoch}.pt")