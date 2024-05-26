# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import gzip
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, r2_score, matthews_corrcoef, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from utils import *
from plot import *

class REC_SUMMARY:
    def __init__(self):
        self.summary = pd.DataFrame(index = None)

    def commit(self, data):
        data_df = pd.DataFrame([data])
        self.summary = pd.concat([self.summary, data_df])

    def save(self):
        self.summary.to_csv('../results/summary.csv', index=False)
        self.save_best()
        
    def load(self):
        self.summary = pd.read_csv('../results/summary.csv')

    def plot(self):
        temp_summary = self.summary.dropna()
        temp_summary.set_index('epoch')[['train_loss', 'train_acc', 'val_loss', 'val_acc']].plot(kind='line')
        plt.grid(True)
        plt.savefig('../figures/loss.png')
        plt.close()
        
    def get_best_epoch(self):
        return self.summary[self.summary['val_loss'] == self.summary['val_loss'].min()]['epoch'].values[0]
    
    def get_best_epoch_acc(self):
        return self.summary[self.summary['val_acc'] == self.summary['val_acc'].max()]['epoch'].values[0]
    
    def display(self):
        print(self.summary)
        
    def save_best(self):
        df = self.summary
        min_val_loss = df['val_loss'].min()
        min_val_loss_row = df[df['val_loss'] == min_val_loss]

        min_val_loss_row_dict = {}
        for col in df.columns:
            if col.startswith('train'):
                if "train" not in min_val_loss_row_dict:
                    min_val_loss_row_dict['train'] = {}
                if col == "train_loss":
                    min_val_loss_row_dict['train'][col] = float(min_val_loss_row[col].values[0])
            elif col.startswith('val'):
                if "val" not in min_val_loss_row_dict:
                    min_val_loss_row_dict['val'] = {}
                min_val_loss_row_dict['val'][col] = float(min_val_loss_row[col].values[0])
            else:
                min_val_loss_row_dict[col] = float(min_val_loss_row[col].values[0])

        with open(f"../results/report.json", 'w') as f:
            json.dump(min_val_loss_row_dict, f, indent=4)
        

class REC:
    """
        Works for every epoch, need to be reset after every epoch
    """
    def __init__(self):
        self.epoch = 0
        self.sample = ""
        self.data = {}
        self.results = {}
        self.metrics = {"max": self.max, "loss": self.loss, "acc": self.acc, "bin_acc": self.bin_acc, "bin_bal_acc": self.bin_bal_acc, "precision": self.precision, "recall": self.recall, "precision_weighted":self.precision_weighted, "recall_weighted":self.recall_weighted, "roc_auc": self.roc_auc, "avg_precision": self.avg_precision, "r2": self.r2, "pearson": self.pearson, "spearman": self.spearman, "matthews_corr": self.matthews_corr, "zero_rate": self.zero_rate, "f1_score": self.f1_score, "f1_score_weighted": self.f1_score_weighted}
        self.plots = {"boxplot": self.boxplot, "box_plot_chr": self.box_plot_chr, "densityplot": self.densityplot, "histogramplot": self.histogramplot, "roc_auc_plot": self.roc_auc_plot, "pr_auc_plot": self.pr_auc_plot, "scatterplot": self.scatterplot}
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def set_sample(self, sample):
        self.sample = sample
        
    def commit(self, tag, y_true, y_pred, misc):
        assert tag in ["train", "val", "blind_test"], "Invalid tag"
        assert y_true.shape == y_pred.shape, "Shape of y_true and y_pred are not same"
        assert y_true.dtype == y_pred.dtype, "Data type of y_true and y_pred are not same"
        if tag not in self.data: self.data[tag] = []
        if tag not in self.results: self.results[tag] = {}
        self.data[tag].append({"y_true": y_true.cpu().detach(), "y_pred": y_pred.cpu().detach(), "misc": misc})
        
    def reset(self):
        self.epoch = 0
        self.sample = ""
        self.data = {}
        self.results = {}
        
    def calc(self, tag, metric_fn, tag2=None):
        print(f"Calc: {tag}, {metric_fn}")
        metric_fn = self.metrics[metric_fn]
        if tag2 is not None: 
            y = torch.cat([d[tag2] for d in self.data[tag]]).view(-1)
            result = metric_fn(y, tag2)
            self.results[tag][f"{metric_fn.__name__}_{tag2}"] = result
            # metric_batches = [metric_fn(d[tag2]) for d in self.data[tag]]
        else: 
            y_true = torch.cat([d["y_true"] for d in self.data[tag]]).view(-1)
            y_pred = torch.cat([d["y_pred"] for d in self.data[tag]]).view(-1)
            result = metric_fn(y_true, y_pred)
            self.results[tag][metric_fn.__name__] = result
            # metric_batches = [metric_fn(d["y_true"], d["y_pred"]) if self.check_data(d["y_true"], d["y_pred"]) else torch.nan for d in self.data[tag]]
        # result = calc_mean(metric_batches, n_segments=False).item()
        return result
    
    def plot(self, tag, plot_fn):
        print(f"Plot: {tag}, {plot_fn}")
        if plot_fn == "box_plot_chr":
            plot_fn = self.plots[plot_fn]
            y_true = torch.cat([d["y_true"] for d in self.data[tag]]).view(-1)
            y_pred = torch.cat([d["y_pred"] for d in self.data[tag]]).view(-1)
            chrom = []
            for d in self.data[tag]:
                shape = d["y_true"].shape[0]
                temp_chr = d["misc"][1][0] # a string
                temp_chr_array = np.full(shape, temp_chr)
                chrom.append(temp_chr_array)
            
            chrom = np.concatenate(chrom)
            return(plot_fn(tag, y_true, y_pred, chrom))
        else:
            plot_fn = self.plots[plot_fn]
            y_true = torch.cat([d["y_true"] for d in self.data[tag]]).view(-1)
            y_pred = torch.cat([d["y_pred"] for d in self.data[tag]]).view(-1)
            return(plot_fn(tag, y_true, y_pred))
        
    def check_data(self, y_true, y_pred):
        if torch.std(y_true) == 0: return False
        else : return True
        
    def load(self, epoch):
        with gzip.open(f'../misc/{epoch}.pkl.gz', 'rb') as f:
            data = pickle.load(f)
        self.epoch = data["epoch"]
        self.sample = data["sample"]
        self.data = data["data"]
        self.results = data["results"]
        
    def view_info(self):
        print(f"Epoch: {self.epoch}")
        print(f"Data: {self.data.keys()}")
        print(f"Results: {self.results.keys()}")
        
    def save(self):
        with gzip.open(f'../misc/{self.epoch}.pkl.gz', 'wb') as f:
            pickle.dump({
                "epoch": self.epoch,
                "sample": self.sample,
                "data": self.data,
                "results": self.results
            }, f, pickle.HIGHEST_PROTOCOL)
        
    def save_results(self):
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('../results/results.csv')
        
        # also save as json with formating
        results_json = results_df.to_json(orient='index')
        with open('../results/results.json', 'w') as f:
            f.write(json.dumps(json.loads(results_json), ensure_ascii=False, indent=4))
            
        
    def save_data(self):
        for tag in self.data.keys():
            for key in ["y_true", "y_pred"]:
                df = pd.DataFrame([d[key].cpu().detach().numpy() for d in self.data[tag]])
                df.to_csv(f'../misc/{self.epoch}_{self.sample}_{tag}_{key}.csv', index=True)
                
    def save_formatted_data(self, tag):
        true_gene_exp_df, pred_gene_exp_df = self.get_formatted_data(tag)
        true_gene_exp_df.to_csv(f'../misc/{self.epoch}_{self.sample}_{tag}_true_gene_exp.csv', sep='\t')
        pred_gene_exp_df.to_csv(f'../misc/{self.epoch}_{self.sample}_{tag}_pred_gene_exp.csv', sep='\t')
        
        
    def get_formatted_data(self, tag, tag2):
        # Does not work with higher batch sizes
        gene_names = getgenelist()
        gene_index = getgeneindex()
        unique_sample_names = list(set([d["misc"][0][0] for d in self.data[tag]]))
        gene_exp_df = pd.DataFrame(0.0, index = gene_names, columns = unique_sample_names)
        for d in self.data[tag]:
            sample_name, chr, cell_type = d["misc"][0][0], d["misc"][1][0], d["misc"][2][0]
            chr_gene_names = list(gene_index[chr].keys())
            gene_exp = d[tag2].cpu().detach().numpy()
            assert len(chr_gene_names) == len(gene_exp), f"Chromosome gene count, true gene exp length and pred gene exp length are not equal for {sample_name}"
            gene_exp_df[sample_name][chr_gene_names] = gene_exp
        return gene_exp_df
        
        
    ### Metrics ###
    
    def max(self, y, tag):
        return torch.max(y).item()
        
    def loss(self, y_true, y_pred):
        criterion = CustomLoss(n_segments=False, use_sampling=False, use_weights=False)
        return criterion(y_pred, y_true).item()
        # return custom_loss(y_pred, y_true, n_segments=False, use_sampling=False, use_weights=False).item()
    
    def acc(self, y_true, y_pred):
        return torch.sum(torch.abs(y_pred - y_true) < 0.1).item() / len(y_true)
    
    def bin_acc(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return accuracy_score(y_true, y_pred)
    
    def bin_bal_acc(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return balanced_accuracy_score(y_true, y_pred)
    
    def precision(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return precision_score(y_true, y_pred, zero_division=1)
    
    def recall(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return recall_score(y_true, y_pred, zero_division=1)
    
    def precision_weighted(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return precision_score(y_true, y_pred, average='weighted', zero_division=1)
    
    def recall_weighted(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return recall_score(y_true, y_pred, average='weighted', zero_division=1)
    
    def avg_precision(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        return average_precision_score(y_true, y_pred, average='weighted')
    
    def f1_score(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return f1_score(y_true, y_pred, zero_division=1)
    
    def f1_score_weighted(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return f1_score(y_true, y_pred, average='weighted', zero_division=1)
    
    def r2(self, y_true, y_pred):
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        return r2_score(y_true, y_pred)
    
    def matthews_corr(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), cont_to_cat(y_pred, "pred").cpu().detach().numpy()
        return matthews_corrcoef(y_true, y_pred)
    
    def pearson(self, y_true, y_pred):
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        pearson_corr, _ = pearsonr(y_true, y_pred)
        return pearson_corr

    def spearman(self, y_true, y_pred):
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        spearman_corr, _ = spearmanr(y_true, y_pred)
        return spearman_corr
    
    def zero_rate(self, y, tag):
        if "true" in tag: y = cont_to_cat(y, "true")
        elif "pred" in tag: y = cont_to_cat(y, "pred")
        return torch.sum(y == 0).item() / len(y)
    
    def roc_auc(self, y_true, y_pred):
        y_true, y_pred = cont_to_cat(y_true, "true").cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        return roc_auc_score(y_true, y_pred, average='weighted')
    
    
    ### Plots ###
    
    def boxplot(self, tag, y_true, y_pred):
        return(plot_absolute_difference_boxplot_manual(y_true, y_pred, str(self.epoch)+"_"+self.sample+"_"+tag))
    
    def box_plot_chr(self, tag, y_true, y_pred, chrom):
        return(plot_absolute_difference_boxplot_for_chromosomes(y_true, y_pred, chrom, str(self.epoch)+"_"+tag))
    
    def densityplot(self, tag, y_true, y_pred):
        if (self.epoch == 1 or tag == "blind_test"): plot_density(y_true, str(self.epoch)+"_"+self.sample+"_"+tag+"_true")
        plot_density(y_pred, str(self.epoch)+"_"+self.sample+"_"+tag+"_pred")   
        return None 
    
    def histogramplot(self, tag, y_true, y_pred):
        if (self.epoch == 1 or tag == "blind_test"): plot_histogram(y_true, str(self.epoch)+"_"+self.sample+"_"+tag+"_true")
        plot_histogram(y_pred, str(self.epoch)+"_"+self.sample+"_"+tag+"_pred")
        return None
    
    def roc_auc_plot(self, tag, y_true, y_pred):
        plot_roc_auc(y_true, y_pred, str(self.epoch)+"_"+self.sample+"_"+tag)
        return None
    
    def pr_auc_plot(self, tag, y_true, y_pred):
        plot_pr_auc(y_true, y_pred, str(self.epoch)+"_"+self.sample+"_"+tag)
        return None
    
    def cluster_gene_exp(self, tag, tag2):
        gene_exp_df = self.get_formatted_data(tag, tag2)
        plot_cluster_gene_exp(gene_exp_df, str(self.epoch)+"_"+self.sample+"_"+tag+"_"+tag2)
        self.kmeansplot(tag, tag2, tag3="all")
        self.kmeansplot(tag, tag2, tag3="UMAP")
        return None
    
    def scatterplot(self, tag, y_true, y_pred):
        plot_scatter(y_true, y_pred, str(self.epoch)+"_"+tag)
        return None
    
    def kmeansplot(self, tag, tag2, tag3):
        # tag = "blind_test"
        # tag2 = "y_pred"
        # Import {MISC_DIR}{data_type}.csv
        data_type = str(self.epoch)+"_"+self.sample+"_"+tag+"_"+tag2+"_"+tag3
        y = pd.read_csv(f'{MISC_DIR}{data_type}.csv', header=0)
        plot_kmeans(y, data_type)
        return None