# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import umap
import torch
import imageio
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch_geometric.utils as utils
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from natsort import natsorted

from utils import *

FIGURE_DIR = "../figures/"
MISC_DIR = "../misc/"

sns.set_theme(style='white', font='sans-serif', font_scale=1, color_codes=True, rc={'font.size': 12})

def plot_graph(data, filename, layout_func=nx.kamada_kawai_layout, node_size=25, node_color_positive='blue', node_color_negative='red', edge_color='k', dpi=300, alpha=1):
    data.edge_index = data.edge_index.transpose(0, 1)
    G = utils.to_networkx(data, to_undirected=True)
    color_map = [node_color_positive if gene_exp > 0 else node_color_negative for gene_exp in data.gene_exp]

    plt.figure(1, figsize=(12, 12), dpi=dpi)
    pos = layout_func(G)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=color_map, alpha=alpha)
    nx.draw_networkx_edges(G, pos, width=0.1, edge_color=edge_color)
    plt.axis('off')
    plt.savefig(f"{FIGURE_DIR}{filename}")
    plt.close()

    return filename

def plot_graphs_and_create_gif(train_loader, n_graphs=10):
    filenames = []
    for i, data in enumerate(train_loader):
        if i >= n_graphs: break
        data = data.to('cpu')
        filename = plot_graph(data, f'{FIGURE_DIR}graph_{i}.png')
        filenames.append(filename)

    with imageio.get_writer(f'{FIGURE_DIR}graphs.gif', mode='I', duration=1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames: os.remove(f"{FIGURE_DIR}{filename}")
    

def plot_absolute_difference_boxplot_for_chromosomes(y_true, y_pred, chrom, data_type):
    # chrom is a numpy array of chromosome string like [chr1, chr1, chr2, ..]
    print("[Start] Absolute Difference Boxplot for Chromosomes.")
    
    # Clip the values to be between 0 and 1
    y_true = torch.clamp(y_true, min=0, max=1)
    y_pred = torch.clamp(y_pred, min=0, max=1)
    
    concatenated_true_values, concatenated_predicted_values = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    
    unique_chrom = np.unique(chrom)
    unique_chrom = natsorted(np.unique(chrom))
    
    unique_chrom_dict = {chrom: i for i, chrom in enumerate(unique_chrom)}
    
    # Create an empty list to store absolute differences for each group
    abs_diff_values = [[] for _ in range(len(unique_chrom))]
    
    for true_val, pred_val, c in zip(concatenated_true_values, concatenated_predicted_values, chrom):
        abs_diff_values[unique_chrom_dict[c]].append(np.abs(true_val - pred_val))
        
    labels = [f"{chrom}" for chrom in unique_chrom]
    
    # set theme
    plt.figure(figsize=(20, 4))
    sns.boxplot(data=abs_diff_values, showfliers=False, whis=0, color='royalblue', medianprops={'color': 'coral'})
    plt.xticks(np.arange(len(labels)), labels)#, rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # get the max of 75th percentile value to set the y-axis limit
    y_75th_percentile = np.max([np.percentile(abs_diff_values[i], 75) for i in range(len(abs_diff_values))])
    # max = np.max([np.max(abs_diff_values[i]) for i in range(len(abs_diff_values))])
    if y_75th_percentile > 1: 
        max = y_75th_percentile + 0.1
        min = 0 - 0.1
    else: 
        max = y_75th_percentile + 0.02
        min = 0 - 0.005
    plt.ylim(min, max)
    plt.savefig(f"{FIGURE_DIR}{data_type}_boxplot_chromosomes.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    print("[Complete] Absolute Difference Boxplot for Chromosomes.")
    


def plot_absolute_difference_boxplot_manual(y_true, y_pred, data_type, return_mean=False):
    """
    Plot the absolute difference boxplot of the true and predicted values.
    Performs grouping manually: 0, (0-0.1], (0.1-0.2], ..., (0.9-1.0), 1

    Parameters
    ----------
    true_values : tensor
        tensor list of true values.
    predicted_values : tensor
        tensor list of predicted values.
    Returns
    -------
    None
    """
    if(not return_mean):
        print("[Start] Absolute Difference Boxplot (Manual).")
    # -----------------
    # Data preparation
    # -----------------
    
    # Clip the values to be between 0 and 1
    y_true = torch.clamp(y_true, min=0, max=1)
    y_pred = torch.clamp(y_pred, min=0, max=1)
    
    concatenated_true_values, concatenated_predicted_values = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    
    # Define the ranges for grouping
    value_ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Create an empty list to store absolute differences for each group
    abs_diff_values = [[] for _ in range(12)]

    # Group the data and calculate absolute differences for each group
    for true_val, pred_val in zip(concatenated_true_values, concatenated_predicted_values):
        for i in range(len(value_ranges)-1):
            if (true_val == 1.0 or true_val == 1):
                abs_diff_values[-1].append(np.abs(true_val - pred_val))
                break
            if (true_val == 0.0 or true_val == 0):
                abs_diff_values[0].append(np.abs(true_val - pred_val))
                break
            if value_ranges[i] < true_val <= value_ranges[i + 1]:
                abs_diff_values[i+1].append(np.abs(true_val - pred_val))
                break
    labels = [f"({value_ranges[i]:.1f}-{value_ranges[i+1]:.1f}]" for i in range(len(value_ranges) - 1)]
    labels.insert(0, "0")
    labels.append("1")
    labels[-2] = "(0.9-1)"
    
    # -----------------
    # Plotting
    # -----------------
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(abs_diff_values, labels=labels, showfliers=False)
    # plt.xlabel('True Gene Expression Values')
    # plt.ylabel('Absolute Difference')
    # plt.title('Absolute Difference between True and Predicted Gene Expressions by Ranges of True Values')
    # plt.grid(True)
    # plt.show()
    # plt.savefig(f"{FIGURE_DIR}{data_type}_boxplot.png")
    # plt.close()
    
    plt.figure(figsize=(20, 4))
    plt.rcParams.update({'font.size': 12})  # Increase the text size
    sns.boxplot(data=abs_diff_values, showfliers=False, whis=0, color='royalblue', medianprops={'color': 'coral'})
    plt.xticks(np.arange(len(labels)), labels)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylim(0 - 0.1, 1 + 0.1)
    # plt.axhline(0.5, color='r', linestyle='loosely dashed', linewidth=0.8)
    plt.savefig(f"{FIGURE_DIR}{data_type}_boxplot_sns.png", dpi=600, bbox_inches='tight')
    plt.close()

    print("[Complete] Absolute Difference Boxplot (Manual).")
    
    # -----------------
    # Calculate mean of absolute differences for each group and its mean again
    # -----------------
    mean_abs_diff_values = [np.mean(abs_diff_values[i]) if abs_diff_values[i] else np.nan for i in range(len(abs_diff_values))]
    print("Mean Absolute difference values of each ranges: ", mean_abs_diff_values)
    mean_of_mean_abs_diff_values = np.nanmean(mean_abs_diff_values)
    print("Mean of mean absolute difference values: ", mean_of_mean_abs_diff_values)
    return mean_of_mean_abs_diff_values


def plot_density(y, data_type):
    """
    Plot the probability density plot of the true or pred values
    Y axis in percentage
    X axis will be the true values between 0 to 1 using a kernel density estimate (KDE).

    Parameters
    ----------
    values : list
        List of values.
    data_type : str
        Type of values, e.g., true or pred.
    
    Returns
    -------
    None
    """

    concatenated_values = y.cpu().detach().numpy()
    plt.figure(figsize=(6.8, 4))
    sns.kdeplot(concatenated_values, fill=True, alpha=0.8, warn_singular=False)
    # plt.xlabel(f'{data_type}')
    # plt.ylabel('Density')
    # plt.title(f'{data_type} Density')
    plt.gca().set_ylabel('')  # This line removes the y-axis label
    plt.xlim(min(concatenated_values), max(concatenated_values))  # Set x-axis limits dynamically # plt.xlim(0, 1)
    plt.xlim(-0.05, 1.05)  # Set x-axis limits dynamically # plt.xlim(0, 1)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(False)
    # plt.legend()
    plt.savefig(os.path.join(f"{FIGURE_DIR}{data_type}_density.png"), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[Complete] {data_type} Density Plot.")
    

def plot_histogram(y, data_type):
    """
    Plot the histogram of the true or pred values
    Y axis in percentage
    X axis will be the true values between 0 to 1

    Parameters
    ----------
    values : list
        List of values.
    data_type : str
        Type of values, e.g., true or pred.
    
    Returns
    -------
    None
    """
    concatenated_values = y.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(concatenated_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel(f'{data_type}')
    plt.ylabel('Frequency')
    plt.title(f'{data_type} Histogram')
    plt.xlim(min(concatenated_values), max(concatenated_values))  # Set x-axis limits dynamically # plt.xlim(0, 1)
    plt.savefig(os.path.join(f"{FIGURE_DIR}{data_type}_histogram.png"))
    plt.show()
    plt.close()
    print(f"[Complete] {data_type} Histogram.")


def plot_cluster_gene_exp(rna_umicount_df, data_type, algorithm='umap'):

    rna_umicount_df = rna_umicount_df.transpose()

    # Now, the row is a sample name, and the columns are the gene names. Each sample is a vector of gene expression. Each samle is a point in the high dimensional space. I want to plot UMAP using umap-learn.
    reducer = umap.UMAP(random_state=42) if algorithm == 'umap' else TSNE(random_state=42)
    rna_umicount_embedding = reducer.fit_transform(rna_umicount_df)

    # Plot the UMAP
    # The color of each point is the cell type. I have a file that contains the cell type of each sample.
    metadata_df = getmetadata()
    metadata_df = metadata_df[['Cellname', 'Cell type']]
    metadata_df = metadata_df.set_index('Cellname')
    rna_celltype_list = []
    # The order of the samples in rna_umicount_df and rna_celltype_df may be different. I need to reorder the rows of rna_celltype_df to match the order of rna_umicount_df so the colors match. Only select the ones that is in rna_umicount_df
    for index, row in rna_umicount_df.iterrows():
        assert index in metadata_df.index, f'{index} is not in metadata_df'
        rna_celltype_list.append(metadata_df.loc[index, 'Cell type'])

    unique_cell_types = sorted(list(set(metadata_df['Cell type'])))

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(rna_umicount_embedding, columns=['X', 'Y'])
    plot_df['Cell type'] = rna_celltype_list
    rna_umicount_df['Cell type'] = rna_celltype_list
    
    # Save the plot_df as a csv file
    plot_df.to_csv(f"{MISC_DIR}{data_type}_UMAP.csv", index=False)
    rna_umicount_df.to_csv(f"{MISC_DIR}{data_type}_all.csv", index=False)

    # Filter the plot_df to only include the selected cell types
    # selected_cell_types = ['In1', 'Oli']
    # plot_df = plot_df[plot_df['Cell type'].isin(selected_cell_types)]

    centroids = plot_df.groupby('Cell type').mean()
    
    # Assuming unique_cell_types is a list of your unique cell types
    num_colors = len(unique_cell_types)
    # Create a color map
    color_map = plt.cm.get_cmap('Set3', num_colors)
    # Generate colors
    palette = [color_map(i) for i in np.linspace(0, 1, num_colors)]

    
    # palette = sns.color_palette(n_colors=len(unique_cell_types))
    color_dict = dict(zip(unique_cell_types, palette))

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})  # Increase the text size
    sns.scatterplot(data=plot_df, x='X', y='Y', hue='Cell type', palette=color_dict, s=10)

    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    
    # for cell_type, (x, y) in centroids.iterrows():
    #     plt.text(x, y, cell_type, fontsize=5, ha='center', color='white', 
    #             bbox=dict(facecolor=color_dict[cell_type], edgecolor='black', boxstyle='round,pad=0.5'))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18) # Place the legend to the right of the figure

    plt.xticks([])
    plt.yticks([])

    plt.savefig(f"{FIGURE_DIR}{data_type}_UMAP.png", dpi=600, bbox_inches='tight')



def plot_roc_auc(y_true, y_pred, data_type):
    """
    Plot the ROC AUC curve for the true and predicted values

    Parameters
    ----------
    y_true : tensor
        tensor list of true values.
    y_pred : tensor
        tensor list of predicted values.
    data_type : str
        Type of values, e.g., true or pred.
    
    Returns
    -------
    None
    """
    y_true = cont_to_cat(y_true, "true")
    fpr, tpr, _ = roc_curve(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    roc_auc = auc(fpr, tpr)
    
    with open(f"{MISC_DIR}{data_type}_roc_auc.pkl", 'wb') as f:
        pickle.dump({"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}, f)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC AUC for {data_type}')
    plt.legend(loc="lower right")
    plt.savefig(f"{FIGURE_DIR}{data_type}_ROC_AUC.png")
    plt.show()
    plt.close()
    print(f"[Complete] {data_type} ROC AUC.")
    
def plot_pr_auc(y_true, y_pred, data_type):
    """
    Plot the average precision recall curve for the true and predicted values

    Parameters
    ----------
    y_true : tensor
        tensor list of true values.
    y_pred : tensor
        tensor list of predicted values.
    data_type : str
        Type of values, e.g., true or pred.
    
    Returns
    -------
    None
    """
    y_true = cont_to_cat(y_true, "true")
    precision, recall, _ = precision_recall_curve(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    pr_auc = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR AUC for {data_type}')
    plt.legend(loc="lower right")
    plt.savefig(f"{FIGURE_DIR}{data_type}_PR_AUC.png")
    plt.show()
    plt.close()
    print(f"[Complete] {data_type} PR AUC.")
    
    
def plot_scatter(y_true, y_pred, data_type):
    """
    Plot the scatter plot of the true and predicted values

    Parameters
    ----------
    y_true : tensor
        tensor list of true values.
    y_pred : tensor
        tensor list of predicted values.
    data_type : str
        Type of values, e.g., true or pred.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), color='blue', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'Scatter Plot for {data_type}')
    plt.savefig(f"{FIGURE_DIR}{data_type}_scatter.png")
    plt.show()
    plt.close()
    print(f"[Complete] {data_type} Scatter Plot.")
    

def plot_kmeans(y, data_type):
    """
    Perform kmeans clustering using sklearn.cluster KMeans and plot the scatter plot of the true or pred values.
    

    Parameters
    ----------
    y : dataframe
        dataframe with columns X,Y,Cell type
    data_type : str
        Type of values, e.g., true or pred.
    
    Returns
    -------
    None
    """
    # Remove cell type column
    X = y.drop('Cell type', axis=1)
    y = y[['Cell type']]
    print(f"[Start] {data_type} KMeans Clustering.")
    # The number of cluster should be counted from the number of unique values in the cell type column
    unique_cell_types = y['Cell type'].unique()
    n_clusters = len(unique_cell_types)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init='auto')
    kmeans.fit(X)
    pred = kmeans.predict(X)
    y['KMeans Cluster'] = pred
    
    # Assign the most frequent KMeans cluster label to each cell type
    cell_type_to_cluster = {}
    for cell_type in unique_cell_types:
        cell_type_data = y[y['Cell type'] == cell_type]
        most_common_cluster, _ = Counter(cell_type_data['KMeans Cluster']).most_common(1)[0]
        cell_type_to_cluster[cell_type] = most_common_cluster
    print(cell_type_to_cluster)
    y['Cell type Encoded'] = y['Cell type'].map(cell_type_to_cluster)
    
    # Calculate the Adjusted rand index(ARI) between the true cell type (Cell type Encoded) and the kmeans cluster label and put it as a column
    y['ARI'] = adjusted_rand_score(y['Cell type Encoded'], y['KMeans Cluster'])
    
    # Calculate the silhouette score
    y['silhouette_score'] = silhouette_score(X, y['KMeans Cluster'], metric='euclidean')
    
    # Save the kmeans cluster to a csv file
    y.to_csv(f"{MISC_DIR}{data_type}_kmeans.csv", index=False)
    
    # Plot the scatter plot
    # check if X column exists 
    X['KMeans Cluster'] = y['KMeans Cluster']
    if 'X' in X.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=X, x='X', y='Y', hue='KMeans Cluster', palette='viridis', s=10)
        plt.title(f'KMeans Clustering for {data_type}')
        plt.savefig(f"{FIGURE_DIR}{data_type}_KMeans.png")
        plt.show()
        plt.close()
    print(f"[Complete] {data_type} KMeans Clustering.")
    
    
    