# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import json
import pickle
import numpy as np
from sklearn.decomposition import PCA

sample_type = "brain"
gene_index = json.load(open(f'../assets/exclusive_gene_index.json', 'r'))
chromosomes = gene_index.keys()

# Step 1
node_features_list = []

for chromosome in chromosomes:
    gene_embedding_filename = f'../assets/exclusive_dnabert2/dnabert2_{chromosome}.npy'
    gene_embedding = np.load(gene_embedding_filename, allow_pickle=True)
    chromHMM_filename = f'../assets/exclusive_chromHMM/chromHMM_{chromosome}.npy'
    chromHMM = np.load(chromHMM_filename, allow_pickle=True)

    node_features = np.concatenate((gene_embedding, chromHMM), axis=1)
    
    # Step 2
    node_features_list.append(node_features)

# Step 3
all_node_features = np.concatenate(node_features_list, axis=0)

# Step 4
pca = PCA(n_components=100, random_state=42)
pca.fit(all_node_features)
all_node_features_pca = pca.transform(all_node_features)

# Save the model
with open('../assets/node_pca_model.pkl', 'wb') as f: pickle.dump(pca, f)

# Cumulative explained variance
cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
# Print cumulative explained variance along with the component index
for i, variance in enumerate(cum_explained_variance, start=1):
    print(f"{i}: {variance}", end=",\t")
print()

# Step 5
start_index = 0
for chromosome, node_features in zip(chromosomes, node_features_list):
    end_index = start_index + node_features.shape[0]
    node_features_pca = all_node_features_pca[start_index:end_index]
    start_index = end_index

    if not os.path.exists('../assets/node_embeddings'): os.makedirs('../assets/node_embeddings')
    pca_node_features_filename = f'../assets/node_embeddings/pca_node_features_{chromosome}.npy'
    np.save(pca_node_features_filename, node_features_pca)
    print(f"Saved PCA node features for chromosome {chromosome}!")


# Step 6: Validate the shape of new pca node features with the original node features
for chromosome in chromosomes:
    gene_embedding_filename = f'../assets/exclusive_dnabert2/dnabert2_{chromosome}.npy'
    gene_embedding = np.load(gene_embedding_filename, allow_pickle=True)
    chromHMM_filename = f'../assets/exclusive_chromHMM/chromHMM_{chromosome}.npy'
    chromHMM = np.load(chromHMM_filename, allow_pickle=True)
    
    node_features = np.concatenate((gene_embedding, chromHMM), axis=1)
   
    pca_node_features_filename = f'../assets/node_embeddings/pca_node_features_{chromosome}.npy'
    pca_node_features = np.load(pca_node_features_filename, allow_pickle=True)
    
    print(f"\nNode features shape for chromosome {chromosome}: {node_features.shape}")
    print(f"PCA node features shape for chromosome {chromosome}: {pca_node_features.shape}")

    assert node_features.shape[0] == pca_node_features.shape[0], "Number of samples mismatch!"

    print(f"Validation successful for chromosome {chromosome}!")
    
