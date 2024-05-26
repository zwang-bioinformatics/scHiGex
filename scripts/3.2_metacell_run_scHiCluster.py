# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import json
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from scHiCluster import *

print(f"Process ID: {os.getpid()}")

n_nodes = 2716 # total number of nodes with all chromosome for GRCm38

with open(f'../assets/cell_types_count.json', 'r') as f:
	cell_types_count = json.load(f)

chrbins = np.loadtxt('../assets/chromosome_binned_sizes_1M.txt', dtype='int')
nchrs = chrbins.shape[0]

output_folder = f"../assets/clusters/"
if not os.path.exists(output_folder): os.makedirs(output_folder)

for cell_type in cell_types_count:

	dir_HiC = f'../assets/binnedHiC/{cell_type}' # sys.argv[1]#"4cells_1M"
	numbers = cell_types_count[cell_type] # int(sys.argv[2]) # number of cells
	cells = []
	hics = []
	
	for line in range(0,numbers):
		edges = np.loadtxt(dir_HiC+"/"+str(line))
		adj = sp.csr_matrix((edges[:,2], (np.int_(edges[:, 0]), np.int_(edges[:, 1]))), shape=(n_nodes, n_nodes))
		adj = adj + adj.T - sp.diags(adj.diagonal())	# make the matrix symmetric
		hics.append(adj)

	pcn1 = 40
	ncells = numbers

	# for each chrs
	#pcaChrs = np.zeros((ncells, pcn1*nchrs))

	pcaChrs = []

	for k in range(nchrs):
		# for each cells
		n = chrbins[k][1] - chrbins[k][0] # number of bins in the chromosome
		if n == 0:	# ignore chrM
			continue
		X = np.zeros((ncells, n*n))
		for i in range(ncells):
			celli = hics[i].toarray()
			cellik = celli[chrbins[k][0]:chrbins[k][1], chrbins[k][0]:chrbins[k][1]]+1 # 1 added so log2(1) = 0 but log2(0) = -inf (part of scHiCluster.py)	
			bi = scHiCluster(cellik) # returns 1D array of length n*n with 1s and 0s (True and False)
			X[i,:] = bi

		# Perform PCA and see how many dimensions are needed to keep 80% of the variance
		print(X.shape)
		ncomp = int(min(X.shape)) #int(min(X.shape) * 0.2) - 1 
		if ncomp > pcn1: ncomp = pcn1

		pca = PCA(n_components=ncomp)

		X2 = pca.fit_transform(X) # ncells x ncomp
  
		print(f"Cumulative Variance explained by {ncomp} dimensions: {pca.explained_variance_ratio_[:pcn1].sum()}")

		pcaChrs.append(X2)

	# Concatenate the PCA of each chromosome and perform PCA again to get the final vector for each cell
	pcaChrs = np.concatenate(pcaChrs, axis=1) # ncells x (pcn1*nchrs)
	print(pcaChrs.shape)
	pca2 = PCA(n_components=min(pcaChrs.shape) - 1)
	pcaChrs = pca2.fit_transform(pcaChrs)
	np.savetxt(f"{output_folder}{cell_type}",pcaChrs,fmt='%1.5f')

# nohup python -u run_scHiCluster.py > run_scHiCluster.log 2>&1 &
# pkill -f run_scHiCluster.py
# pgrep -f run_scHiCluster.py

