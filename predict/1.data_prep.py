# -------------------------------------------------- #
# Author: Bishal Shrestha
# Last Updated: 5/25/2024
# Description: This script generates gene-gene contact matrices, binned HiC matrices, edge features, and data map for the scHiGex model.
# -------------------------------------------------- #

import os
import time
import json
import math
import torch
import numpy as np
import pandas as pd
import multiprocessing
import scipy.sparse as sp
from sklearn.decomposition import PCA
from safetensors import safe_open
from safetensors.torch import save_file

from scHiCluster import *


def edge_gene_gene_contact_matrices(cell_type, pairs_file):
    print(f"\nGenerating gene-gene contact matrices for {cell_type} | {pairs_file}...")
    pairs_files_location = './pairs'
    gene_definition_location = f"../assets/exclusive_gene_definitions.json"
    gene_index_location = f"../assets/exclusive_gene_index.json"
    chromosome_definition_location = f"../assets/exclusive_chrom_definition/"
    chromosome_shapes_location = f'../assets/chrom_definition_shapes.json'
    gene_gene_contact_matrix_location = f"./contact_matrices/"

    if not os.path.exists(gene_gene_contact_matrix_location): os.makedirs(gene_gene_contact_matrix_location)

    with open(gene_definition_location, "r") as f:
        gene_definitions = json.load(f)

    with open(gene_index_location, "r") as f:
        gene_index = json.load(f)
        
    with open(chromosome_shapes_location, "r") as f:
        chromosome_shapes = json.load(f)
        
    chromosomes = list(gene_definitions.keys())

    chromosome_definitions = {}
    for chromosome in chromosomes:
        chromosome_definition_file = f"{chromosome_definition_location}/chrom_definition_{chromosome}.mmap"
        chromosome_definitions[chromosome] = np.memmap(chromosome_definition_file, dtype=np.int8, mode='r', shape=tuple(chromosome_shapes[chromosome]))


    # Read pairs file names
    sample_name = pairs_file#.split("_")[1].split(".")[0]
    
    # Read the pairs file
    try:
        hic = read_pairs_file(f'{pairs_files_location}/{cell_type}/{pairs_file}')
    except:
        print(f"[ERROR] Sample: {sample_name} > Error reading pairs file")
        return
    
    for chromosome in chromosomes:
        
        count_HiC_contacts_between_genes_in_gene_definition = 0
        count_HiC_contacts_between_genes_not_in_gene_definition = 0
        
        # Create a gene-gene contact matrix for each chromosome
        shape_of_gene_gene_contact_matrix = (len(gene_index[chromosome]), len(gene_index[chromosome]))  
        
        if not os.path.exists(f"{gene_gene_contact_matrix_location}/{cell_type}"): os.makedirs(f"{gene_gene_contact_matrix_location}/{cell_type}")
        gene_gene_contact_matrix_file = f"{gene_gene_contact_matrix_location}/{cell_type}/{sample_name}_{chromosome}.mmap"
        
        if os.path.exists(gene_gene_contact_matrix_file): # Check if the gene-gene contact matrix file already exists
            try: # Load the data to see if the file is corrupted
                gene_gene_contact_matrix_memmap = np.memmap(gene_gene_contact_matrix_file, dtype=np.int32, mode='r', shape=tuple(shape_of_gene_gene_contact_matrix))
                gene_gene_contact_matrix = np.array(gene_gene_contact_matrix_memmap)
                print(f"[COMPLETE] Sample: {sample_name}, Chromosome: {chromosome} > Gene-gene contact matrix file already exists")
                continue
            except:
                print(f"[ERROR] Sample: {sample_name}, Chromosome: {chromosome} > Gene-gene contact matrix file is corrupted")
                os.remove(gene_gene_contact_matrix_file)
        
        gene_gene_contact_matrix = np.zeros(shape_of_gene_gene_contact_matrix, dtype=np.int32)
        
        for _, row in hic.iterrows():
            
            # Check for Interchromosome interactions, only consider intrachromosome interactions
            if (row["chr1"] == row["chr2"] == chromosome) != True:
                continue
            
            position1 = row["pos1"] - 1 # Convert to 0-based index
            position2 = row["pos2"] - 1 # Convert to 0-based index
            
            # Check if genes exists in the position1 and position2.
            if (np.any(chromosome_definitions[row["chr1"]][position1] == 1) &
                np.any(chromosome_definitions[row["chr2"]][position2] == 1)) != True:
                count_HiC_contacts_between_genes_not_in_gene_definition += 1
            else:
                count_HiC_contacts_between_genes_in_gene_definition += 1
                # Identify the genes (columns) in the position1 and position2
                genes_indices_0 = np.where(chromosome_definitions[row["chr1"]][position1] == 1)[0]
                genes_indices_1 = np.where(chromosome_definitions[row["chr2"]][position2] == 1)[0]
                
                # Iterate through each gene pair and increment the gene-gene contact matrix
                for gene_index_0 in genes_indices_0:
                    for gene_index_1 in genes_indices_1:
                    
                        # Increment the gene-gene contact matrix
                        if gene_index_0 == gene_index_1:
                            gene_gene_contact_matrix[gene_index_0, gene_index_1] += 1
                            # continue # Do not consider self-interactions (diagonal elements)
                        else:
                            gene_gene_contact_matrix[gene_index_0, gene_index_1] += 1
                            gene_gene_contact_matrix[gene_index_1, gene_index_0] += 1
                        
        # Export the gene-gene contact matrix to a file using numpy memmap
        gene_gene_contact_matrix_memmap = np.memmap(gene_gene_contact_matrix_file, dtype=np.int32, mode='w+', shape=shape_of_gene_gene_contact_matrix)
        gene_gene_contact_matrix_memmap[:] = gene_gene_contact_matrix[:]
        gene_gene_contact_matrix_memmap.flush()
        
        print(f"[COMPLETE] Sample: {sample_name}, Chromosome: {chromosome} > Gene-gene contact matrix file created")


def metacell_scBinnedHiC():
    print("\nGenerating binned HiC matrices...")
    # Convert into 1000000bp resolution
    chromsome_size_file = "../assets/chrom_size.json"
    res = 1000000
    hash = {}
    l = 0

    with open(chromsome_size_file, 'r') as f:
        chromosome_size = json.load(f)
        
    for chromosome in chromosome_size:
        chrom_bead_size = int(chromosome_size[chromosome] / res) # 1M resolution: bead number
        chrom_bead_index = 0 

        for bead_number in range(l, l + chrom_bead_size): # With in the range of chromosome
            hash[chromosome + " " + str(chrom_bead_index)] = bead_number
            chrom_bead_index += 1
        l += chrom_bead_size
    
    # Open json file
    
    cell_types_sample_index = {}
    cell_types_count = {}
    
    for folder in os.listdir("./pairs/"):
        cell_types_sample_index[folder] = {}
        cell_types_count[folder] = len(os.listdir(f"./pairs/{folder}/"))
        for i, file in enumerate(os.listdir(f"./pairs/{folder}/")):
            sample_name = file.split("_")[1].split(".")[0]
            cell_types_sample_index[folder][str(i)] = [file, file]
    
    with open(f'./cell_types_sample_index.json', 'w') as f:
        json.dump(cell_types_sample_index, f)

    json.dump(cell_types_count, open(f'./cell_types_count.json', 'w'))

    selected_cell_types = list(cell_types_sample_index.keys())

    for cell_type in selected_cell_types:
        
        output_folder = f"./binnedHiC/" + cell_type + "/"
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        for sample_index in cell_types_sample_index[cell_type]:
            sample_info = cell_types_sample_index[cell_type][sample_index]
            sample_file = f"./pairs/{cell_type}/"+sample_info[1]

            hic = read_pairs_file(sample_file)
            hash2 = {}
            
            for _, row in hic.iterrows():
                chr_bead_num_1 = int(int(float(row['pos1'])) / res)
                chr_bead_num_2 = int(int(float(row['pos2'])) / res)

                bead_number_1 = hash.get(row['chr1'] + " " + str(chr_bead_num_1))
                bead_number_2 = hash.get(row['chr2'] + " " + str(chr_bead_num_2))
                if bead_number_1 is not None and bead_number_2 is not None:
                    hash2[str(bead_number_1) + " " + str(bead_number_2)] = hash2.get(str(bead_number_1) + " " + str(bead_number_2), 0) + 1
                    hash2[str(bead_number_2) + " " + str(bead_number_1)] = hash2.get(str(bead_number_2) + " " + str(bead_number_1), 0) + 1

            with open(f'{output_folder}{sample_index}', 'w') as f:
                lines = []
                for key in hash2:
                    item = key.split()
                    if int(item[0]) < int(item[1]):
                        lines.append(key + " " + str(hash2[key]))
                    if int(item[0]) == int(item[1]):
                        hic_value = hash2[key] / 2
                        lines.append(key + " " + str(hic_value))
                f.write('\n'.join(lines))
            


def metacell_run_scHiCluster():
    print("\nRunning scHiCluster...")
    n_nodes = 2716 # total number of nodes with all chromosome for GRCm38

    with open(f'./cell_types_count.json', 'r') as f:
        cell_types_count = json.load(f)

    chrbins = np.loadtxt('../assets/chromosome_binned_sizes_1M.txt', dtype='int')
    nchrs = chrbins.shape[0]

    output_folder = f"./clusters/"
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    for cell_type in cell_types_count:

        dir_HiC = f'./binnedHiC/{cell_type}' # sys.argv[1]#"4cells_1M"
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



def metacell_cal_dist_top_20():
    print("\nCalculating distance of top 20 cells...")
    with open(f'./cell_types_count.json', 'r') as f:
        cell_types_count = json.load(f)

    for cell_type in cell_types_count:
        output_folder = f"./metacell/"
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        file = f'./clusters/{cell_type}' # sys.argv[1]
        with open(file, 'r') as f:
            a = [line.strip() for line in f]

        for cell in a:
            hash = {}
            c = 0
            item = cell.split()
            for cell2 in a:
                ite2 = cell2.split()
                l = len(ite2)
                sum = 0
                for i in range(l):
                    sum += (float(item[i]) - float(ite2[i]))**2
                dist = math.sqrt(sum)
                hash[c] = dist
                c += 1
            
            cc = 0
            with open(f'{output_folder}{cell_type}', 'a') as output_file:
                for key in sorted(hash, key=hash.get):
                    if cc < 21:
                        output_file.write(str(key) + " ")
                    cc += 1
                output_file.write("\n")


def read_pairs_file(pairs_file_path):

    pairs_data = []

    with open(pairs_file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.startswith('columns:'):
                continue
            parts = line.strip().split('\t')
            pairs_data.append(parts)

    # Process the parsed data
    # You can create a DataFrame if needed
    pairs_columns = ['readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1', 'strand2', 'phase0', 'phase1']
    df_hi_c_pairs = pd.DataFrame(pairs_data, columns=pairs_columns)

    # Convert pos1 and pos2 columns to integers
    df_hi_c_pairs['pos1'] = df_hi_c_pairs['pos1'].astype(int)
    df_hi_c_pairs['pos2'] = df_hi_c_pairs['pos2'].astype(int)

    return df_hi_c_pairs


def edge_feature_generation(cell_type, meta_cell):
    print(f"\nGenerating edge features for {cell_type} | {meta_cell[0]}...")
    Hi_C_FOLDER = f'./contact_matrices/'
    OUTPUT_FOLDER = f'./edge_features/'  

    chromosomes = []

    cell_types_sample_index = json.load(open(f'./cell_types_sample_index.json', 'r'))

    gene_indexes = json.load(open(f'../assets/exclusive_gene_index.json', 'r'))
    gene_definitions = json.load(open(f'../assets/exclusive_gene_definitions.json', 'r'))
    chrom_sizes = json.load(open(f'../assets/chrom_size.json', 'r'))
    chromosomes = list(gene_indexes.keys())
    
    meta_cell = meta_cell.strip().split()
            
    for chromosome in chromosomes:
        
        if os.path.exists(f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt'):
            # open to check if the file is corrupted
            try:
                safe_open(f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt', framework="pt", device="cpu")
                print(f"\nFile already exists: {cell_type} | {meta_cell[0]} | {chromosome}")
                continue
            except:
                os.remove(f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt')
                
        
        start = time.time()
        
        hics = {}
        shape = (len(gene_indexes[chromosome]), len(gene_indexes[chromosome])) 
        
        for index in meta_cell:
            name = cell_types_sample_index[cell_type][index][0]+'_'+chromosome+'.mmap'
            hics[index] = np.memmap(f'{Hi_C_FOLDER}{cell_type}/{name}', dtype=np.int32, mode='r', shape=shape)
        
        edge_feats = []
        edge_index = []
        
        for gn, gene_index in gene_indexes[chromosome].items():
            for gn2, gene_index2 in gene_indexes[chromosome].items():
                if gene_index == gene_index2:
                    continue
                
                num_of_cells_with_contacts = 0
                temp_edge_feats = []
                mid_point_of_gn = (gene_definitions[chromosome][gn]["end"] + gene_definitions[chromosome][gn]["start"]) / 2
                mid_point_of_gn2 = (gene_definitions[chromosome][gn2]["end"] + gene_definitions[chromosome][gn2]["start"]) / 2
                genomic_distance = math.log10(abs(mid_point_of_gn - mid_point_of_gn2) + 1e-10)
                for index in meta_cell:
                    contact = hics[index][gene_index][gene_index2]
                    temp_edge_feats.append(contact)
                    if contact > 0: num_of_cells_with_contacts += 1
                
                try:
                    assert len(temp_edge_feats) == 21, f"Length of temp_edge_feats is not 21: {len(temp_edge_feats)} | {cell_type} | {meta_cell[0]} | {chromosome} | {gn} | {gn2}"
                except AssertionError as e:
                    if len(temp_edge_feats) == 20:
                        if cell_type == "Ast" or cell_type == "Oli":
                            temp_edge_feats.append(0)

                assert len(temp_edge_feats) == 21, f"Length of temp_edge_feats is not 21: {len(temp_edge_feats)} | {cell_type} | {meta_cell[0]} | {chromosome} | {gn} | {gn2}"
                
                temp_edge_feats.append(num_of_cells_with_contacts/21)
                temp_edge_feats.append(genomic_distance)
                assert len(temp_edge_feats) == 23, f"Length of temp_edge_feats is not 23: {len(temp_edge_feats)} | {cell_type} | {meta_cell[0]} | {chromosome} | {gn} | {gn2}"
                
                if num_of_cells_with_contacts > 0:
                    edge_feats.append(temp_edge_feats)
                    edge_index.append((gene_index, gene_index2))

        # check how many 
        if len(edge_feats) == 0: continue
        
        assert len(edge_feats) == len(edge_index), f"Length of edge_feats and edge_index do not match: {len(edge_feats)} != {len(edge_index)} | {cell_type} | {meta_cell[0]} | {chromosome}"
        
        tensors = {
            "edge_features": torch.tensor(edge_feats, dtype=torch.float),
            "edge_index": torch.tensor(edge_index, dtype=torch.long)
        }
        
        if not os.path.exists(f'{OUTPUT_FOLDER}/{cell_type}'): os.makedirs(f'{OUTPUT_FOLDER}/{cell_type}')
        save_file(tensors, f'{OUTPUT_FOLDER}/{cell_type}/{meta_cell[0]}_{chromosome}.pt')
        
        with open(f'{OUTPUT_FOLDER}/{cell_type}/list', 'a') as f: f.write(f'{meta_cell[0]}_{chromosome}.pt\n')
        
        print(f"\nTime taken for {cell_type} | {meta_cell[0]} | {chromosome}: {(time.time()-start)/60:.2f} min")


def generate_datamap():
    print("\nGenerating data map...")
    data_map = {}
    cell_types_sample_index = json.load(open(f'./cell_types_sample_index.json', 'r'))
    cell_types = list(cell_types_sample_index.keys())
    for cell_type in cell_types:
    
        loc = os.path.join("./edge_features", cell_type)
        
        try: 
            assert os.path.isdir(loc), f"{cell_type} directory not found!"
        except AssertionError:
            return None
        
        file_list = [x for x in os.listdir(loc) if x != "list"]
        file_list.sort()
        with open(os.path.join(loc, "list"), "r") as f: list_f = sorted(f.read().splitlines())
        
        assert file_list == list_f, f"list file does not match the files in {loc}!"
        print(f"Total files in {cell_type}: {len(file_list)}")
        
        for idx, file in enumerate(file_list):
            chr = file.split("_")[1].split(".")[0]
            sample_num = file.split("_")[0]
            sample_name = cell_types_sample_index[cell_type][sample_num]
            file_list[idx] = {'path': os.path.join(loc, file), "chr": chr, "sample_num": sample_num,"sample_name": sample_name, "cell_type": cell_type}
        
        data_map[cell_type] = file_list
    json.dump(data_map, open('./data_map.json', 'w'))

if __name__ == "__main__":
    
    # Generate gene-gene contact matrices
    pairs_files_location = './pairs'
    cell_types = [filename for filename in os.listdir(pairs_files_location)]
    for cell_type in cell_types:
        pairs_files = [filename for filename in os.listdir(f"{pairs_files_location}/{cell_type}")]
        pool = multiprocessing.Pool(processes=25)
        pool.starmap(edge_gene_gene_contact_matrices, [(cell_type, pairs_file) for pairs_file in pairs_files])
        pool.close()
        pool.join()

    # Generate binned HiC matrices
    metacell_scBinnedHiC()
    
    # Run scHiCluster
    metacell_run_scHiCluster()
    
    # Calculate distance of top 20 cells
    metacell_cal_dist_top_20()
    
    # Generate edge features
    for cell_type in cell_types:
        with open(f'./metacell/{cell_type}', 'r') as f:
            meta_cells = f.read().splitlines()
        pool = multiprocessing.Pool(processes=25)
        pool.starmap(edge_feature_generation, [(cell_type, meta_cell) for meta_cell in meta_cells])
        pool.close()
        pool.join()
    
    # Generate data map
    generate_datamap()
        
    