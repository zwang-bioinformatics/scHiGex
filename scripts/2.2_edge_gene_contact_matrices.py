# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import multiprocessing

print("Process ID: ", os.getpid())

pairs_files_location = '../assets/pairs'
gene_definition_location = f"../assets/exclusive_gene_definitions.json"
gene_index_location = f"../assets/exclusive_gene_index.json"
rna_umicount_location = f'../assets/rna_umicount.tsv'
chromosome_definition_location = f"../assets/exclusive_chrom_definition/"
chromosome_shapes_location = f'../assets/chrom_definition_shapes.json'
gene_gene_contact_matrix_location = f"../assets/contact_matrices/"

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
pairs_files = [filename for filename in os.listdir(pairs_files_location)]
print(f"Number of pairs files: {len(pairs_files)}")

# Read the RNA UMI count file
rna_umicount_df = pd.read_csv(rna_umicount_location, delimiter='\t', header=0)
print(f"RNA UMI count shape: {rna_umicount_df.shape}")

# Sample names from the RNA UMI count file
sample_names = rna_umicount_df.columns[1:]

# Check if the sample names in UMI count have corresponding pairs files
selected_pairs_files = [pair_file for pair_file in pairs_files if any(sample_name in pair_file for sample_name in sample_names)]
print(f"Number of selected pairs files: {len(selected_pairs_files)}")
if (len(sample_names) != len(selected_pairs_files)):
    print(f"ERROR: Number of sample names in UMI count and HiC pairs files do not match.")
    sys.exit()

def main(pairs_file):
    
    sample_name = pairs_file.split("_")[1].split(".")[0]
    
    # Read the pairs file
    try:
        hic = read_pairs_file(f'{pairs_files_location}/{pairs_file}')
    except:
        print(f"[ERROR] Sample: {sample_name} > Error reading pairs file")
        return
    
    for chromosome in chromosomes:
        
        start_time = time.time()
        
        count_HiC_contacts_between_genes_in_gene_definition = 0
        count_HiC_contacts_between_genes_not_in_gene_definition = 0
        
        # Create a gene-gene contact matrix for each chromosome
        shape_of_gene_gene_contact_matrix = (len(gene_index[chromosome]), len(gene_index[chromosome]))  
        
        gene_gene_contact_matrix_file = f"{gene_gene_contact_matrix_location}/{sample_name}_{chromosome}.mmap"
        
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
        
        # print(f"[COMPLETE] Sample: {sample_name}, Chromosome: {chromosome} > {(time.time() - start_time) / 60} minutes | # contacts in gene definition: {count_HiC_contacts_between_genes_in_gene_definition} | # no contacts not in gene definition: {count_HiC_contacts_between_genes_not_in_gene_definition}")

        

def read_pairs_file(pairs_file_path):

    pairs_data = []

    with open(pairs_file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.startswith('columns:'):
                continue
            parts = line.strip().split('\t')
            pairs_data.append(parts)

    # Process the parsed data into a DataFrame
    pairs_columns = ['readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1', 'strand2', 'phase0', 'phase1']
    df_hi_c_pairs = pd.DataFrame(pairs_data, columns=pairs_columns)

    # Convert pos1 and pos2 columns to integers
    df_hi_c_pairs['pos1'] = df_hi_c_pairs['pos1'].astype(int)
    df_hi_c_pairs['pos2'] = df_hi_c_pairs['pos2'].astype(int)

    return df_hi_c_pairs



if __name__ == "__main__":
    # pool = multiprocessing.Pool(processes=20)
    # pool.starmap(main, [(pairs_file) for pairs_file in selected_pairs_files])
    for pairs_file in selected_pairs_files:
        main(pairs_file)
    
# nohup python -u HiC_generate_genexgene_contact_matrix.py > output/HiC_generate_genexgene_contact_matrix_brain.log &