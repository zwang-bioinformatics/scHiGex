# -------------------------------------------------- #
# Author: Bishal Shrestha
# Last Updated: 5/25/2024
# -------------------------------------------------- #

from Bio import SeqIO
import os
import json
import sys
import numpy as np
import pandas as pd

from gtfparse import read_gtf


# --------------------------------------------------------
# Preparing the gene definitions: start and end positions of genes
# Ouput: exclusive_gene_definitions.json
# --------------------------------------------------------
def remove_rows_with_all_zeros(df):
    # Select all columns except the first one
    columns_to_check = df.columns[1:]

    # Check if all values in each row (excluding the first column) are equal to 0
    rows_with_all_zeros = df[df[columns_to_check].eq(0).all(axis=1)]

    if not rows_with_all_zeros.empty:
        print("Shape of all_zero_result: ", rows_with_all_zeros.shape)
        print("Rows with all 0 values (excluding the first column):")
        # print(rows_with_all_zeros)
        # Create a new DataFrame without the rows with all zeros
        df_no_zeros = df.drop(rows_with_all_zeros.index)
        print("Shape of df_no_zeros: ", df_no_zeros.shape)
        print("Expected shape of new no zero df: ", (df.shape[0] - rows_with_all_zeros.shape[0], df.shape[1]))
        return df_no_zeros
    else:
        print("No rows with all 0 values (excluding the first column).")
        return df
    
def generate_exclusive_gene_definitions():
    """
    Generate filtered gene definitions (contains start and end position)
    Only use genes that are present in the rna_umicount data
    """
    print("\Generate exclusive gene definitions (contains start and end position)")
    with open('../assets/gene_definitions_vM23.json', 'r') as json_file: gene_definitions = json.load(json_file)
    
    rna_umicount_file_path = f'../assets/rna_umicount.tsv'  # Read the rna umicount data

    rna_umicount_df = pd.read_csv(rna_umicount_file_path, delimiter='\t', header=0) # Read the TSV file into a DataFrame with the first row as column headers
    
    new_rna_umicount_df = remove_rows_with_all_zeros(rna_umicount_df)
    required_gene_list = list(new_rna_umicount_df['gene'])

    chromosomes = list(gene_definitions.keys())
    gene_definitions_filtered = {}
    genes_filtered = []
    for chromosome in chromosomes:
        print("\n> Chromosome: ", chromosome)
        genes = gene_definitions[chromosome].keys()
        gene_definitions_filtered[chromosome] = {}
        print("\tTotal genes: ", len(genes))
        for gene in genes:
            if gene in required_gene_list:
                gene_definitions_filtered[chromosome][gene] = gene_definitions[chromosome][gene]
                genes_filtered += [gene]
        print("\tTotal genes after filter: ", len(gene_definitions_filtered[chromosome].keys()))
    
    print(f"Total required genes: {len(required_gene_list)}")
    print(f"Total genes after filtered: {len(genes_filtered)}")
    if(len(required_gene_list) != len(genes_filtered)):
        print("[ERROR] Gene filtering error")
        print("There are genes in the rna_umicount that does not exist in the gene definitons from gencode.vM10.annotation.gtf")
        # List out the genes that exists in required_gene_list but does not exist in the gene_definitions_filtered
        genes_not_found_in_gene_definitions = []
        for chromosome in chromosomes:
            genes_not_found_in_gene_definitions += list(set(required_gene_list) - set(genes_filtered))
            print(f"Genes not found in gene definitions: {genes_not_found_in_gene_definitions}")
        sys.exit()
    
    # Export gene_definitions to a JSON file
    output_file = f"../assets/exclusive_gene_definitions.json"
    with open(output_file, 'w') as json_file: json.dump(gene_definitions_filtered, json_file, indent=4)

    print(f"Exclusive Gene definitions exported to {output_file}")

# --------------------------------------------------------

def generate_exclusive_gene_index():
    """
    Create a gene index (contains index of genes in its order)
    """
    print("\nFiltering gene index (contains index of genes in its order)")

    # Read the gene definition json file
    with open(f'../assets/exclusive_gene_definitions.json', 'r') as json_file: gene_definitions = json.load(json_file)

    # Create a dictionary to store chromosome-specific gene dictionaries
    gene_index = {}

    # Iterate over each chromosome
    for chromosome, gene_dict in gene_definitions.items():
        gene_index_dict = {}  # Dictionary to store gene name to index mapping
        genes = list(gene_dict.keys())
        
        # Populate gene_index_dict with gene names and their corresponding indices
        for index, gene_name in enumerate(genes):
            gene_index_dict[gene_name] = index
        
        # Add gene_index_dict to chromosome_gene_dicts
        gene_index[chromosome] = gene_index_dict

    # Export chromosome_gene_dicts to a JSON file
    output_file = f"../assets/exclusive_gene_index.json"
    with open(output_file, 'w') as json_file: json.dump(gene_index, json_file, indent=4)
    print(f"Exclusive Gene index exported to {output_file}")

# --------------------------------------------------------

def generate_exclusive_geneome_assembly_indexed():
    """
    Filtering genome assembly sequence index ['AATGC...', 'ATGC...', ..., 'ATGC..']. Here, the number of index for each chromosome depends upon the filtered genes with respect to length.
    Required for generating the dnaBERT2 embeddings.
    """
    print("\nFiltering genome assembly sequence index ['AATGC...', 'ATGC...', ..., 'ATGC..']")

    # Path to the mm10.fa file
    genome_file = "../assets/mm10.fa"

    # Read the genome sequence file
    genome_sequences = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))

    # Read the gene index json file
    with open(f'../assets/exclusive_gene_index.json', 'r') as json_file: gene_index = json.load(json_file)

    # Read the gene definition json file
    with open(f'../assets/exclusive_gene_definitions.json', 'r') as json_file: gene_definitions = json.load(json_file)

    chromosomes = list(gene_index.keys())

    gene_assembly_indexed = {}

    for chromosome in chromosomes:

        # Get the genome sequence for the chromosome
        chromosome_genome_sequence = genome_sequences[chromosome]
        
        gene_assembly_indexed[chromosome] = list(gene_index[chromosome].keys())

        # Checking if the index is in correct order
        for i, gene in enumerate(gene_assembly_indexed[chromosome]):
            if (i != gene_index[chromosome][gene]):
                print("[ERROR] Indexing mismatch")
                sys.exit()

        for i, gene in enumerate(gene_assembly_indexed[chromosome]):
            gene_start_index = gene_definitions[chromosome][gene]["start"] - 1
            gene_end_index = gene_definitions[chromosome][gene]["end"] - 1

            # Check if the desired chromosome is in the file
            if chromosome in genome_sequences:
                gene_assembly_indexed[chromosome][i] = str(chromosome_genome_sequence.seq[gene_start_index:gene_end_index+1])
            else:
                print(f"[ERROR] Chromosome {chromosome} not found in the file. (mm10.fa)")
                sys.exit()

        if not os.path.exists('../assets/exclusive_gene_assembly'): os.makedirs('../assets/exclusive_gene_assembly')
        np.save(f'../assets/exclusive_gene_assembly/exclusive_gene_assembly_{chromosome}.npy', np.array(gene_assembly_indexed[chromosome]))

    print("Indexing of gene assembly complete!!!")

# --------------------------------------------------------

def generate_exclusive_chrom_definitions():
    print("\nGenerating exclusive chromosome definitions")
    flank_size = 0

    # Read the chrom size json file
    with open('../assets/chrom_size.json', 'r') as json_file:
        chrom_size = json.load(json_file)

    # Read the gene definition json file
    with open(f'../assets/exclusive_gene_definitions.json', 'r') as json_file:
        gene_definitions = json.load(json_file)
        
    # Read the gene index json file
    with open(f'../assets/exclusive_gene_index.json', 'r') as json_file:
        gene_index = json.load(json_file)

    chromosomes = list(chrom_size.keys())

    for chromosome in chromosomes:
        
        number_of_genes_on_current_chromosome = len(gene_definitions[chromosome])
        shape_of_chrom_definition = (chrom_size[chromosome], number_of_genes_on_current_chromosome)
        chrom_definition = np.zeros(shape=shape_of_chrom_definition, dtype=np.int8)

        for gene in gene_definitions[chromosome]:
            current_gene_index = gene_index[chromosome][gene]
            
            start_index = gene_definitions[chromosome][gene]["start"] - 1 - flank_size  # Subtract flank_size from start_index
            end_index = gene_definitions[chromosome][gene]["end"] - 1 + flank_size  # Add flank_size to end_index
            start_index = max(0, start_index)  # Ensure start_index is not less than 0
            end_index = min(chrom_size[chromosome] - 1, end_index)  # Ensure end_index does not exceed chromosome size
            
            gene_range = range(start_index, end_index + 1)
            
            # For the respective chrom_range, insert 1 to respective gene index. This means that the gene exists in the respective chrom_range
            chrom_definition[gene_range, current_gene_index] = 1   # Vectorized operation

        # ---------------------------
        # Save the chrom_definition shape information
        # ---------------------------
        # Initialize the folder locations
        folder_location = f'../assets/exclusive_chrom_definition'
        chrom_definition_file = f'{folder_location}/chrom_definition_{chromosome}.mmap'
        
        # Create the folder if it does not exist
        if not os.path.exists(folder_location): os.makedirs(folder_location)
                        
        # Load the shape file
        chrom_definition_config_file = f'../assets/chrom_definition_shapes.json'
        if not os.path.exists(chrom_definition_config_file):
            with open('../assets/chrom_definition_shape.json', 'r') as json_file:
                chrom_definition_shape = json.load(json_file)
        else:
            with open(chrom_definition_config_file, 'r') as json_file:
                chrom_definition_shape = json.load(json_file)
                
        # Update the shape of chrom_definition
        chrom_definition_shape[chromosome] = chrom_definition.shape
        
        # Save the updated shape to the json file
        with open(chrom_definition_config_file, 'w') as json_file: # Export chrom_definition_shape to a JSON file
            json.dump(chrom_definition_shape, json_file, indent=4)
            
        print("Shape of chrom_definition: ", chrom_definition.shape)
        print(f"Chrom definition shape exported to {chrom_definition_config_file}")
        
        # ---------------------------
        # Save chrom_definition as a numpy
        # ---------------------------
        # np.save(f'{sample_type_folder_location}/chrom_definition_{chromosome}.npy', chrom_definition)
        
        # ---------------------------
        # Save chrom_definition as a numpy memmap
        # ---------------------------
        chrom_definition_memmap = np.memmap(chrom_definition_file, dtype=np.int8, mode='w+', shape=chrom_definition.shape)
        assert (isinstance(chrom_definition_memmap, np.ndarray) == True)

        chrom_definition_memmap[:] = chrom_definition[:]
        chrom_definition_memmap.flush()  # flushes changes to disk # del chrom_definition_memmap
        
        print("Chrom definition memmap saved to ", chrom_definition_file)
            
    print("[Complete] Chromosome definition generation complete!!!")
            
# --------------------------------------------------------

def main():
    print ("Process ID: ", os.getpid())

    generate_exclusive_gene_definitions()
    generate_exclusive_gene_index()
    generate_exclusive_geneome_assembly_indexed()
    generate_exclusive_chrom_definitions()

if __name__ == "__main__":
    main()
    
# nohup python -u generate_exclusive_meta_files.py > generate_metadata.txt 2>&1 &
# pkill -f "python -u generate_exclusive_meta_files.py"
# pgrep -f "generate_exclusive_meta_files.py"

