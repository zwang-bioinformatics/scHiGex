# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import logging  # Import the logging module

# Initialize the logging system
logging.basicConfig(filename='../logs/chromHMM.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def readChromHMM(chrom_size):
    logging.info("Reading the ChromHMM mm10_100_segments_browser.bed file as dataframe")
    
    bed_file_path = "../assets/mm10_100_segments_browser.bed"   # Specify the path to your BED file

    chromosomes = list(chrom_size.keys())

    # Define the column names for the BED file
    columns = ["chromosome", "start", "end", "name", "score", "strand", "thick_start", "thick_end", "item_rgb"]

    # Read the BED file into a DataFrame, skipping the first row
    bed_df = pd.read_csv(bed_file_path, sep='\t', header=None, names=columns, skiprows=1)

    # Filter rows based on the "chromosome" column
    filtered_bed_df = bed_df[bed_df['chromosome'].isin(chromosomes)]

    # Print the first few rows of the DataFrame
    return filtered_bed_df


def index_chromHMM(chromHMM_df, chrom_size):
    logging.info("Creating dictionary of chromosomes where each contains ['', '', '255,245,255', ...]")
    chromosomes = list(chrom_size.keys())
    chromHMM_definitions = {}

    for chromosome in chromosomes:
        chromHMM_definitions[chromosome] = np.full(chrom_size[chromosome], "", dtype=object)  # Use object dtype

    for _, row in chromHMM_df.iterrows():
        annotation_segment_range = range(row['start'], row['end'])  # end is not inclusive
        chrN = row['chromosome']

        chromHMM_definitions[chrN][annotation_segment_range] = row['item_rgb']

    # Save chromHMM_definitions as a pickle file
    output_file_path = "../assets/chromHMM_definitions.pkl"
    with open(output_file_path, 'wb') as file:
        pickle.dump(chromHMM_definitions, file)
    logging.info(f"Saved chromHMM_definitions as a pickle file: {output_file_path}")

    return chromHMM_definitions


def calculate_average_rgb(input_list, start_index, end_index):

    # Initialize variables to store the sum and count of RGB values
    sum_rgb = [0, 0, 0]
    count = 0

    # Iterate over the specified range of indices
    for i in range(start_index, end_index + 1):
        rgb_string = input_list[i]
        if rgb_string:
            rgb_values = [int(x) for x in rgb_string.split(',')]
            sum_rgb = [sum(x) for x in zip(sum_rgb, rgb_values)]
            count += 1

    # Calculate the average RGB values
    if count > 0:
        average_rgb = [float(x / count) for x in sum_rgb]
    else:
        average_rgb = [0, 0, 0]

    return average_rgb


def main():

    # Read the chrom size json file
    with open('../assets/chrom_size.json', 'r') as json_file:
        chrom_size = json.load(json_file)
    
    if os.path.exists("../assets/chromHMM_definitions.pkl"):
        with open("../assets/chromHMM_definitions.pkl", 'rb') as file:
            chromHMM_definitions = pickle.load(file)
    else:
        chromHMM_df = readChromHMM(chrom_size)
        chromHMM_definitions = index_chromHMM(chromHMM_df, chrom_size)
    
    # Read the gene index json file
    with open(f'../assets/exclusive_gene_index.json', 'r') as json_file:
        gene_index = json.load(json_file)

    # Read the gene definition json file
    with open(f'../assets/exclusive_gene_definitions.json', 'r') as json_file:
        gene_definitions = json.load(json_file)

    chromosomes = list(chrom_size.keys())
    
    logging.info("Generating rgb node feature each chromosome...")
    for chromosome in chromosomes:
        logging.info("-"*30)
        total_genes = len(gene_index[chromosome].keys())
        logging.info(f"For chromosome {chromosome} of Length: {total_genes}")
        chromHMM_definition = chromHMM_definitions[chromosome]
        genes_chromHMM_list = list(gene_index[chromosome].keys())    # for each chromosome

        # Checking if the index is in correct order
        for i, gene in enumerate(genes_chromHMM_list):
            if (i != gene_index[chromosome][gene]):
                logging.error("[ERROR] Indexing mismatch")
                sys.exit()

        for i, gene in enumerate(genes_chromHMM_list):
            logging.info(f"\t> [{i+1}/{total_genes}]")
            gene_start_index = gene_definitions[chromosome][gene]["start"] - 1
            gene_end_index = gene_definitions[chromosome][gene]["end"] - 1

            genes_chromHMM_list[i] = calculate_average_rgb(chromHMM_definition, gene_start_index, gene_end_index) # returns average [r,g,b]
            genes_chromHMM_list[i] = [element / 255.0 for element in genes_chromHMM_list[i]]    # Scale the RGB values between 0 and 1
        
        logging.info(f"Shape of genes_chromHMM_list: {np.array(genes_chromHMM_list).shape}")
        logging.info(f"genes_chromHMM_list: {np.array(genes_chromHMM_list)}")
        
        if not os.path.exists('../assets/exclusive_chromHMM'): os.makedirs('../assets/exclusive_chromHMM')
        np.save(f'../assets/exclusive_chromHMM/chromHMM_{chromosome}.npy', np.array(genes_chromHMM_list))




if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    execution_time = end - start
    
    days, seconds = divmod(execution_time, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds = (execution_time - int(execution_time)) * 1000

    print("The time of execution of the program is:", (execution_time) * 10**3, "ms")
    print(f"The time of execution of the program is: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds, {int(milliseconds)} ms")
    
    logging.info("The time of execution of the program is: %.2f ms" % (execution_time * 1000))
    logging.info(f"The time of execution of the program is: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds, {int(milliseconds)} ms")

