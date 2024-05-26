# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

"""
- pip uninstall triton (This is an optional package that is automatically installed from requirements.txt)
"""

import os
import time
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import sys
import multiprocessing
import logging  # Import the logging module

# Initialize the logging system
logging.basicConfig(filename='../logs/dnabert2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Returns a list with chunked string
def split_string_by_chunk(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def process_chromosome(chromosome):

    try:
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
        saved_filename = f'../assets/exclusive_gene_assembly/exclusive_gene_assembly_{chromosome}.npy'
        gene_assembly = np.load(saved_filename, allow_pickle=True)

        all_gene_embedding = []
        
        total_genes = len(gene_assembly)
        logging.info(f"Chromosome: {chromosome}")
        logging.info(f"Total genes in chromosome {chromosome}: {total_genes}")

        model.eval()
        with torch.no_grad(): 
            for index, gene_nucleotide_sequence in enumerate(gene_assembly):
                gene_nucleotide_sequence_chunks = split_string_by_chunk(gene_nucleotide_sequence.upper(), 1000)
                logging.info(f"{chromosome} [index: {index+1}/{total_genes}]")
                logging.info(f"\t> Length of gene_nucleotide_sequence: {len(gene_nucleotide_sequence)}")
                logging.info(f"\t> Number of chunks: {len(gene_nucleotide_sequence_chunks)}")

                # gene_embeddings = np.empty((0, 768))  # Initialize an empty array with 768 columns
                gene_embeddings = torch.empty(0, 768)  # Initialize an empty tensor on GPU
                for idx, chunk in enumerate(gene_nucleotide_sequence_chunks):
                    # print(f"{index}: {idx}")
                    inputs = tokenizer(chunk, return_tensors = 'pt')["input_ids"].to(device)
                    hidden_states = model(inputs)[0] # torch.Size([1, sequence_length, 768])

                    # embedding with mean pooling
                    embedding_mean = torch.mean(hidden_states[0], dim=0)
                    embedding_mean = embedding_mean.unsqueeze(0).cpu()  # Add a dimension along dim=0 => torch.Size([1, 768])

                    # gene_embeddings = np.vstack((gene_embeddings, embedding_mean.detach()))
                    gene_embeddings = torch.cat((gene_embeddings, embedding_mean), dim=0)
                    # torch.cuda.empty_cache()

                    
                # logging.info(f"{chromosome} [index: {index+1}/{total_genes}]")
                logging.info(f"\t> Shape of gene_embeddings: {gene_embeddings.shape}")   # torch.Size([# of chunk, 768])

                # Calculate the mean across columns
                # average_embedding = np.mean(gene_embeddings, axis=0)
                average_embedding = torch.mean(gene_embeddings, dim=0)

                logging.info(f"\t> Embedding shape after averaging over column: {average_embedding.shape}")  # torch.Size([768])

                all_gene_embedding.append(average_embedding.cpu().tolist()) 

        logging.info(f"\t all_gene_embedding {chromosome} shape: {np.array(all_gene_embedding).shape}")

        # Check for errors
        if (len(gene_assembly) != len(all_gene_embedding)):
            logging.error(f"[ERROR] Chromosome {chromosome} has uneven gene number for embeddings list.")
            sys.exit()

        # print(f"\t Emedding generation for {chromosome} completed!")
        logging.info(f"Embedding generation for {chromosome} completed!")

        # Save gene_embeddings list to a file
        if not os.path.exists('../assets/exclusive_dnabert2'): os.makedirs('../assets/exclusive_dnabert2')
        np.save(f'../assets/exclusive_dnabert2/dnabert2_{chromosome}.npy', np.array(all_gene_embedding))
        logging.info(f"Embeddings for chromosome {chromosome} saved to file: exclusive_dnabert2/dnabert_{chromosome}.npy")
    
    except Exception as e:
        logging.error(f"An error occurred in process_chromosome for chromosome {chromosome}: \n{str(e)}")
        print(f"An error occurred in process_chromosome for chromosome {chromosome}: \n{str(e)}")
        sys.exit()


def main():
    try:
        # Read the gene definition json file
        with open('../assets/chrom_size.json', 'r') as json_file:
            chrom_size = json.load(json_file)

        chromosomes = chrom_size.keys()

        # # Use multiprocessing to process chromosomes in parallel
        # num_processes = 1#multiprocessing.cpu_count()  # You can adjust this as needed
        # pool = multiprocessing.Pool(processes=num_processes)
        # # Process each chromosome in parallel
        # pool.map(process_chromosome, chromosomes)
        # pool.close()
        # pool.join()

        for chromosome in chromosomes:
            process_chromosome(chromosome)
    
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")



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

# Run command: python dnabert2.py all 2>&1 | tee ../logs/dnabert2.txt
