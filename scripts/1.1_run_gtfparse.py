# -------------------------------------------------- #
# Author: Bishal Shrestha
# Last Updated: 5/25/2024
# -------------------------------------------------- #

import os
import sys
from gtfparse import read_gtf
import json
from tqdm import tqdm


# Set the NUMEXPR_MAX_THREADS environment variable
os.environ["NUMEXPR_MAX_THREADS"] = "128"
def read_GRCm38():
    """
    Returns GTF with essential columns such as "feature", "seqname", "start", "end"
    alongside the names of any optional keys which appeared in the attribute column
    """
    print("\n> Reading GRCm38 GTF data and returning dataframe")
    # check if the file exists
    if not os.path.exists("../assets/gencode.vM23.annotation.gtf"):
        print("[ERROR] gencode.vM23.annotation.gtf file not found")
        sys.exit()
    df = read_gtf("../assets/gencode.vM23.annotation.gtf")
    df_genes_GRCm38 = df[df["feature"] == "gene"]
    return df_genes_GRCm38

def main():
    df_genes_GRCm38 = read_GRCm38()
    
    gene_definitions = {}   # Construct the gene_definitions dictionary
    repetation_count = 0

    for index, row in df_genes_GRCm38.iterrows():
        seqname = row['seqname']
        gene_name = row['gene_name']
        start = row['start']
        end = row['end']

        if seqname not in gene_definitions: gene_definitions[seqname] = {}
        if gene_name in gene_definitions[seqname]: repetation_count += 1
        gene_definitions[seqname][gene_name] = {'start': start, 'end': end}

    # Export gene_definitions to a JSON file
    output_file = "../assets/gene_definitions_vM23.json"
    with open(output_file, 'w') as json_file: json.dump(gene_definitions, json_file, indent=4)
    print(f"Gene definitions exported to {output_file}")

if __name__ == "__main__":
    main()

