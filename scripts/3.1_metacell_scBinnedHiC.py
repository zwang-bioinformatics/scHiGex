# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

# Python
import os
import sys
import json
import pandas as pd

def main():
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
    with open(f'../assets/cell_types_sample_index.json', 'r') as f:
        cell_types_sample_index = json.load(f)

    # selected_cell_types = ["mix late mesenchyme", "blood", "mitosis", "ExE endoderm", "early neurons"]
    selected_cell_types = list(cell_types_sample_index.keys())

    for cell_type in selected_cell_types:
        
        output_folder = f"../assets/binnedHiC/" + cell_type + "/"
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        for sample_index in cell_types_sample_index[cell_type]:
            sample_info = cell_types_sample_index[cell_type][sample_index]
            sample_file = "../assets/pairs/"+sample_info[1]

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

    # --- Export DataFrame to CSV ---
    # df_hi_c_pairs.to_csv('../data/temp/pairs.csv', index=False)

    return df_hi_c_pairs


if __name__ == "__main__":
    main()
    
# nohup python scBinnedHiC.py > scBinnedHiC.log 2>&1 &