# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import math
import json

with open(f'../assets/cell_types_count.json', 'r') as f:
	cell_types_count = json.load(f)

for cell_type in cell_types_count:
    output_folder = f"../assets/metacell/"
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    file = f'../assets/clusters/{cell_type}' # sys.argv[1]
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