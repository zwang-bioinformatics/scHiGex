# -------------------------------------------------- #
# Author: Bishal Shrestha
# -------------------------------------------------- #

import os
import sys
import json
from safetensors import safe_open

def generate_datamap(cell_type):
    
    cell_types_sample_index = json.load(open(f'../assets/cell_types_sample_index.json', 'r'))
    
    loc = os.path.join("../assets/edge_features", cell_type)
    
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
    
    return file_list
        
                
                
if __name__ == "__main__":
    data_map = {}
    cell_types_sample_index = json.load(open(f'../assets/cell_types_sample_index.json', 'r'))
    cell_types = list(cell_types_sample_index.keys())
    for cell_type in cell_types:
        data_map[cell_type] = generate_datamap(cell_type)
    json.dump(data_map, open('../assets/data_map.json', 'w'))
    print("Done!")
