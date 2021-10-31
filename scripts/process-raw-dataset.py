import argparse
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def process_file(file):
    data = pd.read_csv(file)
    data = data.drop(data.columns[0], axis=1)

    new_column_names = {name: name.replace('"', "").replace(" ", "") for name in data.columns}
    data = data.rename(columns=new_column_names)
    
    data = data[data["k"] == 1]
    max_i = data["i"].max()
    max_j = data["j"].max()
    
    matrix = np.zeros((max_i, max_j))
    mask = np.zeros((max_i, max_j))

    for _, row in data.iterrows():
        matrix[int(row["i"])-1, int(row["j"])-1] = row["f"]
        mask[int(row["i"])-1, int(row["j"])-1] = row["Active"]
        
    return matrix, mask


parser = argparse.ArgumentParser()

parser.add_argument("--folder", default="./dataset/raw_dataset/variable_data")
parser.add_argument("--output-folder", default="./dataset/dataset")
parser.add_argument("--only-head", action="store_true")


args = parser.parse_args()

inputs = ["Recharge", "River", "Wells"]
outputs = ["Flow_Front_Face", "Flow_Lower_Face", "Flow_Right_Face", "Head", "Storage"]

if args.only_head:
    outputs = ["Head"]

print("[!] Performing sanity check")
indices = None
for folder in ["input/"+i for i in inputs]+["output/"+o for o in outputs]:
    folder_indices = [name.split("_")[-1].split(".")[0] for name in os.listdir(os.path.join(args.folder, folder))]
    if indices is None: indices = folder_indices
    else:
        for index in indices:
            if index not in folder_indices:
                raise Exception(f"Missing {index} in {folder}")
print("[!] Sanity check ok")

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

dataset = []
print("[!] Bulding dataset")
for index in tqdm(indices):
    elem = dict(
        inputs=dict(),
        outputs=dict()
    )
    for input_folder in inputs:
        matrix, mask = process_file(os.path.join(args.folder, "input", input_folder, f"{input_folder}_{index}.txt"))
        elem["inputs"][input_folder] = dict(
            matrix=matrix,
            mask=mask
        )
    for output_folder in outputs:
        matrix, mask = process_file(os.path.join(args.folder, "output", output_folder, f"{output_folder}_{index}.txt"))
        elem["outputs"][output_folder] = dict(
            matrix=matrix,
            mask=mask
        )
    
    dataset.append(elem)

pickle.dump(dataset, open(os.path.join(args.output_folder, "processed_dataset.pkl"), "wb"))
print("[!] Done")