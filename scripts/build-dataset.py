import argparse
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


def process(elem):
    input_tensors = []
    output_tensors = []
    mask_tensors = []
    for input_name in elem["inputs"]:
        masked_input = np.multiply(
            elem["inputs"][input_name]["matrix"],
            elem["inputs"][input_name]["mask"]
        )
        input_tensors.append(masked_input)
        mask_tensors.append(elem["inputs"][input_name]["mask"])
    
    for output_name in elem["outputs"]:
        masked_output = np.multiply(
            elem["outputs"][output_name]["matrix"],
            elem["outputs"][output_name]["mask"]
        )
        output_tensors.append(masked_output)
        mask_tensors.append(elem["outputs"][output_name]["mask"])

    mask_tensor = mask_tensors[0]
    for test in mask_tensors:
        assert np.all(test == mask_tensor), "different masks"
    
    return np.stack(input_tensors), np.stack(output_tensors), mask_tensor


parser = argparse.ArgumentParser()

parser.add_argument("--input", default="./dataset/dataset/processed_dataset.pkl")
parser.add_argument("--output", default="./dataset/dataset/dataset.pkl")

args = parser.parse_args()

data = pickle.load(open(args.input, "rb"))
X = []
y = []
mask = []

for elem in data:
    X_elem, y_elem, mask_elem = process(elem)
    
    X.append(X_elem)
    y.append(y_elem)
    mask.append(mask_elem)

pickle.dump(dict(X=X, y=y, mask=mask), open(args.output, "wb"))
print("All done")