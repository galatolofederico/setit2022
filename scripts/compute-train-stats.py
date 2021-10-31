import argparse
import pickle
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--dataset-folder", default="./dataset/dataset")

args = parser.parse_args()

train = pickle.load(open(os.path.join(args.dataset_folder, "train.pkl"), "rb"))

y = np.stack(train["y"])
mask = np.stack(train["mask"])
y = y.reshape(y.shape[0], y.shape[1], -1)
mask = mask.reshape(mask.shape[0], -1)
mask = np.repeat(mask[:, np.newaxis, :], y.shape[1], axis=1)

y_mean = np.zeros(y.shape[1])
y_std = np.zeros(y.shape[1])
count = 0
for yy, mm in zip (y, mask):
    count += 1
    for i, (cy, cm) in enumerate(zip(yy, mm)):
        y_mean[i] += cy[cm != 0].mean()
        y_std[i] += cy[cm != 0].std()

y_mean /= count
y_std /= count

print("mean: ", y_mean)
print("std: ", y_std)

json.dump(dict(
    y=dict(
        mean=y_mean.tolist(),
        std=y_std.tolist()
    )
), open(os.path.join(args.dataset_folder, "stats.json"), "w"))