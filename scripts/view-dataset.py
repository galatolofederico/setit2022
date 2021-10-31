import argparse
import pickle
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="./dataset/dataset/dataset.pkl")

args = parser.parse_args()

dataset = pickle.load(open(args.dataset, "rb"))

for X, y, mask in zip(dataset["X"], dataset["y"], dataset["mask"]):
    fig, axes = plt.subplots(2, max(X.shape[0], y.shape[0]))
    for i, (ax, data) in enumerate(zip(axes[0, :], X)):
        ax.set_title(f"input.{i}")
        ax.matshow(data)
    axes[0, i+1].set_title("mask")
    axes[0, i+1].matshow(mask)

    for i, (ax, data) in enumerate(zip(axes[1, :], y)):
        ax.set_title(f"output.{i}")
        ax.matshow(data)
    
    plt.show()
