import argparse
import pickle
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="./dataset/dataset/processed_dataset.pkl")
parser.add_argument("--field", default="inputs.Recharge")

args = parser.parse_args()

dataset = pickle.load(open(args.dataset, "rb"))

for i, elem in enumerate(dataset):
    key1 = args.field.split(".")[0]
    key2 = args.field.split(".")[1]
    
    data = elem[key1][key2]

    
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(f"{args.field} #{i}")

    axes[0].matshow(data["matrix"])
    axes[0].set_title("data")

    axes[1].matshow(data["mask"])
    axes[1].set_title("active mask")
    
    plt.show()